import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - 1. 110M Yoğun Model Yapılandırması (ModelConfig)

/// Apple Silicon (M5 - 24GB Unified Memory) üzerinde sıfırdan eğitilecek
/// 110M parametre sınıfı yoğun (dense) Decoder-only Transformer modelinin hiperparametrelerini tanımlar.
/// Swift 6 Strict Concurrency uyumluluğu için `Sendable` protokolünü uygular.
public struct ModelConfig: Sendable, Codable, Equatable {
    /// Sözlük boyutu (Varsayılan: 32,000)
    public let vocabSize: Int
    /// Gizli katman boyutu (d_model, Varsayılan: 768)
    public let hiddenSize: Int
    /// SwiGLU MLP ara katman boyutu (~8/3 * hiddenSize, Varsayılan: 2,048)
    public let intermediateSize: Int
    /// Transformer blok (katman) sayısı (Varsayılan: 12)
    public let numLayers: Int
    /// Sorgu (Query) dikkat başlığı sayısı (n_heads, Varsayılan: 12 -> Head Dim: 64)
    public let numHeads: Int
    /// Anahtar/Değer (Key/Value) dikkat başlığı sayısı (GQA 3:1, Varsayılan: 4)
    public let numKVHeads: Int
    /// Maksimum bağlam uzunluğu (Sequence Length, Varsayılan: 512)
    public let maxSequenceLength: Int
    /// RMSNorm sayısal kararlılık sabiti (epsilon)
    public let rmsNormEps: Float
    /// RoPE (Rotary Position Embedding) frekans tabanı (theta)
    public let ropeTheta: Float
    /// Giriş Embedding ile Çıkış LM Head ağırlıklarının paylaşılıp paylaşılmayacağı (Weight Tying)
    public let tieWordEmbeddings: Bool

    /// Her bir dikkat başlığının vektör boyutu: d_k = hiddenSize / numHeads (110M varsayılanı: 64)
    public var headDim: Int {
        hiddenSize / numHeads
    }

    public init(
        vocabSize: Int = 32_000,
        hiddenSize: Int = 768,
        intermediateSize: Int = 2_048,
        numLayers: Int = 12,
        numHeads: Int = 12,
        numKVHeads: Int = 4,
        maxSequenceLength: Int = 512,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000.0,
        tieWordEmbeddings: Bool = false
    ) {
        precondition(hiddenSize % numHeads == 0, "hiddenSize (\(hiddenSize)), numHeads (\(numHeads)) değerine tam bölünmelidir.")
        precondition(numHeads % numKVHeads == 0, "numHeads (\(numHeads)), numKVHeads (\(numKVHeads)) değerine tam bölünmelidir.")
        precondition((hiddenSize / numHeads) % 2 == 0, "RoPE uygulaması için headDim çift sayı olmalıdır.")

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.maxSequenceLength = maxSequenceLength
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    /// 110M Yoğun (Dense) Mimari Profili (32.000 sözlük, 12 katman, 768 gizli boyut, 12/4 GQA, 512 bağlam)
    public static func dense110M(vocabSize: Int = 32_000, maxSequenceLength: Int = 512) -> ModelConfig {
        return ModelConfig(
            vocabSize: vocabSize,
            hiddenSize: 768,
            intermediateSize: 2_048,
            numLayers: 12,
            numHeads: 12,
            numKVHeads: 4,
            maxSequenceLength: maxSequenceLength
        )
    }
}

// MARK: - 2. True Mixed Precision (.float32 Master Weights + .bfloat16 Compute) Ağırlık Başlatıcı

/// Dışarıdan hiçbir ağırlık yüklemeden, modelin ana ağırlıklarını (Master Weights)
/// **Float32 (`.float32`)** hassasiyetinde matematiksel dağılımlarla başlatır.
/// İleri ve geri yayılım matris çarpımları katmanlar içinde `.bfloat16` olarak yürütülür.
public enum WeightInitializer: Sendable {

    /// **Xavier (Glorot) Normal Başlatma (Varsayılan: `.float32` Master Weight)**
    public static func xavierNormal(
        shape: [Int],
        fanIn: Int,
        fanOut: Int,
        gain: Float = 1.0,
        dtype: DType = .float32
    ) -> MLXArray {
        let std = gain * sqrt(2.0 / Float(fanIn + fanOut))
        let scale = MLXArray(std).asType(dtype)
        return MLXRandom.normal(shape, dtype: dtype) * scale
    }

    /// **Kaiming (He) Normal Başlatma (Varsayılan: `.float32` Master Weight)**
    public static func kaimingNormal(
        shape: [Int],
        fanIn: Int,
        gain: Float = sqrt(2.0),
        dtype: DType = .float32
    ) -> MLXArray {
        let std = gain / sqrt(Float(fanIn))
        let scale = MLXArray(std).asType(dtype)
        return MLXRandom.normal(shape, dtype: dtype) * scale
    }

    /// **Derinlik Ölçekli Residual (Artık Bağlantı) Başlatma (Varsayılan: `.float32` Master Weight)**
    public static func residualScaledNormal(
        shape: [Int],
        fanIn: Int,
        numLayers: Int,
        dtype: DType = .float32
    ) -> MLXArray {
        let baseStd = 1.0 / sqrt(Float(fanIn))
        let depthScale = 1.0 / sqrt(2.0 * Float(numLayers))
        let std = baseStd * depthScale
        let scale = MLXArray(std).asType(dtype)
        return MLXRandom.normal(shape, dtype: dtype) * scale
    }

    /// **Embedding Başlatma (Varsayılan: `.float32` Master Weight)**
    public static func embeddingNormal(
        vocabSize: Int,
        hiddenSize: Int,
        dtype: DType = .float32
    ) -> MLXArray {
        let std = 1.0 / sqrt(Float(hiddenSize))
        let scale = MLXArray(std).asType(dtype)
        return MLXRandom.normal([vocabSize, hiddenSize], dtype: dtype) * scale
    }
}
