import Foundation
import MLX
import MLXRandom

// MARK: - ModelConfig (Mimari Hiperparametre Yapılandırması)

/// `PersonalFoundation` sıfırdan (from-scratch) eğitilecek Decoder-only Transformer
/// mimarisinin tüm boyutlarını ve matematiksel hiperparametrelerini tanımlayan yapı.
///
/// Swift 6 Strict Concurrency kurallarına tam uyum için `Sendable` protokolünü uygular
/// ve tamamen değer tipi (value-type / immutable) olarak tasarlanmıştır.
public struct ModelConfig: Sendable, Codable, Equatable {
    /// Kelime dağarcığı boyutu (Vocabulary Size - |V|).
    public let vocabSize: Int

    /// Gizli katman / gömme vektör boyutu (Hidden Dimension - d_model).
    public let hiddenSize: Int

    /// SwiGLU FeedForward (MLP) katmanı ara boyutu (Intermediate Dimension - d_ff).
    /// Genellikle `(8/3) * hiddenSize` değerinin 256'nın katına yuvarlanmış halidir.
    public let intermediateSize: Int

    /// Ardışık Transformer bloğu sayısı (Number of Decoder Layers - L).
    public let numLayers: Int

    /// Sorgu (Query) dikkat başlığı sayısı (Number of Attention Heads - H_q).
    public let numHeads: Int

    /// Anahtar/Değer (Key/Value) dikkat başlığı sayısı (Grouped-Query Attention için H_kv).
    /// `numKVHeads == numHeads` ise standart Multi-Head Attention (MHA),
    /// `numKVHeads < numHeads` ise bellek dostu Grouped-Query Attention (GQA) çalışır.
    public let numKVHeads: Int

    /// Maksimum bağlam penceresi uzunluğu (Maximum Sequence Length - T_max).
    public let maxSequenceLength: Int

    /// RMSNorm sayısal kararlılık sabiti (Epsilon - \epsilon).
    public let rmsNormEps: Float

    /// Rotary Position Embedding (RoPE) frekans tabanı (\theta_{base}).
    public let ropeTheta: Float

    /// Giriş Token Embedding ağırlıkları ile çıkış LM Head ağırlıklarının paylaşılıp paylaşılmayacağı.
    /// Apple Silicon 24GB birleşik bellekte (Unified Memory) parametre tasarrufu sağlar.
    public let tieWordEmbeddings: Bool

    /// Ağırlık başlatma (Initialization) için rastgele tohum (Seed).
    public let initSeed: UInt64

    /// Her bir dikkat başlığının vektör boyutu: `d_head = hiddenSize / numHeads`.
    public var headDim: Int {
        hiddenSize / numHeads
    }

    public init(
        vocabSize: Int = 32_000,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 2816,
        numLayers: Int = 16,
        numHeads: Int = 16,
        numKVHeads: Int = 4,
        maxSequenceLength: Int = 2048,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000.0,
        tieWordEmbeddings: Bool = false,
        initSeed: UInt64 = 42
    ) {
        precondition(hiddenSize % numHeads == 0, "hiddenSize (\(hiddenSize)), numHeads (\(numHeads)) değerine tam bölünmelidir.")
        precondition(numHeads % numKVHeads == 0, "numHeads (\(numHeads)), numKVHeads (\(numKVHeads)) değerine tam bölünmelidir.")
        precondition((hiddenSize / numHeads) % 2 == 0, "RoPE rotasyonu için headDim çift sayı olmalıdır.")

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
        self.initSeed = initSeed
    }

    /// Apple Silicon M5 (24GB Unified Memory) üzerinde hızlı prototipleme ve
    /// sıfırdan ön-eğitim doğrulaması için ~25M parametreli mikro yapılandırma.
    public static let microM5 = ModelConfig(
        vocabSize: 16_384,
        hiddenSize: 512,
        intermediateSize: 1536,
        numLayers: 8,
        numHeads: 8,
        numKVHeads: 4,
        maxSequenceLength: 1024
    )

    /// Apple Silicon M5 (24GB Unified Memory) üzerinde tam kapsamlı sıfırdan eğitim için
    /// ~350M parametreli dengeli yapılandırma.
    public static let smallM5 = ModelConfig(
        vocabSize: 32_000,
        hiddenSize: 1024,
        intermediateSize: 2816,
        numLayers: 24,
        numHeads: 16,
        numKVHeads: 4,
        maxSequenceLength: 2048
    )
}

// MARK: - WeightInitializer (Matematiksel Sıfırdan Ağırlık Başlatıcı)

/// Dışarıdan hazır ağırlık (safetensors/HuggingFace) yüklemek yerine,
/// modelin tüm tensörlerini sıfırdan matematiksel dağılımlarla (Xavier/Glorot, Kaiming/He ve GPT-2/Depth-Scaled Residual)
/// başlatan saf matematiksel yardımcı yapı.
public enum WeightInitializer {
    /// Başlatma stratejisi türleri.
    public enum Strategy: Sendable {
        /// **Xavier / Glorot Normal Başlatma**
        /// Doğrusal projeksiyonlar ve dikkat (Q, K, V) katmanlarında ileri ve geri yayılım varyansını korur:
        /// $$\sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan}_{in} + \text{fan}_{out}}}$$
        case xavierNormal(gain: Float = 1.0)

        /// **Xavier / Glorot Uniform Başlatma**
        /// $$a = \text{gain} \times \sqrt{\frac{6}{\text{fan}_{in} + \text{fan}_{out}}}, \quad W \sim \mathcal{U}(-a, a)$$
        case xavierUniform(gain: Float = 1.0)

        /// **Kaiming / He Normal Başlatma**
        /// Doğrusal olmayan (SwiGLU / SiLU) aktivasyon katmanlarında sinyal sönümlenmesini önler:
        /// $$\sigma = \frac{\text{gain}}{\sqrt{\text{fan}_{in}}}$$
        /// SwiGLU / SiLU için önerilen efektif kazanç (gain) $\approx \sqrt{2}$'dir.
        case kaimingNormal(gain: Float = sqrt(2.0))

        /// **Derinlik Ölçekli Artık (Residual Depth-Scaled) Başlatma**
        /// Derin Transformer ağlarında artık bağlantı (residual stream: $x + \text{SubLayer}(x)$) boyunca
        /// varyansın katman sayısı $L$ ile patlamasını önlemek için çıkış projeksiyonlarını ($W_o, W_{down}$)
        /// $\frac{1}{\sqrt{2 \cdot L}}$ çarpanıyla ölçekler:
        /// $$\sigma = \frac{1}{\sqrt{\text{fan}_{in}} \cdot \sqrt{2 \cdot L}}$$
        case residualScaledKaiming(numLayers: Int)

        /// **Token Embedding Başlatma**
        /// Standart normal dağılımdan $\sigma = \frac{1}{\sqrt{d_{model}}}$ ile örneklenir.
        case embedding(hiddenSize: Int)
    }

    /// Verilen boyut (`shape`) ve matematiksel stratejiye göre sıfırdan `MLXArray` ağırlık tensörü üretir.
    ///
    /// - Parameters:
    ///   - shape: Tensör boyutları (Örn: `[outFeatures, inFeatures]` veya `[vocabSize, hiddenSize]`).
    ///   - strategy: Uygulanacak matematiksel varyans koruma formülü.
    ///   - dtype: Tensör veri tipi (Varsayılan: `.float32`; M5 üzerinde karma duyarlıklı eğitim için `.bfloat16` seçilebilir).
    /// - Returns: Matematiksel olarak başlatılmış `MLXArray`.
    public static func initialize(
        shape: [Int],
        strategy: Strategy,
        dtype: DType = .float32
    ) -> MLXArray {
        precondition(!shape.isEmpty, "Tensör boyutu boş olamaz.")

        // 2D ağırlık matrisleri [outFeatures, inFeatures] veya [inFeatures, outFeatures] için fan_in ve fan_out hesabı
        let fanOut = shape[0]
        let fanIn = shape.count > 1 ? shape[1] : shape[0]

        switch strategy {
        case .xavierNormal(let gain):
            let std = gain * sqrt(2.0 / Float(fanIn + fanOut))
            return MLXRandom.normal(shape, dtype: dtype) * MLXArray(std).asType(dtype)

        case .xavierUniform(let gain):
            let bound = gain * sqrt(6.0 / Float(fanIn + fanOut))
            return MLXRandom.uniform(low: -bound, high: bound, shape, dtype: dtype)

        case .kaimingNormal(let gain):
            let std = gain / sqrt(Float(fanIn))
            return MLXRandom.normal(shape, dtype: dtype) * MLXArray(std).asType(dtype)

        case .residualScaledKaiming(let numLayers):
            let baseStd = 1.0 / sqrt(Float(fanIn))
            let depthScale = 1.0 / sqrt(2.0 * Float(max(numLayers, 1)))
            let std = baseStd * depthScale
            return MLXRandom.normal(shape, dtype: dtype) * MLXArray(std).asType(dtype)

        case .embedding(let hiddenSize):
            let std = 1.0 / sqrt(Float(hiddenSize))
            return MLXRandom.normal(shape, dtype: dtype) * MLXArray(std).asType(dtype)
        }
    }
}
