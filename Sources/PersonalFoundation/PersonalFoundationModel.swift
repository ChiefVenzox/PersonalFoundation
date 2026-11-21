import Foundation
import MLX
import MLXNN

// MARK: - 1. CustomLayerNorm (RMSNorm - Root Mean Square Layer Normalization)

/// **RMSNorm (Kök Ortalama Kare Katman Normalizasyonu)**
///
/// Standart LayerNorm'dan farklı olarak ortalama çıkarma ($\mu$ merkezleme) işlemini atlar ve
/// aktivasyonları yalnızca karelerinin ortalamasının karekökü (RMS) ile ölçekler.
/// Bu sayede Apple Silicon GPU üzerinde hem hesaplama yükü (%7-%15) azalır hem de gradyan akışı kararlılığı korunur.
///
/// **Matematiksel Formül:**
/// $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$
/// $$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$
///
/// Burada $\gamma \in \mathbb{R}^{d}$ öğrenilebilir ölçek (weight) parametresidir ve başlangıçta $\mathbf{1}$ vektörüdür.
public final class CustomRMSNorm: Module, UnaryLayer {
    /// Öğrenilebilir ölçek vektörü ($\gamma \in \mathbb{R}^{d_{model}}$).
    public var weight: MLXArray

    /// Sıfıra bölünmeyi önleyen sayısal kararlılık sabiti ($\epsilon$).
    public let eps: Float

    /// Sıfırdan birim ($\mathbf{1}$) ağırlıklarla RMSNorm katmanı oluşturur.
    /// - Parameters:
    ///   - dimensions: Normalizasyon yapılacak son eksenin boyutu ($d_{model}$).
    ///   - eps: Sayısal kararlılık sabiti (Varsayılan: `1e-5`).
    public init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        // Başlangıçta tüm ölçek parametreleri 1.0 olarak başlatılır.
        self.weight = MLXArray.ones([dimensions], dtype: .float32)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Sayısal taşmayı (overflow/underflow) önlemek için ara RMS hesabını Float32 uzayında yapıyoruz.
        let originalDType = x.dtype
        let xFp32 = x.asType(.float32)

        // 1. Her elemanın karesini al: x_i^2
        let squared = xFp32.square()

        // 2. Son eksen (hiddenDimension) boyunca aritmetik ortalamayı hesapla: (1/d) * \sum x_i^2
        let meanSquared = mean(squared, axis: -1, keepDims: true)

        // 3. Ters karekök (Reciprocal Square Root): 1 / \sqrt{meanSquared + \epsilon}
        // rsqrt operatörü Apple Metal GPU üzerinde tek bir donanım komutuyla çalışır.
        let invRMS = rsqrt(meanSquared + MLXArray(eps))

        // 4. Normalize et, orijinal veri tipine döndür ve öğrenilebilir \gamma (weight) ile çarp
        let normalized = (xFp32 * invRMS).asType(originalDType)
        return normalized * weight.asType(originalDType)
    }
}

// MARK: - 2. CustomLinear (Sıfırdan Kaiming / Xavier Başlatmalı Doğrusal Projeksiyon)

/// **CustomLinear (Özel Doğrusal Katman)**
///
/// Hazır ağırlık yüklemek yerine kendi `WeightInitializer` stratejimizle (Xavier veya Kaiming)
/// sıfırdan başlatılan, $y = x W^T + b$ dönüşümünü gerçekleştiren temel matris çarpım katmanı.
/// Modern Foundation modellerde (Pre-Norm + SwiGLU) genellikle `bias` kullanılmaz (`useBias = false`).
public final class CustomLinear: Module, UnaryLayer {
    /// Ağırlık matrisi: $W \in \mathbb{R}^{d_{out} \times d_{in}}$.
    public var weight: MLXArray

    /// İsteğe bağlı sapma (bias) vektörü: $b \in \mathbb{R}^{d_{out}}$.
    public var bias: MLXArray?

    public init(
        inFeatures: Int,
        outFeatures: Int,
        useBias: Bool = false,
        initStrategy: WeightInitializer.Strategy = .xavierNormal()
    ) {
        self.weight = WeightInitializer.initialize(
            shape: [outFeatures, inFeatures],
            strategy: initStrategy
        )
        if useBias {
            self.bias = MLXArray.zeros([outFeatures], dtype: .float32)
        } else {
            self.bias = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [Batch, SeqLen, inFeatures], weight.T: [inFeatures, outFeatures]
        var output = matmul(x, weight.T)
        if let bias {
            output = output + bias
        }
        return output
    }
}

// MARK: - 3. RotaryPositionEmbedding (Sıfırdan RoPE Matematiksel İmplementasyonu)

/// **Rotary Position Embedding (RoPE - Döngüsel Konum Kodlaması)**
///
/// Sorgu ($Q$) ve Anahtar ($K$) vektörlerini, dizideki mutlak konumlarına ($m$) bağlı olarak
/// 2 boyutlu karmaşık düzlem alt-uzaylarında döndürür. Böylece iki konum ($m$ ve $n$) arasındaki
/// iç çarpım ($\langle q_m, k_n \rangle$) yalnızca göreceli mesafeye ($m - n$) bağlı olur.
///
/// **Matematiksel Formül:**
/// Her $i \in \{0, 1, \dots, d_{head}/2 - 1\}$ frekans çifti için açısal frekans:
/// $$\theta_i = \theta_{base}^{-2i / d_{head}}$$
///
/// Konum $m$ için $x = [x_1, x_2]$ ("yarım-bölünmüş" koordinat çiftleri) üzerindeki 2D rotasyon:
/// $$\text{RoPE}(x, m) = \begin{bmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_2 \cos(m\theta) + x_1 \sin(m\theta) \end{bmatrix}$$
public final class RotaryPositionEmbedding: Module {
    public let headDim: Int
    public let maxSequenceLength: Int
    public let base: Float

    /// Önceden hesaplanmış $\cos(m\theta_i)$ tablosu: `[maxSeqLen, headDim / 2]` (Eğitilmez / Non-trainable).
    private let cosCached: MLXArray

    /// Önceden hesaplanmış $\sin(m\theta_i)$ tablosu: `[maxSeqLen, headDim / 2]` (Eğitilmez / Non-trainable).
    private let sinCached: MLXArray

    public init(headDim: Int, maxSequenceLength: Int, base: Float = 10_000.0) {
        precondition(headDim % 2 == 0, "RoPE için headDim (\(headDim)) çift sayı olmalıdır.")
        self.headDim = headDim
        self.maxSequenceLength = maxSequenceLength
        self.base = base

        let halfDim = headDim / 2
        // 1. Frekans üslerini hesapla: \theta_i = 1.0 / (base^(2i / headDim))
        var invFreqValues = [Float](repeating: 0.0, count: halfDim)
        for i in 0..<halfDim {
            let exponent = Float(2 * i) / Float(headDim)
            invFreqValues[i] = 1.0 / pow(base, exponent)
        }
        let invFreq = MLXArray(invFreqValues).reshaped([1, halfDim]) // [1, headDim/2]

        // 2. Zaman/konum adımları vektörü: m \in [0, 1, ..., maxSequenceLength - 1]
        let positions = MLXArray(Array(0..<maxSequenceLength).map { Float($0) })
            .reshaped([maxSequenceLength, 1]) // [maxSeqLen, 1]

        // 3. Dış çarpım (Outer product) ile açılar matrisi: freqs = m * \theta_i -> [maxSeqLen, headDim/2]
        let angles = matmul(positions, invFreq)

        // 4. Kosinüs ve Sinüs önbelleklerini oluştur
        self.cosCached = cos(angles)
        self.sinCached = sin(angles)

        super.init()
        // Önbellek tensörlerinin gradyan takibine (trainableParameters) girmesini engelle
        self.freeze()
    }

    /// Giriş tensörüne (`[Batch, NumHeads, SeqLen, HeadDim]`) sıfırdan RoPE rotasyonu uygular.
    /// - Parameters:
    ///   - x: Sorgu ($Q$) veya Anahtar ($K$) tensörü `[B, H, T, D]`.
    ///   - offset: KV-Cache çıkarımı veya uzun bağlam kaydırması için başlangıç konumu (Varsayılan: `0`).
    /// - Returns: Konumsal olarak döndürülmüş `[B, H, T, D]` tensörü.
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(2)
        let halfDim = headDim / 2

        // İlgili [offset ..< offset + seqLen] aralığındaki cos ve sin dilimlerini al -> [1, 1, T, D/2]
        let cosSlice = cosCached[offset..<(offset + seqLen), 0..<halfDim]
            .reshaped([1, 1, seqLen, halfDim])
            .asType(x.dtype)
        let sinSlice = sinCached[offset..<(offset + seqLen), 0..<halfDim]
            .reshaped([1, 1, seqLen, halfDim])
            .asType(x.dtype)

        // Vektörü son eksende iki yarıya böl: x1 = x[..., :D/2], x2 = x[..., D/2:]
        let x1 = x[0..., 0..., 0..., 0..<halfDim]
        let x2 = x[0..., 0..., 0..., halfDim..<headDim]

        // 2D Karmaşık düzlem rotasyonu:
        // rotated_x1 = x1 * cos - x2 * sin
        // rotated_x2 = x2 * cos + x1 * sin
        let rotatedX1 = (x1 * cosSlice) - (x2 * sinSlice)
        let rotatedX2 = (x2 * cosSlice) + (x1 * sinSlice)

        // İki yarıyı tekrar [B, H, T, D] boyutunda birleştir
        return concatenated([rotatedX1, rotatedX2], axis: -1)
    }
}

// MARK: - 4. CustomMultiHeadAttention (RoPE & Causal Mask Destekli Dikkat Mekanizması)

/// **CustomMultiHeadAttention (RoPE Destekli Çok Başlıklı / Gruplu-Sorgulu Öz-Dikkat)**
///
/// Giriş dizisindeki her token'ın kendisinden önceki token'larla (Causal Autoregressive Masking)
/// bağlamsal ilişkisini hesaplar. Hem standart Multi-Head Attention ($H_q = H_{kv}$) hem de
/// Apple Silicon RAM bant genişliğini optimize eden Grouped-Query Attention ($H_q > H_{kv}$) destekler.
///
/// **Matematiksel Akış:**
/// 1. **Projeksiyonlar:** $Q = X W_Q^T, \quad K = X W_K^T, \quad V = X W_V^T$
/// 2. **RoPE Uygulaması:** $\tilde{Q} = \text{RoPE}(Q), \quad \tilde{K} = \text{RoPE}(K)$
/// 3. **Ölçekli Nokta Çarpım Dikkati (Scaled Dot-Product Attention):**
///    $$\text{Attention}(\tilde{Q}, \tilde{K}, V) = \text{softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_{head}}} + M_{causal}\right) V$$
///    Burada $M_{causal}$ üst üçgen elemanları $-\infty$ (veya `-1e9`), alt üçgen elemanları $0$ olan nedensel maskedir.
/// 4. **Çıkış Projeksiyonu:** $O = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O^T$
public final class CustomMultiHeadAttention: Module {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float

    /// Sorgu (Query) projeksiyonu: $d_{model} \to H_q \cdot d_{head}$ (Xavier Normal).
    public let qProj: CustomLinear
    /// Anahtar (Key) projeksiyonu: $d_{model} \to H_{kv} \cdot d_{head}$ (Xavier Normal).
    public let kProj: CustomLinear
    /// Değer (Value) projeksiyonu: $d_{model} \to H_{kv} \cdot d_{head}$ (Xavier Normal).
    public let vProj: CustomLinear
    /// Çıkış (Output) projeksiyonu: $H_q \cdot d_{head} \to d_{model}$ (Residual Depth-Scaled Kaiming).
    public let oProj: CustomLinear

    /// Sıfırdan kodlanmış Rotary Position Embedding modülü.
    public let rope: RotaryPositionEmbedding

    public init(config: ModelConfig) {
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let qDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim

        // Q, K, V ağırlıkları Xavier Normal ile başlatılır
        self.qProj = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: qDim,
            useBias: false,
            initStrategy: .xavierNormal()
        )
        self.kProj = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: kvDim,
            useBias: false,
            initStrategy: .xavierNormal()
        )
        self.vProj = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: kvDim,
            useBias: false,
            initStrategy: .xavierNormal()
        )

        // Artık akışa (Residual stream) eklenen çıkış katmanı derinlik ölçekli (1 / sqrt(2*L)) başlatılır
        self.oProj = CustomLinear(
            inFeatures: qDim,
            outFeatures: config.hiddenSize,
            useBias: false,
            initStrategy: .residualScaledKaiming(numLayers: config.numLayers)
        )

        self.rope = RotaryPositionEmbedding(
            headDim: config.headDim,
            maxSequenceLength: config.maxSequenceLength,
            base: config.ropeTheta
        )

        super.init()
    }

    /// Nedensel (Causal) üst-üçgen dikkat maskesi oluşturur.
    /// Gelecekteki token'ların ($j > i$) görülmesini engellemek için üst üçgene `-1e9` yazar.
    public static func createCausalMask(seqLen: Int, dtype: DType = .float32) -> MLXArray {
        // Alt üçgeni 1, üst üçgeni 0 olan matris üretmek için indeks karşılaştırması:
        // rowIndices: [seqLen, 1], colIndices: [1, seqLen]
        let indices = MLXArray(0..<seqLen)
        let rowIndices = indices.reshaped([seqLen, 1])
        let colIndices = indices.reshaped([1, seqLen])

        // colIndices > rowIndices olan yerler gelecekteki token'lardır (maskelenmeli)
        let futureMask = colIndices .> rowIndices
        let negInf = MLXArray(Float(-1e9)).asType(dtype)
        let zero = MLXArray(Float(0.0)).asType(dtype)

        // Boyut: [1, 1, seqLen, seqLen] (Batch ve Head eksenlerine broadcast edilir)
        return which(futureMask, negInf, zero).reshaped([1, 1, seqLen, seqLen])
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batchSize = x.dim(0)
        let seqLen = x.dim(1)

        // 1. Q, K, V doğrusal projeksiyonları
        var queries = qProj(x) // [B, T, H_q * D]
        var keys = kProj(x)    // [B, T, H_kv * D]
        var values = vProj(x)  // [B, T, H_kv * D]

        // 2. Çoklu başlık (Multi-Head) boyutuna yeniden şekillendir ve eksenleri [B, H, T, D] yap
        queries = queries.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([batchSize, seqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([batchSize, seqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)

        // 3. Sorgu (Q) ve Anahtar (K) vektörlerine RoPE rotasyonu uygula
        queries = rope(queries)
        keys = rope(keys)

        // 4. Grouped-Query Attention (GQA) desteği:
        // Eğer numKVHeads < numHeads ise, K ve V başlıklarını Q başlık sayısına eşitlemek için tekrarla
        let numGroups = numHeads / numKVHeads
        if numGroups > 1 {
            keys = expandedDimensions(keys, axis: 2)
            keys = broadcast(keys, to: [batchSize, numKVHeads, numGroups, seqLen, headDim])
                .reshaped([batchSize, numHeads, seqLen, headDim])

            values = expandedDimensions(values, axis: 2)
            values = broadcast(values, to: [batchSize, numKVHeads, numGroups, seqLen, headDim])
                .reshaped([batchSize, numHeads, seqLen, headDim])
        }

        // 5. Ölçekli Nokta Çarpım (Scaled Dot-Product): Scores = (Q @ K^T) / \sqrt{d_{head}}
        // queries: [B, H, T, D], keys.T: [B, H, D, T] -> scores: [B, H, T, T]
        var attentionScores = matmul(queries, keys.transposed(0, 1, 3, 2)) * MLXArray(scale).asType(x.dtype)

        // 6. Nedensel Maskeyi (Causal Mask) ekle
        if let mask {
            attentionScores = attentionScores + mask
        } else if seqLen > 1 {
            let causalMask = Self.createCausalMask(seqLen: seqLen, dtype: x.dtype)
            attentionScores = attentionScores + causalMask
        }

        // 7. Softmax ile olasılık dağılımına dönüştür (Sayısal kararlılık için hassas modda)
        let attentionWeights = softmax(attentionScores, axis: -1, precise: true)

        // 8. Değer (Value) vektörleriyle ağırlıklı toplam: [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
        let context = matmul(attentionWeights, values)

        // 9. Başlıkları tekrar birleştir: [B, H, T, D] -> [B, T, H, D] -> [B, T, H * D]
        let mergedContext = context.transposed(0, 2, 1, 3)
            .reshaped([batchSize, seqLen, numHeads * headDim])

        // 10. Çıkış projeksiyonu: W_O
        return oProj(mergedContext)
    }
}

// MARK: - 5. CustomSwiGLUFFN (SwiGLU Aktivasyonlu İleri Besleme Ağı / MLP)

/// **CustomSwiGLUFFN (SwiGLU Feed-Forward Network)**
///
/// Klasik ReLU veya GELU tabanlı 2 katmanlı MLP yerine, Noam Shazeer tarafından geliştirilen
/// **Swish-Gated Linear Unit (SwiGLU)** mimarisini sıfırdan uygular.
/// Üç adet doğrusal projeksiyon ($W_{gate}, W_{up}, W_{down}$) içerir ve kapı (gate) mekanizması
/// sayesinde gradyan akışını ve dil modelleme kapasitesini önemli ölçüde artırır.
///
/// **Matematiksel Formül:**
/// $$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$
/// $$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{gate}^T) \odot (x W_{up}^T) \right) W_{down}^T$$
public final class CustomSwiGLUFFN: Module, UnaryLayer {
    /// Kapı projeksiyonu ($W_{gate} \in \mathbb{R}^{d_{ff} \times d_{model}}$) - Kaiming Normal başlatmalı.
    public let gateProj: CustomLinear

    /// Yukarı projeksiyon ($W_{up} \in \mathbb{R}^{d_{ff} \times d_{model}}$) - Kaiming Normal başlatmalı.
    public let upProj: CustomLinear

    /// Aşağı projeksiyon ($W_{down} \in \mathbb{R}^{d_{model} \times d_{ff}}$) - Derinlik Ölçekli Kaiming başlatmalı.
    public let downProj: CustomLinear

    public init(config: ModelConfig) {
        self.gateProj = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useBias: false,
            initStrategy: .kaimingNormal()
        )
        self.upProj = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useBias: false,
            initStrategy: .kaimingNormal()
        )
        self.downProj = CustomLinear(
            inFeatures: config.intermediateSize,
            outFeatures: config.hiddenSize,
            useBias: false,
            initStrategy: .residualScaledKaiming(numLayers: config.numLayers)
        )
        super.init()
    }

    /// Sıfırdan matematiksel SiLU (Swish-1) aktivasyon fonksiyonu: $\text{SiLU}(z) = z \cdot \sigma(z)$.
    private func customSiLU(_ z: MLXArray) -> MLXArray {
        return z * sigmoid(z)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // 1. Kapı kolu: SiLU(x @ W_gate^T)
        let activatedGate = customSiLU(gateProj(x))

        // 2. Değer kolu: x @ W_up^T
        let upStates = upProj(x)

        // 3. Hadamard (Eleman bazlı) çarpım ve çıkış projeksiyonu: (activatedGate * upStates) @ W_down^T
        return downProj(activatedGate * upStates)
    }
}

// MARK: - 6. TransformerBlock (Pre-Norm Decoder Transformer Bloğu)

/// **TransformerBlock (Pre-Norm Decoder Bloğu)**
///
/// Modern Foundation modellerde eğitim kararlılığını (training stability) maksimize eden
/// **Pre-RMSNorm + Residual Connection (Artık Bağlantı)** topolojisini uygular.
/// Normalizasyon işlemi alt-katmanlardan (Attention ve FFN) *önce* uygulanır,
/// alt-katman çıktısı ise doğrudan ham artık akışa (residual stream) toplanır.
///
/// **Matematiksel Akış:**
/// 1. $$h = x + \text{MultiHeadAttention}(\text{RMSNorm}_{attn}(x))$$
/// 2. $$y = h + \text{SwiGLUFFN}(\text{RMSNorm}_{ffn}(h))$$
public final class TransformerBlock: Module {
    /// Dikkat katmanı öncesi RMSNorm ($\text{RMSNorm}_{attn}$).
    public let attentionNorm: CustomRMSNorm

    /// RoPE destekli Çok Başlıklı Öz-Dikkat katmanı.
    public let attention: CustomMultiHeadAttention

    /// İleri besleme (MLP) katmanı öncesi RMSNorm ($\text{RMSNorm}_{ffn}$).
    public let ffnNorm: CustomRMSNorm

    /// SwiGLU aktivasyonlu İleri Besleme Ağı.
    public let feedForward: CustomSwiGLUFFN

    public init(config: ModelConfig) {
        self.attentionNorm = CustomRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.attention = CustomMultiHeadAttention(config: config)
        self.ffnNorm = CustomRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.feedForward = CustomSwiGLUFFN(config: config)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // 1. Alt-Blok: Pre-Norm + Öz-Dikkat + Artık Bağlantı (Residual Add)
        let normX = attentionNorm(x)
        let attnOut = attention(normX, mask: mask)
        let h = x + attnOut

        // 2. Alt-Blok: Pre-Norm + SwiGLU FFN + Artık Bağlantı (Residual Add)
        let normH = ffnNorm(h)
        let ffnOut = feedForward(normH)
        return h + ffnOut
    }
}

// MARK: - 7. CustomTokenEmbedding (Sıfırdan Token Gömme Katmanı)

/// **CustomTokenEmbedding (Sıfırdan Kelime/Token Gömme Tablosu)**
///
/// Ayrık token kimliklerini (`[Batch, SeqLen]` boyutunda tam sayılar) sürekli vektör uzayına
/// (`[Batch, SeqLen, d_model]`) eşleyen öğrenilebilir arama tablosu ($E \in \mathbb{R}^{|V| \times d_{model}}$).
public final class CustomTokenEmbedding: Module, UnaryLayer {
    /// Gömme ağırlık matrisi: $E \in \mathbb{R}^{|V| \times d_{model}}$.
    public var weight: MLXArray

    public init(vocabSize: Int, hiddenSize: Int) {
        self.weight = WeightInitializer.initialize(
            shape: [vocabSize, hiddenSize],
            strategy: .embedding(hiddenSize: hiddenSize)
        )
        super.init()
    }

    public func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        // MLX indeksleme operatörü ile [Batch, SeqLen] -> [Batch, SeqLen, hiddenSize] vektör çekimi
        return weight[tokenIds]
    }
}

// MARK: - 8. PersonalFoundationModel (Tam Decoder-Only Dil Modeli Mimarisi)

/// **PersonalFoundationModel**
///
/// Apple Silicon (M5, 24GB RAM) üzerinde hiçbir dış bağımlılık veya hazır ağırlık olmadan,
/// tamamen sıfırdan (from-scratch) eğitilmek üzere tasarlanmış Otoregresif Decoder-Only Foundation Model.
///
/// **Mimari Bileşenler:**
/// 1. `tokenEmbedding`: Token ID $\to \mathbb{R}^{d_{model}}$ ($\sigma = 1/\sqrt{d_{model}}$ başlatmalı)
/// 2. `layers`: $L$ adet ardışık `TransformerBlock` (Pre-RMSNorm + RoPE MHA/GQA + SwiGLU FFN)
/// 3. `finalNorm`: Son katman `CustomRMSNorm`
/// 4. `lmHead`: $\mathbb{R}^{d_{model}} \to \mathbb{R}^{|V|}$ kelime dağarcığı logit projeksiyonu
public final class PersonalFoundationModel: Module {
    public let config: ModelConfig

    /// Giriş Token Gömme Tablosu.
    public let tokenEmbedding: CustomTokenEmbedding

    /// $L$ adet Transformer Decoder Bloğu.
    public let layers: [TransformerBlock]

    /// Çıkış öncesi son RMSNorm katmanı.
    public let finalNorm: CustomRMSNorm

    /// Dil Modeli Başlığı (Language Modeling Head): Gizli durumları kelime dağarcığı logit'lerine dönüştürür.
    /// `config.tieWordEmbeddings == true` ise `tokenEmbedding.weight` matrisinin transpozunu kullanır.
    public let lmHead: CustomLinear?

    /// Modeli verilen `ModelConfig` ile sıfırdan matematiksel olarak başlatır.
    public init(config: ModelConfig) {
        self.config = config

        // Deterministik ve tekrarlanabilir sıfırdan ağırlık başlatımı için MLX PRNG tohumunu ayarla
        MLXRandom.seed(config.initSeed)

        self.tokenEmbedding = CustomTokenEmbedding(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize
        )

        var builtLayers = [TransformerBlock]()
        builtLayers.reserveCapacity(config.numLayers)
        for _ in 0..<config.numLayers {
            builtLayers.append(TransformerBlock(config: config))
        }
        self.layers = builtLayers

        self.finalNorm = CustomRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )

        if config.tieWordEmbeddings {
            self.lmHead = nil
        } else {
            self.lmHead = CustomLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.vocabSize,
                useBias: false,
                initStrategy: .xavierNormal()
            )
        }

        super.init()
    }

    /// İleri Besleme (Forward Pass)
    /// - Parameter inputIds: `[Batch, SeqLen]` boyutunda `int32` veya `int64` token ID tensörü.
    /// - Returns: `[Batch, SeqLen, VocabSize]` boyutunda ham olasılık öncesi logit tensörü.
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seqLen = inputIds.dim(1)

        // 1. Token ID'lerini sürekli gömme vektörlerine dönüştür: [B, T, d_model]
        var hiddenStates = tokenEmbedding(inputIds)

        // 2. Tüm katmanlar tarafından paylaşılacak Nedensel Dikkat Maskesini (Causal Mask) bir kez üret
        let mask: MLXArray? = seqLen > 1
            ? CustomMultiHeadAttention.createCausalMask(seqLen: seqLen, dtype: hiddenStates.dtype)
            : nil

        // 3. L adet TransformerBlock üzerinden sırayla geçir
        for layer in layers {
            hiddenStates = layer(hiddenStates, mask: mask)
        }

        // 4. Son RMSNorm normalizasyonu
        hiddenStates = finalNorm(hiddenStates)

        // 5. Kelime dağarcığı uzayına (VocabSize) logit projeksiyonu
        if let lmHead {
            return lmHead(hiddenStates)
        } else {
            // Ağırlık paylaşımı (Weight Tying): hiddenStates @ E^T
            return matmul(hiddenStates, tokenEmbedding.weight.T)
        }
    }

    /// Modeldeki toplam eğitilebilir parametre (skaler ağırlık) sayısını hesaplar.
    public func totalParameterCount() -> Int {
        return trainableParameters().flattened().reduce(0) { acc, pair in
            acc + pair.1.size
        }
    }
}
