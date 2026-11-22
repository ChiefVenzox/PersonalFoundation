import Foundation
import MLX
import MLXNN

// MARK: - 1. Doğrusal Projeksiyon Katmanı (CustomLinear - True Mixed Precision: .float32 Master + .bfloat16 Compute)

/// Sıfırdan eğitilebilir, ana ağırlıkları (`.float32` Master Weights) olarak tutulan,
/// ileri ve geri yayılım matris çarpımlarını (`matmul`) ise Apple Silicon M5 Metal GPU üzerinde
/// **`.bfloat16`** hassasiyetinde yürüten doğrusal katman.
public final class CustomLinear: Module, UnaryLayer {
    public let inFeatures: Int
    public let outFeatures: Int

    /// Ana ağırlık matrisi (Master Weight): Boyut [outFeatures, inFeatures], DType: .float32
    @ParameterInfo(key: "weight") public var weight: MLXArray
    /// Opsiyonel sapma (bias) vektörü: Boyut [outFeatures]
    @ParameterInfo(key: "bias") public var bias: MLXArray?

    public init(
        inFeatures: Int,
        outFeatures: Int,
        useBias: Bool = false,
        dtype: DType = .float32,
        initialWeight: MLXArray? = nil
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures

        if let initialWeight {
            self._weight.wrappedValue = initialWeight.asType(dtype)
        } else {
            self._weight.wrappedValue = WeightInitializer.xavierNormal(
                shape: [outFeatures, inFeatures],
                fanIn: inFeatures,
                fanOut: outFeatures,
                dtype: dtype
            )
        }

        if useBias {
            self._bias.wrappedValue = MLXArray.zeros([outFeatures], dtype: dtype)
        } else {
            self._bias.wrappedValue = nil
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // True Mixed Precision: Master ağırlık .float32 iken matris çarpımını .bfloat16 Tensor çekirdeklerinde çalıştır
        let xBF16 = (x.dtype == .bfloat16) ? x : x.asType(.bfloat16)
        let wBF16 = (weight.dtype == .bfloat16) ? weight : weight.asType(.bfloat16)
        var out = matmul(xBF16, wBF16.T)
        if let bias {
            out = out + bias.asType(.bfloat16)
        }
        return out
    }
}

// MARK: - 2. RMSNorm (True Mixed Precision: .float32 Hesaplama -> .bfloat16 Aktivasyon Çıkışı)

/// **RMSNorm Katmanı (Float32 Master Weight & Varyans Hesabı + BFloat16 Aktivasyon Çıkışı)**
public final class CustomRMSNorm: Module, UnaryLayer {
    public let eps: Float

    /// Öğrenilebilir ölçek (gain / gamma) parametresi: Boyut [hiddenSize], DType: .float32
    @ParameterInfo(key: "weight") public var weight: MLXArray

    public init(dimensions: Int, eps: Float = 1e-5, dtype: DType = .float32) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions], dtype: dtype)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // RMS hesabı ve gamma çarpımı tam Float32 hassasiyetinde yapılır, çıkış .bfloat16 olarak döner
        let xFloat = x.asType(.float32)
        let meanSquared = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let invRMS = rsqrt(meanSquared + MLXArray(eps))
        let normalizedF32 = (xFloat * invRMS) * weight.asType(.float32)
        return normalizedF32.asType(.bfloat16)
    }
}

// MARK: - 3. RoPE (Rotary Position Embedding - Float16)

/// **Rotary Position Embedding (RoPE)**
/// Sorgu (Query) ve Anahtar (Key) vektörlerini 2 boyutlu alt-uzaylarda pozisyona (m) ve
/// frekansa (theta_i) bağlı açıyla `.float16` hassasiyetinde döndürür.
public final class RotaryPositionEmbedding: Module {
    public let dims: Int
    public let maxSeqLen: Int
    public let base: Float

    /// Önceden hesaplanmış Kosinüs ve Sinüs frekans tabloları: Boyut [maxSeqLen, dims / 2]
    private let cosTable: MLXArray
    private let sinTable: MLXArray

    public init(dims: Int, maxSeqLen: Int = 2048, base: Float = 10_000.0, dtype: DType = .bfloat16) {
        precondition(dims % 2 == 0, "RoPE boyutu (headDim) çift sayı olmalıdır.")
        self.dims = dims
        self.maxSeqLen = maxSeqLen
        self.base = base

        let halfDim = dims / 2
        let exponentIndices = MLXArray(stride(from: 0, to: dims, by: 2).map { Float($0) / Float(dims) })
        let invFreq = MLX.pow(MLXArray(base), -exponentIndices)
        let positions = MLXArray((0..<maxSeqLen).map { Float($0) }).reshaped([maxSeqLen, 1])

        // Frekans tablosunu Float32'de hassas hesaplayıp bellekte hedef dtype (.bfloat16) olarak sakla
        let freqs = matmul(positions, invFreq.reshaped([1, halfDim]))
        self.cosTable = MLX.cos(freqs).asType(dtype)
        self.sinTable = MLX.sin(freqs).asType(dtype)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.dim(2)
        let halfDim = dims / 2

        let cosSlice = cosTable[offset..<(offset + seqLen), 0..<halfDim]
            .reshaped([1, 1, seqLen, halfDim])
            .asType(x.dtype)
        let sinSlice = sinTable[offset..<(offset + seqLen), 0..<halfDim]
            .reshaped([1, 1, seqLen, halfDim])
            .asType(x.dtype)

        let x1 = x[0..., 0..., 0..., 0..<halfDim]
        let x2 = x[0..., 0..., 0..., halfDim..<dims]

        let rotatedX1 = (x1 * cosSlice) - (x2 * sinSlice)
        let rotatedX2 = (x2 * cosSlice) + (x1 * sinSlice)

        return concatenated([rotatedX1, rotatedX2], axis: -1)
    }
}

// MARK: - 4. Multi-Head Attention (Float16, RoPE & GQA 4:1 Destekli)

/// **Özel Çok Başlıklı Dikkat Mekanizması (MultiHeadAttention - Float16)**
/// 1B model mimarisinde `numHeads = 16` (`headDim = 128`) ve `numKVHeads = 4` (4:1 GQA) ile
/// KV önbellek ve dikkat ağırlık belleğini %75 azaltır.
public final class CustomMultiHeadAttention: Module {
    public let hiddenSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let numQueriesPerKV: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: CustomLinear
    @ModuleInfo(key: "k_proj") public var kProj: CustomLinear
    @ModuleInfo(key: "v_proj") public var vProj: CustomLinear
    @ModuleInfo(key: "o_proj") public var oProj: CustomLinear
    @ModuleInfo(key: "rope") public var rope: RotaryPositionEmbedding

    public init(config: ModelConfig, dtype: DType = .float32) {
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.numQueriesPerKV = config.numHeads / config.numKVHeads
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let qOutDim = config.numHeads * config.headDim
        let kvOutDim = config.numKVHeads * config.headDim

        self._qProj.wrappedValue = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: qOutDim,
            dtype: dtype,
            initialWeight: WeightInitializer.xavierNormal(
                shape: [qOutDim, config.hiddenSize],
                fanIn: config.hiddenSize,
                fanOut: qOutDim,
                dtype: dtype
            )
        )
        self._kProj.wrappedValue = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: kvOutDim,
            dtype: dtype,
            initialWeight: WeightInitializer.xavierNormal(
                shape: [kvOutDim, config.hiddenSize],
                fanIn: config.hiddenSize,
                fanOut: kvOutDim,
                dtype: dtype
            )
        )
        self._vProj.wrappedValue = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: kvOutDim,
            dtype: dtype,
            initialWeight: WeightInitializer.xavierNormal(
                shape: [kvOutDim, config.hiddenSize],
                fanIn: config.hiddenSize,
                fanOut: kvOutDim,
                dtype: dtype
            )
        )
        self._oProj.wrappedValue = CustomLinear(
            inFeatures: qOutDim,
            outFeatures: config.hiddenSize,
            dtype: dtype,
            initialWeight: WeightInitializer.residualScaledNormal(
                shape: [config.hiddenSize, qOutDim],
                fanIn: qOutDim,
                numLayers: config.numLayers,
                dtype: dtype
            )
        )
        self._rope.wrappedValue = RotaryPositionEmbedding(
            dims: config.headDim,
            maxSeqLen: config.maxSequenceLength,
            base: config.ropeTheta,
            dtype: .bfloat16
        )
        super.init()
    }

    /// **Nedensel (Causal) Üst-Üçgen Dikkat Maskesi (`j > i` -> `-1e9`)**
    /// Gelecekteki tokenların (`colIndices > rowIndices`) görülmesini engellemek için üst üçgeni `-1e9` ile maskeler.
    public static func createCausalMask(
        querySeqLen: Int,
        totalKeySeqLen: Int? = nil,
        dtype: DType = .float32
    ) -> MLXArray {
        let keyLen = totalKeySeqLen ?? querySeqLen
        let pastLength = keyLen - querySeqLen
        let rowIndices = MLXArray(0..<querySeqLen).reshaped([querySeqLen, 1]) + MLXArray(pastLength)
        let colIndices = MLXArray(0..<keyLen).reshaped([1, keyLen])

        // Üst üçgen (j > i) gelecekteki tokenlardır -> -1e9 ile maskelenir; (j <= i) -> 0.0
        let isFuturePosition = colIndices .> rowIndices
        let zero = MLXArray(Float(0.0)).asType(dtype)
        let maskValue: Float = (dtype == .float16) ? -1e4 : -1e9
        let negInf = MLXArray(maskValue).asType(dtype)
        return which(isFuturePosition, negInf, zero).reshaped([1, 1, querySeqLen, keyLen])
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: LayerKVCache? = nil
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let querySeqLen = x.dim(1)

        // 1. BFloat16 Doğrusal Projeksiyonlar: [Batch, SeqLen, HiddenSize]
        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // 2. Başlıkları Ayır ve Kesinlikle [Batch, NumHeads, SeqLen, HeadDim] Sırasına (.transposed(0, 2, 1, 3)) Getir
        queries = queries.reshaped([batchSize, querySeqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([batchSize, querySeqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([batchSize, querySeqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)

        // 3. RoPE Rotasyonu ([Batch, NumHeads, SeqLen, HeadDim] eksen düzeninde)
        let ropeOffset = cache?.offset ?? 0
        queries = rope(queries, offset: ropeOffset)
        keys = rope(keys, offset: ropeOffset)

        // 4. KV-Cache Güncellemesi
        if let cache {
            (keys, values) = cache.updateAndFetch(newKeys: keys, newValues: values)
        }

        let totalKeySeqLen = keys.dim(2)

        // 5. Grouped-Query Attention (GQA): KV başlıklarını axis: 1 (NumHeads ekseni) üzerinde repeated ile genişlet
        if numQueriesPerKV > 1 {
            keys = repeated(keys, count: numQueriesPerKV, axis: 1)
            values = repeated(values, count: numQueriesPerKV, axis: 1)
        }

        // 6. Ölçeklenmiş İç Çarpım: [B, H, L_q, D] x [B, H, D, L_k] -> [B, H, L_q, L_k]
        let keysTransposed = keys.transposed(0, 1, 3, 2)
        let scaleTyped = MLXArray(scale).asType(queries.dtype)
        let rawAttentionScores = matmul(queries, keysTransposed) * scaleTyped

        // 7. Nedensel Maske ve Softmax (Sayısal kararlılık için .float32'de hesaplanıp tekrar .bfloat16 döndürülür)
        var scoresF32 = rawAttentionScores.asType(.float32)
        if let mask {
            scoresF32 = scoresF32 + mask.asType(.float32)
        } else if querySeqLen > 1 {
            let causalMask = Self.createCausalMask(
                querySeqLen: querySeqLen,
                totalKeySeqLen: totalKeySeqLen,
                dtype: .float32
            )
            scoresF32 = scoresF32 + causalMask
        }

        // 8. Float32 Softmax -> BFloat16'ya geri dönüştür ve Value ile çarp
        let attentionWeights = softmax(scoresF32, axis: -1).asType(values.dtype)
        let contextLayer = matmul(attentionWeights, values)

        // 9. Başlıkları [Batch, SeqLen, NumHeads, HeadDim] -> [Batch, SeqLen, HiddenSize] olarak birleştir ve W_O uygula
        let mergedContext = contextLayer
            .transposed(0, 2, 1, 3)
            .reshaped([batchSize, querySeqLen, hiddenSize])

        return oProj(mergedContext)
    }
}

// MARK: - 5. FeedForward / MLP (Float16 SwiGLU)

/// **SwiGLU İleri Besleme Ağı (Float16)**
/// 1B yapılandırmasında `hiddenSize = 2048` -> `intermediateSize = 5504` (~8/3 * hiddenSize).
public final class SwiGLUFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") public var gateProj: CustomLinear
    @ModuleInfo(key: "up_proj") public var upProj: CustomLinear
    @ModuleInfo(key: "down_proj") public var downProj: CustomLinear

    public init(config: ModelConfig, dtype: DType = .float32) {
        self._gateProj.wrappedValue = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            dtype: dtype,
            initialWeight: WeightInitializer.kaimingNormal(
                shape: [config.intermediateSize, config.hiddenSize],
                fanIn: config.hiddenSize,
                dtype: dtype
            )
        )
        self._upProj.wrappedValue = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            dtype: dtype,
            initialWeight: WeightInitializer.kaimingNormal(
                shape: [config.intermediateSize, config.hiddenSize],
                fanIn: config.hiddenSize,
                dtype: dtype
            )
        )
        self._downProj.wrappedValue = CustomLinear(
            inFeatures: config.intermediateSize,
            outFeatures: config.hiddenSize,
            dtype: dtype,
            initialWeight: WeightInitializer.residualScaledNormal(
                shape: [config.hiddenSize, config.intermediateSize],
                fanIn: config.intermediateSize,
                numLayers: config.numLayers,
                dtype: dtype
            )
        )
        super.init()
    }

    private func customSiLU(_ z: MLXArray) -> MLXArray {
        return z * sigmoid(z)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activatedGate = customSiLU(gateProj(x))
        let upStates = upProj(x)
        return downProj(activatedGate * upStates)
    }
}

// MARK: - 6. TransformerBlock (Pre-Norm Decoder Bloğu - True Mixed Precision)

public final class TransformerBlock: Module {
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: CustomRMSNorm
    @ModuleInfo(key: "self_attn") public var attention: CustomMultiHeadAttention
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: CustomRMSNorm
    @ModuleInfo(key: "mlp") public var mlp: SwiGLUFeedForward

    public init(config: ModelConfig, dtype: DType = .float32) {
        self._inputLayerNorm.wrappedValue = CustomRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
            dtype: dtype
        )
        self._attention.wrappedValue = CustomMultiHeadAttention(config: config, dtype: dtype)
        self._postAttentionLayerNorm.wrappedValue = CustomRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
            dtype: dtype
        )
        self._mlp.wrappedValue = SwiGLUFeedForward(config: config, dtype: dtype)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: LayerKVCache? = nil
    ) -> MLXArray {
        let normedInput = inputLayerNorm(x)
        let attnOut = attention(normedInput, mask: mask, cache: cache)
        let hiddenStates = x + attnOut

        let normedHidden = postAttentionLayerNorm(hiddenStates)
        let mlpOut = mlp(normedHidden)
        return hiddenStates + mlpOut
    }
}
