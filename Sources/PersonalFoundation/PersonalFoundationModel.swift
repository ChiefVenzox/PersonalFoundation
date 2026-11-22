import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - 1. Özel Token Embedding Katmanı (CustomTokenEmbedding - True Mixed Precision)

/// **Sıfırdan Token Gömme (Embedding) Katmanı (`.float32` Master Weight -> `.bfloat16` Aktivasyon)**
/// Ana gömme ağırlıklarını `.float32` olarak saklar; ileri yayılımda seçilen token vektörlerini `.bfloat16` olarak döndürür.
public final class CustomTokenEmbedding: Module, UnaryLayer {
    public let vocabSize: Int
    public let hiddenSize: Int

    /// Ana gömme matrisi (Master Weight): Boyut [vocabSize, hiddenSize], DType: .float32
    @ParameterInfo(key: "weight") public var weight: MLXArray

    public init(vocabSize: Int, hiddenSize: Int, dtype: DType = .float32) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self._weight.wrappedValue = WeightInitializer.embeddingNormal(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            dtype: dtype
        )
        super.init()
    }

    public func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        return weight[tokenIds].asType(.bfloat16)
    }
}

// MARK: - 2. Ana Foundation Model (PersonalFoundationModel - True Mixed Precision)

/// **PersonalFoundationModel (True Mixed Precision: `.float32` Master Weights + `.bfloat16` Compute)**
/// Apple Silicon M5 (24GB RAM) üzerinde ana ağırlıkları `.float32` hassasiyetinde tutarak küçük öğrenme
/// oranlarında 16-bit mantissa yuvarlama kaybını tamamen ortadan kaldırır; ileri ve geri yayılım
/// matris çarpımlarını ise `.bfloat16` Tensor çekirdeklerinde yüksek hızla çalıştırır.
public final class PersonalFoundationModel: Module {
    public let config: ModelConfig
    public let dtype: DType

    @ModuleInfo(key: "embed_tokens") public var embedTokens: CustomTokenEmbedding
    @ModuleInfo(key: "layers") public var layers: [TransformerBlock]
    @ModuleInfo(key: "norm") public var finalNorm: CustomRMSNorm
    @ModuleInfo(key: "lm_head") public var lmHead: CustomLinear

    public init(config: ModelConfig = ModelConfig(), seed: UInt64 = 42, dtype: DType = .float32) {
        self.config = config
        self.dtype = dtype
        MLXRandom.seed(seed)

        self._embedTokens.wrappedValue = CustomTokenEmbedding(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            dtype: dtype
        )

        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            TransformerBlock(config: config, dtype: dtype)
        }

        self._finalNorm.wrappedValue = CustomRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
            dtype: dtype
        )

        self._lmHead.wrappedValue = CustomLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.vocabSize,
            dtype: dtype,
            initialWeight: WeightInitializer.xavierNormal(
                shape: [config.vocabSize, config.hiddenSize],
                fanIn: config.hiddenSize,
                fanOut: config.vocabSize,
                dtype: dtype
            )
        )

        super.init()
    }

    /// Modelin tüm ağırlıklarını verilen tohum (`seed`) ve veri tipi (`dtype`, varsayılan `.float16`) ile
    /// sıfırdan Kaiming/Xavier dağılımlarına göre katman katman başlatır ve Birleşik Bellekte материalize eder.
    /// Katman bazlı `eval` uygulaması sayesinde 1B model başlatılırken geçici bellek tepe noktası (peak RAM) oluşmaz.
    public func initializeWeightsFromScratch(seed: UInt64, dtype: DType? = nil) {
        let targetDType = dtype ?? self.dtype
        MLXRandom.seed(seed)

        let qOutDim = config.numHeads * config.headDim
        let kvOutDim = config.numKVHeads * config.headDim

        // 1. Token Embedding
        let embedWeight = WeightInitializer.embeddingNormal(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            dtype: targetDType
        )
        self.update(parameters: ModuleParameters.unflattened([("embed_tokens.weight", embedWeight)]))
        eval(self.embedTokens)

        // 2. Her bir TransformerBlock katmanını sırayla başlat ve hemen bellekte değerlendir
        for i in 0..<config.numLayers {
            let prefix = "layers.\(i)"
            let layerParams: [(String, MLXArray)] = [
                ("\(prefix).input_layernorm.weight", MLXArray.ones([config.hiddenSize], dtype: targetDType)),
                ("\(prefix).post_attention_layernorm.weight", MLXArray.ones([config.hiddenSize], dtype: targetDType)),
                ("\(prefix).self_attn.q_proj.weight", WeightInitializer.xavierNormal(
                    shape: [qOutDim, config.hiddenSize], fanIn: config.hiddenSize, fanOut: qOutDim, dtype: targetDType
                )),
                ("\(prefix).self_attn.k_proj.weight", WeightInitializer.xavierNormal(
                    shape: [kvOutDim, config.hiddenSize], fanIn: config.hiddenSize, fanOut: kvOutDim, dtype: targetDType
                )),
                ("\(prefix).self_attn.v_proj.weight", WeightInitializer.xavierNormal(
                    shape: [kvOutDim, config.hiddenSize], fanIn: config.hiddenSize, fanOut: kvOutDim, dtype: targetDType
                )),
                ("\(prefix).self_attn.o_proj.weight", WeightInitializer.residualScaledNormal(
                    shape: [config.hiddenSize, qOutDim], fanIn: qOutDim, numLayers: config.numLayers, dtype: targetDType
                )),
                ("\(prefix).mlp.gate_proj.weight", WeightInitializer.kaimingNormal(
                    shape: [config.intermediateSize, config.hiddenSize], fanIn: config.hiddenSize, dtype: targetDType
                )),
                ("\(prefix).mlp.up_proj.weight", WeightInitializer.kaimingNormal(
                    shape: [config.intermediateSize, config.hiddenSize], fanIn: config.hiddenSize, dtype: targetDType
                )),
                ("\(prefix).mlp.down_proj.weight", WeightInitializer.residualScaledNormal(
                    shape: [config.hiddenSize, config.intermediateSize], fanIn: config.intermediateSize, numLayers: config.numLayers, dtype: targetDType
                ))
            ]
            self.update(parameters: ModuleParameters.unflattened(layerParams))
            eval(self.layers[i])
        }

        // 3. Final RMSNorm ve LM Head
        let headParams: [(String, MLXArray)] = [
            ("norm.weight", MLXArray.ones([config.hiddenSize], dtype: targetDType)),
            ("lm_head.weight", WeightInitializer.xavierNormal(
                shape: [config.vocabSize, config.hiddenSize],
                fanIn: config.hiddenSize,
                fanOut: config.vocabSize,
                dtype: targetDType
            ))
        ]
        self.update(parameters: ModuleParameters.unflattened(headParams))
        eval(self)
    }

    /// Yeni bir çıkarım (Inference) oturumu için boş bir `ModelKVCache` oluşturur.
    public func makeKVCache() -> ModelKVCache {
        return ModelKVCache(numLayers: config.numLayers)
    }

    /// **Float16 İleri Besleme (Forward Pass)**
    /// - Parameters:
    ///   - inputIds: `[BatchSize, SeqLen]` boyutunda token indeksleri (`Int32`)
    ///   - cache: Otoregresif üretim (Inference) sırasında geçmiş K/V durumlarını tutan opsiyonel `ModelKVCache`.
    /// - Returns: `[BatchSize, SeqLen, VocabSize]` boyutunda `.float16` ham olasılık skorları (Logits)
    public func callAsFunction(_ inputIds: MLXArray, cache: ModelKVCache? = nil) -> MLXArray {
        let querySeqLen = inputIds.dim(1)
        let pastSeqLen = cache?.currentSequenceLength ?? 0
        let totalKeySeqLen = pastSeqLen + querySeqLen

        // 1. Token ID -> BFloat16 Gömme Vektörleri: [Batch, QuerySeqLen, HiddenSize]
        var hiddenStates = embedTokens(inputIds)

        // 2. Float32 Hassasiyetli Nedensel Maske (j > i -> -1e9)
        let causalMask: MLXArray? = querySeqLen > 1
            ? CustomMultiHeadAttention.createCausalMask(
                querySeqLen: querySeqLen,
                totalKeySeqLen: totalKeySeqLen,
                dtype: .float32
            )
            : nil

        // 3. TransformerBlock katmanlarından BFloat16 ileri yayılım
        for (index, layer) in layers.enumerated() {
            let layerCache = cache?.layerCaches[index]
            hiddenStates = layer(hiddenStates, mask: causalMask, cache: layerCache)
        }

        // 4. Son RMSNorm (İç hesap Float32, çıkış BFloat16)
        hiddenStates = finalNorm(hiddenStates)

        // 5. Kelime dağarcığı boyutuna projeksiyon (Çıkış: [Batch, SeqLen, VocabSize], DType: .bfloat16)
        if config.tieWordEmbeddings {
            return matmul(hiddenStates, embedTokens.weight.asType(.bfloat16).T)
        } else {
            return lmHead(hiddenStates)
        }
    }

    /// **75 Adımlık Tek-Batch Aşırı Öğrenme Doğrulaması (Overfit Sanity Check - True Mixed Precision)**
    /// Ana eğitim başlamadan önce geçici bir 110M (`.float32` Master Weights + `.bfloat16` Compute) model kopyası üzerinde
    /// tek bir 512-tokenlik `(inputs: tokens[0..<511], targets: tokens[1..<512])` batch ile 75 adım eğitim çalıştırır.
    /// `Loss < 1.8` eşiğine ulaştığı anda geçici model kapsamdan çıkarılarak bellekten temizlenir ve ana eğitime geçilir.
    public static func runOverfitSanityCheck(
        config: ModelConfig,
        singleBatch: (inputs: MLXArray, targets: MLXArray),
        steps: Int = 75,
        targetLossThreshold: Float = 1.8
    ) -> (passed: Bool, initialLoss: Float, finalLoss: Float) {
        print("\n🧪 [Sanity Check] Geçici 110M (True Mixed Precision: .float32 Master + .bfloat16 Compute) Model Üzerinde \(steps) Adım Tek-Batch Overfit Testi Başlıyor (Hedef: Loss < \(targetLossThreshold))...")
        let sanityModel = PersonalFoundationModel(config: config, seed: 777, dtype: .float32)
        eval(sanityModel)

        let scheduler = LRScheduler(
            warmupSteps: 5,
            totalSteps: steps,
            maxLR: 2.5e-3,
            minLR: 5e-4
        )
        let trainCfg = TrainingConfig(
            learningRate: 2.5e-3,
            weightDecay: 0.0,
            gradientAccumulationSteps: 1,
            clipGradNorm: 1.0
        )
        let sanityTrainer = PreTrainer(
            model: sanityModel,
            config: trainCfg,
            lrScheduler: scheduler
        )

        var initialLoss: Float = 0.0
        var finalLoss: Float = 0.0

        for step in 1...steps {
            let metrics = sanityTrainer.trainMicroStep(
                inputs: singleBatch.inputs,
                targets: singleBatch.targets
            )
            if step == 1 {
                initialLoss = metrics.averageLoss
            }
            finalLoss = metrics.averageLoss

            if step == 1 || step % 15 == 0 || step == steps || finalLoss < targetLossThreshold {
                print(String(
                    format: "   • [Sanity Adım %02d/%02d] Loss: %.4f | PPL: %8.2f | GradNorm: %.4f",
                    step,
                    steps,
                    metrics.averageLoss,
                    metrics.perplexity,
                    metrics.gradientNorm
                ))
            }

            // Hedef (< 1.8) yakalandığı anda erken tamamla ve hemen ana eğitime geç
            if finalLoss < targetLossThreshold {
                break
            }
        }

        let passed = finalLoss < targetLossThreshold
        return (passed, initialLoss, finalLoss)
    }

    /// Modeldeki toplam eğitilebilir parametre sayısını hesaplar.
    public func totalParameterCount() -> Int {
        return parameters().flattened().reduce(0) { acc, item in
            acc + item.1.size
        }
    }

    /// Model ağırlıklarının bellekte kapladığı gerçek bayt miktarını döndürür.
    public func allocatedParameterBytes() -> Int {
        return parameters().flattened().reduce(0) { acc, item in
            acc + item.1.nbytes
        }
    }

    /// Model ağırlıklarının bellekte kapladığı alanı Megabayt (MB) cinsinden döndürür.
    public func allocatedParameterMB() -> Double {
        return Double(allocatedParameterBytes()) / (1024.0 * 1024.0)
    }
}
