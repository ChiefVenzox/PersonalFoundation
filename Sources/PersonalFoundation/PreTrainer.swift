import Foundation
import MLX
import MLXNN
import MLXOptimizers

// MARK: - TrainingConfig (Ön-Eğitim Hiperparametreleri)

/// Sıfırdan Ön-Eğitim (Pre-Training) döngüsünün öğrenme oranı, ağırlık çürümesi ve
/// Apple Silicon 24GB RAM sınırlarını korumak için kullanılan Gradyan Birikimi (Gradient Accumulation) yapılandırması.
///
/// Swift 6 Strict Concurrency kurallarına uygun olarak `Sendable` değer tipidir.
public struct TrainingConfig: Sendable, Codable, Equatable {
    /// AdamW başlangıç öğrenme oranı ($\eta$).
    public let learningRate: Float

    /// AdamW ağırlık çürümesi katsayısı ($\lambda$ - Weight Decay).
    public let weightDecay: Float

    /// AdamW birinci ve ikinci moment üssel hareketli ortalama katsayıları ($\beta_1, \beta_2$).
    public let betas: (Float, Float)

    /// AdamW sayısal kararlılık sabiti ($\epsilon$).
    public let eps: Float

    /// **Gradyan Birikimi Adım Sayısı (Gradient Accumulation Steps - $K$)**
    ///
    /// Apple Silicon M5 (24GB Unified Memory) üzerinde büyük efektif yığın boyutlarına
    /// (`EffectiveBatchSize = MicroBatchSize * gradientAccumulationSteps`) ulaşırken
    /// GPU belleğinin (VRAM) taşmasını önler. Her mikro-yığında gradyanlar $\frac{1}{K}$ ile ölçeklenerek
    /// biriktirilir ve yalnızca $K$ mikro-adımda bir `optimizer.update` çağrılır.
    public let gradientAccumulationSteps: Int

    /// Gradyan patlamasını (Exploding Gradients) önlemek için maksimum L2 gradyan normu eşiği.
    public let maxGradNorm: Float?

    public init(
        learningRate: Float = 3e-4,
        weightDecay: Float = 0.01,
        betas: (Float, Float) = (0.9, 0.95),
        eps: Float = 1e-8,
        gradientAccumulationSteps: Int = 4,
        maxGradNorm: Float? = 1.0
    ) {
        precondition(gradientAccumulationSteps >= 1, "gradientAccumulationSteps en az 1 olmalıdır.")
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.betas = betas
        self.eps = eps
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.maxGradNorm = maxGradNorm
    }

    enum CodingKeys: String, CodingKey {
        case learningRate, weightDecay, beta1, beta2, eps, gradientAccumulationSteps, maxGradNorm
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.learningRate = try container.decode(Float.self, forKey: .learningRate)
        self.weightDecay = try container.decode(Float.self, forKey: .weightDecay)
        let b1 = try container.decode(Float.self, forKey: .beta1)
        let b2 = try container.decode(Float.self, forKey: .beta2)
        self.betas = (b1, b2)
        self.eps = try container.decode(Float.self, forKey: .eps)
        self.gradientAccumulationSteps = try container.decode(Int.self, forKey: .gradientAccumulationSteps)
        self.maxGradNorm = try container.decodeIfPresent(Float.self, forKey: .maxGradNorm)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(weightDecay, forKey: .weightDecay)
        try container.encode(betas.0, forKey: .beta1)
        try container.encode(betas.1, forKey: .beta2)
        try container.encode(eps, forKey: .eps)
        try container.encode(gradientAccumulationSteps, forKey: .gradientAccumulationSteps)
        try container.encodeIfPresent(maxGradNorm, forKey: .maxGradNorm)
    }

    public static func == (lhs: TrainingConfig, rhs: TrainingConfig) -> Bool {
        lhs.learningRate == rhs.learningRate &&
        lhs.weightDecay == rhs.weightDecay &&
        lhs.betas.0 == rhs.betas.0 &&
        lhs.betas.1 == rhs.betas.1 &&
        lhs.eps == rhs.eps &&
        lhs.gradientAccumulationSteps == rhs.gradientAccumulationSteps &&
        lhs.maxGradNorm == rhs.maxGradNorm
    }
}

// MARK: - StepMetrics (Eğitim Adımı Metrikleri)

/// Tek bir efektif eğitim adımının (veya mikro-adımın) telemetri çıktısı (`Sendable`).
public struct StepMetrics: Sendable {
    /// Tamamlanan optimizasyon (ağırlık güncelleme) adımı.
    public let step: Int

    /// Ortalama Çapraz Entropi Kaybı (Cross-Entropy Loss).
    public let loss: Float

    /// Perplexity (Şaşkınlık skoru): $\text{PPL} = e^{\text{Loss}}$.
    public let perplexity: Float

    /// İlgili adımda hesaplanan toplam L2 gradyan normu.
    public let gradNorm: Float
}

// MARK: - PreTrainer (Sıfırdan Ön-Eğitim ve Gradyan Birikimi Motoru)

/// **PreTrainer (Sıfırdan Dil Modeli Ön-Eğitim Sınıfı)**
///
/// `PersonalFoundationModel` mimarisini otoregresif sonraki-token tahmini (Causal Language Modeling)
/// göreviyle sıfırdan eğiten sınıf.
///
/// **İçerdiği Çekirdek Matematiksel İşlemler:**
/// 1. **Nedensel Kaydırma (Causal Shift):** Giriş dizisi $x_{0:T-1}$ iken hedef dizisi $y_{1:T}$ olarak hizalanır.
/// 2. **Sıfırdan Cross-Entropy Loss:** Log-Sum-Exp kararlılığı ile $-\frac{1}{N}\sum \log P(y_t \mid x_{\le t})$ hesabı.
/// 3. **Otomatik Türev (Autograd - `valueAndGrad`):** Tüm Transformer blokları üzerinden geriye yayılım.
/// 4. **Gradient Accumulation (Gradyan Birikimi):** $K$ mikro-yığın boyunca gradyan ağaçlarını (`NestedDictionary`)
///    GPU belleğini şişirmeden biriktirme ve `eval()` ile tembel grafiği (lazy computation graph) temizleme.
/// 5. **AdamW Optimizasyonu:** `MLXOptimizers.AdamW` ile ayrık ağırlık çürümesi ve parametre güncellemesi.
public final class PreTrainer {
    public let model: PersonalFoundationModel
    public let optimizer: AdamW
    public let config: TrainingConfig

    /// Şimdiye kadar gerçekleştirilen toplam ağırlık güncelleme (optimizer step) sayısı.
    public private(set) var currentStep: Int = 0

    /// Şimdiye kadar işlenen toplam mikro-yığın (micro-batch) sayısı.
    public private(set) var microStepCount: Int = 0

    /// Gradyan birikimi (Gradient Accumulation) için ara gradyan ağacı tamponu.
    private var accumulatedGradients: ModuleParameters? = nil

    /// Birikim döngüsü boyunca ortalama kaybı takip etmek için kümülatif toplam.
    private var accumulatedLoss: Float = 0.0

    public init(model: PersonalFoundationModel, config: TrainingConfig = TrainingConfig()) {
        self.model = model
        self.config = config
        self.optimizer = AdamW(
            learningRate: config.learningRate,
            betas: config.betas,
            eps: config.eps,
            weightDecay: config.weightDecay
        )
    }

    // MARK: - Sıfırdan Matematiksel Cross-Entropy Loss

    /// **Otoregresif Çapraz Entropi Kaybı (Causal Cross-Entropy Loss)**
    ///
    /// Sayısal taşmayı (numerical overflow) önlemek için **Log-Sum-Exp** özdeşliğini kullanarak
    /// logit'ler üzerinden doğrudan negatif log-olabilirlik (Negative Log-Likelihood) hesaplar:
    ///
    /// $$\log \text{softmax}(z_i) = z_i - \left( m + \log \sum_{j=1}^{|V|} e^{z_j - m} \right), \quad m = \max_j(z_j)$$
    /// $$\mathcal{L}_{CE} = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log \text{softmax}(z_{b,t})[y_{b,t}]$$
    ///
    /// - Parameters:
    ///   - logits: Modelin ürettiği `[Batch, SeqLen, VocabSize]` boyutundaki ham logit tensörü.
    ///   - targets: Gerçek sonraki token ID'lerini içeren `[Batch, SeqLen]` boyutundaki hedef tensörü.
    /// - Returns: Skaler (0-boyutlu) ortalama kayıp (`MLXArray`).
    public static func causalCrossEntropyLoss(logits: MLXArray, targets: MLXArray) -> MLXArray {
        // Sayısal kararlılık için Float32 uzayına geç
        let logitsFp32 = logits.asType(.float32)

        // 1. Log-Sum-Exp: m + log(\sum exp(z - m))
        let maxLogits = max(logitsFp32, axis: -1, keepDims: true)
        let shiftedLogits = logitsFp32 - stopGradient(maxLogits)
        let logSumExp = maxLogits + log(sum(exp(shiftedLogits), axis: -1, keepDims: true))

        // 2. Log-Olasılık dağılımı: log P(v | x) = z_v - logSumExp -> [Batch, SeqLen, VocabSize]
        let logProbs = logitsFp32 - logSumExp

        // 3. Hedef token indekslerindeki (targets) log-olasılıkları çek (Gather / takeAlong)
        // targets: [Batch, SeqLen] -> expandedTargets: [Batch, SeqLen, 1]
        let expandedTargets = expandedDimensions(targets, axis: -1)
        let targetLogProbs = takeAlong(logProbs, expandedTargets, axis: -1).squeezed(axis: -1)

        // 4. Ortalama Negatif Log-Olabilirlik: -mean(targetLogProbs)
        return -mean(targetLogProbs)
    }

    // MARK: - Gradyan Birikimi ve Küresel L2 Norm Kırpma Yardımcıları

    /// İki `ModuleParameters` (gradyan ağacı) yapısını eleman bazında toplar: `G_{acc} = G_{acc} + G_{micro}`.
    private static func addGradients(
        _ accumulated: ModuleParameters,
        _ newGrads: ModuleParameters
    ) -> ModuleParameters {
        var mergedFlat = [String: MLXArray]()
        let accFlat = Dictionary(uniqueKeysWithValues: accumulated.flattened())
        for (key, grad) in newGrads.flattened() {
            if let existing = accFlat[key] {
                mergedFlat[key] = existing + grad
            } else {
                mergedFlat[key] = grad
            }
        }
        return ModuleParameters.unflattened(mergedFlat)
    }

    /// Tüm gradyan ağacını sabit bir skaler (`scale = 1.0 / K`) ile çarpar.
    private static func scaleGradients(
        _ grads: ModuleParameters,
        factor: Float
    ) -> ModuleParameters {
        let scaleArray = MLXArray(factor)
        return grads.mapValues { $0 * scaleArray }
    }

    /// Küresel L2 gradyan normunu ($\|\nabla \mathcal{L}\|_2 = \sqrt{\sum_p \|g_p\|_2^2}$) hesaplar ve
    /// `maxNorm` değerini aşıyorsa tüm gradyanları orantılı olarak kırpar (Global Gradient Clipping).
    private static func clipGradNorm(
        _ grads: ModuleParameters,
        maxNorm: Float?
    ) -> (clippedGrads: ModuleParameters, totalNorm: Float) {
        var sumSquares = MLXArray(Float(0.0))
        for (_, grad) in grads.flattened() {
            sumSquares = sumSquares + sum(grad.asType(.float32).square())
        }
        let totalNormArray = sqrt(sumSquares)
        eval(totalNormArray)
        let totalNorm = totalNormArray.item(Float.self)

        guard let maxNorm, totalNorm > maxNorm, totalNorm > 0 else {
            return (grads, totalNorm)
        }

        let clipCoef = maxNorm / (totalNorm + 1e-6)
        let clipped = scaleGradients(grads, factor: clipCoef)
        return (clipped, totalNorm)
    }

    // MARK: - Eğitim Adımları (Micro-Step & Full Gradient Accumulation Step)

    /// Giriş token dizisini (`[Batch, SeqLen]`) otoregresif olarak `(inputs: x[:, :-1], targets: x[:, 1:])`
    /// şeklinde kaydırarak ayırır.
    public static func prepareAutoregressivePair(tokenBatch: MLXArray) -> (inputs: MLXArray, targets: MLXArray) {
        let seqLen = tokenBatch.dim(1)
        precondition(seqLen >= 2, "Otoregresif eğitim için dizi uzunluğu (seqLen) en az 2 olmalıdır.")
        let inputs = tokenBatch[0..., 0..<(seqLen - 1)]
        let targets = tokenBatch[0..., 1..<seqLen]
        return (inputs, targets)
    }

    /// **Tek Bir Mikro-Yığın İşleme ve Gradyan Biriktirme (`trainMicroStep`)**
    ///
    /// Bu fonksiyon her çağrıldığında:
    /// 1. İleri besleme (Forward pass) ve Cross-Entropy Loss hesaplar.
    /// 2. `valueAndGrad` ile geriye yayılım (Backpropagation) yaparak gradyanları çıkarır.
    /// 3. Gradyanları $\frac{1}{K}$ (`1 / gradientAccumulationSteps`) ile ölçekleyip `accumulatedGradients` içinde biriktirir.
    /// 4. **Bellek Yönetimi (M5 24GB RAM):** Tembel hesaplama grafiğinin (lazy graph) bellekte birikmesini
    ///    önlemek için her mikro-adımda biriken gradyanları `eval()` ile somutlaştırır.
    /// 5. Eğer $K$ mikro-adım tamamlandıysa `optimizer.update(model:gradients:)` ile ağırlıkları günceller.
    ///
    /// - Parameters:
    ///   - inputs: `[MicroBatch, SeqLen]` boyutunda giriş token ID'leri.
    ///   - targets: `[MicroBatch, SeqLen]` boyutunda hedef (bir sonraki) token ID'leri.
    /// - Returns: Eğer bu mikro-adımda ağırlık güncellemesi (`optimizer.update`) tetiklendiyse `StepMetrics`,
    ///            henüz birikim devam ediyorsa `nil` döner.
    @discardableResult
    public func trainMicroStep(inputs: MLXArray, targets: MLXArray) -> StepMetrics? {
        let accumSteps = config.gradientAccumulationSteps

        // İleri besleme + Kayıp fonksiyonunu MLX otomatik türev (valueAndGrad) sarmalayıcısına bağla
        let lossAndGradFn = valueAndGrad(model: model) { [targets] (model: PersonalFoundationModel, x: MLXArray) -> MLXArray in
            let logits = model(x)
            return Self.causalCrossEntropyLoss(logits: logits, targets: targets)
        }

        // 1. İleri ve Geri Yayılım (Forward & Backward Pass)
        let (lossArray, rawGrads) = lossAndGradFn(model, inputs)

        // 2. Gradyanları 1 / K ile ölçekle (böylece K adım sonunda toplam tam ortalamaya eşit olur)
        let scaledGrads = Self.scaleGradients(rawGrads, factor: 1.0 / Float(accumSteps))

        // 3. Gradyan Birikimi (Gradient Accumulation)
        if let existingGrads = accumulatedGradients {
            accumulatedGradients = Self.addGradients(existingGrads, scaledGrads)
        } else {
            accumulatedGradients = scaledGrads
        }

        // 4. Apple Silicon Birleşik Bellek (24GB RAM) koruması:
        // Tembel hesaplama grafiğini burada değerlendirerek (eval) önceki mikro-adımın ara aktivasyonlarını serbest bırak.
        if let currentAcc = accumulatedGradients {
            eval(lossArray, currentAcc)
        }

        let microLoss = lossArray.item(Float.self)
        accumulatedLoss += microLoss / Float(accumSteps)
        microStepCount += 1

        // 5. K mikro-adım tamamlandıysa ağırlıkları güncelle (Optimizer Step)
        if microStepCount % accumSteps == 0, let finalGrads = accumulatedGradients {
            let (clippedGrads, gradNorm) = Self.clipGradNorm(finalGrads, maxNorm: config.maxGradNorm)

            // AdamW ile model ağırlıklarını güncelle
            optimizer.update(model: model, gradients: clippedGrads)

            // Güncellenen model parametrelerini ve optimizer durumunu GPU üzerinde somutlaştır
            eval(model, optimizer)

            currentStep += 1
            let stepLoss = accumulatedLoss
            let perplexity = exp(min(stepLoss, 20.0))

            // Bir sonraki birikim penceresi için tamponları sıfırla
            accumulatedGradients = nil
            accumulatedLoss = 0.0

            return StepMetrics(
                step: currentStep,
                loss: stepLoss,
                perplexity: perplexity,
                gradNorm: gradNorm
            )
        }

        return nil
    }

    /// Verilen mikro-yığın dizisi (`microBatches`) üzerinde tam bir Gradyan Birikimi (Gradient Accumulation)
    /// adımı çalıştırır. Her eleman `[MicroBatchSize, SeqLen + 1]` boyutunda ham token dizisidir.
    public func trainAccumulatedStep(microBatches: [MLXArray]) -> StepMetrics {
        precondition(
            microBatches.count == config.gradientAccumulationSteps,
            "Verilen mikro-yığın sayısı (\(microBatches.count)), gradientAccumulationSteps (\(config.gradientAccumulationSteps)) ile eşleşmelidir."
        )

        var lastMetrics: StepMetrics? = nil
        for batch in microBatches {
            let (inputs, targets) = Self.prepareAutoregressivePair(tokenBatch: batch)
            if let metrics = trainMicroStep(inputs: inputs, targets: targets) {
                lastMetrics = metrics
            }
        }

        guard let completedMetrics = lastMetrics else {
            fatalError("Gradyan birikimi döngüsü tamamlanmasına rağmen optimizer adımı tetiklenmedi.")
        }
        return completedMetrics
    }
}
