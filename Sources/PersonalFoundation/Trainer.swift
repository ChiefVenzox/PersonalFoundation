import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// MARK: - 1. Eğitim Yapılandırması ve Metrikler (Swift 6 Sendable)

/// Ön eğitim (Pre-training) hiperparametrelerini tutan `Sendable` ve `Codable` yapı.
public struct TrainingConfig: Sendable, Codable, Equatable {
    /// AdamW varsayılan (veya tepe) öğrenme oranı (Learning Rate)
    public let learningRate: Float
    /// AdamW ağırlık çürümesi (Weight Decay - L2 Regularization)
    public let weightDecay: Float
    /// AdamW 1. moment (momentum) katsayısı
    public let beta1: Float
    /// AdamW 2. moment (varyans) katsayısı
    public let beta2: Float
    /// AdamW için (beta1, beta2) tuple erişimi
    public var betas: (Float, Float) { (beta1, beta2) }
    /// AdamW sayısal kararlılık sabiti
    public let eps: Float
    /// **Gradyan Birikimi Adım Sayısı (Gradient Accumulation Steps)**
    /// M5 (24GB RAM) belleğini aşmadan büyük efektif Batch Size elde etmek için,
    /// ağırlık güncellemesi yapmadan önce kaç mikro-batch gradyanının biriktirileceğini belirler.
    public let gradientAccumulationSteps: Int
    /// **Küresel Gradyan Kırpma Limiti (Global L2 Gradient Clipping)**
    /// 1B ölçekli model eğitiminde gradyan patlamalarını (Exploding Gradients) önlemek için varsayılan `1.0` olarak ayarlanır.
    public let clipGradNorm: Float?

    /// Geriye dönük uyumluluk için `maxGradNorm` erişimi
    public var maxGradNorm: Float? { clipGradNorm }

    public init(
        learningRate: Float = 3e-4,
        weightDecay: Float = 0.1,
        beta1: Float = 0.9,
        beta2: Float = 0.95,
        eps: Float = 1e-6,
        gradientAccumulationSteps: Int = 4,
        clipGradNorm: Float? = 1.0
    ) {
        precondition(gradientAccumulationSteps >= 1, "gradientAccumulationSteps en az 1 olmalıdır.")
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.clipGradNorm = clipGradNorm
    }
}

/// Her bir optimizer adımının çıktısını taşıyan `Sendable` sonuç yapısı.
public struct TrainingStepMetrics: Sendable {
    public let step: Int
    public let learningRate: Float
    public let averageLoss: Float
    public let perplexity: Float
    public let gradientNorm: Float
    public let activeMemoryMB: Double
}

// MARK: - 2. Float16 Alt-Normal Sıfırlanmasını (Underflow) Önleyen AdamW Alt Sınıfı

/// **Float16 Güvenli AdamW (`Float16SafeAdamW: AdamW`)**
/// Standart `MLXOptimizers.Adam` içerisinde `update = learningRate * m / (sqrt(v) + eps)` ifadesi
/// soldan sağa `(learningRate * m)` sırasıyla `.float16` uzayında hesaplandığında,
/// küçük öğrenme oranlarında (`1e-5 .. 3e-4`) pay `(learningRate * m) < 6e-8` olduğu için
/// bölme işleminden önce **Float16 alt-normal sınırının altına düşerek `0.0` (underflow)** olur ve ağırlıklar güncellenemez.
///
/// Bu `AdamW` alt sınıfı:
/// - Model ağırlıklarını (`parameter`) ve momentum durumlarını (`m`, `v`) bellekte **`.float16`** olarak saklar (8 GB GPU RAM sınırı korunur).
/// - Ancak adım içi `(m, v)` güncellemesini ve `m / (sqrt(v) + eps)` oranını **`.float32`** hassasiyetinde hesaplayıp
///   sonucu tekrar `.float16`'ya yazar. Böylece 1B Float16 modelde gradyan sönümlenmesi tamamen ortadan kalkar.
/// **BFloat16 / Float16 Karma Hassasiyetli Güvenli AdamW (`Float16SafeAdamW: AdamW`)**
/// - Model ağırlıklarını (`parameter`) ve ileri/geri yayılım aktivasyonlarını **`.bfloat16`** olarak tutar (`~238 MB`).
/// - Birinci (`m`) ve ikinci (`v`) moment durumlarını ve ağırlık güncelleme adımını **tam `.float32`** uzayında hesaplar.
/// - `.float32` güncellenmiş ağırlığı `.bfloat16` (7-bit mantissa) formatına yazarken **Yansız Stokastik Yuvarlama**
///   (*Unbiased Stochastic Rounding*) uygulayarak $0.5\text{ ULP}$ altındaki ince güncellemelerin yutulmasını (`Loss ~ 6.23` platosu) önler.
public final class Float16SafeAdamW: AdamW {
    override public func applySingle(
        gradient: MLXArray,
        parameter: MLXArray,
        state: AdamState
    ) -> (MLXArray, AdamState) {
        let targetDType = parameter.dtype
        let (b1, b2) = betas

        let exported = Dictionary(uniqueKeysWithValues: exportState(state))
        let mStored = exported["m"] ?? MLXArray.zeros(parameter.shape, dtype: .float32)
        let vStored = exported["v"] ?? MLXArray.zeros(parameter.shape, dtype: .float32)
        let stepPrev = exported["step"] ?? MLXArray(Float(0.0))
        let nextStep = stepPrev.asType(.float32) + MLXArray(Float(1.0))

        // 1. Gradyan, ağırlık ve momentleri Float32 uzayında işle
        let g32 = gradient.asType(.float32)
        let p32 = parameter.asType(.float32)
        let mPrev32 = mStored.asType(.float32)
        let vPrev32 = vStored.asType(.float32)

        // 2. Birinci (m) ve ikinci (v) momentleri tam Float32 hassasiyetinde güncelle
        let m32 = mPrev32 * b1 + g32 * (1.0 - b1)
        let v32 = vPrev32 * b2 + square(g32) * (1.0 - b2)

        // 3. Bias Correction (Başlangıç ve Resume adımlarında moment gecikmesini önler)
        let biasCorrection1 = 1.0 - pow(MLXArray(b1), nextStep)
        let biasCorrection2 = 1.0 - pow(MLXArray(b2), nextStep)
        let mHat32 = m32 / biasCorrection1
        let sqrtVHat32 = sqrt(v32 / biasCorrection2)

        // 4. Decoupled Weight Decay ve Ağırlık Güncellemesi (Float32 uzayında)
        let decayedP32 = p32 * (1.0 - learningRate * weightDecay)
        let stepUpdate32 = learningRate * (mHat32 / (sqrtVHat32 + eps))
        let updatedP32 = decayedP32 - stepUpdate32

        // 5. BFloat16 / Float16 Yansız Stokastik Yuvarlama (Unbiased Stochastic Rounding):
        // Float32'den BFloat16 (7-bit mantissa) veya Float16 (10-bit mantissa) formatına dönerken
        // 0.5 ULP altındaki ince ağırlık güncellemelerinin yutulmasını önler.
        let newParam: MLXArray
        if targetDType == .bfloat16 || targetDType == .float16 {
            let ulpFactor: Float = (targetDType == .bfloat16) ? (1.0 / 128.0) : (1.0 / 1024.0)
            let dither = MLXRandom.uniform(low: Float(-0.5), high: Float(0.5), updatedP32.shape) * abs(updatedP32) * ulpFactor
            let hasUpdate = abs(stepUpdate32) .> MLXArray(Float(1e-12))
            newParam = which(hasUpdate, updatedP32 + dither, updatedP32).asType(targetDType)
        } else {
            newParam = updatedP32.asType(targetDType)
        }

        // 6. Momentleri Float32 hassasiyetinde, model ağırlığını ise BFloat16 (targetDType) olarak kaydet
        let newState = importState([("m", m32), ("v", v32), ("step", nextStep)])
        return (newParam, newState)
    }
}

// MARK: - 3. Ön Eğitim ve SFT Sınıfı (Trainer / PreTrainer - Float16 Karma Hassasiyet)

/// `PreTrainer` ve `Trainer` isimlendirmelerinin her ikisini de destekleyen tür eşadı.
public typealias PreTrainer = Trainer

/// **PersonalFoundationModel 1B Ölçekli Float16 Ön Eğitim ve SFT Motoru**
/// - İleri besleme, ağırlık, gradyan ve optimizer durumlarını **Float16 (`.float16`)** olarak tutar (~8 GB GPU RAM).
/// - Cross-Entropy Loss, Küresel L2 Gradyan Normu ve AdamW oran adımını sayısal kararlılık için **Float32 (`.float32`)** uzayında yapar.
public final class Trainer {
    public let model: PersonalFoundationModel
    public let optimizer: AdamW
    public let config: TrainingConfig
    public let lrScheduler: LRScheduler?
    public private(set) var currentStep: Int = 0

    public init(
        model: PersonalFoundationModel,
        config: TrainingConfig = TrainingConfig(),
        lrScheduler: LRScheduler? = nil
    ) {
        self.model = model
        self.config = config
        self.lrScheduler = lrScheduler

        let initialLR = lrScheduler?.getLR(step: 1) ?? config.learningRate
        self.optimizer = Float16SafeAdamW(
            learningRate: initialLR,
            betas: config.betas,
            eps: config.eps,
            weightDecay: config.weightDecay
        )
    }

    // MARK: - Matematiksel Maskelenmiş Cross-Entropy Loss (Float32 Stabiliteli)

    /// **Maskelenmiş Otoregresif Nedensel Dil Modeli Kaybı (Float32 Cast + Log-Sum-Exp)**
    /// Modelden `.float16` olarak çıkan `[Batch, SeqLen, 32000]` boyutlu logitleri,
    /// `exp()` ve `logSumExp()` taşmalarını önlemek için **zorunlu olarak `.float32`'ye yükseltir (cast eder)**.
    public static func causalCrossEntropyLoss(
        logits: MLXArray,
        targets: MLXArray,
        ignoreIndex: Int32 = -100
    ) -> MLXArray {
        // KRİTİK: Float16 logitleri Loss hesabı sırasında sayısal kararlılık için Float32'ye yükselt
        let logitsF32 = logits.asType(.float32)

        // 1. Geçerli (maskelenmemiş) hedef pozisyonların boolean maskesi: targets != -100
        let validMask = targets .!= MLXArray(ignoreIndex)

        // 2. takeAlong indekslemesinde negatif indeks (-100) erişimini önlemek için güvenli `0` indeksi ata
        let safeTargets = which(validMask, targets, MLXArray(Int32(0)))

        // 3. Float32 uzayında Log-Sum-Exp: log(sum(exp(logits)))
        let logNormalizer = logSumExp(logitsF32, axis: -1)

        // 4. Doğru hedef sınıfın logit değerini seç: z_y
        let expandedTargets = expandedDimensions(safeTargets, axis: -1)
        let targetLogits = takeAlong(logitsF32, expandedTargets, axis: -1).squeezed(axis: -1)

        // 5. Ham Negatif Log-Olabilirlik: NLL = logSumExp(z) - z_y
        let rawTokenLosses = logNormalizer - targetLogits

        // 6. Maskelenmiş (-100) pozisyonların kaybını `which` ile 0.0 yap
        let maskedLosses = which(validMask, rawTokenLosses, MLXArray(Float(0.0)))

        // 7. Toplam kaybı yalnızca geçerli token sayısına böl
        let validTokenCount = maximum(sum(validMask.asType(.float32)), MLXArray(Float(1.0)))
        return sum(maskedLosses) / validTokenCount
    }

    // MARK: - Float16 Koruyucu Gradyan Ağacı Yardımcıları

    /// İki gradyan ağacını (`ModuleParameters`) kendi veri tiplerini (`.float16`) koruyarak toplar.
    private static func addGradients(
        _ lhs: ModuleParameters,
        _ rhs: ModuleParameters
    ) -> ModuleParameters {
        let lhsFlat = lhs.flattened()
        let rhsDict = Dictionary(uniqueKeysWithValues: rhs.flattened())

        var combined: [(String, MLXArray)] = []
        combined.reserveCapacity(lhsFlat.count)

        for (key, lhsArray) in lhsFlat {
            if let rhsArray = rhsDict[key] {
                combined.append((key, lhsArray + rhsArray.asType(lhsArray.dtype)))
            } else {
                combined.append((key, lhsArray))
            }
        }
        return ModuleParameters.unflattened(combined)
    }

    /// Gradyan ağacındaki tüm tensörleri kendi `.float16` veri tipini bozmadan `scale` ile çarpar.
    /// (Not: Float32 skaler ile doğrudan çarpılırsa MLX tensörü Float32'ye terfi ettirip RAM kullanımını 2 katına çıkarır;
    /// bu yüzden skaler değer tensörün kendi `dtype` tipine dönüştürülür.)
    private static func scaleGradients(
        _ grads: ModuleParameters,
        by scale: Float
    ) -> ModuleParameters {
        return grads.mapValues { g in
            g * MLXArray(scale).asType(g.dtype)
        }
    }

    /// **Küresel L2 Gradyan Normu Hesaplama ve Kırpma (`clipGradNorm = 1.0`)**
    /// 1 Milyar parametrenin kareler toplamı Float16 üst sınırını (`65504.0`) aşacağı için,
    /// norm kareler toplamı **Float32** uzayında biriktirilir; kırpma katsayısı (`clipCoef`) ise
    /// gradyanların `.float16` tipini korumak için her tensörün kendi tipine (`g.dtype`) cast edilerek uygulanır.
    private static func clipGradientNorm(
        _ grads: ModuleParameters,
        maxNorm: Float?
    ) -> (clippedGrads: ModuleParameters, totalNorm: MLXArray) {
        let flatGrads = grads.flattened().map { $0.1 }
        var sumSquaresF32 = MLXArray(Float(0.0))

        for g in flatGrads {
            let gF32 = g.asType(.float32)
            sumSquaresF32 = sumSquaresF32 + sum(gF32 * gF32)
        }
        let totalNormF32 = sqrt(sumSquaresF32)

        guard let maxNorm else {
            return (grads, totalNormF32)
        }

        // clipCoef = min(1.0, maxNorm / (totalNorm + 1e-6))
        let clipCoefF32 = minimum(
            MLXArray(Float(1.0)),
            MLXArray(maxNorm) / (totalNormF32 + MLXArray(Float(1e-6)))
        )

        let clipped = grads.mapValues { g in
            g * clipCoefF32.asType(g.dtype)
        }
        return (clipped, totalNormF32)
    }

    // MARK: - Tek Mikro-Adım Eğitimi (trainMicroStep)

    @discardableResult
    public func trainMicroStep(
        inputs: MLXArray,
        targets: MLXArray
    ) -> TrainingStepMetrics {
        return trainStepWithAccumulation(microBatches: [(inputs: inputs, targets: targets)])
    }

    // MARK: - Gradyan Birikimli 1B Float16 Eğitim Adımı

    @discardableResult
    public func trainStepWithAccumulation(
        microBatches: [(inputs: MLXArray, targets: MLXArray)]
    ) -> TrainingStepMetrics {
        precondition(!microBatches.isEmpty, "En az bir mikro-batch sağlanmalıdır.")

        model.train(true)
        let nextStep = currentStep + 1

        // 1. LRScheduler varsa o anki adımın öğrenme oranını al ve AdamW optimizer.learningRate değerini güncelle
        let activeLR = lrScheduler?.getLR(step: nextStep) ?? config.learningRate
        optimizer.learningRate = activeLR

        let numMicroSteps = microBatches.count
        let accumulationScale = 1.0 / Float(numMicroSteps)

        let lossAndGradFn = valueAndGrad(model: model) { (model: PersonalFoundationModel, args: [MLXArray]) -> [MLXArray] in
            let inputs = args[0]
            let targets = args[1]
            let logits = model(inputs) // Float16 İleri Besleme
            let rawLoss = Self.causalCrossEntropyLoss(logits: logits, targets: targets) // Float32 Loss
            return [rawLoss]
        }

        var accumulatedGradients: ModuleParameters? = nil
        var totalLossValue: Float = 0.0

        for (inputs, targets) in microBatches {
            // 2. Float16 İleri Besleme + Float32 Loss + Float16 Geriye Yayılım
            let (losses, microGrads) = lossAndGradFn(model, [inputs, targets])
            let microLoss = losses[0]

            // 3. Gradyanları Float16 tipini koruyarak ölçekle ve biriktir
            let scaledMicroGrads = Self.scaleGradients(microGrads, by: accumulationScale)
            if let existingGrads = accumulatedGradients {
                accumulatedGradients = Self.addGradients(existingGrads, scaledMicroGrads)
            } else {
                accumulatedGradients = scaledMicroGrads
            }

            // 4. KRİTİK BELLEK YÖNETİMİ (M5 24GB RAM Koruması):
            // Her mikro-batch sonunda tembel grafiği hemen değerlendir ve ara aktivasyon önbelleğini temizle
            if let gradsToEval = accumulatedGradients {
                eval(microLoss, gradsToEval)
            }

            totalLossValue += microLoss.item(Float.self) * accumulationScale
        }

        guard let finalGradients = accumulatedGradients else {
            fatalError("Gradyan birikimi başarısız oldu.")
        }

        // 5. Küresel Gradyan Kırpma (clipGradNorm = 1.0)
        let (clippedGradients, gradNormArray) = Self.clipGradientNorm(
            finalGradients,
            maxNorm: config.clipGradNorm
        )

        // 6. AdamW Ağırlık Güncellemesi
        optimizer.update(model: model, gradients: clippedGradients)

        // 7. Güncellenen 1B Float16 model parametrelerini ve optimizer durumunu değerlendir
        eval(model, optimizer, gradNormArray)

        currentStep = nextStep
        let gradNorm = gradNormArray.item(Float.self)
        let perplexity = exp(min(20.0, totalLossValue))

        // 8. Uzun süreli (50+ adım) 1B akışlı eğitimde bellek sızıntısını / tampon şişmesini önlemek için
        //    serbest kalan geçici Metal önbellek tamponlarını temizle
        MLX.Memory.clearCache()
        let activeMemMB = Double(MLX.Memory.activeMemory) / (1024.0 * 1024.0)

        return TrainingStepMetrics(
            step: currentStep,
            learningRate: activeLR,
            averageLoss: totalLossValue,
            perplexity: perplexity,
            gradientNorm: gradNorm,
            activeMemoryMB: activeMemMB
        )
    }

    public static func makeCausalBatch(tokenSequence: MLXArray) -> (inputs: MLXArray, targets: MLXArray) {
        let seqLenMinusOne = tokenSequence.dim(1) - 1
        precondition(seqLenMinusOne >= 1, "Sekans uzunluğu en az 2 olmalıdır (Giriş + Hedef).")
        let inputs = tokenSequence[0..., 0..<seqLenMinusOne]
        let targets = tokenSequence[0..., 1..<(seqLenMinusOne + 1)]
        return (inputs, targets)
    }
}
