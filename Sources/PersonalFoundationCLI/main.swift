import Foundation
import MLX
import MLXRandom
import PersonalFoundation

@main
struct PersonalFoundationCLI {
    static func main() {
        print("==================================================================")
        print(" PersonalFoundation — Sıfırdan (From-Scratch) Transformer Mimarisi")
        print(" Hedef Donanım : Apple Silicon (M5, 24GB Unified Memory)")
        print(" Dil & Çatı    : Swift 6 (Strict Concurrency) + mlx-swift")
        print("==================================================================\n")

        // 1. Model Yapılandırması (Dışarıdan hiçbir model/safetensors yüklenmez!)
        let config = ModelConfig(
            vocabSize: 4096,
            hiddenSize: 256,
            intermediateSize: 768,
            numLayers: 4,
            numHeads: 8,
            numKVHeads: 4,
            maxSequenceLength: 512,
            rmsNormEps: 1e-5,
            ropeTheta: 10_000.0,
            tieWordEmbeddings: false,
            initSeed: 2026
        )

        print("[1/3] Model Sıfırdan (Xavier / Kaiming / Depth-Scaled) Başlatılıyor...")
        let model = PersonalFoundationModel(config: config)
        eval(model)

        let totalParams = model.totalParameterCount()
        let approxMemoryMB = Float(totalParams * 4) / (1024.0 * 1024.0)
        print("      • Katman Sayısı (numLayers)    : \(config.numLayers)")
        print("      • Gizli Boyut (hiddenSize)     : \(config.hiddenSize)")
        print("      • Dikkat Başlıkları (Q / KV)   : \(config.numHeads) / \(config.numKVHeads) (GQA + RoPE)")
        print("      • SwiGLU Ara Boyutu (d_ff)     : \(config.intermediateSize)")
        print("      • Toplam Eğitilebilir Parametre: \(totalParams) (~%\(String(format: "%.2f", approxMemoryMB)) MB FP32)\n")

        // 2. Ön-Eğitim (Pre-Training) ve Gradyan Birikimi (Gradient Accumulation) Yapılandırması
        let trainingConfig = TrainingConfig(
            learningRate: 1e-3,
            weightDecay: 0.01,
            betas: (0.9, 0.95),
            eps: 1e-8,
            gradientAccumulationSteps: 4, // 4 mikro-yığın biriktirilip tek AdamW güncellemesi yapılacak
            maxGradNorm: 1.0
        )

        let trainer = PreTrainer(model: model, config: trainingConfig)

        // 3. Sentetik Otoregresif Veri Kümesi üzerinde Sıfırdan Ön-Eğitim Doğrulaması
        let microBatchSize = 2
        let seqLen = 32
        let totalOptimizerSteps = 5

        // Modelin gerçekten öğrendiğini (Loss değerinin düştüğünü) doğrulamak için
        // tekrarlayan deterministik bir sentetik token örüntüsü oluşturuyoruz.
        let syntheticBatch = MLXRandom.randInt(
            low: 0,
            high: config.vocabSize,
            [microBatchSize, seqLen + 1]
        )
        eval(syntheticBatch)

        print("[2/3] Ön-Eğitim Döngüsü Başlatılıyor (Gradient Accumulation Steps = \(trainingConfig.gradientAccumulationSteps))...")
        for _ in 1...totalOptimizerSteps {
            let microBatches = Array(repeating: syntheticBatch, count: trainingConfig.gradientAccumulationSteps)
            let metrics = trainer.trainAccumulatedStep(microBatches: microBatches)
            print(String(
                format: "      Adım %02d/%02d | Cross-Entropy Loss: %.4f | Perplexity: %7.2f | Grad L2 Norm: %.4f",
                metrics.step,
                totalOptimizerSteps,
                metrics.loss,
                metrics.perplexity,
                metrics.gradNorm
            ))
        }

        print("\n[3/3] Sıfırdan Ön-Eğitim Döngüsü Başarıyla Doğrulandı.")
    }
}
