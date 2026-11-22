import Foundation
import MLX
import MLXRandom
import PersonalFoundation

/// Apple Silicon (M5 - 24GB RAM) 110M (32.000 Vocab | True Mixed Precision: .float32 Master + .bfloat16 Compute)
/// Açık Kaynak Türkçe Sohbet Külliyatı (ChatCorpusFetcher -> turkish_chat_corpus.jsonl),
/// -100 Maskeli SFT Eğitimi (%95 Train / %5 Val, 1.200 Adım, v3 -> v4) ve Özgün Sohbet Doğrulaması
@main
struct PersonalFoundationTrainApp {
    static func main() async {
        print("======================================================================================================================")
        print("   PersonalFoundation 110M v4 (32K Vocab | True Mixed Precision) — Açık Kaynak Türkçe Chat SFT & %5 Val Loss Takibi")
        print("======================================================================================================================")

        let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let chatCorpusURL = currentDir.appendingPathComponent("turkish_chat_corpus.jsonl")
        let tokenizerURL = currentDir.appendingPathComponent("personalfoundation_110m_tokenizer.json")
        let v3PretrainURL = currentDir.appendingPathComponent("personalfoundation_110m_pretrain_v3.safetensors")
        let v2PretrainURL = currentDir.appendingPathComponent("personalfoundation_110m_pretrain_v2.safetensors")
        let v4ChatCheckpointURL = currentDir.appendingPathComponent("personalfoundation_110m_chat_v4.safetensors")

        // 1. Açık Kaynak Türkçe Sohbet Külliyatı Toplayıcı (turkish_chat_corpus.jsonl >= 2.500 çift ise atla, yoksa 3.500+ çift indir)
        print("\n[Aşama 1] Açık Kaynak Türkçe Talimat/Sohbet Külliyatı ('\(chatCorpusURL.lastPathComponent)') Hazırlanıyor...")
        let chatSamples: [ChatSample]
        do {
            chatSamples = try await ChatCorpusFetcher.ensureChatCorpusExists(
                at: chatCorpusURL,
                minValidPairs: 2_500,
                targetPairs: 3_600
            )
        } catch {
            print("❌ Sohbet Külliyatı Hatası: \(error.localizedDescription)")
            return
        }

        // 2. 32.000 Kelimelik BPE Sözlüğü (personalfoundation_110m_tokenizer.json) Yükle
        print("\n[Aşama 2] Kayıtlı 32.000 Kelimelik BPE Sözlük Yükleniyor -> \(tokenizerURL.lastPathComponent)")
        let tokenizer: BPETokenizer
        do {
            tokenizer = try BPETokenizer.load(from: tokenizerURL)
            precondition(SpecialToken.imStart.rawValue == 0, "<|im_start|> ID 0 olmalıdır.")
            precondition(SpecialToken.imEnd.rawValue == 1, "<|im_end|> ID 1 olmalıdır.")
            precondition(SpecialToken.user.rawValue == 2, "<|user|> ID 2 olmalıdır.")
            precondition(SpecialToken.assistant.rawValue == 3, "<|assistant|> ID 3 olmalıdır.")
            print("  ✅ 32K Sözlük Yüklendi | Vocab Size: \(tokenizer.vocabSize) | Özel Tokenler: <|im_start|>=0, <|im_end|>=1, <|user|>=2, <|assistant|>=3")
        } catch {
            print("❌ Sözlük Yükleme Hatası (\(tokenizerURL.path)): \(error.localizedDescription)")
            return
        }

        // 3. 110M Temel Modeli (v3) True Mixed Precision (.float32 Master + .bfloat16 Compute) Olarak Yükle
        let config110M = ModelConfig(
            vocabSize: 32_000,
            hiddenSize: 768,
            intermediateSize: 2_048,
            numLayers: 12,
            numHeads: 12,
            numKVHeads: 4,
            maxSequenceLength: 512
        )

        let baseCheckpointURL = FileManager.default.fileExists(atPath: v3PretrainURL.path)
            ? v3PretrainURL
            : v2PretrainURL

        print("\n[Aşama 3] 110M Temel Model (.float32 Master + .bfloat16 Compute) Başlatılıyor ve v3 Ağırlıkları Yükleniyor -> \(baseCheckpointURL.lastPathComponent)")
        let model = PersonalFoundationModel(config: config110M, seed: 2026, dtype: .float32)
        eval(model)

        do {
            let loadedHeader = try CheckpointManager.loadWeights(into: model, from: baseCheckpointURL)
            let activeRAM_MB = Double(MLX.Memory.activeMemory) / (1024.0 * 1024.0)
            print(String(
                format: "  ✅ v3 Temel Model Yüklendi | Kayıtlı Adım: %d | Temel Model Val Loss: %.4f | Tensör: %d",
                loadedHeader.step,
                loadedHeader.averageLoss,
                loadedHeader.tensorCount
            ))
            print(String(
                format: "  • Parametre Sayısı: %d (~%.1fM) | Master Ağırlık Belleği: %.2f MB Float32 | Aktif GPU RAM: %.0f MB",
                model.totalParameterCount(),
                Double(model.totalParameterCount()) / 1e6,
                model.allocatedParameterMB(),
                activeRAM_MB
            ))
        } catch {
            print("❌ Temel Model Yükleme Hatası (\(baseCheckpointURL.path)): \(error.localizedDescription)")
            return
        }

        // 4. ChatDataLoader (-100 Maskeli & %95 Train / %5 Validation Ayrımlı) Kurulumu
        let batchSize = 4
        let seqLength = 512
        let accumulationSteps = 2 // 4 * 2 = 8 tam sohbet / adım

        print("\n[Aşama 4] Sohbet Külliyatı -100 Maskeli ChatML Formatına Dönüştürülüyor (%95 Train / %5 Validation)...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let chatLoader = ChatDataLoader(
            samples: chatSamples,
            tokenizer: tokenizer,
            sequenceLength: seqLength,
            microBatchSize: batchSize,
            trainSplitRatio: 0.95,
            seed: 2026
        )
        let loadElapsed = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(
            format: "  ✅ ChatDataLoader Hazır (%.2f sn) | Toplam Geçerli Sohbet: %d | %%95 Train: %d | %%5 Val: %d",
            loadElapsed,
            chatLoader.encodedSamples.count,
            chatLoader.trainSamples.count,
            chatLoader.valSamples.count
        ))

        if let firstSample = chatLoader.trainSamples.first {
            let unmaskedTargetIds = firstSample.targets.filter { $0 != ChatDataLoader.ignoreIndex }
            let unmaskedPreview = tokenizer.decode(Array(unmaskedTargetIds.prefix(18)), skipSpecialTokens: false)
            print("  • Maskeleme Doğrulaması : Kullanıcı kısmı -100 ile maskelendi | İlk Örnek Aktif Hedef Önizleme: \"\(unmaskedPreview)...\"")
        }

        // Başlangıç Masked Validation Loss ölçümü (Adım 0)
        let initialVal = chatLoader.evaluateValidationLoss(model: model, numBatches: 8)
        print(String(
            format: "  📊 [Başlangıç Doğrulama] Adım 0000 | Masked Val Loss: %.4f | Val PPL: %8.2f",
            initialVal.valLoss,
            initialVal.valPerplexity
        ))

        // 5. 1.200 Adımlık Masked SFT Eğitim Döngüsü
        // batchSize = 4, seqLength = 512, gradientAccumulationSteps = 2,
        // totalSteps = 1200, warmupSteps = 80, maxLR = 1.5e-4, minLR = 1e-5, clipGradNorm = 1.0
        let totalSteps = 1_200
        let checkpointInterval = 200

        let lrScheduler = LRScheduler(
            warmupSteps: 80,
            totalSteps: totalSteps,
            maxLR: 1.5e-4,
            minLR: 1e-5
        )

        let sftConfig = TrainingConfig(
            learningRate: 1.5e-4,
            weightDecay: 0.05,
            gradientAccumulationSteps: accumulationSteps,
            clipGradNorm: 1.0
        )

        let trainer = PreTrainer(
            model: model,
            config: sftConfig,
            lrScheduler: lrScheduler
        )

        print("\n[Aşama 5] 110M True Mixed Precision 1.200 Adım Masked Chat SFT Başlıyor (Warmup: 80 | MaxLR: 1.5e-4 -> MinLR: 1e-5)...")
        print("----------------------------------------------------------------------------------------------------------------------------")

        var bestValLoss: Float = Float.infinity

        for step in 1...totalSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()
            let microBatches = chatLoader.nextAccumulationGroup(
                accumulationSteps: sftConfig.gradientAccumulationSteps
            )
            let metrics = trainer.trainStepWithAccumulation(microBatches: microBatches)
            let stepDuration = CFAbsoluteTimeGetCurrent() - stepStart

            if step == 1 || step % 25 == 0 || step == totalSteps {
                print(
                    String(
                        format: "Adım %04d/%04d | LR: %.6f | Masked Train Loss: %.4f | Train PPL: %7.2f | GradNorm: %.4f | GPU RAM: %.0f MB (%.2f sn)",
                        metrics.step,
                        totalSteps,
                        metrics.learningRate,
                        metrics.averageLoss,
                        metrics.perplexity,
                        metrics.gradientNorm,
                        metrics.activeMemoryMB,
                        stepDuration
                    )
                )
            }

            // Her 200 adımda bir %5 Val üzerinde Validation Loss hesapla ve personalfoundation_110m_chat_v4.safetensors kaydet
            if step % checkpointInterval == 0 {
                let valMetrics = chatLoader.evaluateValidationLoss(model: model, numBatches: 8)
                let isBest = valMetrics.valLoss <= bestValLoss
                if isBest {
                    bestValLoss = valMetrics.valLoss
                }

                print(
                    String(
                        format: "🔎 [Validation - Adım %04d] Train Loss: %.4f (PPL: %.2f) | Val Loss: %.4f (Val PPL: %.2f)%@",
                        step,
                        metrics.averageLoss,
                        metrics.perplexity,
                        valMetrics.valLoss,
                        valMetrics.valPerplexity,
                        isBest ? " ⭐️ [Yeni En İyi]" : ""
                    )
                )

                do {
                    let report = try CheckpointManager.save(
                        model: model,
                        step: step,
                        averageLoss: valMetrics.valLoss,
                        to: v4ChatCheckpointURL
                    )
                    print(String(
                        format: "💾 [Ara Kayıt - Adım %04d] '%@' kaydedildi (Val Loss: %.4f | Boyut: %.2f MB)",
                        step,
                        v4ChatCheckpointURL.lastPathComponent,
                        valMetrics.valLoss,
                        report.sizeMB
                    ))
                } catch {
                    print("⚠️ [Ara Kayıt Hatası - Adım \(step)]: \(error.localizedDescription)")
                }
            }
        }
        print("----------------------------------------------------------------------------------------------------------------------------")

        // 6. Eğitim Sonu Özgün Sohbet Doğrulaması (Eğitim Setinde Birebir Geçmeyen 3 Serbest Soru)
        print("\n[Aşama 6] 110M v4 KV-Cache Özgün Türkçe Sohbet Doğrulaması (temperature: 0.7, topP: 0.9, repetitionPenalty: 1.18, durma: <|im_end|>)...")
        MLXRandom.seed(42)
        let generator = TextGenerator(model: model)

        let testUserQuestions = [
            "Yaşım 27 oldu artık hayat zorlaştı be!",
            "Hayat bazen insanı yorsa da sence ne yapmalıyız?",
            "Biyolojide mayoz bölünme neden önemlidir, kısaca açıklar mısın?"
        ]

        let chatSamplingConfig = SamplingConfig(
            temperature: 0.7,
            topK: 40,
            topP: 0.9,
            repetitionPenalty: 1.18,
            maxNewTokens: 80,
            eosTokenId: SpecialToken.imEnd.rawValue // <|im_end|> (ID: 1) üretildiği anda durur
        )

        for (idx, question) in testUserQuestions.enumerated() {
            let prompt = ChatSample.formatInferencePrompt(question: question)
            let output = generator.generateText(
                prompt: prompt,
                tokenizer: tokenizer,
                config: chatSamplingConfig,
                addBOS: false
            )

            print("----------------------------------------------------------------------------------------------------------------------------")
            print("Soru \(idx + 1)          : \"\(question)\"")
            print("ChatML Prompt   : \(prompt)")
            print("110M v4 Asistan : \(output.generatedText)")
            print(String(
                format: "KV-Cache Hızı   : Prefill: %.2f ms | Decode: %.1f token/sn | Üretilen Token: %d",
                output.result.prefillTimeMs,
                output.result.tokensPerSecond,
                output.result.generatedTokens.count
            ))
        }
        print("----------------------------------------------------------------------------------------------------------------------------")
        print("=== PersonalFoundation 110M Chat SFT v4 Eğitimi ve Doğrulaması Başarıyla Tamamlandı ===")
    }
}
