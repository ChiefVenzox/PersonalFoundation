import Foundation
import Darwin
import MLX
import MLXRandom
import PersonalFoundation

// MARK: - 1. Çok Turlu Konuşma Hafızası Yöneticisi (MultiTurnChatMemory)

/// Tek bir Soru-Cevap konuşma turunu ChatML (`<|im_start|>...<|im_end|>`) formatında tutan `Sendable` yapı.
struct ConversationTurn: Sendable {
    let userQuestion: String
    let assistantAnswer: String
    let formattedChatML: String
    let tokenIds: [Int32]
}

/// **Çok Turlu Konuşma Hafızası (Multi-Turn History Manager)**
/// Önceki soruları ve asistan yanıtlarını:
///   `<|im_start|><|user|> {user} <|im_end|><|im_start|><|assistant|> {assistant} <|im_end|>`
/// formatında biriktirir. Toplam bağlam uzunluğu modelin `maxSequenceLength` (512) sınırına yaklaştığında
/// en eski konuşma turlarını kayan pencereden (*sliding window*) otomatik olarak düşürür.
struct MultiTurnChatMemory: Sendable {
    let maxSequenceLength: Int
    /// Yeni üretilecek yanıt için pencerede ayrılan güvenlik payı (token sayısı)
    let reservedResponseTokens: Int
    private(set) var activeTurns: [ConversationTurn] = []
    private(set) var evictedTurnCount: Int = 0

    init(maxSequenceLength: Int, reservedResponseTokens: Int = 120) {
        self.maxSequenceLength = maxSequenceLength
        self.reservedResponseTokens = reservedResponseTokens
    }

    /// Aktif pencerede birikmiş geçmiş turların toplam token sayısı
    var historyTokenCount: Int {
        activeTurns.reduce(0) { $0 + $1.tokenIds.count }
    }

    /// Yeni kullanıcı sorusu geldiğinde `maxSequenceLength` sınırını kontrol eder,
    /// sınır aşılıyorsa en eski turları pencereden düşürür ve tam ChatML istemini (Prompt) döndürür.
    mutating func buildPromptAndTrimWindow(
        for newQuestion: String,
        tokenizer: BPETokenizer
    ) -> (fullPrompt: String, promptTokenCount: Int, droppedTurnsInThisStep: Int) {
        let currentTurnPrompt = ChatSample.formatInferencePrompt(question: newQuestion)
        let currentTurnTokens = tokenizer.encode(currentTurnPrompt, addBOS: false, addEOS: false)

        let maxPromptBudget = max(32, maxSequenceLength - reservedResponseTokens)

        var dropped = 0
        while !activeTurns.isEmpty && (historyTokenCount + currentTurnTokens.count) > maxPromptBudget {
            activeTurns.removeFirst()
            evictedTurnCount += 1
            dropped += 1
        }

        let historyPrefix = activeTurns.map { $0.formattedChatML }.joined()
        let combinedPrompt = historyPrefix + currentTurnPrompt
        let totalPromptTokens = historyTokenCount + currentTurnTokens.count

        return (combinedPrompt, totalPromptTokens, dropped)
    }

    /// Tamamlanan turu `<|im_start|><|user|> ... <|im_end|><|im_start|><|assistant|> ... <|im_end|>` olarak hafızaya ekler.
    mutating func appendCompletedTurn(
        question: String,
        answer: String,
        tokenizer: BPETokenizer
    ) {
        let formatted = "<|im_start|><|user|> \(question) <|im_end|><|im_start|><|assistant|> \(answer) <|im_end|>"
        let tokens = tokenizer.encode(formatted, addBOS: false, addEOS: false)
        let turn = ConversationTurn(
            userQuestion: question,
            assistantAnswer: answer,
            formattedChatML: formatted,
            tokenIds: tokens
        )
        activeTurns.append(turn)
    }

    /// `/temizle` komutu ile tüm konuşma geçmişini sıfırlar.
    mutating func clear() {
        activeTurns.removeAll()
        evictedTurnCount = 0
    }
}

// MARK: - 2. PersonalFoundationChat (110M v4 BFloat16 Canlı Sohbet Motoru)

@main
struct PersonalFoundationChatApp {
    static func main() {
        let startupBegin = CFAbsoluteTimeGetCurrent()

        // Komut satırı argümanları ayrıştırma (--prompt, --json, --temperature, --max-tokens, --model-dir)
        var cliPrompt: String? = nil
        var isJSONOutput = false
        var cliTemperature: Float = 0.7
        var cliMaxTokens: Int = 120
        var customModelDir: URL? = nil

        var argIdx = 1
        while argIdx < CommandLine.arguments.count {
            let arg = CommandLine.arguments[argIdx]
            if (arg == "--prompt" || arg == "-p") && argIdx + 1 < CommandLine.arguments.count {
                cliPrompt = CommandLine.arguments[argIdx + 1]
                argIdx += 1
            } else if arg == "--json" {
                isJSONOutput = true
            } else if (arg == "--temperature" || arg == "-t") && argIdx + 1 < CommandLine.arguments.count {
                if let val = Float(CommandLine.arguments[argIdx + 1]) { cliTemperature = val }
                argIdx += 1
            } else if (arg == "--max-tokens" || arg == "-m") && argIdx + 1 < CommandLine.arguments.count {
                if let val = Int(CommandLine.arguments[argIdx + 1]) { cliMaxTokens = val }
                argIdx += 1
            } else if (arg == "--model-dir" || arg == "-d") && argIdx + 1 < CommandLine.arguments.count {
                customModelDir = URL(fileURLWithPath: CommandLine.arguments[argIdx + 1])
                argIdx += 1
            }
            argIdx += 1
        }

        let isInteractive = (cliPrompt == nil)

        if isInteractive {
            print("============================================================================================")
            print("   PersonalFoundation 110M v4 (32K Vocab | BFloat16) — Canlı KV-Cache Türkçe Sohbet Motoru")
            print("============================================================================================")
        }

        var searchDirs: [URL] = []
        if let customModelDir { searchDirs.append(customModelDir) }
        if let envDir = ProcessInfo.processInfo.environment["PERSONAL_FOUNDATION_DIR"] {
            searchDirs.append(URL(fileURLWithPath: envDir))
        }
        searchDirs.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        searchDirs.append(URL(fileURLWithPath: "/Users/hakankaba/Desktop/PersonalFoundation"))

        var resolvedTokenizerURL: URL? = nil
        var resolvedCheckpointURL: URL? = nil

        for dir in searchDirs {
            let tok = dir.appendingPathComponent("personalfoundation_110m_tokenizer.json")
            if FileManager.default.fileExists(atPath: tok.path) && resolvedTokenizerURL == nil {
                resolvedTokenizerURL = tok
            }
            let ckpt4 = dir.appendingPathComponent("personalfoundation_110m_chat_v4.safetensors")
            let ckpt3 = dir.appendingPathComponent("personalfoundation_110m_pretrain_v3.safetensors")
            let ckpt2 = dir.appendingPathComponent("personalfoundation_110m_pretrain_v2.safetensors")
            if FileManager.default.fileExists(atPath: ckpt4.path) && resolvedCheckpointURL == nil {
                resolvedCheckpointURL = ckpt4
            } else if FileManager.default.fileExists(atPath: ckpt3.path) && resolvedCheckpointURL == nil {
                resolvedCheckpointURL = ckpt3
            } else if FileManager.default.fileExists(atPath: ckpt2.path) && resolvedCheckpointURL == nil {
                resolvedCheckpointURL = ckpt2
            }
        }

        guard let tokenizerURL = resolvedTokenizerURL, let checkpointURL = resolvedCheckpointURL else {
            fputs("❌ Başlatma Hatası: Tokenizer ('personalfoundation_110m_tokenizer.json') veya model ağırlıkları (.safetensors) bulunamadı.\n", stderr)
            exit(1)
        }

        // 1. Doğrudan 32K BPE Sözlüğü ve v4 Safetensors Ağırlıklarını (.bfloat16 Çıkarım Modunda) Yükle
        let tokenizer: BPETokenizer
        let model: PersonalFoundationModel
        let header: CheckpointHeader

        do {
            tokenizer = try BPETokenizer.load(from: tokenizerURL)
            (model, header) = try CheckpointManager.load(from: checkpointURL, dtype: .bfloat16)
        } catch {
            fputs("❌ Başlatma Hatası: '\(tokenizerURL.lastPathComponent)' veya '\(checkpointURL.lastPathComponent)' yüklenemedi: \(error.localizedDescription)\n", stderr)
            exit(1)
        }

        let startupElapsedSec = CFAbsoluteTimeGetCurrent() - startupBegin
        let startupElapsedMs = startupElapsedSec * 1000.0
        let activeRAM_MB = Double(MLX.Memory.activeMemory) / (1024.0 * 1024.0)

        let generator = TextGenerator(model: model)

        // CLI / Ruby / Programatik Tek Seferlik Çağrı Modu
        if let prompt = cliPrompt {
            let formattedPrompt = ChatSample.formatInferencePrompt(question: prompt)
            let samplingConfig = SamplingConfig(
                temperature: cliTemperature,
                topK: 40,
                topP: 0.9,
                repetitionPenalty: 1.18,
                maxNewTokens: cliMaxTokens,
                eosTokenId: SpecialToken.imEnd.rawValue
            )

            let result = generator.generateText(
                prompt: formattedPrompt,
                tokenizer: tokenizer,
                config: samplingConfig,
                addBOS: false
            )

            if isJSONOutput {
                let jsonDict: [String: Any] = [
                    "prompt": prompt,
                    "response": result.generatedText,
                    "tokens_generated": result.result.generatedTokens.count,
                    "prefill_time_ms": result.result.prefillTimeMs,
                    "tokens_per_second": result.result.tokensPerSecond
                ]
                if let data = try? JSONSerialization.data(withJSONObject: jsonDict, options: [.sortedKeys]),
                   let str = String(data: data, encoding: .utf8) {
                    print(str)
                }
            } else {
                print(result.generatedText)
            }
            return
        }

        // İnteraktif REPL Bilgilendirme
        print(String(
            format: "✅ Model & 32K Sözlük Yüklendi | Açılış Süresi: %.2f sn (%.0f ms)",
            startupElapsedSec,
            startupElapsedMs
        ))
        print(String(
            format: "   • Checkpoint : %@ (Adım: %d | Val Loss: %.4f)",
            checkpointURL.lastPathComponent,
            header.step,
            header.averageLoss
        ))
        print(String(
            format: "   • Mimari     : %d Parametre (~%.2f MB BFloat16) | Sözlük: %d token | MaxSeqLen: %d",
            model.totalParameterCount(),
            model.allocatedParameterMB(),
            tokenizer.vocabSize,
            model.config.maxSequenceLength
        ))
        print(String(format: "   • Aktif RAM  : %.0f MB Birleşik Bellek (Unified Memory)", activeRAM_MB))
        print("--------------------------------------------------------------------------------------------")
        print("Komutlar: '/temizle' -> Konuşma hafızasını sıfırlar | '/cikis' -> Programdan çıkar")
        print("--------------------------------------------------------------------------------------------")

        var memory = MultiTurnChatMemory(
            maxSequenceLength: model.config.maxSequenceLength,
            reservedResponseTokens: 120
        )

        let samplingConfig = SamplingConfig(
            temperature: 0.7,
            topK: 40,
            topP: 0.9,
            repetitionPenalty: 1.18,
            maxNewTokens: 120,
            eosTokenId: SpecialToken.imEnd.rawValue
        )

        // 2. İnteraktif Terminal Sohbet Döngüsü (REPL)
        while true {
            print("\n👤 Kullanıcı > ", terminator: "")
            fflush(stdout)

            guard let rawInput = readLine() else {
                print("\nGirdi akışı (EOF) sonlandı. Güvenli çıkış yapılıyor...")
                break
            }

            let userInput = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
            if userInput.isEmpty {
                continue
            }

            if userInput.lowercased() == "/cikis" || userInput.lowercased() == "/çıkış" {
                print("👋 PersonalFoundationChat oturumu güvenli şekilde sonlandırıldı.")
                break
            }

            if userInput.lowercased() == "/temizle" {
                memory.clear()
                MLX.Memory.clearCache()
                print("🧹 Konuşma hafızası ve KV bağlam penceresi sıfırlandı.")
                continue
            }

            // 3. Çok Turlu Bağlam Penceresini Hazırla (maxSequenceLength=512 aşılırsa en eski turu düşür)
            let (fullPrompt, promptTokenCount, _) = memory.buildPromptAndTrimWindow(
                for: userInput,
                tokenizer: tokenizer
            )

            print("🤖 Asistan   > ", terminator: "")
            fflush(stdout)

            // 4. Gerçek Zamanlı UTF-8 Tamponlu Canlı Daktilo Akışı (Streaming Generation)
            let streamOutput = generator.generateTextStreaming(
                prompt: fullPrompt,
                tokenizer: tokenizer,
                config: samplingConfig,
                addBOS: false,
                onTextChunk: { textPiece in
                    print(textPiece, terminator: "")
                    fflush(stdout)
                }
            )

            print("")

            // 5. Tamamlanan Turu Çok Turlu Konuşma Hafızasına Ekle
            memory.appendCompletedTurn(
                question: userInput,
                answer: streamOutput.generatedText,
                tokenizer: tokenizer
            )

            let totalContextTokens = promptTokenCount + streamOutput.result.generatedTokens.count
            print(String(
                format: "📊 [Prefill: %.2f ms | Decode: %.1f token/sn | Toplam Bağlam: %d/%d token (Aktif Hafıza: %d tur)]",
                streamOutput.result.prefillTimeMs,
                streamOutput.result.tokensPerSecond,
                totalContextTokens,
                model.config.maxSequenceLength,
                memory.activeTurns.count
            ))
            fflush(stdout)
        }
    }
}
