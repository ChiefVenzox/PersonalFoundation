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

        print("============================================================================================")
        print("   PersonalFoundation 110M v4 (32K Vocab | BFloat16) — Canlı KV-Cache Türkçe Sohbet Motoru")
        print("============================================================================================")

        let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let tokenizerURL = currentDir.appendingPathComponent("personalfoundation_110m_tokenizer.json")
        let ckpt110MV4 = currentDir.appendingPathComponent("personalfoundation_110m_chat_v4.safetensors")
        let ckpt110MV3 = currentDir.appendingPathComponent("personalfoundation_110m_pretrain_v3.safetensors")
        let ckpt110MV2 = currentDir.appendingPathComponent("personalfoundation_110m_pretrain_v2.safetensors")

        let checkpointURL: URL
        if FileManager.default.fileExists(atPath: ckpt110MV4.path) {
            checkpointURL = ckpt110MV4
        } else if FileManager.default.fileExists(atPath: ckpt110MV3.path) {
            checkpointURL = ckpt110MV3
        } else {
            checkpointURL = ckpt110MV2
        }

        // 1. Hiçbir eğitim yapmadan doğrudan 32K BPE Sözlüğü ve v4 Safetensors Ağırlıklarını (.bfloat16 Çıkarım Modunda) Yükle
        let tokenizer: BPETokenizer
        let model: PersonalFoundationModel
        let header: CheckpointHeader

        do {
            tokenizer = try BPETokenizer.load(from: tokenizerURL)
            (model, header) = try CheckpointManager.load(from: checkpointURL, dtype: .bfloat16)
        } catch {
            print("❌ Başlatma Hatası: '\(tokenizerURL.lastPathComponent)' veya '\(checkpointURL.lastPathComponent)' yüklenemedi.")
            print("   Detay: \(error.localizedDescription)")
            return
        }

        let startupElapsedSec = CFAbsoluteTimeGetCurrent() - startupBegin
        let startupElapsedMs = startupElapsedSec * 1000.0
        let activeRAM_MB = Double(MLX.Memory.activeMemory) / (1024.0 * 1024.0)

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

        let generator = TextGenerator(model: model)
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
