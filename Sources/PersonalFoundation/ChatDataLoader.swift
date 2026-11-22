import Foundation
import MLX

// MARK: - 1. Soru-Cevap Veri Modeli (ChatSample)

/// **Sohbet / Talimat İnce Ayarı (Chat / Instruction Fine-Tuning) Örneği**
/// Tek bir Soru-Cevap (User -> Assistant) etkileşimini temsil eden `Sendable` yapı.
public struct ChatSample: Sendable, Codable, Equatable {
    public let question: String
    public let answer: String

    public init(question: String, answer: String) {
        self.question = question
        self.answer = answer
    }

    /// Çıkarım (Inference) sırasında modele verilecek standart sohbet istemi (Prompt) formatı:
    /// `<|im_start|><|user|> {question} <|im_end|><|im_start|><|assistant|>`
    public static func formatInferencePrompt(question: String) -> String {
        return "<|im_start|><|user|> \(question) <|im_end|><|im_start|><|assistant|>"
    }

    /// Eğitim sırasında kullanılan tam sohbet metni:
    /// `<|im_start|><|user|> {question} <|im_end|><|im_start|><|assistant|> {answer} <|im_end|>`
    public var formattedConversation: String {
        return "<|im_start|><|user|> \(question) <|im_end|><|im_start|><|assistant|> \(answer) <|im_end|>"
    }
}

// MARK: - 2. %95 Train / %5 Validation Ayrımlı & -100 Maskeli Sohbet Veri Yükleyicisi (ChatDataLoader)

/// **Maskelenmiş Kayıp (Masked Loss) ve %95 Train / %5 Val Destekli Sohbet Veri Yükleyicisi (`ChatDataLoader`)**
/// Soru-Cevap çiftlerini şu yapıya dönüştürür:
///   `<|im_start|><|user|> {user} <|im_end|><|im_start|><|assistant|> {assistant} <|im_end|>`
///
/// **Matematiksel Hedef Maskeleme Kuralı:**
/// - `<|assistant|>` tokenine kadar olan tüm kullanıcı kısmı (`0 ..< L_prompt - 1`) ve sağdaki `<pad>` dolguları
///   hedeflerde **`-100` (`ignoreIndex`)** ile maskelenir.
/// - Model yalnızca `<|assistant|>` etiketinden sonraki asistan yanıtını ve `<|im_end|>` (`ID: 1`) durdurma tokenini
///   tahmin etmekten Cross-Entropy Loss ve gradyan hesaplar.
/// - Veri kümesinin **%95'i eğitim (`trainSamples`)**, **%5'i doğrulama (`valSamples`)** olarak ayrılır.
public final class ChatDataLoader {
    /// Kayıp fonksiyonunda yoksayılacak (maskelenecek) hedef indeks değeri
    public static let ignoreIndex: Int32 = -100

    /// Tüm kodlanmış örnekler (`trainSamples + valSamples`)
    public let encodedSamples: [(inputs: [Int32], targets: [Int32])]
    /// %95 Eğitim örnekleri
    public let trainSamples: [(inputs: [Int32], targets: [Int32])]
    /// %5 Doğrulama (Validation) örnekleri (Eğitim sırasında modele gösterilmez)
    public let valSamples: [(inputs: [Int32], targets: [Int32])]
    /// Sabit sekans uzunluğu ($L_{\text{seq}}$, Varsayılan: `512`)
    public let sequenceLength: Int
    /// Mikro-batch boyutu ($B_{\text{micro}}$, Varsayılan: `4`)
    public let microBatchSize: Int

    private var trainCursor: Int = 0
    private var valCursor: Int = 0
    private var shuffledTrainIndices: [Int]
    private var rngState: UInt64

    public init(
        samples: [ChatSample],
        tokenizer: BPETokenizer,
        sequenceLength: Int = 512,
        microBatchSize: Int = 4,
        trainSplitRatio: Double = 0.95,
        seed: UInt64 = 2026
    ) {
        precondition(!samples.isEmpty, "ChatDataLoader için en az bir soru-cevap örneği gereklidir.")
        precondition(sequenceLength >= 4, "sequenceLength en az 4 olmalıdır.")
        precondition(microBatchSize >= 1, "microBatchSize en az 1 olmalıdır.")

        self.sequenceLength = sequenceLength
        self.microBatchSize = microBatchSize
        self.rngState = seed == 0 ? 0x9E3779B97F4A7C15 : seed

        let padId = SpecialToken.pad.rawValue
        let maskId = Self.ignoreIndex
        let imStartId = SpecialToken.imStart.rawValue
        let imEndId = SpecialToken.imEnd.rawValue
        let userId = SpecialToken.user.rawValue
        let assistantId = SpecialToken.assistant.rawValue
        let limitId = Int32(tokenizer.vocabSize)

        // 3.500+ çifti saniyenin altında kodlamak için kelime önbellekli (memoized) BPE kodlayıcı
        var wordCache: [String: [Int32]] = [:]
        wordCache.reserveCapacity(32_768)

        func encodePlainFast(_ text: String) -> [Int32] {
            let chunks = BPETokenizer.preTokenize(text)
            var out: [Int32] = []
            out.reserveCapacity(chunks.count * 2)
            for chunk in chunks {
                if let cached = wordCache[chunk] {
                    out.append(contentsOf: cached)
                } else {
                    let enc = tokenizer.encode(chunk, addBOS: false, addEOS: false).map { id in
                        (id >= 0 && id < limitId) ? id : SpecialToken.unk.rawValue
                    }
                    wordCache[chunk] = enc
                    out.append(contentsOf: enc)
                }
            }
            return out
        }

        var prepared: [(inputs: [Int32], targets: [Int32])] = []
        prepared.reserveCapacity(samples.count)

        for sample in samples {
            // 1. Prompt: <|im_start|><|user|> {user} <|im_end|><|im_start|><|assistant|>
            let userBodyTokens = encodePlainFast(" " + sample.question + " ")
            var promptTokens: [Int32] = []
            promptTokens.reserveCapacity(userBodyTokens.count + 5)
            promptTokens.append(imStartId)
            promptTokens.append(userId)
            promptTokens.append(contentsOf: userBodyTokens)
            promptTokens.append(imEndId)
            promptTokens.append(imStartId)
            promptTokens.append(assistantId)

            // 2. Answer: {assistant} <|im_end|>
            let answerBodyTokens = encodePlainFast(" " + sample.answer + " ")
            var answerTokens: [Int32] = []
            answerTokens.reserveCapacity(answerBodyTokens.count + 1)
            answerTokens.append(contentsOf: answerBodyTokens)
            answerTokens.append(imEndId)

            let fullTokens = promptTokens + answerTokens
            let totalCount = fullTokens.count

            // Çok uzun (> sequenceLength) örnekleri ele ki her örnek <|im_end|> ile eksiksiz bitsin
            guard totalCount >= 6 && totalCount <= (sequenceLength + 1) else {
                continue
            }

            let promptLen = promptTokens.count
            let pairLen = min(sequenceLength, totalCount - 1)

            var inputRow = [Int32](repeating: padId, count: sequenceLength)
            var targetRow = [Int32](repeating: maskId, count: sequenceLength)

            for t in 0..<pairLen {
                inputRow[t] = fullTokens[t]
                // t == promptLen - 1 olduğunda inputRow[t] == <|assistant|> tokenıdır
                // ve fullTokens[t + 1] asistan cevabının İLK tokenıdır.
                if t >= (promptLen - 1) {
                    targetRow[t] = fullTokens[t + 1]
                } else {
                    targetRow[t] = maskId
                }
            }

            prepared.append((inputs: inputRow, targets: targetRow))
        }

        self.encodedSamples = prepared

        if prepared.count >= 20 {
            let splitIdx = max(1, min(prepared.count - 1, Int(Double(prepared.count) * trainSplitRatio)))
            self.trainSamples = Array(prepared[0..<splitIdx])
            self.valSamples = Array(prepared[splitIdx..<prepared.count])
        } else {
            self.trainSamples = prepared
            self.valSamples = prepared
        }

        self.shuffledTrainIndices = Array(0..<self.trainSamples.count)
        shuffleTrainOrder()
    }

    /// Epoch başında eğitim indekslerini deterministik PRNG ile karıştırır.
    private func shuffleTrainOrder() {
        guard shuffledTrainIndices.count > 1 else { return }
        for i in stride(from: shuffledTrainIndices.count - 1, through: 1, by: -1) {
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            let j = Int(rngState % UInt64(i + 1))
            shuffledTrainIndices.swapAt(i, j)
        }
    }

    /// `%95 trainSamples` içerisinden tek bir mikro-batch (`(inputs: MLXArray, targets: MLXArray)`) döndürür.
    public func nextMicroBatch() -> (inputs: MLXArray, targets: MLXArray) {
        var inputBuffer: [Int32] = []
        var targetBuffer: [Int32] = []
        inputBuffer.reserveCapacity(microBatchSize * sequenceLength)
        targetBuffer.reserveCapacity(microBatchSize * sequenceLength)

        for _ in 0..<microBatchSize {
            if trainCursor >= shuffledTrainIndices.count {
                trainCursor = 0
                shuffleTrainOrder()
            }
            let sampleIdx = shuffledTrainIndices[trainCursor]
            trainCursor += 1

            let sample = trainSamples[sampleIdx]
            inputBuffer.append(contentsOf: sample.inputs)
            targetBuffer.append(contentsOf: sample.targets)
        }

        let inputsArray = MLXArray(inputBuffer, [microBatchSize, sequenceLength])
        let targetsArray = MLXArray(targetBuffer, [microBatchSize, sequenceLength])
        return (inputs: inputsArray, targets: targetsArray)
    }

    /// `PreTrainer.trainStepWithAccumulation` için `accumulationSteps` adet mikro-batch döndürür.
    public func nextAccumulationGroup(accumulationSteps: Int) -> [(inputs: MLXArray, targets: MLXArray)] {
        return (0..<accumulationSteps).map { _ in nextMicroBatch() }
    }

    /// `%5 valSamples` üzerinden bir doğrulama mikro-batch'i döndürür.
    public func nextValidationBatch() -> (inputs: MLXArray, targets: MLXArray) {
        var inputBuffer: [Int32] = []
        var targetBuffer: [Int32] = []
        inputBuffer.reserveCapacity(microBatchSize * sequenceLength)
        targetBuffer.reserveCapacity(microBatchSize * sequenceLength)

        for _ in 0..<microBatchSize {
            if valCursor >= valSamples.count {
                valCursor = 0
            }
            let sample = valSamples[valCursor]
            valCursor += 1

            inputBuffer.append(contentsOf: sample.inputs)
            targetBuffer.append(contentsOf: sample.targets)
        }

        let inputsArray = MLXArray(inputBuffer, [microBatchSize, sequenceLength])
        let targetsArray = MLXArray(targetBuffer, [microBatchSize, sequenceLength])
        return (inputs: inputsArray, targets: targetsArray)
    }

    /// Modelin eğitimde hiç görmediği `%5 valSamples` üzerinde yalnızca `<|assistant|> ... <|im_end|>`
    /// yanıtları için Maskelenmiş Doğrulama Kaybını (**Masked Val Loss**) ve **Val Perplexity** değerini hesaplar.
    public func evaluateValidationLoss(
        model: PersonalFoundationModel,
        numBatches: Int = 8
    ) -> (valLoss: Float, valPerplexity: Float) {
        let batches = max(1, numBatches)
        model.train(false)
        defer { model.train(true) }

        var totalLoss: Float = 0.0
        for _ in 0..<batches {
            let valBatch = nextValidationBatch()
            let logits = model(valBatch.inputs)
            let lossArray = Trainer.causalCrossEntropyLoss(
                logits: logits,
                targets: valBatch.targets,
                ignoreIndex: Self.ignoreIndex
            )
            eval(lossArray)
            totalLoss += lossArray.item(Float.self)
        }

        MLX.Memory.clearCache()
        let avgValLoss = totalLoss / Float(batches)
        let valPPL = exp(min(20.0, avgValLoss))
        return (avgValLoss, valPPL)
    }
}
