import Foundation
import MLX

// MARK: - Belge Sınırlı (<|im_end|>) ve %95 Train / %5 Validation Ayrımlı Veri Yükleyici (CorpusDataLoader)

/// **Belge Sınırlı (`<|im_end|>`) ve Eğitim/Doğrulama Ayrımlı Külliyat Veri Yükleyici (`CorpusDataLoader`)**
/// 1. `corpus.txt` içindeki her paragraf/makale bloğunu `BPETokenizer` ile tokenlaştırırken
///    sonuna **`<|im_end|>` (`ID: 1`)** sınır tokeni ekler; böylece model cümle ve paragraf bitişlerini net öğrenir.
/// 2. Toplam token dizisinin **%95'ini eğitim (`trainTokens`)**, **%5'ini doğrulama (`valTokens`)** olarak ayırır.
/// 3. Eğitim adımlarında yalnızca `trainTokens` üzerinden rastgele `seqLength = 512` pencereleri çeker;
///    doğrulama aşamasında ise hiç görülmemiş `valTokens` üzerinde **Validation Loss** hesaplar.
public final class CorpusDataLoader {
    /// Tüm külliyatın toplam token sayısı (`trainTokens.count + valTokens.count`)
    public let totalTokenCount: Int
    /// %95 Eğitim token dizisi
    public let trainTokens: [Int32]
    /// %5 Doğrulama (Validation) token dizisi (Eğitim sırasında modele hiç gösterilmez)
    public let valTokens: [Int32]
    /// Çekilecek pencere uzunluğu (Varsayılan: `512` -> `inputs: 511`, `targets: 511`)
    public let seqLength: Int
    /// Her mikro-adımda çekilecek örnek sayısı (`batchSize`, Varsayılan: `4`)
    public let batchSize: Int

    /// Eğitim penceresi PRNG durumu
    private var trainRngState: UInt64
    /// Doğrulama penceresi PRNG durumu
    private var valRngState: UInt64

    /// Geriye dönük uyumluluk için `tokens` özelliği (`trainTokens` döndürür)
    public var tokens: [Int32] { trainTokens }

    public init(
        corpusURL: URL,
        tokenizer: BPETokenizer,
        seqLength: Int = 512,
        batchSize: Int = 4,
        trainSplitRatio: Double = 0.95,
        seed: UInt64 = 2026
    ) throws {
        precondition(seqLength >= 2, "seqLength en az 2 olmalıdır.")
        precondition(batchSize >= 1, "batchSize en az 1 olmalıdır.")
        precondition(trainSplitRatio > 0.5 && trainSplitRatio < 1.0, "trainSplitRatio (0.5, 1.0) aralığında olmalıdır.")

        let text = try String(contentsOf: corpusURL, encoding: .utf8)
        let lines = text.components(separatedBy: .newlines)

        // Her paragraf/belge bloğunu kelime önbellekli (memoized) BPE ile kodla ve sonuna <|im_end|> (ID: 1) ekle
        let imEndId = SpecialToken.imEnd.rawValue
        let limitId = Int32(tokenizer.vocabSize)
        var wordCache: [String: [Int32]] = [:]
        wordCache.reserveCapacity(40_000)

        var allTokens: [Int32] = []
        allTokens.reserveCapacity(text.utf8.count / 3)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let chunks = BPETokenizer.preTokenize(trimmed)
            guard !chunks.isEmpty else { continue }

            for chunk in chunks {
                if let cached = wordCache[chunk] {
                    allTokens.append(contentsOf: cached)
                } else {
                    let encoded = tokenizer.encode(chunk, addBOS: false, addEOS: false).map { id in
                        (id >= 0 && id < limitId) ? id : SpecialToken.unk.rawValue
                    }
                    wordCache[chunk] = encoded
                    allTokens.append(contentsOf: encoded)
                }
            }
            // Paragraf / belge sonuna <|im_end|> (ID: 1) sınır tokeni ekle
            allTokens.append(imEndId)
        }

        // Külliyat çok küçükse güvenli minimum uzunluğa tamamla
        let minRequired = seqLength * 25
        if !allTokens.isEmpty && allTokens.count < minRequired {
            let base = allTokens
            while allTokens.count < minRequired {
                allTokens.append(contentsOf: base)
            }
        }
        precondition(allTokens.count >= minRequired, "Külliyat eğitim ve %5 validation ayrımı için yeterli token içermiyor.")

        // %95 Eğitim (trainTokens) ve %5 Doğrulama (valTokens) ayrımı
        let splitIdx = max(seqLength * 2, min(allTokens.count - (seqLength * 2), Int(Double(allTokens.count) * trainSplitRatio)))
        self.trainTokens = Array(allTokens[0..<splitIdx])
        self.valTokens = Array(allTokens[splitIdx..<allTokens.count])
        self.totalTokenCount = allTokens.count
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.trainRngState = seed == 0 ? 0x9E3779B97F4A7C15 : seed
        self.valRngState = (seed ^ 0xBF58476D1CE4E5B9) == 0 ? 0x94D049BB133111EB : (seed ^ 0xBF58476D1CE4E5B9)
    }

    private func nextTrainWindowOffset() -> Int {
        trainRngState = trainRngState &* 6364136223846793005 &+ 1442695040888963407
        let maxStart = trainTokens.count - seqLength
        guard maxStart > 0 else { return 0 }
        return Int(trainRngState % UInt64(maxStart + 1))
    }

    private func nextValWindowOffset() -> Int {
        valRngState = valRngState &* 6364136223846793005 &+ 1442695040888963407
        let maxStart = valTokens.count - seqLength
        guard maxStart > 0 else { return 0 }
        return Int(valRngState % UInt64(maxStart + 1))
    }

    /// `%95 trainTokens` kümesinden `batchSize` adet `seqLength` (512) uzunluğunda rastgele pencere çeker (`inputs[0..<511]`, `targets[1..<512]`).
    public func nextBatch() -> (inputs: MLXArray, targets: MLXArray) {
        let contextLen = seqLength - 1
        var inputFlat: [Int32] = []
        var targetFlat: [Int32] = []
        inputFlat.reserveCapacity(batchSize * contextLen)
        targetFlat.reserveCapacity(batchSize * contextLen)

        for _ in 0..<batchSize {
            let offset = nextTrainWindowOffset()
            inputFlat.append(contentsOf: trainTokens[offset..<(offset + contextLen)])
            targetFlat.append(contentsOf: trainTokens[(offset + 1)..<(offset + seqLength)])
        }

        let inputsArray = MLXArray(inputFlat, [batchSize, contextLen])
        let targetsArray = MLXArray(targetFlat, [batchSize, contextLen])
        return (inputs: inputsArray, targets: targetsArray)
    }

    /// `%5 valTokens` (eğitimde hiç görülmemiş doğrulama kümesi) üzerinden bir doğrulama batch'i çeker.
    public func nextValidationBatch() -> (inputs: MLXArray, targets: MLXArray) {
        let contextLen = seqLength - 1
        var inputFlat: [Int32] = []
        var targetFlat: [Int32] = []
        inputFlat.reserveCapacity(batchSize * contextLen)
        targetFlat.reserveCapacity(batchSize * contextLen)

        for _ in 0..<batchSize {
            let offset = nextValWindowOffset()
            inputFlat.append(contentsOf: valTokens[offset..<(offset + contextLen)])
            targetFlat.append(contentsOf: valTokens[(offset + 1)..<(offset + seqLength)])
        }

        let inputsArray = MLXArray(inputFlat, [batchSize, contextLen])
        let targetsArray = MLXArray(targetFlat, [batchSize, contextLen])
        return (inputs: inputsArray, targets: targetsArray)
    }

    /// `PreTrainer.trainStepWithAccumulation` için `accumulationSteps` adet eğitim mikro-batch grubu döndürür.
    public func nextAccumulationGroup(accumulationSteps: Int = 2) -> [(inputs: MLXArray, targets: MLXArray)] {
        return (0..<accumulationSteps).map { _ in nextBatch() }
    }

    /// Modelin hiç görmediği `%5 valTokens` üzerinde gradyan hesaplamadan ortalama **Validation Loss** ve **Perplexity** ölçer.
    public func evaluateValidationLoss(
        model: PersonalFoundationModel,
        numBatches: Int = 4
    ) -> (valLoss: Float, valPerplexity: Float) {
        let batches = max(1, numBatches)
        model.train(false)
        defer { model.train(true) }

        var totalLoss: Float = 0.0
        for _ in 0..<batches {
            let valBatch = nextValidationBatch()
            let logits = model(valBatch.inputs)
            let lossArray = Trainer.causalCrossEntropyLoss(logits: logits, targets: valBatch.targets)
            eval(lossArray)
            totalLoss += lossArray.item(Float.self)
        }

        MLX.Memory.clearCache()
        let avgValLoss = totalLoss / Float(batches)
        let valPPL = exp(min(20.0, avgValLoss))
        return (avgValLoss, valPPL)
    }
}
