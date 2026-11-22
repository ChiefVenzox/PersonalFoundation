import Foundation

// MARK: - 1. Özel Sohbet ve Kontrol Token Kimlikleri (Special Tokens)

/// **Özel Sohbet ve Kontrol Tokenları (Special Tokens)**
/// Sözlüğün en başındaki indeksler (`0..<8`) sohbet ve kontrol tokenlarına sabitlenmiştir:
/// - `0`: `<|im_start|>` (Mesaj / Sohbet turu başlangıcı)
/// - `1`: `<|im_end|>`   (Mesaj / Sohbet turu bitişi)
/// - `2`: `<|user|>`     (Kullanıcı rol etiketi)
/// - `3`: `<|assistant|>`(Asistan rol etiketi)
/// - `4`: `<pad>`        (Dolgu tokenı)
/// - `5`: `<unk>`        (Bilinmeyen token yedeği)
/// - `6`: `<bos>`        (Sekans başlangıcı)
/// - `7`: `<eos>`        (Sekans sonu)
///
/// Ham UTF-8 baytları (`0x00...0xFF`) ise `8..<264` aralığına eşlenir.
public enum SpecialToken: Int32, Sendable, Codable, CaseIterable {
    case imStart = 0
    case imEnd = 1
    case user = 2
    case assistant = 3
    case pad = 4
    case unk = 5
    case bos = 6
    case eos = 7

    public var textRepresentation: String {
        switch self {
        case .imStart: return "<|im_start|>"
        case .imEnd: return "<|im_end|>"
        case .user: return "<|user|>"
        case .assistant: return "<|assistant|>"
        case .pad: return "<pad>"
        case .unk: return "<unk>"
        case .bos: return "<bos>"
        case .eos: return "<eos>"
        }
    }

    public static let tagToToken: [(tag: String, token: SpecialToken)] = SpecialToken.allCases.map {
        ($0.textRepresentation, $0)
    }
}

/// İki ardışık token kimliğinden oluşan BPE çifti.
public struct TokenPair: Hashable, Sendable, Codable {
    public let first: Int32
    public let second: Int32

    public init(_ first: Int32, _ second: Int32) {
        self.first = first
        self.second = second
    }
}

/// Öğrenilmiş tek bir BPE birleştirme kuralı: `(first, second) -> mergedId`
public struct BPEMergeRule: Sendable, Codable, Equatable {
    public let pair: TokenPair
    public let mergedId: Int32
    public let rank: Int
}

// MARK: - 2. Sıfırdan 32.000 Kelimelik Byte-Level BPE Tokenizer (BPETokenizer)

/// **PersonalFoundation 32.000 Kelimelik Byte-Pair Encoding (BPE) Tokenizer**
/// Ters İndeksli (Inverted-Index) ve Frekans Kovalı (Frequency-Bucketed) $O(1)$ birleştirme motoruyla
/// 4–5 MB'lık gerçek Türkçe külliyat üzerinde 32.000 kelimelik sözlüğü saniyeler içinde eğitir.
public final class BPETokenizer: Sendable, Codable {
    /// Özel token sayısı (`8`)
    public static let byteOffset: Int32 = Int32(SpecialToken.allCases.count)
    /// Temel sözlük boyutu: 8 Özel Token + 256 UTF-8 Baytı = 264
    public static let baseVocabSize: Int = Int(byteOffset) + 256
    /// Maksimum sözlük üst sınırı (`32,000`)
    public static let maxAllowedVocabSize: Int = 32_000

    /// Öğrenilmiş birleştirme kuralları (Öncelik sırasına göre)
    public let mergeRules: [BPEMergeRule]

    /// Hızlı arama için `[TokenPair: BPEMergeRule]` sözlüğü
    private let mergeLookup: [TokenPair: BPEMergeRule]

    /// Her bir Token ID'nin karşılık geldiği ham UTF-8 bayt dizisi (`[Int32: [UInt8]]`)
    private let idToBytes: [Int32: [UInt8]]

    /// Toplam sözlük boyutu (`264 + mergeRules.count`)
    public var vocabSize: Int {
        Self.baseVocabSize + mergeRules.count
    }

    // MARK: - Başlatıcı (Initializer)

    public init(mergeRules: [BPEMergeRule] = []) {
        let sortedRules = mergeRules.sorted { $0.rank < $1.rank }
        self.mergeRules = sortedRules

        var lookup: [TokenPair: BPEMergeRule] = [:]
        lookup.reserveCapacity(sortedRules.count)

        var bytesTable: [Int32: [UInt8]] = [:]
        bytesTable.reserveCapacity(Self.baseVocabSize + sortedRules.count)

        for byteVal in 0..<256 {
            let tokenId = Int32(byteVal) + Self.byteOffset
            bytesTable[tokenId] = [UInt8(byteVal)]
        }

        for rule in sortedRules {
            lookup[rule.pair] = rule
            let leftBytes = bytesTable[rule.pair.first] ?? []
            let rightBytes = bytesTable[rule.pair.second] ?? []
            bytesTable[rule.mergedId] = leftBytes + rightBytes
        }

        self.mergeLookup = lookup
        self.idToBytes = bytesTable
    }

    // MARK: - Codable (JSON Serileştirme & Geri Yükleme)

    private enum CodingKeys: String, CodingKey {
        case formatVersion
        case vocabSize
        case specialTokens
        case mergeRules
    }

    public convenience init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rules = try container.decode([BPEMergeRule].self, forKey: .mergeRules)
        self.init(mergeRules: rules)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(2, forKey: .formatVersion)
        try container.encode(vocabSize, forKey: .vocabSize)

        var specialMap: [String: Int32] = [:]
        for special in SpecialToken.allCases {
            specialMap[special.textRepresentation] = special.rawValue
        }
        try container.encode(specialMap, forKey: .specialTokens)
        try container.encode(mergeRules, forKey: .mergeRules)
    }

    /// Eğitilmiş BPE sözlüğünü diske JSON olarak kaydeder.
    public func save(to fileURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: fileURL, options: .atomic)
    }

    /// Diskteki JSON sözlük dosyasından `BPETokenizer` nesnesini yükler.
    public static func load(from fileURL: URL) throws -> BPETokenizer {
        let data = try Data(contentsOf: fileURL)
        return try JSONDecoder().decode(BPETokenizer.self, from: data)
    }

    // MARK: - Ön-Kelimeleme (Pre-Tokenization)

    public static func preTokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }

        var chunks: [String] = []
        var currentChunk = ""
        var lastWasWhitespace = false

        for char in text {
            if char.isWhitespace {
                if !currentChunk.isEmpty && !lastWasWhitespace {
                    chunks.append(currentChunk)
                    currentChunk = ""
                }
                currentChunk.append(char)
                lastWasWhitespace = true
            } else if char.isPunctuation || char.isSymbol {
                if !currentChunk.isEmpty {
                    chunks.append(currentChunk)
                    currentChunk = ""
                }
                chunks.append(String(char))
                lastWasWhitespace = false
            } else {
                currentChunk.append(char)
                lastWasWhitespace = false
            }
        }

        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }

        return chunks
    }

    // MARK: - Dosyadan 32.000 Kelimelik Hızlı BPE Eğitimi (trainFromFile)

    /// `corpus.txt` dosyasının ilk `sampleByteLimit` (Varsayılan: `4.5 MB`) temsili bloğunu okuyarak
    /// `targetVocabSize = 32000` hedefine ulaşana kadar Ters İndeksli (Inverted-Index) BPE birleştirmeleri yapar.
    public static func trainFromFile(
        url: URL,
        targetVocabSize: Int = 32_000,
        sampleByteLimit: Int = 4_500_000,
        minFrequency: Int = 1
    ) throws -> BPETokenizer {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let rawData = try handle.read(upToCount: sampleByteLimit) ?? Data()
        // UTF-8 karakter ortasından bölünmeyi önlemek için son satır sonuna (\n) kadar kes
        let newlineByte = UInt8(ascii: "\n")
        let safeData: Data
        if let lastNewline = rawData.lastIndex(of: newlineByte), lastNewline > 0 {
            safeData = rawData.subdata(in: 0..<lastNewline)
        } else {
            safeData = rawData
        }

        let sampleText = String(decoding: safeData, as: UTF8.self)
        return train(
            corpus: [sampleText],
            targetVocabSize: targetVocabSize,
            minFrequency: minFrequency
        )
    }

    /// **Ters İndeksli (Inverted-Index) + Frekans Kovalı Yüksek Hızlı BPE Eğitim Algoritması**
    /// Her birleştirmede tüm kelimeleri baştan taramak yerine, yalnızca `bestPair` çiftini içeren
    /// benzersiz kelimeleri (`pairToWords[bestPair]`) günceller. Bu sayede 32.000 birleştirme saniyeler sürer.
    public static func train(
        corpus: [String],
        targetVocabSize: Int = 32_000,
        minFrequency: Int = 1
    ) -> BPETokenizer {
        precondition(
            targetVocabSize >= baseVocabSize && targetVocabSize <= maxAllowedVocabSize,
            "targetVocabSize [\(baseVocabSize), \(maxAllowedVocabSize)] aralığında olmalıdır."
        )

        let maxMerges = targetVocabSize - baseVocabSize
        if maxMerges == 0 {
            return BPETokenizer(mergeRules: [])
        }

        // 1. Benzersiz kelime parçalarını ve külliyattaki frekanslarını çıkar
        var wordFrequencies: [String: Int] = [:]
        for text in corpus {
            for chunk in preTokenize(text) {
                wordFrequencies[chunk, default: 0] += 1
            }
        }

        var wordTokens: [[Int32]] = []
        var wordFreqs: [Int] = []
        wordTokens.reserveCapacity(wordFrequencies.count)
        wordFreqs.reserveCapacity(wordFrequencies.count)

        for (word, freq) in wordFrequencies {
            let tokens = Array(word.utf8).map { Int32($0) + byteOffset }
            if tokens.count >= 2 {
                wordTokens.append(tokens)
                wordFreqs.append(freq)
            }
        }

        // 2. Başlangıç Çift Frekansları (pairCounts), Ters İndeks (pairToWords) ve Frekans Kovaları (freqBuckets)
        var pairCounts: [TokenPair: Int] = [:]
        var pairToWords: [TokenPair: Set<Int>] = [:]
        var freqBuckets: [Int: [TokenPair]] = [:]
        var currentMaxFreq = 0

        for wIdx in 0..<wordTokens.count {
            let tokens = wordTokens[wIdx]
            let freq = wordFreqs[wIdx]
            for i in 0..<(tokens.count - 1) {
                let pair = TokenPair(tokens[i], tokens[i + 1])
                pairCounts[pair, default: 0] += freq
                pairToWords[pair, default: []].insert(wIdx)
            }
        }

        for (pair, count) in pairCounts where count >= minFrequency {
            freqBuckets[count, default: []].append(pair)
            if count > currentMaxFreq {
                currentMaxFreq = count
            }
        }

        var learnedRules: [BPEMergeRule] = []
        learnedRules.reserveCapacity(maxMerges)

        // Yardımcı: Bir kelimedeki tüm komşu çiftleri verilen çarpanla (+freq veya -freq) güncelle
        func addWordPairContributions(wordIndex: Int, delta: Int) {
            let tokens = wordTokens[wordIndex]
            guard tokens.count >= 2 else { return }
            for i in 0..<(tokens.count - 1) {
                let pair = TokenPair(tokens[i], tokens[i + 1])
                let updated = (pairCounts[pair] ?? 0) + delta
                if updated <= 0 {
                    pairCounts.removeValue(forKey: pair)
                } else {
                    pairCounts[pair] = updated
                    if delta > 0 {
                        pairToWords[pair, default: []].insert(wordIndex)
                        if updated >= minFrequency {
                            freqBuckets[updated, default: []].append(pair)
                            if updated > currentMaxFreq {
                                currentMaxFreq = updated
                            }
                        }
                    }
                }
            }
        }

        // 3. O(1) Kovalı En Sık Çift Seçimi ve Artımlı (Incremental) Güncelleme
        for rank in 0..<maxMerges {
            var selectedPair: TokenPair? = nil

            while currentMaxFreq >= minFrequency {
                if var bucket = freqBuckets[currentMaxFreq], !bucket.isEmpty {
                    let candidate = bucket.removeLast()
                    if bucket.isEmpty {
                        freqBuckets.removeValue(forKey: currentMaxFreq)
                    } else {
                        freqBuckets[currentMaxFreq] = bucket
                    }

                    // Tembel silme (lazy deletion) doğrulaması: Aday çiftin güncel frekansı bu kovaya mı ait?
                    if let actualCount = pairCounts[candidate], actualCount == currentMaxFreq {
                        selectedPair = candidate
                        break
                    }
                } else {
                    freqBuckets.removeValue(forKey: currentMaxFreq)
                    currentMaxFreq -= 1
                }
            }

            guard let bestPair = selectedPair else {
                break
            }

            let newTokenId = Int32(baseVocabSize + rank)
            let rule = BPEMergeRule(pair: bestPair, mergedId: newTokenId, rank: rank)
            learnedRules.append(rule)

            guard let affectedWordIndices = pairToWords.removeValue(forKey: bestPair) else {
                pairCounts.removeValue(forKey: bestPair)
                continue
            }
            pairCounts.removeValue(forKey: bestPair)

            for wIdx in affectedWordIndices {
                let tokens = wordTokens[wIdx]
                guard tokens.count >= 2 else { continue }

                // Kelime gerçekten bestPair içeriyor mu kontrol et
                var containsBest = false
                for i in 0..<(tokens.count - 1) {
                    if tokens[i] == bestPair.first && tokens[i + 1] == bestPair.second {
                        containsBest = true
                        break
                    }
                }
                guard containsBest else { continue }

                let freq = wordFreqs[wIdx]
                // Eski çift katkılarını düş
                addWordPairContributions(wordIndex: wIdx, delta: -freq)

                // Kelime içinde bestPair -> newTokenId birleştirmesini uygula
                var merged: [Int32] = []
                merged.reserveCapacity(tokens.count)
                var i = 0
                while i < tokens.count {
                    if i < tokens.count - 1 &&
                        tokens[i] == bestPair.first &&
                        tokens[i + 1] == bestPair.second {
                        merged.append(newTokenId)
                        i += 2
                    } else {
                        merged.append(tokens[i])
                        i += 1
                    }
                }
                wordTokens[wIdx] = merged

                // Yeni oluşan çift katkılarını ekle
                addWordPairContributions(wordIndex: wIdx, delta: freq)
            }

            if (rank + 1) % 5000 == 0 {
                print("   • [BPETokenizer] \(rank + 1) / \(maxMerges) birleştirme tamamlandı (Güncel Sözlük: \(baseVocabSize + rank + 1))...")
            }
        }

        return BPETokenizer(mergeRules: learnedRules)
    }

    // MARK: - Metin Kodlama (Encode: String -> [Int32])

    private func encodeChunk(_ chunk: String) -> [Int32] {
        var tokens = Array(chunk.utf8).map { Int32($0) + Self.byteOffset }
        guard tokens.count >= 2, !mergeRules.isEmpty else {
            return tokens
        }

        while tokens.count >= 2 {
            var bestRule: BPEMergeRule? = nil

            for i in 0..<(tokens.count - 1) {
                let pair = TokenPair(tokens[i], tokens[i + 1])
                if let rule = mergeLookup[pair] {
                    if bestRule == nil || rule.rank < bestRule!.rank {
                        bestRule = rule
                    }
                }
            }

            guard let ruleToApply = bestRule else {
                break
            }

            var newTokens: [Int32] = []
            newTokens.reserveCapacity(tokens.count)

            var i = 0
            while i < tokens.count {
                if i < tokens.count - 1 &&
                    tokens[i] == ruleToApply.pair.first &&
                    tokens[i + 1] == ruleToApply.pair.second {
                    newTokens.append(ruleToApply.mergedId)
                    i += 2
                } else {
                    newTokens.append(tokens[i])
                    i += 1
                }
            }
            tokens = newTokens
        }

        return tokens
    }

    /// Büyük külliyatları (`corpus.txt`) kelime önbelleği (memoization) kullanarak çok hızlı şekilde token dizisine dönüştürür.
    public func encodeCorpus(_ text: String) -> [Int32] {
        let chunks = Self.preTokenize(text)
        var cache: [String: [Int32]] = [:]
        cache.reserveCapacity(32_768)

        var output: [Int32] = []
        output.reserveCapacity(chunks.count * 2)

        let limitId = Int32(self.vocabSize)
        for chunk in chunks {
            if let cached = cache[chunk] {
                output.append(contentsOf: cached)
            } else {
                let encoded = encodeChunk(chunk).map { id in
                    (id >= 0 && id < limitId) ? id : SpecialToken.unk.rawValue
                }
                cache[chunk] = encoded
                output.append(contentsOf: encoded)
            }
        }
        return output
    }

    private func encodeTextDetectingSpecialTags(_ text: String) -> [Int32] {
        guard !text.isEmpty else { return [] }

        var tokens: [Int32] = []
        var remaining = text[...]

        while !remaining.isEmpty {
            var earliestMatch: (range: Range<Substring.Index>, token: SpecialToken)? = nil

            for (tag, specialToken) in SpecialToken.tagToToken {
                if let range = remaining.range(of: tag) {
                    if earliestMatch == nil || range.lowerBound < earliestMatch!.range.lowerBound {
                        earliestMatch = (range, specialToken)
                    }
                }
            }

            if let match = earliestMatch {
                let prefixText = String(remaining[..<match.range.lowerBound])
                if !prefixText.isEmpty {
                    for chunk in Self.preTokenize(prefixText) {
                        tokens.append(contentsOf: encodeChunk(chunk))
                    }
                }
                tokens.append(match.token.rawValue)
                remaining = remaining[match.range.upperBound...]
            } else {
                let tailText = String(remaining)
                for chunk in Self.preTokenize(tailText) {
                    tokens.append(contentsOf: encodeChunk(chunk))
                }
                break
            }
        }

        return tokens
    }

    public func encode(_ text: String, addBOS: Bool = false, addEOS: Bool = false) -> [Int32] {
        var result: [Int32] = []
        if addBOS {
            result.append(SpecialToken.bos.rawValue)
        }

        let limitId = Int32(self.vocabSize)
        for tokenId in encodeTextDetectingSpecialTags(text) {
            if tokenId >= 0 && tokenId < limitId {
                result.append(tokenId)
            } else {
                result.append(SpecialToken.unk.rawValue)
            }
        }

        if addEOS {
            result.append(SpecialToken.eos.rawValue)
        }
        return result
    }

    // MARK: - Token Çözme (Decode: [Int32] -> String)

    public func decode(_ tokenIds: [Int32], skipSpecialTokens: Bool = true) -> String {
        var outputParts: [String] = []
        var byteBuffer: [UInt8] = []

        func flushByteBuffer() {
            if !byteBuffer.isEmpty {
                outputParts.append(String(decoding: byteBuffer, as: UTF8.self))
                byteBuffer.removeAll(keepingCapacity: true)
            }
        }

        for id in tokenIds {
            if let special = SpecialToken(rawValue: id) {
                if !skipSpecialTokens {
                    flushByteBuffer()
                    outputParts.append(special.textRepresentation)
                }
            } else if let bytes = idToBytes[id] {
                byteBuffer.append(contentsOf: bytes)
            } else if !skipSpecialTokens {
                flushByteBuffer()
                outputParts.append(SpecialToken.unk.textRepresentation)
            }
        }

        flushByteBuffer()
        return outputParts.joined()
    }

    public func tokenToString(_ tokenId: Int32) -> String {
        if let special = SpecialToken(rawValue: idToSpecialRaw(tokenId)) {
            return special.textRepresentation
        }
        guard let bytes = idToBytes[tokenId] else {
            return "<unk:\(tokenId)>"
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func idToSpecialRaw(_ tokenId: Int32) -> Int32 {
        return tokenId
    }

    public func rawBytes(for tokenId: Int32) -> [UInt8] {
        if SpecialToken(rawValue: tokenId) != nil {
            return []
        }
        return idToBytes[tokenId] ?? []
    }
}

// MARK: - 3. Çok Baytlı Türkçe Karakterler İçin Güvenli UTF-8 Akış Çözücüsü (UTF8StreamingDecoder)

public struct UTF8StreamingDecoder: Sendable {
    private var pendingBytes: [UInt8] = []
    private var isAtStartOfResponse: Bool = true

    public init() {}

    private static func expectedUTF8SequenceLength(for leadingByte: UInt8) -> Int {
        if (leadingByte & 0b1000_0000) == 0b0000_0000 {
            return 1
        } else if (leadingByte & 0b1110_0000) == 0b1100_0000 {
            return 2
        } else if (leadingByte & 0b1111_0000) == 0b1110_0000 {
            return 3
        } else if (leadingByte & 0b1111_1000) == 0b1111_0000 {
            return 4
        } else {
            return 1
        }
    }

    public mutating func appendToken(_ tokenId: Int32, tokenizer: BPETokenizer, skipSpecialTokens: Bool = true) -> String {
        if let special = SpecialToken(rawValue: tokenId) {
            if skipSpecialTokens {
                return ""
            } else {
                let flushed = flush()
                return flushed + special.textRepresentation
            }
        }

        let newBytes = tokenizer.rawBytes(for: tokenId)
        guard !newBytes.isEmpty else { return "" }
        pendingBytes.append(contentsOf: newBytes)

        var index = 0
        var validEndIndex = 0
        let totalCount = pendingBytes.count

        while index < totalCount {
            let leadByte = pendingBytes[index]
            let seqLen = Self.expectedUTF8SequenceLength(for: leadByte)

            if index + seqLen <= totalCount {
                index += seqLen
                validEndIndex = index
            } else {
                break
            }
        }

        guard validEndIndex > 0 else {
            return ""
        }

        let completeSlice = Array(pendingBytes[0..<validEndIndex])
        pendingBytes.removeFirst(validEndIndex)

        var decodedChunk = String(decoding: completeSlice, as: UTF8.self)
        if isAtStartOfResponse && !decodedChunk.isEmpty {
            if decodedChunk.hasPrefix(" ") {
                decodedChunk.removeFirst()
            }
            if !decodedChunk.isEmpty {
                isAtStartOfResponse = false
            }
        }
        return decodedChunk
    }

    public mutating func flush() -> String {
        guard !pendingBytes.isEmpty else { return "" }
        let str = String(decoding: pendingBytes, as: UTF8.self)
        pendingBytes.removeAll(keepingCapacity: true)
        return str
    }

    public mutating func reset() {
        pendingBytes.removeAll(keepingCapacity: true)
        isAtStartOfResponse = true
    }
}
