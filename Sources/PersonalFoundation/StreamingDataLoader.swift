import Foundation
import MLX

// MARK: - Büyük Veri Disk Akışlı Veri Yükleyici (StreamingDataLoader)

/// **Diskten Akışlı Büyük Veri Yükleyicisi (`StreamingDataLoader`)**
/// Yüzlerce gigabaytlık `.txt` veya `.jsonl` veri kümelerini RAM'e yüklemeden,
/// `FileHandle` üzerinden sabit boyutlu (`chunkSizeBytes = 64 KB`) bayt blokları halinde diskten satır satır okur.
///
/// Çalışma Prensibi ($O(1)$ Sabit Sistem Belleği):
/// 1. Diskten yalnızca ihtiyaç duyuldukça `64 KB` ham bayt okunur ve satırlara (`\n`) bölünür.
/// 2. `.jsonl` formatındaki satırlarda `"text"` alanı otomatik ayrıştırılır; `.txt` satırları doğrudan alınır.
/// 3. Her satır `BPETokenizer` ile anlık olarak tokenlaştırılıp küçük bir halka tampona (`rollingTokenBuffer`) eklenir.
/// 4. Tamponda `sequenceLength + 1` (Örn: `1025` veya `2049`) token biriktiğinde `(inputs, targets)` bloğu kesilir
///    ve tüketilen tokenlar tampondan silinir (`removeFirst`). Böylece RAM kullanımı dosya boyutundan bağımsız olarak sabit kalır.
public final class StreamingDataLoader {
    public let fileURL: URL
    public let tokenizer: BPETokenizer
    public let sequenceLength: Int
    public let microBatchSize: Int
    public let readChunkSizeBytes: Int

    /// Disk okuma tutamacı (`FileHandle`)
    private let fileHandle: FileHandle
    /// Satır sonu (`\n`) henüz gelmemiş kısmi UTF-8 bayt tamponu
    private var pendingByteBuffer = Data()
    /// Yalnızca aktif pencereyi tutan küçük kayan token tamponu
    private var rollingTokenBuffer: [Int32] = []

    /// İstatistikler
    public private(set) var totalBytesStreamed: Int = 0
    public private(set) var totalTokensStreamed: Int = 0
    public private(set) var epochsCompleted: Int = 0

    /// `.jsonl` satırları için hafif JSON şeması (`{"text": "..."}`)
    private struct JSONLTextRecord: Decodable {
        let text: String
    }

    public init(
        fileURL: URL,
        tokenizer: BPETokenizer,
        sequenceLength: Int = 1024,
        microBatchSize: Int = 1,
        readChunkSizeBytes: Int = 64 * 1024
    ) throws {
        precondition(sequenceLength >= 2, "sequenceLength en az 2 olmalıdır.")
        precondition(microBatchSize >= 1, "microBatchSize en az 1 olmalıdır.")
        precondition(FileManager.default.fileExists(atPath: fileURL.path), "Veri dosyası bulunamadı: \(fileURL.path)")

        self.fileURL = fileURL
        self.tokenizer = tokenizer
        self.sequenceLength = sequenceLength
        self.microBatchSize = microBatchSize
        self.readChunkSizeBytes = readChunkSizeBytes
        self.fileHandle = try FileHandle(forReadingFrom: fileURL)
        self.rollingTokenBuffer.reserveCapacity((sequenceLength + 1) * 2)
    }

    deinit {
        try? fileHandle.close()
    }

    /// Tek bir satırı (`.txt` veya `.jsonl`) metin olarak ayıklar ve BPE tokenlarına çevirip tampona ekler.
    private func processLineData(_ lineData: Data) {
        guard !lineData.isEmpty else { return }

        let rawString = String(decoding: lineData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawString.isEmpty else { return }

        let extractedText: String
        if rawString.hasPrefix("{"),
           let jsonData = rawString.data(using: .utf8),
           let record = try? JSONDecoder().decode(JSONLTextRecord.self, from: jsonData) {
            extractedText = record.text
        } else {
            extractedText = rawString
        }

        guard !extractedText.isEmpty else { return }

        let encoded = tokenizer.encode(extractedText, addBOS: true, addEOS: true)
        rollingTokenBuffer.append(contentsOf: encoded)
        totalTokensStreamed += encoded.count
    }

    /// Diskten bir sonraki `readChunkSizeBytes` bloğunu okur ve tam satırları `rollingTokenBuffer` içine tokenize eder.
    /// Dosya sonuna (EOF) ulaşıldığında otomatik olarak dosyanın başına (`offset: 0`) sarar.
    private func refillTokenBufferFromDisk() throws {
        let newlineByte = UInt8(ascii: "\n")

        while rollingTokenBuffer.count < (sequenceLength + 1) {
            // 1. Önce mevcut `pendingByteBuffer` içinde tam satır (`\n`) var mı kontrol et
            while let newlineIndex = pendingByteBuffer.firstIndex(of: newlineByte) {
                let lineData = pendingByteBuffer.subdata(in: 0..<newlineIndex)
                pendingByteBuffer.removeSubrange(0...newlineIndex)
                processLineData(lineData)

                if rollingTokenBuffer.count >= (sequenceLength + 1) {
                    return
                }
            }

            // 2. Diskten yeni bir bayt bloğu oku
            let chunk = try fileHandle.read(upToCount: readChunkSizeBytes) ?? Data()

            if chunk.isEmpty {
                // Dosya sonuna (EOF) ulaşıldı: Kalan son satırı işle
                if !pendingByteBuffer.isEmpty {
                    processLineData(pendingByteBuffer)
                    pendingByteBuffer.removeAll(keepingCapacity: true)
                }

                // Yeni Epoch için dosya imlecini başa (0. bayta) sar
                try fileHandle.seek(toOffset: 0)
                epochsCompleted += 1
            } else {
                totalBytesStreamed += chunk.count
                pendingByteBuffer.append(chunk)
            }
        }
    }

    /// Tampondan `sequenceLength + 1` uzunluğunda tek bir sekans çeker ve tüketilen tokenları bellekten siler.
    private func popNextSequencePair() throws -> (inputs: [Int32], targets: [Int32]) {
        let requiredCount = sequenceLength + 1
        if rollingTokenBuffer.count < requiredCount {
            try refillTokenBufferFromDisk()
        }

        let inputSlice = Array(rollingTokenBuffer[0..<sequenceLength])
        let targetSlice = Array(rollingTokenBuffer[1..<requiredCount])

        // Tüketilen `sequenceLength` adet tokenı halka tampondan düşür (RAM şişmesini engeller!)
        // Son token (requiredCount - 1), bir sonraki bloğun ilk giriş tokenı olarak korunur (kesintisiz akış).
        rollingTokenBuffer.removeFirst(sequenceLength)

        return (inputs: inputSlice, targets: targetSlice)
    }

    /// Diskten akışlı olarak `[microBatchSize, sequenceLength]` boyutunda bir sonraki mikro-batch'i üretir.
    public func nextMicroBatch() throws -> (inputs: MLXArray, targets: MLXArray) {
        var inputFlat: [Int32] = []
        var targetFlat: [Int32] = []
        inputFlat.reserveCapacity(microBatchSize * sequenceLength)
        targetFlat.reserveCapacity(microBatchSize * sequenceLength)

        for _ in 0..<microBatchSize {
            let pair = try popNextSequencePair()
            inputFlat.append(contentsOf: pair.inputs)
            targetFlat.append(contentsOf: pair.targets)
        }

        let inputsArray = MLXArray(inputFlat, [microBatchSize, sequenceLength])
        let targetsArray = MLXArray(targetFlat, [microBatchSize, sequenceLength])
        return (inputs: inputsArray, targets: targetsArray)
    }

    /// `PreTrainer.trainStepWithAccumulation` için `accumulationSteps` adet mikro-batch'i diskten okuyarak döndürür.
    public func nextAccumulationGroup(accumulationSteps: Int) throws -> [(inputs: MLXArray, targets: MLXArray)] {
        var group: [(inputs: MLXArray, targets: MLXArray)] = []
        group.reserveCapacity(accumulationSteps)
        for _ in 0..<accumulationSteps {
            group.append(try nextMicroBatch())
        }
        return group
    }
}
