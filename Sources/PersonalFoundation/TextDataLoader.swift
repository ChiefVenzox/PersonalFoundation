import Foundation
import MLX

// MARK: - Kayan Pencereli Ön Eğitim Veri Yükleyicisi (TextDataLoader)

/// **Otoregresif Ön Eğitim İçin Metin Veri Yükleyicisi (`TextDataLoader`)**
/// Ham metin belgelerini `BPETokenizer` ile tek bir kesintisiz token akışına (`[Int32]`) dönüştürür
/// ve kayan pencere (*sliding window*) yöntemiyle `(inputs: [MicroBatchSize, SeqLen], targets: [MicroBatchSize, SeqLen])`
/// boyutunda `MLXArray` mikro-batch grupları üretir.
///
/// Matematiksel Otoregresif Hizalama:
///   Uzunluğu $L + 1$ olan her pencere $W = [w_0, w_1, \dots, w_L]$ için:
///   - `inputs`  ($X$): $[w_0, w_1, \dots, w_{L-1}]$
///   - `targets` ($Y$): $[w_1, w_2, \dots, w_L]$
public final class TextDataLoader {
    /// Külliyatın tamamının tokenize edilmiş düz dizisi
    public let tokenStream: [Int32]
    /// Her eğitim örneğinin bağlam uzunluğu ($L$)
    public let sequenceLength: Int
    /// Her mikro-batch içindeki örnek sayısı ($B_{\text{micro}}$)
    public let microBatchSize: Int
    /// Pencere kaydırma adımı (`stride`)
    public let stride: Int

    /// Çıkarılan tüm başlangıç pencere ofsetleri
    private var windowOffsets: [Int]
    /// Mevcut okuma imleci
    private var cursor: Int = 0

    /// Ham metin dizisinden (`documents`) ve eğitilmiş `BPETokenizer` nesnesinden veri yükleyici oluşturur.
    public init(
        documents: [String],
        tokenizer: BPETokenizer,
        sequenceLength: Int,
        microBatchSize: Int,
        stride: Int? = nil
    ) {
        precondition(sequenceLength >= 1, "sequenceLength en az 1 olmalıdır.")
        precondition(microBatchSize >= 1, "microBatchSize en az 1 olmalıdır.")

        var stream: [Int32] = []
        for doc in documents where !doc.isEmpty {
            let encoded = tokenizer.encode(doc, addBOS: true, addEOS: true)
            stream.append(contentsOf: encoded)
        }

        // Eğer toplam token sayısı (sequenceLength + 1) değerinden küçükse akışı tekrarla
        let minRequiredTokens = sequenceLength + 1 + (microBatchSize * (stride ?? sequenceLength))
        if !stream.isEmpty && stream.count < minRequiredTokens {
            let original = stream
            while stream.count < minRequiredTokens {
                stream.append(contentsOf: original)
            }
        }

        self.tokenStream = stream
        self.sequenceLength = sequenceLength
        self.microBatchSize = microBatchSize
        let effectiveStride = max(1, stride ?? (sequenceLength / 2))
        self.stride = effectiveStride

        // Tüm geçerli pencere başlangıç indekslerini hesapla
        let maxStart = max(0, stream.count - (sequenceLength + 1))
        self.windowOffsets = Array(Swift.stride(from: 0, through: maxStart, by: effectiveStride))
    }

    /// Toplam kullanılabilir pencere (örnek) sayısı.
    public var totalWindows: Int {
        windowOffsets.count
    }

    /// Tek bir mikro-batch (`(inputs: MLXArray, targets: MLXArray)`) çeker.
    /// İmleç sona ulaştığında otomatik olarak başa sarar (Epoch döngüsü).
    public func nextMicroBatch() -> (inputs: MLXArray, targets: MLXArray) {
        precondition(!windowOffsets.isEmpty, "Veri kümesi pencere oluşturmak için yeterli token içermiyor.")

        var inputBuffer: [Int32] = []
        var targetBuffer: [Int32] = []
        inputBuffer.reserveCapacity(microBatchSize * sequenceLength)
        targetBuffer.reserveCapacity(microBatchSize * sequenceLength)

        for _ in 0..<microBatchSize {
            if cursor >= windowOffsets.count {
                cursor = 0
            }
            let startIdx = windowOffsets[cursor]
            cursor += 1

            let inputEnd = startIdx + sequenceLength
            let targetStart = startIdx + 1
            let targetEnd = targetStart + sequenceLength

            inputBuffer.append(contentsOf: tokenStream[startIdx..<inputEnd])
            targetBuffer.append(contentsOf: tokenStream[targetStart..<targetEnd])
        }

        let inputsArray = MLXArray(inputBuffer, [microBatchSize, sequenceLength])
        let targetsArray = MLXArray(targetBuffer, [microBatchSize, sequenceLength])
        return (inputs: inputsArray, targets: targetsArray)
    }

    /// `Trainer.trainStepWithAccumulation` için `accumulationSteps` adet mikro-batch grubunu tek seferde döndürür.
    public func nextAccumulationGroup(accumulationSteps: Int) -> [(inputs: MLXArray, targets: MLXArray)] {
        return (0..<accumulationSteps).map { _ in nextMicroBatch() }
    }
}
