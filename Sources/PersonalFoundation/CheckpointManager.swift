import Foundation
import MLX
import MLXNN

// MARK: - 1. 1B Float16 Safetensors Checkpoint Veri Yapıları

/// Checkpoint üst verilerini taşıyan `Sendable` yapı.
public struct CheckpointHeader: Sendable, Codable, Equatable {
    public let formatVersion: Int
    public let createdAtISO8601: String
    public let step: Int
    public let averageLoss: Float
    public let dtype: String
    public let tensorCount: Int
    public let modelConfig: ModelConfig
}

/// Kayıt işlemi tamamlandığında diske yazılan dosyanın boyut ve konum bilgilerini döndüren `Sendable` rapor.
public struct CheckpointSaveReport: Sendable {
    public let fileURL: URL
    public let byteCount: Int64
    public let sizeMB: Double
    public let sizeGB: Double
    public let tensorCount: Int
}

/// Checkpoint kaydetme/yükleme sırasında oluşabilecek disk ve veri hataları.
public enum CheckpointError: Error, LocalizedError, Sendable {
    case directoryNotWritable(path: String)
    case insufficientDiskSpace(requiredMB: Double, availableMB: Double)
    case saveFailed(underlying: String)
    case fileNotFound(path: String)
    case missingModelConfigMetadata
    case loadFailed(underlying: String)

    public var errorDescription: String? {
        switch self {
        case let .directoryNotWritable(path):
            return "Disk yazma izni hatası: '\(path)' dizinine yazma izni yok."
        case let .insufficientDiskSpace(requiredMB, availableMB):
            return String(
                format: "Yetersiz disk alanı: Checkpoint için en az %.1f MB gerekli, ancak diskte %.1f MB boş alan var.",
                requiredMB,
                availableMB
            )
        case let .saveFailed(underlying):
            return "Checkpoint diske kaydedilirken hata oluştu: \(underlying)"
        case let .fileNotFound(path):
            return "Checkpoint dosyası bulunamadı: '\(path)'"
        case .missingModelConfigMetadata:
            return "Safetensors dosyasında 'model_config_json' üst verisi (metadata) bulunamadı."
        case let .loadFailed(underlying):
            return "Checkpoint diskten yüklenirken hata oluştu: \(underlying)"
        }
    }
}

// MARK: - 2. 1B Float16 Uyumlu Checkpoint Yöneticisi (CheckpointManager)

/// **1B Float16 Uyumlu Sıfır-Kopya (Zero-Copy) Safetensors Checkpoint Yöneticisi**
/// `MLX.save(arrays:metadata:url:)` C++/Metal köprüsünü kullanarak, 1B modelin `.float16`
/// ağırlıklarını Swift `[Float]` dizilerine kopyalamadan (RAM'i hiç şişirmeden) doğrudan
/// Birleşik Bellekten (Unified Memory) diske `.safetensors` (`F16` - 2 bayt/parametre) formatında yazar.
public enum CheckpointManager: Sendable {

    /// Verilen URL'nin uzantısını `.safetensors` olarak garanti altına alır.
    /// (Örn: `.pfckpt` verilse bile MLX C++ `mlx_save_safetensors` motoru için `.safetensors` kullanılır.)
    public static func resolvedSafetensorsURL(for url: URL) -> URL {
        if url.pathExtension.lowercased() == "safetensors" {
            return url
        }
        return url.deletingPathExtension().appendingPathExtension("safetensors")
    }

    /// **1B Modeli Float16 (.float16) Olarak Diske Kaydet**
    /// - Parameters:
    ///   - model: Kaydedilecek `PersonalFoundationModel`
    ///   - step: Eğitim adımı
    ///   - averageLoss: Son eğitim kaybı
    ///   - fileURL: Hedef dosya URL'si (`.safetensors`)
    /// - Returns: Diske yazılan dosyanın MB/GB boyutlarını içeren `CheckpointSaveReport`
    @discardableResult
    public static func save(
        model: PersonalFoundationModel,
        step: Int,
        averageLoss: Float,
        to fileURL: URL
    ) throws -> CheckpointSaveReport {
        let targetURL = resolvedSafetensorsURL(for: fileURL)
        let parentDir = targetURL.deletingLastPathComponent()

        // 1. Yazma izni kontrolü
        guard FileManager.default.isWritableFile(atPath: parentDir.path) else {
            throw CheckpointError.directoryNotWritable(path: parentDir.path)
        }

        // 2. Tüm parametreleri düzleştir ve modelin 16-bit hedef tipinde (.bfloat16 veya .float16 - 2 bayt/parametre) koru
        let targetDType = (model.dtype == .bfloat16 || model.dtype == .float16) ? model.dtype : .bfloat16
        let dtypeLabel = (targetDType == .bfloat16) ? "bfloat16" : "float16"
        let flatParams = model.parameters().flattened()
        var savedArrays: [String: MLXArray] = [:]
        savedArrays.reserveCapacity(flatParams.count)

        var estimatedBytes: Int64 = 0
        for (key, array) in flatParams {
            let typedTensor = (array.dtype == targetDType) ? array : array.asType(targetDType)
            savedArrays[key] = typedTensor
            estimatedBytes += Int64(typedTensor.size * 2) // 16-bit = 2 bayt
        }

        // 3. Disk boş alan kontrolü
        if let resourceValues = try? parentDir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let availableBytes = resourceValues.volumeAvailableCapacityForImportantUsage {
            let requiredBytes = estimatedBytes + (10 * 1024 * 1024) // +10 MB güvenlik payı
            if availableBytes < requiredBytes {
                throw CheckpointError.insufficientDiskSpace(
                    requiredMB: Double(requiredBytes) / (1024.0 * 1024.0),
                    availableMB: Double(availableBytes) / (1024.0 * 1024.0)
                )
            }
        }

        // 4. Tembel (lazy) tensörleri diske yazmadan önce değerlendir
        eval(Array(savedArrays.values))

        // 5. ModelConfig ve eğitim durumunu Safetensors Metadata başlığına JSON olarak göm
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let configData = try encoder.encode(model.config)
        let configJSONString = String(decoding: configData, as: UTF8.self)
        let createdAt = ISO8601DateFormatter().string(from: Date())

        let metadata: [String: String] = [
            "format_version": "3",
            "architecture": "PersonalFoundation",
            "dtype": dtypeLabel,
            "step": String(step),
            "average_loss": String(averageLoss),
            "created_at": createdAt,
            "tensor_count": String(savedArrays.count),
            "model_config_json": configJSONString
        ]

        // 6. MLX.save(arrays:metadata:url:) ile sıfır-kopya (Zero-Copy) BFloat16/Float16 Safetensors kaydı
        do {
            try MLX.save(arrays: savedArrays, metadata: metadata, url: targetURL)
        } catch {
            throw CheckpointError.saveFailed(underlying: error.localizedDescription)
        }

        // 7. Diske yazılan gerçek dosya boyutunu (MB / GB) ölç
        let attributes = try FileManager.default.attributesOfItem(atPath: targetURL.path)
        let fileSizeBytes = (attributes[.size] as? Int64) ?? estimatedBytes
        let sizeMB = Double(fileSizeBytes) / (1024.0 * 1024.0)
        let sizeGB = sizeMB / 1024.0

        return CheckpointSaveReport(
            fileURL: targetURL,
            byteCount: fileSizeBytes,
            sizeMB: sizeMB,
            sizeGB: sizeGB,
            tensorCount: savedArrays.count
        )
    }

    /// **Mevcut Bir 1B Float16 Modele `.safetensors` Ağırlıklarını Doğrudan Yükle (Bellek Dostu / In-Place)**
    /// İkinci bir 1B model kopyası oluşturmadan, diskteki `.float16` tensörleri doğrudan mevcut `model` üzerine yükler.
    @discardableResult
    public static func loadWeights(
        into model: PersonalFoundationModel,
        from fileURL: URL
    ) throws -> CheckpointHeader {
        let targetURL = resolvedSafetensorsURL(for: fileURL)

        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw CheckpointError.fileNotFound(path: targetURL.path)
        }

        let loadedArrays: [String: MLXArray]
        let metadata: [String: String]

        do {
            (loadedArrays, metadata) = try MLX.loadArraysAndMetadata(url: targetURL)
        } catch {
            throw CheckpointError.loadFailed(underlying: error.localizedDescription)
        }

        let targetDType = model.dtype
        let currentFlat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var restoredParams: [(String, MLXArray)] = []
        restoredParams.reserveCapacity(loadedArrays.count)

        for (key, tensor) in loadedArrays {
            let typedTensor = (tensor.dtype == targetDType) ? tensor : tensor.asType(targetDType)
            if let expectedTensor = currentFlat[key] {
                if typedTensor.shape == expectedTensor.shape {
                    restoredParams.append((key, typedTensor))
                } else if typedTensor.ndim == 2 &&
                            expectedTensor.ndim == 2 &&
                            typedTensor.dim(1) == expectedTensor.dim(1) &&
                            typedTensor.dim(0) >= expectedTensor.dim(0) {
                    // Örn: [32000, 2048] boyutlu ön eğitim embedding/lm_head tensörünü
                    // dinamik [tokenizer.vocabSize, 2048] boyutuna dilimleyerek hizala
                    let sliced = typedTensor[0..<expectedTensor.dim(0), 0...]
                    restoredParams.append((key, sliced))
                }
            }
        }

        model.update(parameters: ModuleParameters.unflattened(restoredParams))
        eval(model)
        MLX.Memory.clearCache()

        let step = Int(metadata["step"] ?? "0") ?? 0
        let averageLoss = Float(metadata["average_loss"] ?? "0.0") ?? 0.0
        let formatVersion = Int(metadata["format_version"] ?? "2") ?? 2
        let createdAt = metadata["created_at"] ?? ""
        let savedDType = metadata["dtype"] ?? "float16"

        return CheckpointHeader(
            formatVersion: formatVersion,
            createdAtISO8601: createdAt,
            step: step,
            averageLoss: averageLoss,
            dtype: savedDType,
            tensorCount: restoredParams.count,
            modelConfig: model.config
        )
    }

    /// **Yerel `.safetensors` Dosyasından 110M / 1B Modeli Geri Yükle (Varsayılan: `.float32` Master Weights)**
    public static func load(
        from fileURL: URL,
        dtype: DType = .float32
    ) throws -> (model: PersonalFoundationModel, header: CheckpointHeader) {
        let targetURL = resolvedSafetensorsURL(for: fileURL)

        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw CheckpointError.fileNotFound(path: targetURL.path)
        }

        let loadedArrays: [String: MLXArray]
        let metadata: [String: String]

        do {
            (loadedArrays, metadata) = try MLX.loadArraysAndMetadata(url: targetURL)
        } catch {
            throw CheckpointError.loadFailed(underlying: error.localizedDescription)
        }

        guard let configJSON = metadata["model_config_json"],
              let configData = configJSON.data(using: .utf8) else {
            throw CheckpointError.missingModelConfigMetadata
        }

        let modelConfig = try JSONDecoder().decode(ModelConfig.self, from: configData)
        let step = Int(metadata["step"] ?? "0") ?? 0
        let averageLoss = Float(metadata["average_loss"] ?? "0.0") ?? 0.0
        let formatVersion = Int(metadata["format_version"] ?? "2") ?? 2
        let createdAt = metadata["created_at"] ?? ""
        let savedDType = metadata["dtype"] ?? "float16"

        var restoredParams: [(String, MLXArray)] = []
        restoredParams.reserveCapacity(loadedArrays.count)

        for (key, tensor) in loadedArrays {
            let typedTensor = (tensor.dtype == dtype) ? tensor : tensor.asType(dtype)
            restoredParams.append((key, typedTensor))
        }

        let model = PersonalFoundationModel(config: modelConfig, dtype: dtype)
        model.update(parameters: ModuleParameters.unflattened(restoredParams))
        eval(model)

        let header = CheckpointHeader(
            formatVersion: formatVersion,
            createdAtISO8601: createdAt,
            step: step,
            averageLoss: averageLoss,
            dtype: savedDType,
            tensorCount: restoredParams.count,
            modelConfig: modelConfig
        )

        return (model, header)
    }
}
