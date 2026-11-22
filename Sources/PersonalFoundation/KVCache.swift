import Foundation
import MLX

// MARK: - 1. Tek Katman İçin KV-Cache (LayerKVCache)

/// **Tek Bir Transformer Katmanı İçin Anahtar-Değer Önbelleği (Key-Value Cache)**
/// Otoregresif metin üretiminde (Inference / Decoding) her yeni token üretilirken,
/// geçmiş tokenların Anahtar (Key) ve Değer (Value) projeksiyonlarını tekrar hesaplamak ($O(T^2)$ maliyet)
/// yerine bellekte saklar ve yeni tokenın $K, V$ tensörlerini mevcut önbelleğe ekler ($O(T)$ maliyet).
///
/// Tensör Boyutları:
///   - `keys`:   `[BatchSize, NumKVHeads, CachedSeqLen, HeadDim]`
///   - `values`: `[BatchSize, NumKVHeads, CachedSeqLen, HeadDim]`
public final class LayerKVCache {
    /// Önbellekteki Anahtar (Key) tensörü (RoPE uygulanmış halde saklanır)
    public private(set) var keys: MLXArray?
    /// Önbellekteki Değer (Value) tensörü
    public private(set) var values: MLXArray?
    /// Şu ana kadar işlenmiş toplam sekans uzunluğu (RoPE pozisyon ofseti için kullanılır)
    public private(set) var offset: Int = 0

    public init() {}

    /// Yeni hesaplanan `newKeys` ve `newValues` tensörlerini önbelleğe ekler ve
    /// o ana kadarki tüm `[K_geçmiş, K_yeni]` ve `[V_geçmiş, V_yeni]` tensörlerini döndürür.
    ///
    /// - Parameters:
    ///   - newKeys: `[BatchSize, NumKVHeads, NewSeqLen, HeadDim]` (RoPE uygulanmış)
    ///   - newValues: `[BatchSize, NumKVHeads, NewSeqLen, HeadDim]`
    /// - Returns: Güncellenmiş `(allKeys, allValues)` çifti (`[BatchSize, NumKVHeads, TotalSeqLen, HeadDim]`)
    public func updateAndFetch(newKeys: MLXArray, newValues: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        let addedSeqLen = newKeys.dim(2)

        let updatedKeys: MLXArray
        let updatedValues: MLXArray

        if let existingKeys = keys, let existingValues = values {
            // Sekans ekseni (axis: 2) boyunca geçmiş K/V ile yeni K/V tensörlerini birleştir
            updatedKeys = concatenated([existingKeys, newKeys], axis: 2)
            updatedValues = concatenated([existingValues, newValues], axis: 2)
        } else {
            // İlk adım (Prefill aşaması)
            updatedKeys = newKeys
            updatedValues = newValues
        }

        self.keys = updatedKeys
        self.values = updatedValues
        self.offset += addedSeqLen

        return (updatedKeys, updatedValues)
    }

    /// Önbelleği sıfırlar.
    public func reset() {
        self.keys = nil
        self.values = nil
        self.offset = 0
    }
}

// MARK: - 2. Tüm Model İçin KV-Cache Yöneticisi (ModelKVCache)

/// Modeldeki tüm `TransformerBlock` katmanlarının `LayerKVCache` nesnelerini yöneten kapsayıcı sınıf.
public final class ModelKVCache {
    public let layerCaches: [LayerKVCache]

    /// Modelin katman sayısı (`numLayers`) kadar boş `LayerKVCache` oluşturur.
    public init(numLayers: Int) {
        self.layerCaches = (0..<numLayers).map { _ in LayerKVCache() }
    }

    /// Şu ana kadar önbelleğe alınmış toplam token sayısı.
    public var currentSequenceLength: Int {
        layerCaches.first?.offset ?? 0
    }

    /// Tüm katman önbelleklerini temizler.
    public func reset() {
        for cache in layerCaches {
            cache.reset()
        }
    }
}
