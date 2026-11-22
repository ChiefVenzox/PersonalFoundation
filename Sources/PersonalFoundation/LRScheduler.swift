import Foundation

// MARK: - Doğrusal Isınma + Kosinüs Soğutma Öğrenme Oranı Zamanlayıcısı (LRScheduler)

/// **Linear Warmup + Cosine Annealing Learning Rate Scheduler**
/// Sıfırdan (from-scratch) Transformer ön eğitiminde iki kritik sorunu çözer:
/// 1. **Linear Warmup ($0 \le t \le T_{\text{warmup}}$)**:
///    Eğitimin başında ağırlıklar rastgele (Kaiming/Xavier) olduğu ve AdamW'nin ikinci moment ($v_t$)
///    tahmini henüz ısınmadığı için, yüksek öğrenme oranı erken gradyan patlamasına yol açar.
///    Bu aşamada öğrenme oranı $0$'dan $\eta_{\max}$ değerine doğrusal olarak çıkarılır:
///    $$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$
///
/// 2. **Cosine Annealing ($T_{\text{warmup}} < t \le T_{\text{total}}$)**:
///    Isınma tamamlandıktan sonra öğrenme oranı bir kosinüs yarım dalgası boyunca pürüzsüzce
///    $\eta_{\max}$ değerinden $\eta_{\min}$ değerine indirilir. Bu sayede model eğitimin sonunda
///    dar yerel minimumlarda salınım (*oscillation* / tekrar döngüsü) yapmadan kararlı şekilde yakınsar:
///    $$p = \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \in [0, 1]$$
///    $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \cdot \left(1 + \cos(\pi \cdot p)\right)$$
public struct LRScheduler: Sendable, Codable, Equatable {
    /// Doğrusal ısınma (Linear Warmup) adım sayısı ($T_{\text{warmup}}$)
    public let warmupSteps: Int
    /// Toplam eğitim adım sayısı ($T_{\text{total}}$)
    public let totalSteps: Int
    /// Tepe (Maksimum) öğrenme oranı ($\eta_{\max}$)
    public let maxLR: Float
    /// Taban (Minimum) öğrenme oranı ($\eta_{\min}$)
    public let minLR: Float

    public init(
        warmupSteps: Int = 20,
        totalSteps: Int = 100,
        maxLR: Float = 1e-3,
        minLR: Float = 1e-5
    ) {
        precondition(warmupSteps >= 0, "warmupSteps negatif olamaz.")
        precondition(totalSteps > warmupSteps, "totalSteps (\(totalSteps)), warmupSteps (\(warmupSteps)) değerinden büyük olmalıdır.")
        precondition(maxLR >= minLR && minLR >= 0.0, "0 <= minLR <= maxLR koşulu sağlanmalıdır.")

        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.maxLR = maxLR
        self.minLR = minLR
    }

    /// Verilen eğitim adımındaki (`step`) güncel öğrenme oranını ($\eta_t$) hesaplar.
    ///
    /// - Parameter step: Mevcut eğitim adımı (`0` ise `0.0`, `1...warmupSteps` arası doğrusal artış,
    ///                   `warmupSteps...totalSteps` arası kosinüs düşüşü).
    /// - Returns: Güncel öğrenme oranı (`Float`)
    public func getLR(step: Int) -> Float {
        // Adım 0 veya negatifse henüz ısınma başlangıcı (0.0)
        if step <= 0 {
            return 0.0
        }

        // 1. Aşama: Doğrusal Isınma (Linear Warmup) -> 0'dan maxLR'ye doğrusal tırmanış
        if warmupSteps > 0 && step <= warmupSteps {
            let warmupProgress = Float(step) / Float(warmupSteps)
            return maxLR * warmupProgress
        }

        // Toplam adım sayısı aşıldıysa taban öğrenme oranında (minLR) sabit kal
        if step >= totalSteps {
            return minLR
        }

        // 2. Aşama: Kosinüs Soğutma (Cosine Annealing) -> maxLR'den minLR'ye pürüzsüz kosinüs inişi
        let decaySteps = Float(totalSteps - warmupSteps)
        let currentDecayStep = Float(step - warmupSteps)
        let progress = min(1.0, max(0.0, currentDecayStep / decaySteps))

        // cosineDecay = 0.5 * (1.0 + cos(pi * progress))
        let cosineDecay = 0.5 * (1.0 + cos(Float.pi * progress))
        return minLR + (maxLR - minLR) * cosineDecay
    }
}
