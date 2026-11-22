import Foundation
import MLX
import MLXRandom

// MARK: - 1. Örnekleme Yapılandırması (SamplingConfig)

/// Otoregresif metin üretimi (Inference) sırasında olasılık dağılımını şekillendiren hiperparametreler.
/// Swift 6 Strict Concurrency için `Sendable` uyumludur.
public struct SamplingConfig: Sendable, Codable, Equatable {
    /// **Sıcaklık (Temperature - $T$)**
    /// Logitleri örnekleme öncesinde ölçekler ($z'_i = z_i / T$).
    /// - $T = 0.0$: Tamamen deterministik (Greedy Decoding - en yüksek olasılıklı token seçilir).
    /// - $0 < T < 1.0$: Dağılımı keskinleştirir (daha tutarlı ve odaklı üretim).
    /// - $T > 1.0$: Dağılımı düzleştirir (daha yaratıcı ve çeşitli üretim).
    public let temperature: Float

    /// **Top-K Filtreleme**
    /// Yalnızca en yüksek olasılıklı $K$ adet aday tokenı tutar, geri kalan tüm sözlüğü $-\infty$ ile maskeler.
    /// `nil` veya `0` ise devre dışıdır.
    public let topK: Int?

    /// **Top-P (Nucleus / Çekirdek) Örnekleme**
    /// Olasılıkları büyükten küçüğe sıralar ve kümülatif olasılık toplamı $P$ eşiğine ulaşana kadarki
    /// en küçük aday kümesini (çekirdeği) seçer ($0.0 < P \le 1.0$).
    public let topP: Float

    /// **Tekrar Cezası (Repetition Penalty - $\theta_{\text{rep}}$)**
    /// Daha önce üretilmiş tokenların tekrar seçilme olasılığını düşürür (`1.0` = ceza yok, `1.1 - 1.2` = standart).
    /// Pozitif logitleri $\theta_{\text{rep}}$ değerine böler, negatif logitleri $\theta_{\text{rep}}$ ile çarpar.
    public let repetitionPenalty: Float

    /// Üretilecek maksimum yeni token sayısı
    public let maxNewTokens: Int

    /// Üretimi sonlandıracak Bitiş Token Kimliği (End-of-Sequence Token ID)
    public let eosTokenId: Int32?

    public init(
        temperature: Float = 0.8,
        topK: Int? = 40,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.1,
        maxNewTokens: Int = 32,
        eosTokenId: Int32? = nil
    ) {
        precondition(temperature >= 0.0, "temperature negatif olamaz.")
        precondition(topP > 0.0 && topP <= 1.0, "topP (0.0, 1.0] aralığında olmalıdır.")
        precondition(repetitionPenalty >= 1.0, "repetitionPenalty en az 1.0 olmalıdır.")
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxNewTokens = maxNewTokens
        self.eosTokenId = eosTokenId
    }
}

/// Üretim performans ve çıktı metriklerini taşıyan `Sendable` sonuç yapısı.
public struct GenerationResult: Sendable {
    /// Başlangıç (Prompt) tokenları dahil tam token dizisi
    public let allTokens: [Int32]
    /// Yalnızca model tarafından yeni üretilen tokenlar
    public let generatedTokens: [Int32]
    /// Prefill (Prompt işleme) süresi (milisaniye)
    public let prefillTimeMs: Double
    /// Decode (Token üretim) hızı (Token / Saniye)
    public let tokensPerSecond: Double
}

// MARK: - 2. KV-Cache Destekli Otoregresif Üretici (TextGenerator)

/// **PersonalFoundationModel Otoregresif Çıkarım (Inference) Motoru**
/// İki aşamalı (**Prefill** + **KV-Cache Decode**) üretim döngüsü ve
/// sıfırdan matematiksel örnekleme (`Repetition Penalty`, `Temperature`, `Top-K`, `Top-P Nucleus`) algoritmalarını içerir.
public final class TextGenerator {
    public let model: PersonalFoundationModel

    public init(model: PersonalFoundationModel) {
        self.model = model
    }

    // MARK: - Matematiksel Örnekleme (Sampling) Algoritmaları

    /// Son adımın ham logit vektöründen (`[VocabSize]`) bir sonraki token ID'sini örnekler.
    ///
    /// Matematiksel Sıra:
    /// 1. **Tekrar Cezası (Repetition Penalty)**:
    ///    $z_t \leftarrow \begin{cases} z_t / \theta_{\text{rep}} & \text{if } z_t > 0 \\ z_t \cdot \theta_{\text{rep}} & \text{if } z_t \le 0 \end{cases}$
    /// 2. **Greedy Decoding ($T = 0$)**: $\hat{y} = \arg\max_i z_i$
    /// 3. **Temperature Ölçekleme**: $z'_i = z_i / T$
    /// 4. **Top-K & Top-P (Nucleus) Filtreleme**:
    ///    $p_i = \frac{\exp(z'_i - \max_j z'_j)}{\sum_k \exp(z'_k - \max_j z'_j)}$
    ///    Azalan sırada sıralanmış olasılıklarda $\sum_{m=0}^{j-1} p_{(m)} \ge P$ olan kuyruk tokenları elenir.
    /// 5. **Kategorik (Multinomial) Örnekleme**: Yeniden normalize edilmiş çekirdek dağılımından rastgele seçim yapılır.
    public static func sampleNextToken(
        logits: MLXArray,
        previousTokens: [Int32],
        config: SamplingConfig
    ) -> Int32 {
        let logitsF32 = logits.asType(.float32)
        eval(logitsF32)
        var logitValues = logitsF32.asArray(Float.self)
        let vocabSize = logitValues.count

        // 1. Tekrar Cezası (Repetition Penalty) Uygulaması
        if config.repetitionPenalty > 1.0 && !previousTokens.isEmpty {
            let uniqueSeen = Set(previousTokens)
            for tokenId in uniqueSeen {
                let idx = Int(tokenId)
                if idx >= 0 && idx < vocabSize {
                    let currentLogit = logitValues[idx]
                    logitValues[idx] = currentLogit > 0
                        ? (currentLogit / config.repetitionPenalty)
                        : (currentLogit * config.repetitionPenalty)
                }
            }
        }

        // 2. Greedy Decoding (Sıcaklık == 0 ise en yüksek logit'li tokenı doğrudan seç)
        if config.temperature <= 1e-6 {
            var bestIdx = 0
            var bestVal = -Float.infinity
            for (i, val) in logitValues.enumerated() where val > bestVal {
                bestVal = val
                bestIdx = i
            }
            return Int32(bestIdx)
        }

        // 3. Temperature Ölçekleme: z'_i = z_i / T
        let invTemp = 1.0 / config.temperature
        for i in 0..<vocabSize {
            logitValues[i] *= invTemp
        }

        // İndeks ve ölçeklenmiş logit çiftlerini büyükten küçüğe sırala
        var indexedLogits: [(index: Int, logit: Float)] = logitValues.enumerated().map { ($0.offset, $0.element) }
        indexedLogits.sort { $0.logit > $1.logit }

        // 4. Top-K Kesmesi: Yalnızca en yüksek K adayı tut
        if let topK = config.topK, topK > 0, topK < indexedLogits.count {
            indexedLogits = Array(indexedLogits.prefix(topK))
        }

        // 5. Sayısal kararlı Softmax ile olasılıkları hesapla
        let maxLogit = indexedLogits[0].logit
        var probs: [(index: Int, prob: Float)] = []
        probs.reserveCapacity(indexedLogits.count)

        var sumExp: Float = 0.0
        for item in indexedLogits {
            let e = exp(item.logit - maxLogit)
            probs.append((item.index, e))
            sumExp += e
        }

        for i in 0..<probs.count {
            probs[i].prob /= sumExp
        }

        // 6. Top-P (Nucleus) Kesmesi: Kümülatif olasılığı `topP` eşiğine ulaşan çekirdek kümeyi tut
        if config.topP < 1.0 {
            var cumulativeProb: Float = 0.0
            var nucleusCount = 0
            for item in probs {
                cumulativeProb += item.prob
                nucleusCount += 1
                if cumulativeProb >= config.topP {
                    break
                }
            }
            probs = Array(probs.prefix(max(1, nucleusCount)))
        }

        // 7. Çekirdek (Nucleus) Olasılıklarını Yeniden Normalize Et ve Kategorik Örnekle
        let retainedSum = probs.reduce(Float(0.0)) { $0 + $1.prob }
        let r = MLXRandom.uniform(low: Float(0.0), high: retainedSum, [1]).item(Float.self)

        var runningSum: Float = 0.0
        for item in probs {
            runningSum += item.prob
            if r <= runningSum {
                return Int32(item.index)
            }
        }

        return Int32(probs.last?.index ?? 0)
    }

    // MARK: - İki Aşamalı (Prefill + KV-Cache Decode) Üretim Döngüsü

    /// Verilen başlangıç (`promptTokens`) dizisinden başlayarak KV-Cache ile yeni tokenlar üretir.
    ///
    /// - Parameters:
    ///   - promptTokens: Başlangıç token kimlikleri (`[Int32]`)
    ///   - config: Örnekleme yapılandırması (`SamplingConfig`)
    ///   - onTokenGenerated: Her yeni token üretildiğinde anlık akış (streaming) için çağrılan opsiyonel kapanış (closure).
    /// - Returns: Üretilen tokenlar ve hız metriklerini içeren `GenerationResult`.
    public func generate(
        promptTokens: [Int32],
        config: SamplingConfig = SamplingConfig(),
        onTokenGenerated: ((Int32) -> Void)? = nil
    ) -> GenerationResult {
        precondition(!promptTokens.isEmpty, "Prompt token dizisi boş olamaz.")

        model.train(false)
        let kvCache = model.makeKVCache()

        var allTokens = promptTokens
        var generatedTokens: [Int32] = []
        generatedTokens.reserveCapacity(config.maxNewTokens)

        // --- AŞAMA 1: PREFILL (Tüm Prompt'u Paralel İşle ve KV-Cache'i Doldur) ---
        let prefillStart = CFAbsoluteTimeGetCurrent()

        // Boyut: [1, PromptSeqLen]
        let promptArray = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        let prefillLogits = model(promptArray, cache: kvCache)

        // Son pozisyonun logit vektörünü al: [VocabSize]
        let lastPromptIndex = promptTokens.count - 1
        var nextTokenLogits = prefillLogits[0, lastPromptIndex, 0...]
        eval(nextTokenLogits)

        let prefillTimeMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000.0

        // İlk yeni tokenı örnekle
        let firstNewToken = Self.sampleNextToken(
            logits: nextTokenLogits,
            previousTokens: allTokens,
            config: config
        )
        allTokens.append(firstNewToken)
        generatedTokens.append(firstNewToken)
        onTokenGenerated?(firstNewToken)

        if let eos = config.eosTokenId, firstNewToken == eos {
            return GenerationResult(
                allTokens: allTokens,
                generatedTokens: generatedTokens,
                prefillTimeMs: prefillTimeMs,
                tokensPerSecond: 0.0
            )
        }

        // --- AŞAMA 2: KV-CACHE DECODE (Her Adımda Yalnızca Tek Token [1, 1] İşle) ---
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var currentToken = firstNewToken

        for _ in 1..<config.maxNewTokens {
            //Maksimum bağlam uzunluğu sınırını kontrol et
            if kvCache.currentSequenceLength >= model.config.maxSequenceLength {
                break
            }

            // Yalnızca son üretilen tek tokenı modele ver: Boyut [1, 1]
            // Geçmiş tokenların K ve V tensörleri kvCache içerisinden O(1) sürede çekilir!
            let singleTokenInput = MLXArray([currentToken]).reshaped([1, 1])
            let stepLogits = model(singleTokenInput, cache: kvCache)

            nextTokenLogits = stepLogits[0, 0, 0...]
            eval(nextTokenLogits)

            let sampledToken = Self.sampleNextToken(
                logits: nextTokenLogits,
                previousTokens: allTokens,
                config: config
            )

            allTokens.append(sampledToken)
            generatedTokens.append(sampledToken)
            onTokenGenerated?(sampledToken)
            currentToken = sampledToken

            if let eos = config.eosTokenId, sampledToken == eos {
                break
            }
        }

        let decodeElapsedSec = max(1e-6, CFAbsoluteTimeGetCurrent() - decodeStart)
        let decodeTokenCount = max(0, generatedTokens.count - 1)
        let tokensPerSec = Double(decodeTokenCount) / decodeElapsedSec

        return GenerationResult(
            allTokens: allTokens,
            generatedTokens: generatedTokens,
            prefillTimeMs: prefillTimeMs,
            tokensPerSecond: tokensPerSec
        )
    }

    /// Doğrudan bir metin (`prompt: String`) alır, `BPETokenizer` ile kodlar,
    /// KV-Cache ile yeni tokenlar üretir ve çözülmüş (`decoded`) tam metni döndürür.
    /// Not: Eğer `prompt` `<|im_start|>` sohbet etiketiyle başlıyorsa `addBOS` otomatik olarak devre dışı bırakılır.
    public func generateText(
        prompt: String,
        tokenizer: BPETokenizer,
        config: SamplingConfig = SamplingConfig(),
        addBOS: Bool = false
    ) -> (fullText: String, generatedText: String, result: GenerationResult) {
        let shouldAddBOS = addBOS && !prompt.hasPrefix(SpecialToken.imStart.textRepresentation)
        let promptTokens = tokenizer.encode(prompt, addBOS: shouldAddBOS, addEOS: false)
        let result = generate(promptTokens: promptTokens, config: config)
        let fullText = tokenizer.decode(result.allTokens, skipSpecialTokens: false)
        let generatedText = tokenizer
            .decode(result.generatedTokens, skipSpecialTokens: true)
            .trimmingCharacters(in: .whitespaces)
        return (fullText, generatedText, result)
    }

    /// **Gerçek Zamanlı UTF-8 Tamponlu Canlı Metin Akışı (Streaming Generation)**
    /// Üretilen her yeni tokenı `UTF8StreamingDecoder` üzerinden geçirerek çok baytlı Türkçe karakterleri
    /// (`ğ, ü, ş, ı, ö, ç`) yarım bayt bölünmesine uğratmadan anlık olarak `onTextChunk` kapanışına (closure) iletir.
    public func generateTextStreaming(
        prompt: String,
        tokenizer: BPETokenizer,
        config: SamplingConfig = SamplingConfig(),
        addBOS: Bool = false,
        onTextChunk: (String) -> Void
    ) -> (fullText: String, generatedText: String, result: GenerationResult) {
        let shouldAddBOS = addBOS && !prompt.hasPrefix(SpecialToken.imStart.textRepresentation)
        let promptTokens = tokenizer.encode(prompt, addBOS: shouldAddBOS, addEOS: false)

        var streamDecoder = UTF8StreamingDecoder()
        var accumulatedResponse = ""

        let result = withoutActuallyEscaping(onTextChunk) { escapingCallback in
            generate(
                promptTokens: promptTokens,
                config: config,
                onTokenGenerated: { tokenId in
                    // <|im_end|> veya eosTokenId geldiğinde ekrana basma
                    if let eos = config.eosTokenId, tokenId == eos {
                        return
                    }
                    let textPiece = streamDecoder.appendToken(tokenId, tokenizer: tokenizer, skipSpecialTokens: true)
                    if !textPiece.isEmpty {
                        accumulatedResponse.append(textPiece)
                        escapingCallback(textPiece)
                    }
                }
            )
        }

        let remainingTail = streamDecoder.flush()
        if !remainingTail.isEmpty {
            accumulatedResponse.append(remainingTail)
            onTextChunk(remainingTail)
        }

        let fullText = tokenizer.decode(result.allTokens, skipSpecialTokens: false)
        return (fullText, accumulatedResponse.trimmingCharacters(in: .whitespaces), result)
    }
}
