import Foundation

// MARK: - Açık Kaynak Türkçe Talimat & Sohbet Külliyatı Toplayıcı (ChatCorpusFetcher)

/// **HuggingFace Datasets Server Üzerinden Gerçek Türkçe Talimat/Sohbet Külliyatı Toplayıcı (`ChatCorpusFetcher`)**
/// Hiçbir elle yazılmış (hardcoded) soru-cevap örneği kullanmadan, yalnızca `Foundation` (`URLSession`, `JSONSerialization`) ile:
/// 1. Çalışma dizininde `turkish_chat_corpus.jsonl` mevcutsa ve en az `minRequiredPairs` (Varsayılan: `2.500`) geçerli
///    `{"user": "...", "assistant": "..."}` satırı içeriyorsa indirmeyi otomatik atlar ve doğrudan diskten okur.
/// 2. Dosya yoksa veya eksikse HuggingFace Datasets Server API üzerinden açık kaynak Türkçe talimat/sohbet
///    veri setlerinden (`TFLai/Turkish-Alpaca`, `malhajar/alpaca-gpt4-tr`, `merve/turkish_instructions`,
///    `matrixportalx/aya-turkish-alpaca` vb.) otomatik şema algılama (`instruction/input/output`, `prompt/response`,
///    `conversations/messages`) ile en az **3.500 adet** temiz Türkçe soru-cevap çifti indirip `turkish_chat_corpus.jsonl` olarak kaydeder.
public struct ChatCorpusFetcher: Sendable {

    /// JSONL dosya satır şeması: `{"user": "...", "assistant": "..."}`
    public struct JSONLChatRecord: Sendable, Codable, Equatable {
        public let user: String
        public let assistant: String

        public init(user: String, assistant: String) {
            self.user = user
            self.assistant = assistant
        }
    }

    private struct DatasetSource: Sendable {
        let dataset: String
        let defaultConfig: String
        let defaultSplit: String
    }

    /// Açık kaynak Türkçe talimat ve sohbet veri setleri listesi
    private static let candidateSources: [DatasetSource] = [
        DatasetSource(dataset: "TFLai/Turkish-Alpaca", defaultConfig: "default", defaultSplit: "train"),
        DatasetSource(dataset: "malhajar/alpaca-gpt4-tr", defaultConfig: "default", defaultSplit: "train"),
        DatasetSource(dataset: "merve/turkish_instructions", defaultConfig: "default", defaultSplit: "train"),
        DatasetSource(dataset: "matrixportalx/aya-turkish-alpaca", defaultConfig: "default", defaultSplit: "train"),
        DatasetSource(dataset: "Aeala/ShareGPT_Vicuna_unfiltered", defaultConfig: "default", defaultSplit: "train"),
        DatasetSource(dataset: "CohereForAI/aya_dataset", defaultConfig: "default", defaultSplit: "train")
    ]

    // MARK: - Ana Kontrol, İndirme ve Yükleme Metodu

    /// `turkish_chat_corpus.jsonl` dosyasını kontrol eder; en az `minValidPairs` (`2.500`) çift varsa doğrudan diskten yükler.
    /// Yoksa veya eksikse açık kaynaklardan `targetPairs` (`3.500+`) adet Türkçe soru-cevap çifti indirip diske yazar ve döndürür.
    @discardableResult
    public static func ensureChatCorpusExists(
        at fileURL: URL,
        minValidPairs: Int = 2_500,
        targetPairs: Int = 3_600
    ) async throws -> [ChatSample] {
        // 1. Diskte mevcut dosya varsa oku ve geçerli çift sayısını kontrol et
        var collectedSamples = loadExistingJSONL(from: fileURL)
        var seenPrompts = Set<String>(collectedSamples.map { $0.question.lowercased() })

        if collectedSamples.count >= minValidPairs {
            print(String(
                format: "✅ [ChatCorpusFetcher] Mevcut '%@' bulundu (%d >= %d geçerli Türkçe sohbet çifti). İndirme otomatik atlandı.",
                fileURL.lastPathComponent,
                collectedSamples.count,
                minValidPairs
            ))
            return collectedSamples
        }

        print(String(
            format: "🌐 [ChatCorpusFetcher] '%@' eksik veya yetersiz (%d / %d çift). HuggingFace açık Türkçe talimat veri setlerinden %d+ çift indiriliyor...",
            fileURL.lastPathComponent,
            collectedSamples.count,
            minValidPairs,
            targetPairs
        ))

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = 20.0
        sessionConfig.timeoutIntervalForResource = 30.0
        sessionConfig.httpMaximumConnectionsPerHost = 6
        let session = URLSession(configuration: sessionConfig)

        // 2. Her veri seti kaynağı üzerinden paralel offset pencereleriyle (length=100) Türkçe talimat çiftleri çek
        for source in candidateSources {
            if collectedSamples.count >= targetPairs {
                break
            }

            // Veri setinin gerçek config ve split adını doğrula
            let (resolvedConfig, resolvedSplit) = await resolveConfigAndSplit(for: source, session: session)
            print("   • Kaynak Taranıyor: \(source.dataset) (config: \(resolvedConfig), split: \(resolvedSplit))...")

            var offset = 0
            let batchSize = 100
            let parallelRequests = 4
            var emptyCycles = 0

            while collectedSamples.count < targetPairs && offset < 25_000 && emptyCycles < 3 {
                let currentStartOffset = offset
                offset += batchSize * parallelRequests

                let batchPairs: [ChatSample] = await withTaskGroup(of: [ChatSample].self) { group in
                    for i in 0..<parallelRequests {
                        let reqOffset = currentStartOffset + (i * batchSize)
                        group.addTask {
                            await fetchRowsBatch(
                                dataset: source.dataset,
                                config: resolvedConfig,
                                split: resolvedSplit,
                                offset: reqOffset,
                                length: batchSize,
                                session: session
                            )
                        }
                    }

                    var combined: [ChatSample] = []
                    for await pairs in group {
                        combined.append(contentsOf: pairs)
                    }
                    return combined
                }

                var addedInBatch = 0
                for sample in batchPairs {
                    let key = sample.question.lowercased()
                    if !seenPrompts.contains(key) {
                        seenPrompts.insert(key)
                        collectedSamples.append(sample)
                        addedInBatch += 1
                    }
                }

                if addedInBatch > 0 {
                    emptyCycles = 0
                    let percent = min(100.0, (Double(collectedSamples.count) / Double(targetPairs)) * 100.0)
                    print(String(
                        format: "📥 [ChatCorpusFetcher] İlerleme: %d / %d (%%%.1f) temiz Türkçe soru-cevap çifti toplandı (%@)...",
                        collectedSamples.count,
                        targetPairs,
                        percent,
                        source.dataset
                    ))
                } else {
                    emptyCycles += 1
                }
            }
        }

        // 3. Eğer mevcut corpus.txt varsa ve hedef sayıya ulaşılamadıysa corpus.txt'deki temiz Türkçe paragraflardan
        //    doğal soru-cevap / açıklama çiftleriyle takviye et (Tamamen gerçek indirilen Türkçe metinlerden, sıfır hardcoded!)
        if collectedSamples.count < minValidPairs {
            let fallbackCorpusURL = fileURL.deletingLastPathComponent().appendingPathComponent("corpus.txt")
            if FileManager.default.fileExists(atPath: fallbackCorpusURL.path),
               let rawCorpus = try? String(contentsOf: fallbackCorpusURL, encoding: .utf8) {
                let extraPairs = extractInstructionPairsFromDownloadedCorpus(
                    rawCorpus: rawCorpus,
                    neededCount: targetPairs - collectedSamples.count,
                    seenPrompts: &seenPrompts
                )
                collectedSamples.append(contentsOf: extraPairs)
                if !extraPairs.isEmpty {
                    print("📥 [ChatCorpusFetcher] İndirilen Türkçe külliyattan \(extraPairs.count) ek açıklama çifti çıkarıldı (Toplam: \(collectedSamples.count) çift).")
                }
            }
        }

        // 4. Tüm temiz çiftleri `{"user": "...", "assistant": "..."}` JSONL formatında diske kaydet
        try saveAsJSONL(samples: collectedSamples, to: fileURL)
        print(String(
            format: "✅ [ChatCorpusFetcher] '%@' diske kaydedildi (Toplam Geçerli Çift: %d).",
            fileURL.lastPathComponent,
            collectedSamples.count
        ))

        return collectedSamples
    }

    // MARK: - JSONL Disk Okuma & Yazma

    public static func loadExistingJSONL(from fileURL: URL) -> [ChatSample] {
        guard FileManager.default.fileExists(atPath: fileURL.path),
              let content = try? String(contentsOf: fileURL, encoding: .utf8),
              !content.isEmpty else {
            return []
        }

        let decoder = JSONDecoder()
        var samples: [ChatSample] = []
        let lines = content.components(separatedBy: .newlines)
        samples.reserveCapacity(lines.count)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { continue }
            if let record = try? decoder.decode(JSONLChatRecord.self, from: data),
               isValidTurkishPair(user: record.user, assistant: record.assistant) {
                samples.append(ChatSample(question: record.user, answer: record.assistant))
            }
        }
        return samples
    }

    public static func saveAsJSONL(samples: [ChatSample], to fileURL: URL) throws {
        let encoder = JSONEncoder()
        var jsonlLines: [String] = []
        jsonlLines.reserveCapacity(samples.count)

        for sample in samples {
            let record = JSONLChatRecord(user: sample.question, assistant: sample.answer)
            if let data = try? encoder.encode(record) {
                jsonlLines.append(String(decoding: data, as: UTF8.self))
            }
        }

        let payload = jsonlLines.joined(separator: "\n") + "\n"
        try payload.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    // MARK: - HuggingFace Datasets Server API Çözümleyicileri

    private static func resolveConfigAndSplit(
        for source: DatasetSource,
        session: URLSession
    ) async -> (config: String, split: String) {
        let encodedDataset = source.dataset.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? source.dataset
        let splitsURLString = "https://datasets-server.huggingface.co/splits?dataset=\(encodedDataset)"
        guard let url = URL(string: splitsURLString) else {
            return (source.defaultConfig, source.defaultSplit)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("PersonalFoundation/4.0 (AppleSilicon; MLX-Swift)", forHTTPHeaderField: "User-Agent")

        guard let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let splits = root["splits"] as? [[String: Any]],
              let firstSplit = splits.first else {
            return (source.defaultConfig, source.defaultSplit)
        }

        // Varsa "turkish" veya "tr" içeren config'i tercih et, yoksa ilk geçerli config/split'i al
        for item in splits {
            if let cfg = item["config"] as? String,
               let sp = item["split"] as? String,
               (cfg.lowercased().contains("tr") || cfg.lowercased().contains("tur")) {
                return (cfg, sp)
            }
        }

        let cfg = (firstSplit["config"] as? String) ?? source.defaultConfig
        let sp = (firstSplit["split"] as? String) ?? source.defaultSplit
        return (cfg, sp)
    }

    private static func fetchRowsBatch(
        dataset: String,
        config: String,
        split: String,
        offset: Int,
        length: Int,
        session: URLSession
    ) async -> [ChatSample] {
        let encDataset = dataset.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? dataset
        let encConfig = config.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? config
        let encSplit = split.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? split

        let urlString = "https://datasets-server.huggingface.co/rows?dataset=\(encDataset)&config=\(encConfig)&split=\(encSplit)&offset=\(offset)&length=\(length)"
        guard let url = URL(string: urlString) else { return [] }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("PersonalFoundation/4.0 (AppleSilicon; MLX-Swift)", forHTTPHeaderField: "User-Agent")

        guard let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = root["rows"] as? [[String: Any]] else {
            return []
        }

        var parsedSamples: [ChatSample] = []
        parsedSamples.reserveCapacity(rows.count)

        for rowWrapper in rows {
            guard let row = rowWrapper["row"] as? [String: Any] else { continue }
            if let pair = extractChatPairFromRow(row) {
                parsedSamples.append(pair)
            }
        }

        return parsedSamples
    }

    // MARK: - Otomatik Şema Algılayıcı (Alpaca / Prompt-Response / Conversations-Messages)

    private static func extractChatPairFromRow(_ row: [String: Any]) -> ChatSample? {
        // Dil alanı varsa ve Türkçe değilse atla
        if let lang = (row["language"] as? String) ?? (row["lang"] as? String) {
            let lower = lang.lowercased()
            if !lower.contains("tur") && lower != "tr" {
                return nil
            }
        }

        var rawUser: String? = nil
        var rawAssistant: String? = nil

        // Şema 1: Alpaca Formatı (`instruction` + opsiyonel `input` + `output`)
        if let instruction = (row["instruction"] as? String) ?? (row["talimat"] as? String) ?? (row["soru"] as? String),
           let output = (row["output"] as? String) ?? (row["çıktı"] as? String) ?? (row["cevap"] as? String) ?? (row["response"] as? String) {
            let inputField = ((row["input"] as? String) ?? (row["girdi"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if !inputField.isEmpty && inputField.lowercased() != "none" && inputField.lowercased() != "yok" && inputField.lowercased() != "null" {
                rawUser = "\(instruction.trimmingCharacters(in: .whitespacesAndNewlines)) \(inputField)"
            } else {
                rawUser = instruction
            }
            rawAssistant = output
        }
        // Şema 2: Prompt / Response veya Question / Answer Formatı
        else if let prompt = (row["prompt"] as? String) ?? (row["question"] as? String) ?? (row["inputs"] as? String) ?? (row["user"] as? String),
                let response = (row["response"] as? String) ?? (row["answer"] as? String) ?? (row["targets"] as? String) ?? (row["assistant"] as? String) ?? (row["completion"] as? String) {
            rawUser = prompt
            rawAssistant = response
        }
        // Şema 3: Conversations veya Messages Dizisi Formatı (`[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`)
        else if let convs = (row["conversations"] as? [[String: Any]]) ?? (row["messages"] as? [[String: Any]]),
                convs.count >= 2 {
            for i in 0..<(convs.count - 1) {
                let first = convs[i]
                let second = convs[i + 1]
                let role1 = ((first["from"] as? String) ?? (first["role"] as? String) ?? "").lowercased()
                let role2 = ((second["from"] as? String) ?? (second["role"] as? String) ?? "").lowercased()
                if (role1 == "human" || role1 == "user") && (role2 == "gpt" || role2 == "assistant" || role2 == "bot") {
                    rawUser = (first["value"] as? String) ?? (first["content"] as? String)
                    rawAssistant = (second["value"] as? String) ?? (second["content"] as? String)
                    break
                }
            }
        }

        guard let u = rawUser, let a = rawAssistant else {
            return nil
        }

        let cleanUser = normalizeSingleLineText(u)
        let cleanAssistant = normalizeSingleLineText(a)

        guard isValidTurkishPair(user: cleanUser, assistant: cleanAssistant) else {
            return nil
        }

        return ChatSample(question: cleanUser, answer: cleanAssistant)
    }

    // MARK: - Kalite, Uzunluk (<= 450 Token) ve Türkçe Doğrulama Filtresi

    private static func normalizeSingleLineText(_ text: String) -> String {
        return text
            .replacingOccurrences(of: "\r\n", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Çok kısa (< 10 karakter) veya çok uzun (> 450 token / ~1.350 karakter) çiftleri ve
    /// bozuk/kod yığını içeren satırları eler.
    private static func isValidTurkishPair(user: String, assistant: String) -> Bool {
        // 1. Minimum karakter uzunluğu kontrolü (>= 10 karakter)
        guard user.count >= 10, assistant.count >= 15 else {
            return false
        }

        // 2. Maksimum uzunluk kontrolü (seqLength = 512 içine <|im_end|> ile birlikte tam sığması için <= ~450 token / 1350 karakter)
        guard (user.count + assistant.count) <= 1_350, assistant.count <= 1_000 else {
            return false
        }

        // 3. Harf ve boşluk oranı kontrolü (Kod, JSON veya bozuk sembol yığınlarını eler)
        var letterSpaceCount = 0
        var totalScalars = 0
        for scalar in assistant.unicodeScalars {
            totalScalars += 1
            if CharacterSet.letters.contains(scalar) || CharacterSet.whitespaces.contains(scalar) {
                letterSpaceCount += 1
            }
        }
        guard totalScalars > 0, (Double(letterSpaceCount) / Double(totalScalars)) >= 0.80 else {
            return false
        }

        return true
    }

    /// Ağ bağlantısı kesilirse veya HF API hız sınırına takılırsa, halihazırda açık kaynaklardan indirilmiş olan
    /// `corpus.txt` paragraflarındaki çok-cümleli Türkçe metinlerden doğal bağlam-tamamlama / açıklama çiftleri üretir.
    private static func extractInstructionPairsFromDownloadedCorpus(
        rawCorpus: String,
        neededCount: Int,
        seenPrompts: inout Set<String>
    ) -> [ChatSample] {
        guard neededCount > 0 else { return [] }
        var extra: [ChatSample] = []
        extra.reserveCapacity(neededCount)

        let lines = rawCorpus.components(separatedBy: .newlines)
        for line in lines {
            if extra.count >= neededCount { break }
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.count >= 120 && trimmed.count <= 950 else { continue }

            // Paragrafı ilk cümlesinden (konu/bağlam) ve devamındaki açıklama cümlelerinden ikiye ayır
            if let dotRange = trimmed.range(of: ". ") {
                let firstSentence = String(trimmed[..<dotRange.upperBound]).trimmingCharacters(in: .whitespaces)
                let continuation = String(trimmed[dotRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                if isValidTurkishPair(user: firstSentence, assistant: continuation) {
                    let key = firstSentence.lowercased()
                    if !seenPrompts.contains(key) {
                        seenPrompts.insert(key)
                        extra.append(ChatSample(question: firstSentence, answer: continuation))
                    }
                }
            }
        }
        return extra
    }
}
