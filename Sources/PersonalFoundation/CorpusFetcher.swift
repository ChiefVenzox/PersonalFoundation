import Foundation

// MARK: - Modern Türkçe Düz Yazı Filtresi ve 18+ MB Külliyat Toplayıcı (CorpusFetcher)

/// **Canlı Açık Kaynaklardan Modern Türkçe Düz Yazı Toplayıcı ve Sıkı Kalite Filtresi (`CorpusFetcher`)**
/// Üçüncü parti hiçbir kütüphane kullanmadan yalnızca `Foundation` (`URLSession`, `JSONSerialization`, `NSRegularExpression`) ile:
/// 1. Mevcut `corpus.txt` satırlarını sıkı bir kalite filtresinden geçirir:
///    - `(d. ...)`, `(ö. ...)`, `[1]`, `Pn./Ky` gibi dipnot, kısaltma ve referans yığınlarını regex ile temizler.
///    - **Harf ve Boşluk Oranı < %85** olan (aşırı rakam, parantez, noktalama veya kısaltma dizisi içeren) satırları
///      ve **40 karakterden kısa** satırları tamamen eler.
/// 2. Temizlenmiş `corpus.txt` boyutu **18.0 MB** altındaysa Türkçe Wikipedia (`tr.wikipedia.org`),
///    Türkçe Vikisöz (`tr.wikiquote.org`), Türkçe Vikikaynak (`tr.wikisource.org`) ve Wikimedia TR kaynaklarından
///    modern Türkçe düz yazı paragrafları çekerek **18.0+ MB** hedefine tamamlar.
public struct CorpusFetcher: Sendable {

    public enum FetchError: Error, LocalizedError, Sendable {
        case invalidURL(String)
        case fileWriteFailed(String)

        public var errorDescription: String? {
            switch self {
            case let .invalidURL(url):
                return "Geçersiz API uç noktası: \(url)"
            case let .fileWriteFailed(reason):
                return "corpus.txt dosyasına yazılırken hata oluştu: \(reason)"
            }
        }
    }

    // MARK: - Regex Tabanlı Dipnot, Kısaltma ve Ansiklopedik Gürültü Temizleyici

    /// Derlenmiş regex kalıpları:
    /// - `(d. 1920)`, `(ö. 1995)`, `(doğum ...)`, `(ölümü ...)` biyografik tarih parantezleri
    /// - `[1]`, `[24]`, `[kaynak belirtilmeli]` dipnot referansları
    /// - `Pn./Ky`, `Bkz.`, `vb./vs.` gibi eğik çizgili/noktalı kısaltma kalıntıları
    /// - `''` veya `'''` Vikimetin tırnak işaretleri ve boş parantezler `( )`
    private static let noiseRegexPatterns: [NSRegularExpression] = {
        let patterns = [
            "\\((?:[^)]*\\b(?:d\\.|ö\\.|doğum|doğumu|ölüm|ölümü|MÖ|MS)\\b[^)]*)\\)",
            "\\[[0-9]+\\]",
            "\\[(?:kaynak belirtilmeli|kaynak|not\\s*[0-9]*)\\]",
            "\\b[A-ZÇĞİÖŞÜa-zçğıöşü]{1,4}\\.?/[A-ZÇĞİÖŞÜa-zçğıöşü]{1,4}\\.?",
            "'{2,}",
            "\\(\\s*[;,./-]*\\s*\\)"
        ]
        return patterns.compactMap { try? NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }
    }()

    /// Mevcut `corpus.txt` dosyasını sıkı %85 harf/boşluk ve >= 40 karakter düz yazı filtresinden geçirir.
    @discardableResult
    public static func cleanExistingCorpusFile(at fileURL: URL) throws -> Int64 {
        guard FileManager.default.fileExists(atPath: fileURL.path),
              let existingText = try? String(contentsOf: fileURL, encoding: .utf8),
              !existingText.isEmpty else {
            return 0
        }

        let beforeBytes = Int64(existingText.utf8.count)
        let cleanedText = cleanAndFilterParagraphs([existingText])
        let finalContent = cleanedText.isEmpty ? "" : (cleanedText + "\n")

        try finalContent.write(to: fileURL, atomically: true, encoding: .utf8)
        let afterBytes = Int64(finalContent.utf8.count)

        print(String(
            format: "🧹 [CorpusFetcher] Mevcut '%@' sıkı modern düz yazı filtresinden (%%85+ harf/boşluk, >=40 krk) geçirildi: %.2f MB -> %.2f MB.",
            fileURL.lastPathComponent,
            Double(beforeBytes) / (1024.0 * 1024.0),
            Double(afterBytes) / (1024.0 * 1024.0)
        ))
        return afterBytes
    }

    /// Çalışma dizinindeki `corpus.txt` dosyasının boyutunu kontrol eder;
    /// dosya mevcut ve boyutu `minSizeMB` (Varsayılan: `18.0 MB`) üzerindeyse temizleme/indirme yapmadan doğrudan mevcut dosyayı kullanır.
    public static func ensureCorpusExists(
        at fileURL: URL,
        minSizeMB: Double = 18.0,
        targetSizeMB: Double = 18.4
    ) async throws {
        let minBytes = Int64(minSizeMB * 1024.0 * 1024.0)
        let targetBytes = Int64(targetSizeMB * 1024.0 * 1024.0)

        // 1. Dosya zaten >= 18.0 MB ise hiçbir işlem yapmadan doğrudan mevcut dosyayı kullan
        var currentBytes: Int64 = 0
        if FileManager.default.fileExists(atPath: fileURL.path),
           let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
           let fileSize = attrs[.size] as? Int64 {
            currentBytes = fileSize
        }

        if currentBytes >= minBytes {
            let existingMB = Double(currentBytes) / (1024.0 * 1024.0)
            print(String(
                format: "✅ [CorpusFetcher] Mevcut '%@' hazır (%.2f MB >= %.2f MB). İndirme otomatik atlandı, doğrudan diskteki dosya kullanılıyor.",
                fileURL.lastPathComponent,
                existingMB,
                minSizeMB
            ))
            return
        }

        // Dosya 18.0 MB altındaysa önce mevcut içeriği filtrele, sonra eksik kısmı indir
        if FileManager.default.fileExists(atPath: fileURL.path) {
            currentBytes = (try? cleanExistingCorpusFile(at: fileURL)) ?? 0
        }

        if currentBytes >= minBytes {
            let existingMB = Double(currentBytes) / (1024.0 * 1024.0)
            print(String(
                format: "✅ [CorpusFetcher] Sıkı filtrelenmiş '%@' hazır (%.2f MB >= %.2f MB). Yeni indirme atlanıyor.",
                fileURL.lastPathComponent,
                existingMB,
                minSizeMB
            ))
            return
        }

        print(String(
            format: "🌐 [CorpusFetcher] Temiz '%@' boyutu %.2f MB (Hedef: %.2f MB). Modern Türkçe düz yazı kaynaklarından tamamlanıyor...",
            fileURL.lastPathComponent,
            Double(currentBytes) / (1024.0 * 1024.0),
            minSizeMB
        ))

        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }

        guard let fileHandle = try? FileHandle(forWritingTo: fileURL) else {
            throw FetchError.fileWriteFailed("FileHandle yazma modunda açılamadı: \(fileURL.path)")
        }
        defer { try? fileHandle.close() }
        _ = try? fileHandle.seekToEnd()

        let sessionConfig = URLSessionConfiguration.ephemeral
        sessionConfig.timeoutIntervalForRequest = 20.0
        sessionConfig.timeoutIntervalForResource = 30.0
        sessionConfig.httpMaximumConnectionsPerHost = 6
        let session = URLSession(configuration: sessionConfig)

        // Yeni makaleler çekmek için offset 12.000'den başlar
        var hfOffset = 12_000
        let hfBatchLength = 100
        var consecutiveFailures = 0

        let trWikiURL = "https://tr.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&grnlimit=20&prop=extracts&explaintext=1&format=json"
        let wikiquoteURL = "https://tr.wikiquote.org/w/api.php?action=query&generator=random&grnnamespace=0&grnlimit=20&prop=extracts&explaintext=1&format=json"
        let wikisourceURL = "https://tr.wikisource.org/w/api.php?action=query&generator=random&grnnamespace=0&grnlimit=20&prop=extracts&explaintext=1&format=json"

        while currentBytes < targetBytes && consecutiveFailures < 15 {
            let batchStartOffset = hfOffset
            hfOffset += hfBatchLength * 2

            let fetchedChunks: [String] = await withTaskGroup(of: [String]?.self) { group in
                // 1. Türkçe Wikipedia (Modern Düz Yazı, Kültür, Felsefe & Toplum) - 3 paralel istek
                group.addTask { await fetchMediaWikiExtracts(urlString: trWikiURL, session: session) }
                group.addTask { await fetchMediaWikiExtracts(urlString: trWikiURL, session: session) }
                group.addTask { await fetchMediaWikiExtracts(urlString: trWikiURL, session: session) }

                // 2. Türkçe Vikisöz & Vikikaynak (İnsani İfade, Deneme & Edebiyat) - 3 paralel istek
                group.addTask { await fetchMediaWikiExtracts(urlString: wikiquoteURL, session: session) }
                group.addTask { await fetchMediaWikiExtracts(urlString: wikiquoteURL, session: session) }
                group.addTask { await fetchMediaWikiExtracts(urlString: wikisourceURL, session: session) }

                // 3. Yüksek Hacimli Türkçe Metin Akışı (Offset 12.000+)
                if batchStartOffset <= 60_000 {
                    let hfURL1 = "https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.tr&split=train&offset=\(batchStartOffset)&length=\(hfBatchLength)"
                    let hfURL2 = "https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.tr&split=train&offset=\(batchStartOffset + hfBatchLength)&length=\(hfBatchLength)"
                    group.addTask { await fetchHuggingFaceWikipediaRows(urlString: hfURL1, session: session) }
                    group.addTask { await fetchHuggingFaceWikipediaRows(urlString: hfURL2, session: session) }
                }

                var combined: [String] = []
                for await result in group {
                    if let texts = result, !texts.isEmpty {
                        combined.append(contentsOf: texts)
                    }
                }
                return combined
            }

            let cleanedBatch = cleanAndFilterParagraphs(fetchedChunks)
            if !cleanedBatch.isEmpty, let data = (cleanedBatch + "\n").data(using: .utf8) {
                try? fileHandle.write(contentsOf: data)
                currentBytes += Int64(data.count)
                consecutiveFailures = 0

                let progressMB = Double(currentBytes) / (1024.0 * 1024.0)
                let percent = min(100.0, (Double(currentBytes) / Double(minBytes)) * 100.0)
                print(String(
                    format: "📥 [CorpusFetcher] İlerleme: %.2f MB / %.2f MB (%%%.1f) modern Türkçe düz yazı kaydedildi...",
                    progressMB,
                    minSizeMB,
                    percent
                ))
            } else {
                consecutiveFailures += 1
            }
        }

        let finalMB = Double(currentBytes) / (1024.0 * 1024.0)
        print(String(
            format: "✅ [CorpusFetcher] 18+ MB Modern Türkçe düz yazı külliyatı hazır: %@ (Toplam Boyut: %.2f MB)",
            fileURL.lastPathComponent,
            finalMB
        ))
    }

    // MARK: - API İstek Yardımcıları

    private static func fetchHuggingFaceWikipediaRows(
        urlString: String,
        session: URLSession
    ) async -> [String]? {
        guard let url = URL(string: urlString) else { return nil }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("PersonalFoundation/3.0 (AppleSilicon; MLX-Swift)", forHTTPHeaderField: "User-Agent")

        guard let (data, response) = try? await session.data(for: request),
              let httpResp = response as? HTTPURLResponse,
              (200..<300).contains(httpResp.statusCode) else {
            return nil
        }

        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = root["rows"] as? [[String: Any]] else {
            return nil
        }

        var extracted: [String] = []
        extracted.reserveCapacity(rows.count)

        for rowWrapper in rows {
            if let rowDict = rowWrapper["row"] as? [String: Any],
               let text = rowDict["text"] as? String,
               !text.isEmpty {
                extracted.append(text)
            }
        }
        return extracted
    }

    private static func fetchMediaWikiExtracts(
        urlString: String,
        session: URLSession
    ) async -> [String]? {
        guard let url = URL(string: urlString) else { return nil }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("PersonalFoundation/3.0 (AppleSilicon; MLX-Swift)", forHTTPHeaderField: "User-Agent")

        guard let (data, response) = try? await session.data(for: request),
              let httpResp = response as? HTTPURLResponse,
              (200..<300).contains(httpResp.statusCode) else {
            return nil
        }

        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let query = root["query"] as? [String: Any],
              let pages = query["pages"] as? [String: Any] else {
            return nil
        }

        var extracted: [String] = []
        extracted.reserveCapacity(pages.count)

        for (_, pageValue) in pages {
            if let pageDict = pageValue as? [String: Any],
               let extract = pageDict["extract"] as? String,
               !extract.isEmpty {
                extracted.append(extract)
            }
        }
        return extracted
    }

    // MARK: - Sıkı Modern Türkçe Düz Yazı Filtresi (>= %85 Harf/Boşluk & >= 40 Karakter)

    /// Satırları şu kurallarla filtreler:
    /// 1. 40 karakterden kısa satırlar elenir.
    /// 2. Dipnot, tarih parantezleri ve `Pn./Ky` türü kısaltmalar regex ile silinir.
    /// 3. Harf ve boşluk karakterlerinin toplam karakter sayısına oranı **%85'in altında** olan
    ///    (sayı yığını, parantez, noktalama veya kısaltma dizisi içeren) satırlar elenir.
    public static func cleanAndFilterParagraphs(_ rawTexts: [String]) -> String {
        var validLines: [String] = []

        for raw in rawTexts {
            let lines = raw.components(separatedBy: .newlines)
            for line in lines {
                var trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.count < 40 {
                    continue
                }

                // Yapısal başlık, kaynakça, tablo veya liste satırlarını ele
                if trimmed.hasPrefix("==") ||
                    trimmed.hasSuffix("==") ||
                    trimmed.hasPrefix("Kategori:") ||
                    trimmed.hasPrefix("Kaynakça") ||
                    trimmed.hasPrefix("Dış bağlantılar") ||
                    trimmed.hasPrefix("Ayrıca bakınız") ||
                    trimmed.hasPrefix("*") ||
                    trimmed.hasPrefix("|") ||
                    trimmed.hasPrefix("{") ||
                    trimmed.hasPrefix("#") {
                    continue
                }

                // HTML etiket kalıntılarını temizle
                if trimmed.contains("<") && trimmed.contains(">") {
                    trimmed = stripSimpleHTMLTags(from: trimmed)
                }

                // Regex ile dipnot, kısaltma (Pn./Ky) ve tarih parantezlerini temizle
                for regex in noiseRegexPatterns {
                    let range = NSRange(trimmed.startIndex..<trimmed.endIndex, in: trimmed)
                    trimmed = regex.stringByReplacingMatches(in: trimmed, options: [], range: range, withTemplate: "")
                }

                let words = trimmed
                    .components(separatedBy: .whitespaces)
                    .filter { !$0.isEmpty }

                // En az 6 kelimeli ve 40+ karakterli tam cümle/paragraf şartı
                guard words.count >= 6 else {
                    continue
                }

                let collapsed = words
                    .joined(separator: " ")
                    .replacingOccurrences(of: " ,", with: ",")
                    .replacingOccurrences(of: " .", with: ".")

                guard collapsed.count >= 40, isCleanModernTurkishProse(collapsed) else {
                    continue
                }

                validLines.append(collapsed)
            }
        }

        return validLines.joined(separator: "\n")
    }

    /// Satırın harf + boşluk oranının >= %85 olduğunu, eğik çizgili kısaltma yığını içermediğini
    /// ve kurallı cümle noktalamasına sahip olduğunu doğrular.
    private static func isCleanModernTurkishProse(_ text: String) -> Bool {
        guard text.count >= 40 else { return false }

        // Eğik çizgi (Pn./Ky, 1920/21 vb.) veya köşeli parantez kalıntısı varsa ele
        if text.contains("/") || text.contains("[") || text.contains("]") {
            return false
        }

        // Cümle sonu noktalaması (. ! ? …) içermeli
        let hasSentenceEnding = text.hasSuffix(".") || text.hasSuffix("!") || text.hasSuffix("?") || text.hasSuffix("…") || text.hasSuffix(".”") || text.hasSuffix(".\"")
        guard hasSentenceEnding else { return false }

        var letterOrSpaceCount = 0
        var totalScalarCount = 0

        for scalar in text.unicodeScalars {
            totalScalarCount += 1
            if CharacterSet.letters.contains(scalar) || CharacterSet.whitespaces.contains(scalar) {
                letterOrSpaceCount += 1
            }
        }

        guard totalScalarCount > 0 else { return false }

        // KRİTİK KURAL: Harf ve boşluk oranı >= %85 olmalı
        let letterSpaceRatio = Double(letterOrSpaceCount) / Double(totalScalarCount)
        return letterSpaceRatio >= 0.85
    }

    private static func stripSimpleHTMLTags(from input: String) -> String {
        var result = ""
        result.reserveCapacity(input.count)
        var insideTag = false

        for ch in input {
            if ch == "<" {
                insideTag = true
            } else if ch == ">" {
                insideTag = false
            } else if !insideTag {
                result.append(ch)
            }
        }
        return result
    }
}
