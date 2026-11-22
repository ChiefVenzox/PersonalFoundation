# PersonalFoundation

**PersonalFoundation**, Apple Silicon (M-serisi çipler, Unified Memory) üzerinde **mlx-swift** ve **Swift 6 Strict Concurrency** kullanılarak tamamen sıfırdan (*from-scratch*) inşa edilmiş özel bir Dil Modeli (Decoder-only Transformer) mimarisi ve eğitim altyapısıdır.

Dış kütüphanelerden önceden eğitilmiş ağırlık indirmeden; kendi Transformer katmanlarını, BPE tokenizer'ını, veri yükleyicilerini, KV Cache mekanizmasını, eğitim (pre-training) ve ince ayar (SFT) döngülerini barındırır.

---

## 🚀 Temel Özellikler

- **Sıfırdan Transformer Mimarisi (PersonalFoundationModel)**:
  - **RMSNorm**: Sayısal kararlılık için optimize edilmiş katman normalizasyonu.
  - **Rotary Position Embedding (RoPE)**: Göreli pozisyon kodlaması.
  - **Grouped-Query Attention (GQA)**: 12 Query / 4 Key-Value dikkat başlığı (3:1 oran) ile hızlı çıkarım ve düşük bellek tüketimi.
  - **SwiGLU MLP**: `(Swish(xW_gate) * xW_up) W_down` formülüyle yüksek kapasiteli ileri besleme bloğu.
  - **KV Cache**: `KVCache` sınıfı ile oto-regresif üretimde $O(1)$ adım karmaşıklığı.
- **True Mixed Precision**:
  - Master Weights `.float32` hassasiyetinde tutulurken matris çarpımları ve gradyan hesapları donanım hızlandırmalı `.bfloat16` üzerinde koşar.
- **Özel BPE Tokenizer (Byte-Pair Encoding)**:
  - Byte-düzeyinde kelime dağarcığı ve özel chat token'ları (`<|im_start|>`, `<|im_end|>`, `<|user|>`, `<|assistant|>`).
  - Hızlı JSON ve `.pftok` serileştirme/deserileştirme desteği.
- **Eğitim & SFT Döngüsü**:
  - `Trainer`: Gradient Accumulation, Gradient Clipping (`maxGradNorm`), Cosine Annealing LR Scheduler (`LRScheduler`) ve AdamW optimizasyonu.
  - Maskeli SFT Eğitimi (`-100` maskesi ile kullanıcı komutlarını loss hesabından hariç tutarak sadece asistan yanıtlarını öğrenme).
  - Checkpoint yönetimi (`CheckpointManager`) ile safetensors formatında ağırlık kaydetme ve geri yükleme.
- **Apple Silicon & Swift 6**:
  - Tüm yapılar `Sendable` protokolüne tam uyumlu, veri yarışlarından (data-race) arındırılmış modern eşzamanlılık tasarımı.

---

## 📐 Mimari Özellikleri (110M Varsayılanı)

| Parametre | Değer |
| :--- | :--- |
| **Toplam Parametre** | ~110 Milyon |
| **Sözlük Boyutu (Vocab Size)** | 32.000 |
| **Gizli Boyut ($d_{model}$)** | 768 |
| **Ara Katman Boyutu ($d_{ff}$)** | 2.048 (SwiGLU) |
| **Katman Sayısı ($L$)** | 12 Blok |
| **Dikkat Başlıkları (Heads)** | 12 Query, 4 Key/Value (GQA) |
| **Head Dim** | 64 |
| **Bağlam Uzunluğu (Seq Len)** | 512 |
| **Pozisyon Kodlama** | RoPE ($\theta = 10.000$) |

---

## 🛠️ Kurulum & Derleme

### Gereksinimler
- macOS 14+ (Sonoma veya Sequoia)
- Apple Silicon işlemci (M1/M2/M3/M4/M5)
- Xcode 16+ veya Swift 6.0 Araç Zinciri

### Projeyi Derleme
```bash
git clone https://github.com/ChiefVenzox/PersonalFoundation.git
cd PersonalFoundation
swift build
```

---

## 💻 Kullanım

### 1. Modeli Eğitme (Pre-Training & SFT)
Eğitim sürecini (veri kümesi hazırlığı, tokenizer yüklemesi ve iterasyonları) başlatmak için:

```bash
swift run -c release PersonalFoundationTrain
```

### 2. Sohbet & Çıkarım (Inference)
Eğitilen model ile interaktif terminal sohbeti gerçekleştirmek için:

```bash
swift run -c release PersonalFoundationChat
```

---

## 📁 Proje Yapısı

```
PersonalFoundation/
├── Package.swift
├── Sources/
│   ├── PersonalFoundation/             # Çekirdek Kütüphane
│   │   ├── ModelConfig.swift           # Model mimarisi ve ağırlık başlatıcılar
│   │   ├── PersonalFoundationModel.swift # Transformer mimarisi
│   │   ├── Layers.swift                # RMSNorm, RoPE, MultiHeadAttention, SwiGLU
│   │   ├── BPETokenizer.swift          # Byte-Pair Encoding Tokenizer
│   │   ├── KVCache.swift               # Çıkarım için Key-Value önbelleği
│   │   ├── Generator.swift             # Top-p, Temperature ile metin üretimi
│   │   ├── Trainer.swift               # AdamW, Gradient Accumulation & Loss
│   │   ├── LRScheduler.swift           # Warmup & Cosine decay
│   │   ├── CheckpointManager.swift     # Safetensors kayıt yöneticisi
│   │   ├── CorpusFetcher.swift         # Külliyat toplayıcı
│   │   ├── ChatCorpusFetcher.swift     # SFT sohbet verisi toplayıcı
│   │   ├── TextDataLoader.swift        # Token akış veri yükleyicisi
│   │   ├── StreamingDataLoader.swift   # Bellek dostu akış yükleyici
│   │   ├── CorpusDataLoader.swift      # Ham metin yükleyici
│   │   └── ChatDataLoader.swift        # Maskeli SFT veri yükleyici
│   ├── PersonalFoundationTrain/        # Eğitim Uygulaması
│   │   └── main.swift
│   └── PersonalFoundationChat/         # İnteraktif Sohbet Uygulaması
│       └── main.swift
├── turkish_chat_corpus.jsonl           # Türkçe SFT eğitim veri kümesi
├── personalfoundation_110m_tokenizer.json # 32K Tokenizer sözlüğü
└── LICENSE
```

---

## 📄 Lisans
Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
