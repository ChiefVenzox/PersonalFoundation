# PersonalFoundation

<p align="center">
  <a href="https://github.com/ChiefVenzox/PersonalFoundation/actions/workflows/ci.yml"><img src="https://github.com/ChiefVenzox/PersonalFoundation/actions/workflows/ci.yml/badge.svg" alt="CI - Swift Build & Test" /></a>
  <a href="https://github.com/ChiefVenzox/PersonalFoundation/actions/workflows/ruby-gem.yml"><img src="https://github.com/ChiefVenzox/PersonalFoundation/actions/workflows/ruby-gem.yml/badge.svg" alt="Ruby Gem CI" /></a>
  <img src="https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat-square&logo=swift" alt="Swift 6.0" />
  <img src="https://img.shields.io/badge/MLX-mlx--swift-blue.svg?style=flat-square" alt="MLX Swift" />
  <img src="https://img.shields.io/badge/Ruby-gem%20v0.0.1-red.svg?style=flat-square&logo=ruby" alt="Ruby Gem" />
  <img src="https://img.shields.io/badge/Platform-Apple%20Silicon-silver.svg?style=flat-square&logo=apple" alt="Apple Silicon" />
  <img src="https://img.shields.io/badge/GitHub%20Packages-personal__foundation-blueviolet.svg?style=flat-square" alt="GitHub Packages" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square&logo=opensourceinitiative" alt="MIT License" />
  <img src="https://img.shields.io/badge/Security-Policy-brightgreen.svg?style=flat-square" alt="Security" />
</p>

**PersonalFoundation**, Apple Silicon (M-serisi çipler, Unified Memory) üzerinde **mlx-swift** ve **Swift 6 Strict Concurrency** kullanılarak tamamen sıfırdan (*from-scratch*) inşa edilmiş özel bir Dil Modeli (Decoder-only Transformer) mimarisi ve eğitim altyapısıdır.

Dış kütüphanelerden önceden eğitilmiş ağırlık indirmeden; kendi Transformer katmanlarını, BPE tokenizer'ını, veri yükleyicilerini, KV Cache mekanizmasını, eğitim (pre-training) ve ince ayar (SFT) döngülerini barındırır.

---

## 📖 Proje Dokümantasyonu & Wiki

Kapsamlı teknik açıklamalar ve matematiksel formülasyonlar için projenin **[GitHub Wiki](https://github.com/ChiefVenzox/PersonalFoundation/wiki)** sayfalarını inceleyebilirsiniz:
- 🏛️ **[Mimari Tasarım (Architecture)](https://github.com/ChiefVenzox/PersonalFoundation/wiki/Architecture)**: RMSNorm, RoPE, Grouped-Query Attention (GQA), SwiGLU ve True Mixed Precision detayları.
- 🔤 **[BPE Tokenizer](https://github.com/ChiefVenzox/PersonalFoundation/wiki/Tokenizer)**: 32K sözlük, özel chat tokenları (`<|im_start|>`, vb.) ve serileştirme.
- 🏋️ **[Eğitim & SFT (Training & Fine-Tuning)](https://github.com/ChiefVenzox/PersonalFoundation/wiki/Training-and-Fine-Tuning)**: Gradient Accumulation, AdamW, Cosine Annealing LR ve -100 hedef maskeli SFT eğitimi.
- 💬 **[Çıkarım & Sohbet (Inference & Chat)](https://github.com/ChiefVenzox/PersonalFoundation/wiki/Inference-and-Chat)**: KV Cache, Top-p ve Temperature örneklemesi ile interaktif terminal sohbeti.

---

## 🚀 Temel Özellikler

- **Sıfırdan Transformer Mimarisi (PersonalFoundationModel)**:
  - **RMSNorm**: Sayısal kararlılık için optimize edilmiş katman normalizasyonu.
  - **Rotary Position Embedding (RoPE)**: Göreli pozisyon kodlaması ($\theta = 10.000$).
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

Tek komutla doğrudan yanıt üretmek için:
```bash
swift run PersonalFoundationChat --prompt "Yapay zeka nedir?" --json
```

---

## 💎 Ruby Entegrasyonu & Gem Paketi (`personal_foundation`)

PersonalFoundation modelini **Ruby** projelerinizden doğrudan çalıştırmak ve entegre etmek için `personal_foundation` gem'i hazırlanmıştır.

### Kurulum (GitHub Packages)
```bash
gem install personal_foundation --source "https://rubygems.pkg.github.com/ChiefVenzox"
```

Veya `Gemfile` dosyanıza ekleyerek:
```ruby
source "https://rubygems.pkg.github.com/ChiefVenzox" do
  gem "personal_foundation", "~> 0.0.1"
end
```

### Ruby ile Kullanım Örneği
```ruby
require "personal_foundation"

# 1. Hızlı Yanıt Üretimi
response = PersonalFoundation.generate(
  "Türkiye'nin başkenti neresidir?",
  temperature: 0.7,
  max_tokens: 80
)

puts "Yanıt : #{response.text}"
puts "Hız   : #{response.tokens_per_second.round(1)} token/sn"
puts "Token : #{response.tokens_generated}"

# 2. Doğrudan Metin Olarak Sohbet
cevap = PersonalFoundation.chat("Atatürk kimdir?")
puts cevap
```

Detaylar için [ruby/README.md](ruby/README.md) dosyasını inceleyebilirsiniz.

---

## 👥 Contributors & Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ChiefVenzox">
        <img src="https://avatars.githubusercontent.com/u/173578558?v=4" width="100px;" alt="Hakan Kaba"/><br />
        <sub><b>ChiefVenzox (Hakan Kaba)</b></sub>
      </a><br />
      <sub>Creator & Lead AI Architect</sub>
    </td>
    <td align="center">
      <a href="https://github.com/JennyDarling">
        <img src="https://avatars.githubusercontent.com/u/261972903?v=4" width="100px;" alt="Melda Lara Washington"/><br />
        <sub><b>JennyDarling (Melda Lara Washington)</b></sub>
      </a><br />
      <sub>Core Contributor & Researcher</sub>
    </td>
    <td align="center">
      <a href="https://github.com/HNMLabs">
        <img src="https://avatars.githubusercontent.com/u/261977521?v=4" width="100px;" alt="HNM Labs"/><br />
        <sub><b>HNMLabs (HNM Labs)</b></sub>
      </a><br />
      <sub>Research Organization & Engineering</sub>
    </td>
  </tr>
</table>

Ayrıntılı katkı listesi için [CONTRIBUTORS.md](CONTRIBUTORS.md) dosyasını inceleyebilirsiniz.

---

## 🛡️ Güvenlik ve Katkıda Bulunma
- **Güvenlik Politikası:** Güvenlik bildirimleri için lütfen [SECURITY.md](SECURITY.md) dosyasını okuyun.
- **Katkı Rehberi:** Projeye katkıda bulunmak için [CONTRIBUTING.md](CONTRIBUTING.md) ve [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) belgelerini inceleyebilirsiniz.

---

## 📄 Lisans
Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
