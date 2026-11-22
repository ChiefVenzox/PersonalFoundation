# PersonalFoundation Ruby Gem

The official Ruby package for **PersonalFoundation** — the from-scratch Decoder-only Transformer LLM designed for Apple Silicon (MLX).

---

## 🚀 Installation

### From GitHub Packages:
```bash
gem install personal_foundation --source "https://rubygems.pkg.github.com/ChiefVenzox"
```

### In your Gemfile:
```ruby
source "https://rubygems.pkg.github.com/ChiefVenzox" do
  gem "personal_foundation", "~> 0.0.1"
end
```
Or directly from GitHub:
```ruby
gem "personal_foundation", git: "https://github.com/ChiefVenzox/PersonalFoundation.git", glob: "ruby/*.gemspec"
```

---

## 💻 Usage

### 1. Simple Generation
```ruby
require "personal_foundation"

response = PersonalFoundation.generate("Merhaba! Kendini tanıtır mısın?", temperature: 0.7, max_tokens: 100)

puts response.text
puts "Speed: #{response.tokens_per_second.round(1)} tokens/sec"
puts "Tokens: #{response.tokens_generated}"
```

### 2. Fast Chat String
```ruby
answer = PersonalFoundation.chat("Yapay zeka nedir?")
puts answer
```

### 3. Custom Configuration
```ruby
PersonalFoundation.configure do |config|
  config.temperature = 0.8
  config.max_tokens = 150
  config.model_dir = "/path/to/PersonalFoundation"
end
```

### 4. Command Line Interface (CLI)
When the gem is installed, you can use the `personal_foundation` CLI:
```bash
personal_foundation "Atatürk kimdir?"
personal_foundation -p "İstanbul'u anlat" --json
```

---

## 📄 License
MIT License.
