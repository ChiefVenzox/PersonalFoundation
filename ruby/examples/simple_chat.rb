#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "../lib/personal_foundation"

puts "=== PersonalFoundation Ruby Client Demo ==="

prompt = "Türkiye'nin başkenti neresidir?"
puts "\nSoru: #{prompt}"
puts "Model yanıtı hesaplanıyor..."

response = PersonalFoundation.generate(prompt, temperature: 0.7, max_tokens: 60)

puts "\nYanıt: #{response.text}"
puts "\nİstatistikler:"
puts " - Üretilen Token: #{response.tokens_generated}"
puts " - Prefill Süresi: #{response.prefill_time_ms&.round(2)} ms"
puts " - Üretim Hızı   : #{response.tokens_per_second&.round(1)} token/sn"
