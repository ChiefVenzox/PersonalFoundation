# frozen_string_literal: true

module PersonalFoundation
  class Response
    attr_reader :prompt, :text, :tokens_generated, :prefill_time_ms, :tokens_per_second, :raw

    def initialize(payload)
      @raw = payload
      @prompt = payload["prompt"]
      @text = payload["response"]
      @tokens_generated = payload["tokens_generated"]
      @prefill_time_ms = payload["prefill_time_ms"]
      @tokens_per_second = payload["tokens_per_second"]
    end

    def to_s
      @text.to_s
    end

    def inspect
      "#<PersonalFoundation::Response text=#{text.inspect} tokens=#{tokens_generated} tps=#{tokens_per_second&.round(1)}>"
    end
  end
end
