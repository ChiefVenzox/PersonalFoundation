# frozen_string_literal: true

module PersonalFoundation
  class Configuration
    attr_accessor :binary_path, :model_dir, :temperature, :max_tokens

    DEFAULT_PROJECT_PATH = File.expand_path("../../..", __dir__)

    def initialize
      @binary_path = find_default_binary
      @model_dir = ENV["PERSONAL_FOUNDATION_DIR"] || DEFAULT_PROJECT_PATH
      @temperature = 0.7
      @max_tokens = 120
    end

    private

    def find_default_binary
      candidates = [
        ENV["PERSONAL_FOUNDATION_BIN"],
        File.join(DEFAULT_PROJECT_PATH, ".build/debug/PersonalFoundationChat"),
        File.join(DEFAULT_PROJECT_PATH, ".build/release/PersonalFoundationChat"),
        File.join(DEFAULT_PROJECT_PATH, ".build/arm64-apple-macosx/debug/PersonalFoundationChat"),
        File.join(DEFAULT_PROJECT_PATH, ".build/arm64-apple-macosx/release/PersonalFoundationChat")
      ].compact

      candidates.find { |p| File.exist?(p) && File.executable?(p) } || "swift run PersonalFoundationChat"
    end
  end
end
