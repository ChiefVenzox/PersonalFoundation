# frozen_string_literal: true

require "json"
require "open3"

module PersonalFoundation
  class Client
    attr_reader :config

    def initialize(configuration = nil)
      @config = configuration || PersonalFoundation.configuration || Configuration.new
    end

    def generate(prompt, temperature: nil, max_tokens: nil, model_dir: nil)
      temp = temperature || config.temperature
      max_tok = max_tokens || config.max_tokens
      dir = model_dir || config.model_dir

      cmd = build_command(prompt, temp: temp, max_tokens: max_tok, model_dir: dir, json: true)
      stdout, stderr, status = Open3.capture3(*cmd, chdir: dir)

      unless status.success?
        raise ExecutionError, "Failed to run PersonalFoundationChat (exit code #{status.exitstatus}): #{stderr}"
      end

      # Extract the JSON line from stdout
      json_line = stdout.lines.reverse.find { |line| line.strip.start_with?("{") && line.strip.end_with?("}") }
      unless json_line
        raise ExecutionError, "Invalid JSON output from model runner:\n#{stdout}\nSTDERR:\n#{stderr}"
      end

      data = JSON.parse(json_line)
      Response.new(data)
    end

    def chat(prompt, **options)
      generate(prompt, **options).text
    end

    private

    def build_command(prompt, temp:, max_tokens:, model_dir:, json: true)
      bin = config.binary_path
      args = []
      args += ["--prompt", prompt]
      args += ["--temperature", temp.to_s]
      args += ["--max-tokens", max_tokens.to_s]
      args += ["--model-dir", model_dir] if model_dir
      args += ["--json"] if json

      if bin.start_with?("swift run")
        # swift run command
        ["swift", "run", "PersonalFoundationChat", *args]
      else
        [bin, *args]
      end
    end
  end
end
