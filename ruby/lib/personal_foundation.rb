# frozen_string_literal: true

require_relative "personal_foundation/version"
require_relative "personal_foundation/errors"
require_relative "personal_foundation/configuration"
require_relative "personal_foundation/response"
require_relative "personal_foundation/client"

module PersonalFoundation
  class << self
    attr_writer :configuration

    def configuration
      @configuration ||= Configuration.new
    end

    def configure
      yield(configuration) if block_given?
    end

    def reset_configuration!
      @configuration = Configuration.new
    end

    def client
      @client ||= Client.new(configuration)
    end

    def generate(prompt, **options)
      client.generate(prompt, **options)
    end

    def chat(prompt, **options)
      client.chat(prompt, **options)
    end
  end
end
