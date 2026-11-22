# frozen_string_literal: true

require_relative "lib/personal_foundation/version"

Gem::Specification.new do |spec|
  spec.name          = "personal_foundation"
  spec.version       = PersonalFoundation::VERSION
  spec.authors       = ["ChiefVenzox", "JennyDarling", "HNMLabs"]
  spec.email         = ["hakankaba1921@gmail.com", "contact@hnmlabs.com"]

  spec.summary       = "Ruby client and runner for the PersonalFoundation Apple Silicon Foundation Model"
  spec.description   = "PersonalFoundation is a from-scratch Decoder-only Transformer LLM designed for Apple Silicon via MLX. This gem provides native Ruby integration, CLI generation, and streaming support."
  spec.homepage      = "https://github.com/ChiefVenzox/PersonalFoundation"
  spec.license       = "MIT"
  spec.required_ruby_version = ">= 2.6.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "https://github.com/ChiefVenzox/PersonalFoundation/releases"
  spec.metadata["github_repo"] = "https://github.com/ChiefVenzox/PersonalFoundation"

  spec.files = Dir["lib/**/*.rb", "exe/*", "README.md", "LICENSE"]
  spec.bindir        = "exe"
  spec.executables   = ["personal_foundation"]
  spec.require_paths = ["lib"]
end
