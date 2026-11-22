# frozen_string_literal: true

module PersonalFoundation
  class Error < StandardError; end
  class BinaryNotFoundError < Error; end
  class ExecutionError < Error; end
  class ModelNotFoundError < Error; end
end
