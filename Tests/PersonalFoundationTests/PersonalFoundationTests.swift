import Foundation
import Testing
import MLX
@testable import PersonalFoundation

@Suite("PersonalFoundation Core Architecture Tests")
struct PersonalFoundationTests {

    init() {
        if GPU.metallib == nil {
            let fm = FileManager.default
            let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
            let cwdMetallib = cwd.appendingPathComponent("default.metallib")

            if fm.fileExists(atPath: cwdMetallib.path) {
                GPU.metallib = cwdMetallib
            } else {
                let buildDir = cwd.appendingPathComponent(".build")
                if let enumerator = fm.enumerator(
                    at: buildDir,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles]
                ) {
                    while let fileURL = enumerator.nextObject() as? URL {
                        if fileURL.lastPathComponent == "default.metallib" {
                            GPU.metallib = fileURL
                            break
                        }
                    }
                }
            }
        }
    }

    @Test("ModelConfig Presets and Invariants")
    func testModelConfigInvariants() {
        let config = ModelConfig.dense110M()
        #expect(config.vocabSize == 32_000)
        #expect(config.hiddenSize == 768)
        #expect(config.numLayers == 12)
        #expect(config.numHeads == 12)
        #expect(config.numKVHeads == 4)
        #expect(config.headDim == 64)
        #expect(config.maxSequenceLength == 512)
        #expect(config.headDim % 2 == 0)
        #expect(config.numHeads % config.numKVHeads == 0)
    }

    @Test("LRScheduler Warmup and Cosine Decay")
    func testLRScheduler() {
        let scheduler = LRScheduler(
            warmupSteps: 10,
            totalSteps: 100,
            maxLR: 1e-3,
            minLR: 1e-5
        )

        // Step 0: Warmup start
        #expect(scheduler.getLR(step: 0) == 0.0)

        // Step 5: Midway through warmup
        let midWarmupLR = scheduler.getLR(step: 5)
        #expect(midWarmupLR > 0.0 && midWarmupLR < 1e-3)

        // Step 10: Peak LR at warmup end
        #expect(abs(scheduler.getLR(step: 10) - 1e-3) < 1e-6)

        // Step 100: Reached minLR
        #expect(abs(scheduler.getLR(step: 100) - 1e-5) < 1e-6)
    }

    @Test("Special Tokens ID Mapping")
    func testSpecialTokens() {
        #expect(SpecialToken.imStart.rawValue == 0)
        #expect(SpecialToken.imEnd.rawValue == 1)
        #expect(SpecialToken.user.rawValue == 2)
        #expect(SpecialToken.assistant.rawValue == 3)
    }

    @Test("CustomRMSNorm Shape and Normalization")
    func testCustomRMSNorm() {
        let norm = CustomRMSNorm(dimensions: 64)
        let input = MLXArray.ones([2, 10, 64], dtype: .float32)
        let output = norm(input)
        eval(output)
        #expect(output.shape == [2, 10, 64])
    }

    @Test("ModelKVCache Initialization and Reset")
    func testModelKVCache() {
        let cache = ModelKVCache(numLayers: 4)
        #expect(cache.currentSequenceLength == 0)
        #expect(cache.layerCaches.count == 4)

        cache.reset()
        #expect(cache.currentSequenceLength == 0)
    }
}
