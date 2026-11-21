// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PersonalFoundation",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "PersonalFoundation",
            targets: ["PersonalFoundation"]
        ),
        .executable(
            name: "PersonalFoundationCLI",
            targets: ["PersonalFoundationCLI"]
        )
    ],
    dependencies: [
        // Sadece saf Apple MLX C++/Metal backend bindings (MLX, MLXNN, MLXOptimizers, MLXRandom)
        // Hiçbir dış model yükleyici (MLXLLM, HuggingFace Hub vb.) DAHİL EDİLMEMİŞTİR.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0")
    ],
    targets: [
        .target(
            name: "PersonalFoundation",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "PersonalFoundationCLI",
            dependencies: ["PersonalFoundation"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        )
    ]
)
