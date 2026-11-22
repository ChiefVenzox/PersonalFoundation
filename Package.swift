// swift-tools-version: 6.0
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
            name: "PersonalFoundationTrain",
            targets: ["PersonalFoundationTrain"]
        ),
        .executable(
            name: "PersonalFoundationChat",
            targets: ["PersonalFoundationChat"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0")
    ],
    targets: [
        .target(
            name: "PersonalFoundation",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "PersonalFoundationTrain",
            dependencies: ["PersonalFoundation"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "PersonalFoundationChat",
            dependencies: ["PersonalFoundation"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "PersonalFoundationTests",
            dependencies: ["PersonalFoundation"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        )
    ]
)
