# Contributing to PersonalFoundation

Thank you for your interest in contributing to **PersonalFoundation**! We welcome contributions from developers, researchers, and AI enthusiasts.

---

## 📜 Code of Conduct
By participating in this project, you agree to abide by the terms of our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 🛠️ Getting Started

1. **Fork the Repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/<your-username>/PersonalFoundation.git
   cd PersonalFoundation
   ```
3. **Verify the build**:
   ```bash
   swift build
   ```

---

## 📐 Development Guidelines

- **Swift 6 Strict Concurrency**: All code must compile cleanly under Swift 6 strict concurrency settings (`.enableExperimentalFeature("StrictConcurrency")`). All public types crossing actor boundaries must conform to `Sendable`.
- **Zero External Model Loaders**: PersonalFoundation is built strictly from scratch using raw mathematical layers in `mlx-swift`. Do not introduce dependencies on external pre-trained model repositories (e.g., HuggingFace transformers, weights downloaders).
- **Mixed Precision**: Computations must adhere to True Mixed Precision (`.float32` master weights + `.bfloat16` forward/backward compute) for GPU efficiency on Apple Silicon.
- **Code Style**:
  - Use clear, idiomatic Swift naming conventions.
  - Document public APIs with docstrings explaining mathematical motivations and tensor shapes (e.g., `[B, T, D]`).

---

## 🔄 Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with descriptive messages:
   ```bash
   git commit -m "feat: add rotary embedding optimization"
   ```
3. Ensure the project builds without warnings or errors:
   ```bash
   swift build
   ```
4. Push to your branch and open a **Pull Request** targeting the `main` branch.
5. Provide a summary of changes, motivation, and any test results in the PR description.
