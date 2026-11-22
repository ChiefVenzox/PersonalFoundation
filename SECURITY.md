# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

We actively support and provide security updates for the latest stable release of **PersonalFoundation**.

---

## Reporting a Vulnerability

The PersonalFoundation team takes the security of our foundation models, tokenizers, and training infrastructure seriously.

If you believe you have discovered a security vulnerability or critical flaw in PersonalFoundation (including unsafe memory handling, buffer vulnerabilities in BPE serialization, or dependency issues in MLX bindings):

1. **Do not create a public GitHub Issue.** Public disclosure puts the community at risk.
2. Please privately report the vulnerability to the project maintainers:
   - **Primary Contact:** [contact@hnmlabs.com](mailto:contact@hnmlabs.com)
   - **GitHub Security Advisory:** [Open a private Security Advisory on GitHub](https://github.com/ChiefVenzox/PersonalFoundation/security/advisories/new)

### What to Include in Your Report
- A detailed description of the vulnerability.
- Steps to reproduce the issue or a proof-of-concept (PoC) code.
- Impact assessment (e.g., potential memory corruption, denial of service, privilege escalation).
- Any proposed mitigations or fixes.

### Response Timeline
- **Initial Response:** Within 48 hours of receipt.
- **Triage & Status Update:** Within 5 business days.
- **Patch Release & Advisory:** A coordinated security release will be deployed, followed by public acknowledgment (unless anonymity is requested).

Thank you for helping keep PersonalFoundation and the open-source community secure!
