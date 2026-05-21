# AudioProcesser - Audio Processing & Learning Project

A complete audio processing learning journey from traditional DSP to deep learning, spanning **digital filter design → classical noise reduction → deep learning denoising → voice conversion** with full-stack engineering practice.

[中文](README.md) | English

---

## Panorama

```
AudioProcesser/
│
├── Document/                          # Theoretical foundation (32+ Chinese docs)
│   ├── 前置知识/                       # Signals & systems, filters, Attention/Transformer
│   ├── DASP/                          # Digital audio signal processing examples
│   ├── FilterDesignInfo/              # Filter design references (PDFs)
│   ├── Voice Signals Process/         # Speech analysis, noise suppressor design
│   ├── Paper/                         # Research papers
│   └── Book/                          # Reference books
│
├── DSP_Filter_Design/                 # Traditional DSP filter implementations
│   ├── IIRFilter/                     # Butterworth filters (C++/Python)
│   ├── FIRFIlter/                     # FIR filters + FFT visualization
│   ├── FIRSimulation/                 # scipy.signal simulation
│   └── CalculatorCore/                # IIR coefficient engine (Qt/C++)
│
├── FrameworkLearning/                 # Deep learning for audio curriculum
│   ├── GTCRN-Learning/                # Full GTCRN tutorial (with RNNoise reference)
│   ├── ML/                            # ML fundamentals (activations, UNet, MNIST)
│   └── RVC-Learning/                  # RVC voice conversion notes
│
├── Archived_Workshop/                 # Engineering practice
│   ├── NoisyPrint/                    # Spectral subtraction denoising
│   ├── DeepFilterDemo/                # DeepFilter C++ implementation
│   ├── gtcrn_onnx_runtime/            # GTCRN ONNX inference (C++17)
│   ├── MeanVC/                        # Streaming zero-shot voice conversion
│   ├── Noise_Reduction_Benchmark/     # Objective NR evaluation
│   ├── RealTime-Mic-Algorithm-Testing-Platform/  # Qt real-time audio platform
│   └── ...
│
├── Test_Audio/                        # Test audio + sample rate conversion tools
│
└── third_party/                       # Submodules
    ├── DeepFilterNet/                 # Deep filtering (Rust + Python)
    └── gtcrn/                         # Official GTCRN implementation
```

---

## Four Generations of Noise Reduction

| Gen | Technology | Location | Latency | Compute | Quality |
|-----|------------|---------|---------|---------|---------|
| 1️⃣ | FIR / IIR | `DSP_Filter_Design/` | Very Low | Very Low | Basic |
| 2️⃣ | Spectral Subtraction | `Archived_Workshop/NoisyPrint/` | Low | Low | Medium |
| 3️⃣ | DeepFilterNet (CRNN) | `third_party/DeepFilterNet/` | Medium | Medium | Good |
| 4️⃣ | GTCRN (Group Conv + RNN) | `third_party/gtcrn/` | Low | Very Low | Excellent |

## Voice Conversion

| Project | Stack | Description |
|---------|-------|-------------|
| MeanVC | DiT + CFM + WavLM + Vocos | Streaming zero-shot VC, 2-step generation |
| RVC-Learning | Study notes | NSF-HiFiGAN vocoder, etc. |

## Engineering Practice

| Project | Stack | Description |
|---------|-------|-------------|
| gtcrn_onnx_runtime | C++17 + ONNX Runtime | GTCRN native inference with custom STFT/ISTFT |
| DeepFilterDemo | C++ | Real-time DeepFilter denoising |
| RealTime-Mic | Qt/C++ + WASAPI | Mic capture + algorithm plugin |
| NoisyPrint | Python | Full spectral subtraction pipeline + visualization |
| Noise_Reduction_Benchmark | Python | DNSMOS / NISQA / PESQ / STOI evaluation |

## Third-Party Submodules

- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** — Rust/Python deep complex-spectral filtering
- **[gtcrn](https://github.com/Xiaobin-Rong/gtcrn)** — ICASSP 2024 ultra-lightweight real-time denoising
- **BenchMark** — MOS evaluation toolset

---

## Learning Roadmap

```
Start ──→ Document/前置知识/          Signals & systems, filter theory, FFT
          │
Practice → DSP_Filter_Design/        FIR / IIR filter design & simulation
          │
Advanced → Document/前置知识/         Attention / Transformer / Conformer
          │
          └──→ FrameworkLearning/     GTCRN architecture, RNNoise reference
          │
Deep dive → Archived_Workshop/       MeanVC voice conversion, ONNX inference
          │
Deploy  ─→ third_party/              DeepFilterNet / GTCRN official code
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05 | MeanVC docs, README restructure |
| 2026-02 | Project structure reorganization |
| 2025-11 | Advanced audio analysis (pitch, cepstrum) |
| 2025-09 | GTCRN lightweight denoising integration |
| 2025-04 | NR evaluation system established |
| 2025-01 | WebRTC industrial denoising |
| 2024-06 | Project initialization |

---

## License

[GNU General Public License v3](LICENSE)
