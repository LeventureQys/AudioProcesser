# AudioProcessor - Audio Processing and Machine Learning Project

## Project Overview

This project contains code and practices from my learning journey in audio processing and machine learning. It covers implementations from traditional digital signal processing to modern deep learning audio enhancement algorithms, including filter design, noise cancellation, voice conversion, and more.

## Project Structure

```
AudioProcesser/
├── DSP_Filter_Design/          # DSP filter design implementations
│   ├── FIRFIlter/              # FIR filter implementation and testing
│   ├── FIRSimulation/          # Python-based filter simulation
│   └── IIRFilter/              # IIR filter design and calculation core
├── FrameworkLearning/          # Framework learning
│   ├── GTCRN-Learning/         # GTCRN lightweight noise reduction tutorial
│   └── RVC-Learning/           # RVC voice conversion learning
├── Document/                   # Learning materials and references
│   ├── DASP/                   # Audio signal processing fundamentals
│   ├── FilterDesignInfo/       # Filter design references
│   ├── Book/                   # Reference books
│   ├── Paper/                  # Research papers
│   ├── RVC/                    # RVC-related materials
│   ├── Voice Signals Process/  # Voice signal processing
│   └── ...                     # Other theoretical documents
├── Test_Audio/                 # Test audio files
│   ├── AudioSample-16000hz/    # 16kHz sample rate audio
│   ├── AudioSample-48000hz/    # 48kHz sample rate audio
│   └── ...                     # Other audio resources and utility scripts
├── Archived_Workshop/          # Archived projects (early practice code)
│   ├── DeepFilterDemo/         # C++ implementation of DeepFilter noise reduction
│   ├── gtcrn_onnx_runtime/     # ONNX inference version of GTCRN noise reduction
│   ├── Noise_Reduction_Benchmark/ # Noise reduction algorithm evaluation framework
│   ├── NoisyPrint/             # Spectral subtraction-based noise cancellation
│   └── RealTime-Mic-Algorithm-Testing-Platform/ # Qt real-time algorithm testing platform
├── LICENSE
├── README.md
└── README.en.md
```

### Module Descriptions

**1. DSP Filter Design (`DSP_Filter_Design/`)**
- `FIRFIlter/` - FIR filter implementation and testing
- `FIRSimulation/` - Python-based filter simulation
- `IIRFilter/` - IIR filter design and calculation core

**2. Framework Learning (`FrameworkLearning/`)**
- `GTCRN-Learning/` - GTCRN lightweight noise reduction network architecture, optimization, and practice
- `RVC-Learning/` - RVC voice conversion framework learning

**3. Learning Materials (`Document/`)**
- Signal processing fundamentals (DASP, filter design principles, etc.)
- Research papers and reference books
- Specialized topics on RVC, voice signal processing, etc.

**4. Test Audio (`Test_Audio/`)**
- Audio samples at various sample rates (16kHz, 48kHz)
- Sample rate conversion utility scripts

**5. Archived Projects (`Archived_Workshop/`)**

Early practice code, archived for reference:
- `DeepFilterDemo/` - C++ implementation of DeepFilter noise reduction
- `gtcrn_onnx_runtime/` - ONNX inference implementation of GTCRN noise reduction
- `Noise_Reduction_Benchmark/` - Objective evaluation framework for noise reduction algorithms
- `NoisyPrint/` - Spectral subtraction-based noise cancellation
- `RealTime-Mic-Algorithm-Testing-Platform/` - Qt-based real-time audio algorithm testing platform

## Algorithm Comparison

| Algorithm Type | Example Method | Latency | Computation | Effect | Use Case |
|---------------|---------------|---------|------------|--------|----------|
| Traditional Filter | FIR/IIR | Very Low | Very Low | Basic | Simple Noise |
| Spectral Subtraction | NoisyPrint | Low | Low | Medium | Stationary Noise |
| Machine Learning | DeepFilter | Medium | Medium | Good | Complex Environment |
| Lightweight DL | GTCRN | Low | Medium | Excellent | Real-time Communication |

## Project Features

1. **Complete Algorithm Evolution** - Implementations from traditional filters to deep learning noise reduction
2. **Rich Learning Resources** - Systematic algorithm tutorials and theoretical materials
3. **Engineering Practice Reference** - Archived projects contain runnable engineering code

## Learning Suggestions

1. **Beginner** - Start with signal processing fundamentals in `Document/`
2. **Practice** - Try filter design in `DSP_Filter_Design/`
3. **Advanced** - Learn deep learning noise reduction in `FrameworkLearning/GTCRN-Learning/`
4. **Reference** - Review engineering implementations in `Archived_Workshop/`

## Changelog

**February 2026** - Project structure reorganization, archived early projects, streamlined directories

**January 2026** - Project documentation cleanup and restructuring

**November 2025** - Added advanced audio analysis features (pitch detection, etc.)

**September 2025** - Integrated GTCRN lightweight noise reduction algorithm

**April 2025** - Established noise reduction algorithm evaluation system

**January 2025** - Integrated WebRTC industrial-grade noise reduction solution

**December 2024** - Added FastASR speech recognition

**October 2024** - Enhanced traditional algorithms (spectral subtraction)

**July 2024** - Core feature development (DeepFilter, WASAPI, RVC, etc.)

**June 2024** - Project initialization

## License

MIT License - see [LICENSE](LICENSE) file for details

---

*Last Updated: February 24, 2026*
*Project Status: Active Development*