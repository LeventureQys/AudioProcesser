# AudioProcessor - Audio Processing and Machine Learning Project

## Project Overview

This project contains code and practices from my learning journey in audio processing and machine learning. It covers implementations from traditional digital signal processing to modern deep learning audio enhancement algorithms, including filter design, noise cancellation, voice conversion, and more.

## Project Structure

### Main Modules

**1. Digital Signal Processing (DASP)**
- `DASP/` - Basic audio signal processing algorithms (FIR/IIR filters, resampling, etc.)
- `FIRFilter/` - FIR filter implementation and testing
- `IIRFilter/` - IIR filter design and calculation core
- `FIRSimulation/` - Python-based filter simulation

**2. Audio Processing Basics**
- `Audio/` - Various test audio files and sample rate conversion tools
- `Voice Process/` - Voice signal analysis, like pitch detection
- `NoisyPrint/` - Spectral subtraction-based noise cancellation
- `hubert_onnx/` - Hubert speech feature extraction model

**3. Machine Learning Noise Reduction**
- `DeepFilterDemo/` - C++ implementation of DeepFilter noise reduction
- `gtcrn_onnx_runtime/` - ONNX version of GTCRN lightweight noise reduction
- `GTCRN-Learning/` - GTCRN algorithm learning tutorial
- `Webrtc_NoisyReduce/` - WebRTC official noise reduction algorithm

**4. Real-time Audio Processing**
- `RealTime-Mic-Algorithm-Testing-Platform/` - Qt-based real-time algorithm testing platform
- `WASAPI/` - Windows audio driver development practices

**5. Voice Conversion and Recognition**
- `RVC/` - VITS-based voice conversion tool
- `PaddleSpeech/` - PaddleSpeech related code

**6. Evaluation Tools**
- `Noise_Reduction_Benchmark/` - Noise reduction algorithm performance evaluation framework

**7. Learning Materials**
- `Document/` - Signal processing theory, filter design, papers, and other references

## Quick Start

### Requirements
- Python 3.8+
- C++ environment (CMake 3.10+, optional Qt)

### Install Dependencies
```bash
pip install numpy scipy matplotlib torch onnx onnxruntime librosa soundfile
```

### Usage Examples

**Run spectral subtraction noise reduction:**
```bash
cd NoisyPrint
python Process.py
```

**Test FIR filter:**
```bash
cd FIRFilter
python Main.py
```

**Voice conversion example:**
```bash
cd RVC
python firstProject.py
```

**Build C++ project:**
```bash
cd DeepFilterDemo/Demo
mkdir build && cd build
cmake ..
cmake --build .
```

## Algorithm Comparison

| Algorithm Type | Example Method | Latency | Computation | Effect | Use Case |
|---------------|---------------|---------|------------|--------|----------|
| Traditional Filter | FIR/IIR | Very Low | Very Low | Basic | Simple Noise |
| Spectral Subtraction | NoisyPrint | Low | Low | Medium | Stationary Noise |
| Machine Learning | DeepFilter | Medium | Medium | Good | Complex Environment |
| Lightweight DL | GTCRN | Low | Medium | Excellent | Real-time Communication |
| Industrial Grade | WebRTC | Low | Low | Good | Real-time Communication |

## Project Features

1. **Complete Algorithm Evolution** - Implementations from traditional methods to deep learning
2. **Real-time Processing Support** - Low-latency real-time audio processing capabilities
3. **Industrial-grade Implementations** - Includes mature solutions like WebRTC
4. **Rich Learning Resources** - Systematic algorithm tutorials and theoretical materials

## Project Progress

**Implemented:**
- Traditional filter design (FIR/IIR)
- Spectral subtraction noise cancellation
- Machine learning noise reduction (DeepFilter, GTCRN, etc.)
- WebRTC noise reduction algorithm
- RVC voice conversion
- Real-time audio testing platform

**Planned:**
- More deep learning models
- Cloud inference support
- Mobile optimization
- Automated evaluation system

## Changelog

**January 2026** - Project documentation cleanup and restructuring

**November 2025** - Added advanced audio analysis features (pitch detection, etc.)

**September 2025** - Integrated GTCRN lightweight noise reduction algorithm

**April 2025** - Established noise reduction algorithm evaluation system

**January 2025** - Integrated WebRTC industrial-grade noise reduction solution

**December 2024** - Added FastASR speech recognition

**October 2024** - Enhanced traditional algorithms (spectral subtraction)

**July 2024** - Core feature development (DeepFilter, WASAPI, RVC, etc.)

**June 2024** - Project initialization

## Contribution Guidelines

Welcome suggestions and code improvements! If you have issues, please report via GitHub Issue.

Development notes:
- Python code follows PEP8 standards
- C++ code uses Google C++ style
- Please add documentation for new features

## Learning Suggestions

1. **Beginner** - Start with signal processing basics in `Document/`
2. **Practice** - Try filters in `DASP/` and `FIRFilter/`
3. **Advanced** - Learn deep learning noise reduction in `GTCRN-Learning/`
4. **Engineering** - Test algorithms with `RealTime-Mic-Algorithm-Testing-Platform/`

## License

MIT License - see [LICENSE](LICENSE) file for details

---

*Last Updated: January 26, 2026*  
*Project Status: Active Development*