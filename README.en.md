# AudioProcessor - Audio Processing and Machine Learning Project

## 📋 Project Overview

This is a comprehensive audio processing and machine learning project covering the complete implementation from basic digital signal processing to advanced machine learning audio enhancement algorithms. The project includes multiple independent modules involving traditional filter design, deep learning noise reduction, real-time audio processing, voice conversion, and other technologies.

## 🏗️ Project Structure

### Core Module Classification

```
AudioProcessor/
├── 📁 1. Digital Signal Processing (DASP)
│   ├── DASP/                 # Audio digital signal processing simulation code
│   ├── FIRFilter/           # FIR filter implementation and simulation
│   ├── FIRSimulation/       # Python FIR filter simulation
│   └── IIRFilter/           # IIR filter design and application
│
├── 📁 2. Audio Processing Fundamentals
│   ├── Audio/              # Test audio files and sample rate conversion
│   ├── Voice Process/      # Voice signal analysis (pitch detection, etc.)
│   ├── NoisyPrint/         # Spectral subtraction noise reduction algorithm
│   └── hubert_onnx/        # Hubert speech feature extraction model
│
├── 📁 3. Machine Learning Audio Enhancement
│   ├── DeepFilterDemo/     # DeepFilter machine learning noise reduction Demo
│   ├── gtcrn_onnx_runtime/ # GTCRN low-latency noise reduction algorithm ONNX implementation
│   ├── GTCRN-Learning/     # GTCRN algorithm systematic learning tutorial
│   └── Webrtc_NoisyReduce/ # WebRTC noise reduction algorithm implementation
│
├── 📁 4. Real-time Audio Processing Platform
│   ├── RealTime-Mic-Algorithm-Testing-Platform/  # Qt real-time algorithm testing platform
│   └── WASAPI/             # Windows WASAPI audio driver development
│
├── 📁 5. Voice Conversion and Recognition
│   ├── RVC/               # VITS-based RVC voice conversion
│   └── PaddleSpeech/      # PaddleSpeech audio machine learning framework
│
├── 📁 6. Evaluation and Benchmarking
│   └── Noise_Reduction_Benchmark/  # Noise reduction algorithm benchmarking solution
│
└── 📁 7. Documentation and Learning Resources
    └── Document/          # Audio processing theoretical knowledge, papers, reference books
```

## 🔧 Module Detailed Description

### 1. Digital Signal Processing (DASP)

**DASP/**
- FIR_LowPassFilter.py: FIR low-pass filter implementation
- IIR_LowpassFilter.py: IIR low-pass filter implementation
- IIR_Level2ButterworthFilter.py: Second-order Butterworth filter
- resample_rebuild.py: Resampling and signal reconstruction
- test.py: FIR filter coefficient generation test

**FIRFilter/**
- Main.py: FIR filter main program
- ToolBox.py: Signal processing utility functions

**FIRSimulation/**
- FIRFilter/API.py: FIR filter Python API interface
- FIRFilter/HighPass.py: FIR high-pass filter implementation

**IIRFilter/**
- ButterWorth high-pass and low-pass examples.py: Filter visualization
- CalculateCore.cpp: IIR filter calculation core (C++)
- CalculateCore.h: IIR filter calculation core header file
- different-level-butterworth.py: Different order Butterworth filter design
- magnitude-squared-function.py: Magnitude squared function calculation
- images/: Filter response images
- 怎么求解IIR butter-worth-filter.md: IIR Butterworth filter design documentation
- 设计并应用一个IIR-ButterWorth-Filter示例.md: IIR filter application example

### 2. Audio Processing Fundamentals

**Audio/**
- AudioSample-16000hz/: 16kHz sample rate test audio
- AudioSample-48000hz/: 48kHz sample rate test audio
- mp3/: MP3 format test audio
- voice/: Voice test audio (including m4a, wav, pcm formats)
- DownRate.py: Downsampling tool
- UpRate.py: Upsampling tool
- processPCM.py: PCM audio processing tool

**Voice Process/**
- pitch_analyse.py: Pitch detection and cepstrum analysis
- 基音检测、复倒谱检测.png: Analysis result visualization

**NoisyPrint/**
- Process.py: Spectral subtraction noise reduction with overlapping windows
- Process_NoneSplit.py: Spectral subtraction noise reduction without frame splitting
- test.py: Test script
- ToolBox.py: Audio processing utility functions
- AudioSource/: Test audio source files
- README.md: Project documentation

**hubert_onnx/**
- hubert_eval.py: Hubert model evaluation script
- hubert_export.py: Hubert model export to ONNX format
- test.py: Test script

### 3. Machine Learning Audio Enhancement

**DeepFilterDemo/**
- Demo/: C++ implementation of DeepFilter noise reduction Demo
  - main.cpp: Main program
  - model/: Pre-trained models
  - lib/, include/: Dependency libraries and header files
  - CMakeLists.txt: Build configuration
- RealTimeDemo/: Real-time DeepFilter noise reduction demonstration
- local/: Local dependency libraries

**gtcrn_onnx_runtime/**
- api/api.h: API interface definition
- demo/main.cpp: Demonstration program
- src/src.cpp: Source code implementation
- STFT/: Short-time Fourier transform implementation
- wav_reader/: WAV file reader
- model/: ONNX model files
- onnx/: ONNX runtime dependencies
- main.cpp: Main program entry
- CMakeLists.txt: Build configuration

**GTCRN-Learning/**
- Chapter1-8/: Systematic GTCRN learning tutorial chapters
- GTCRN学习提纲.md: Complete learning outline
- README.md: Project documentation

**Webrtc_NoisyReduce/**
- AudioProcessing/: WebRTC audio processing core module
- ENC/: Audio encoding related modules
- calculate_FFT_table.m: FFT table calculation script
- UpRate.py: Sample rate increase tool

### 4. Real-time Audio Processing Platform

**RealTime-Mic-Algorithm-Testing-Platform/**
- Qt5/: Qt5 version real-time algorithm testing platform
- Qt6/: Qt6 version real-time algorithm testing platform
- Supports rapid integration and testing of custom audio processing algorithms
- Provides complete functions including audio I/O, visualization, parameter adjustment

**WASAPI/**
- AudioCapture/: Audio capture example
- AudioRecorder_Demo/: Audio recording demonstration
- EnumerateDevices/: Audio device enumeration example
- ReadMe.md: Project documentation

### 5. Voice Conversion and Recognition

**RVC/**
- assets/: Pre-trained model files
- result/: Conversion result audio
- firstProject.py: RVC voice conversion main program

**PaddleSpeech/**
- Document/: PaddleSpeech related documentation

### 6. Evaluation and Benchmarking

**Noise_Reduction_Benchmark/**
- Objective-BenchMark/BenchMark/: Objective benchmarking tools
  - Includes Python scripts for various evaluation metrics
  - ONNX model evaluation
  - Test data and configuration files
- ReadMe.md: Benchmarking solution documentation

### 7. Documentation and Learning Resources

**Document/**
- Book/: Reference books (Modern Speech Processing Technology and Applications)
- FilterDesignInfo/: Filter design reference materials
- Paper/: Academic papers (GTCRN, PerceptNet, RVC, etc.)
- RVC/: RVC related parameter documentation
- Voice Signals Process/: Voice signal processing documentation
- 前置知识/: Signal and system basic knowledge
- 工程开发/: Software development guidelines
- 旧日谈/: Technical history and experience sharing
- 降噪算法参数/: Algorithm parameter configuration documentation
- 预畸变计算.md: Predistortion calculation documentation
- Various technical notes and development documents

## 🚀 Quick Start

### Environment Requirements

#### Python Environment
```bash
# Recommended Python 3.8+
pip install numpy scipy matplotlib
pip install torch onnx onnxruntime
pip install librosa soundfile
```

#### C++ Environment
- CMake 3.10+
- Qt5/Qt6 (optional, for GUI applications)
- Visual Studio 2019+ or GCC 7+

### Basic Usage Examples

#### 1. Run Spectral Subtraction Noise Reduction
```bash
cd NoisyPrint
python Process.py
```

#### 2. Test FIR Filter
```bash
cd FIRFilter
python Main.py
```

#### 3. Run RVC Voice Conversion
```bash
cd RVC
python firstProject.py
```

#### 4. Build C++ Project
```bash
# DeepFilterDemo
cd DeepFilterDemo/Demo
mkdir build && cd build
cmake ..
cmake --build .
```

## 📊 Algorithm Performance Comparison

| Algorithm Type | Representative Algorithm | Latency | Computation Cost | Noise Reduction Effect | Application Scenario |
|---------------|------------------------|---------|----------------|------------------------|---------------------|
| Traditional Filter | FIR/IIR | Very Low | Very Low | Basic | Simple noise suppression |
| Spectral Subtraction | NoisyPrint | Low | Low | Medium | Stationary noise |
| Machine Learning | DeepFilter | Medium | Medium | Good | Complex environmental noise |
| Lightweight DL | GTCRN | Low | Medium | Excellent | Real-time communication |
| Industrial Grade | WebRTC | Low | Low | Good | Real-time communication |

## 🔬 Technical Highlights

### 1. Complete Algorithm Evolution Path
- Complete implementation from traditional filters to deep learning methods
- Each algorithm includes theoretical background and actual code implementation

### 2. Real-time Processing Capability
- Supports millisecond-level latency real-time audio processing
- Provides hardware-level audio I/O support

### 3. Industrial-grade Implementation
- Complete implementation of industrial standard algorithms like WebRTC
- Engineering optimization focusing on performance and stability

### 4. Rich Learning Resources
- Systematic GTCRN learning tutorial
- Detailed signal processing theory documentation
- Practical engineering development experience sharing

## 📈 Project Progress

### Implemented Features
- ✅ Traditional filter design (FIR/IIR)
- ✅ Spectral subtraction noise reduction algorithm
- ✅ DeepFilter machine learning noise reduction
- ✅ GTCRN lightweight noise reduction
- ✅ WebRTC noise reduction algorithm
- ✅ RVC voice conversion
- ✅ Real-time audio testing platform

### Planned Features
- 🔄 More deep learning model integration
- 🔄 Cloud inference support
- 🔄 Mobile deployment optimization
- 🔄 Automated evaluation framework

## 🤝 Contribution Guidelines

Welcome to contribute code, documentation, or improvement suggestions!

### Contribution Methods
1. Submit Issues to report problems or suggest features
2. Fork the project and submit Pull Requests
3. Improve existing algorithm implementations
4. Add new audio processing algorithms
5. Improve documentation and tutorial content

### Development Standards
- Python code follows PEP8 standards
- C++ code follows Google C++ Style Guide
- Ensure basic tests pass before submitting code
- Add corresponding documentation for new features

## 📚 Learning Resources

### Recommended Learning Path
1. **Beginner Stage**: Learn signal processing basics in `Document/`
2. **Practice Stage**: Try filter implementations in `DASP/` and `FIRFilter/`
3. **Advanced Stage**: Learn deep learning noise reduction in `GTCRN-Learning/`
4. **Engineering Practice**: Test algorithms using `RealTime-Mic-Algorithm-Testing-Platform/`

### Reference Books
- `Document/Book/Modern Speech Processing Technology and Applications.pdf`
- Academic papers and documentation related to filter design

## 📄 License

This project uses MIT License - see [LICENSE](LICENSE) file for details

## 📞 Contact

For questions or suggestions, please contact through:
- Submit GitHub Issue
- Check project documentation for more information

---

**Last Updated**: January 26, 2026  
**Project Status**: Active Development