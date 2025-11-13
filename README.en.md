# FIRFilter_Venture

# Preface

Learning and Practice of Filter-Related Knowledge

# Update Notes
| Update Content    | Update Date   | Author    | 
| :-------------: | :-----------: | ------------: |
| DeepFilter Remove Environmental Noise CLI RealTime| 2024.7.18|Venture| 
|   DeepFilter Remove Environmental Noise Demo   | 2024.7.15      | Venture |
|   WASAPI Practice Demo   | 2024.7.15      | Venture     | 
|   RVC Voice Conversion Tool Practice Based on VITS   | 2024.7.12      | Venture  |
|   Sampling and Reconstruction Code Practice   | 2024.7.10      | Venture     |
| Several Common FIR Filter Practices     | 2024.7.10      | Venture     |
| Several Common IIR Filter Practices     | 2024.7.10       |  Venture      |
| Spectral Subtraction Example|2024.10.30 | Venture|
| FastASR - Fast STT | 2024.12.5 | Venture|
| WebRTC Noise Reduction Solution | 2025.1.23 | Venture |
| Noise Suppressor Algorithm and Porting| 2025.1.23 | Venture |
| MOS - Benchmark Testing for Noise Reduction Algorithms | 2025.4.28 | Venture |
| Implementation Methods for Complex Cepstrum Detection and Pitch Detection | 2025.11.12 | Venture |
| GTCRN - A Low-Latency, Low-Overhead Machine Learning Noise Reduction Algorithm| 2025.9.22 | Venture |
| Testing Platform for Real-Time Audio Algorithms | 2025.11.12 | Venture|

# Folder Description
| Folder Name   | Content Description   | Author    |
| :-------------: | :-----------: | ------------: |
|   Audio   | Test Audio, Including Recordings and Music      | Venture     |
|   DASP   | Simulation Code Related to Audio Digital Signal Processing      | Venture     |
|   DeepFilter   | Machine Learning Noise Reduction Module Demo      | Venture     |
|   Document   | Notes on Audio Digital Signal Processing, with a Lot of Content, Including Chinese and English Reference Books, Refined Extracts      | Venture     |
| PaddleSpeech     | Content Related to the Audio Machine Learning Framework PaddleSpeech      | Venture     |
| RVC     | RVC Model Testing Based on VITS       |  Venture      |
| WASAPI     | WASAPI Practical Project, Mainly for Subsequent RealTime Driver Development Research       |  Venture      |
| NoisyPrint| Spectral Subtraction Example | Venture|
| Webrtc_NoisyReduce| WebRtc Noise Reduction Module | Venture |
| Noise_Reduction_Benchmark | Benchmark Testing for Noise Reduction Algorithms | Venture|
|gtcrn_onnx_runtime| Low-Latency Noise Reduction Algorithm| Venture |
| RealTime-Mic-Algorithm-Testing-Platform| Testing Platform for Real-Time Audio Algorithms, Dependent on Qt | Venture

## Introduction
The FIR filter simulation and computation core is developed purely in C++, with Qt interface for display.

The content in FIRSimulation is Python simulation of FIR filters using Python's scipy library, including various types of filters such as Peaking, Low-Pass, High-Pass, Band-Pass, High Shelf, Low Shelf, Notch, etc.

1. IIR Peaking Filter 
<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142435.png"/>

2. IIR LowShelf Filter 
<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142609.png"/>
3. IIR HighShelf Filter

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142638.png"/>

4. IIR LowPassFilter 

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142709.png"/>

5. IIR HighPass Filter

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142737.png"/>

6. FIR PeakingFilter
<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142803.png"/>

7. FIR LowShelf Filter

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142827.png"/>
8. FIR HighShelf Filter
<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142903.png"/>
9. FIR LowPass Filter 

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142928.png"/>
10. FIR HighPass Filter

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142948.png"/>