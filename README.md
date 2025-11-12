# FIRFilter_Venture

# 前言

滤波器相关知识学习与实践

# 更新说明
| 更新内容    | 更新时间   | 作者    | 
| :-------------: | :-----------: | ------------: |
| DeepFilter去除环境音 CLI RealTime| 2024.7.18|Venture| 
|   DeepFilter去除环境音Demo   | 2024.7.15      | Venture |
|   WASAPI实践Demo   | 2024.7.15      | Venture     | 
|   基于VITS的RVC语音转换工具实践   | 2024.7.12      | Venture  |
|   采样与重构的代码实践   | 2024.7.10      | Venture     |
| 几种常见的FIR滤波器实践     | 2024.7.10      | Venture     |
| 几种常见的IIR滤波器实践     | 2024.7.10       |  Venture      |
| 谱减法示例|2024.10.30 | Venture|
| FastASR - 快速STT | 2024.12.5 | Venture|
| webrtc的降噪方案 | 2025.1.23 | Venture |
| Noise Suppressor 算法及移植| 2025.1.23 | Venture |
| MOS - 降噪算法的基准测试 | 2025.4.28 | Venture |
| 复倒谱检测、基音检测的实现方式 | 2025.11.12 | Venture |
| GTCRN - 一种低延迟、低开销的机器学习降噪算法| 2025.9.22 | Venture |
| 测试实时音频算法的算法平台 | 2025.11.12 | Venture|

# 文件夹说明
| 文件夹名称   | 内容说明   | 作者    |
| :-------------: | :-----------: | ------------: |
|   Audio   | 测试用音频，包括录音和音乐      | Venture     |
|   DASP   | 音频数字信号处理相关的仿真代码      | Venture     |
|   DeepFilter   | 机器学习降噪模块Demo      | Venture     |
|   Document   | 音频数字信号处理相关的笔记，内容比较多，包括中英文参考书籍，提炼的比较精炼      | Venture     |
| PaddleSpeech     | 音频机器学习框架PaddleSpeech相关内容      | Venture     |
| RVC     | 基于VITS的RVC模型测试       |  Venture      |
| WASAPI     | WASAPI实操项目，主要用于后续RealTime驱动开发调研       |  Venture      |
| NoisyPrint|谱减法示例 | Venture|
| Webrtc_NoisyReduce| WebRtc降噪模块 | Venture |
| Noise_Reduction_Benchmark | 降噪算法基准测试 | Venture|
|gtcrn_onnx_runtime| 低延迟降噪算法| Venture |
| RealTime-Mic-Algorithm-Testing-Platform| 测试实时音频算法的算法平台，依赖Qt | Venture

## 介绍
FIR滤波器仿真与计算核心，使用纯C++开发，Qt界面做显示

FIRSimulation中的内容为Python对FIR滤波器的仿真，使用Python的scipy库，其中包括各种类型的滤波器，包括Peaking、Low-Pass、High-Pass、Band-Pass、High Shelf,Low ShelfNotch等


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
8. FIR LowPass Filter 

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142928.png"/>
 10. FIR HighPass Filter

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701142948.png"/>
