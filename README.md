# AudioProcessor - 音频处理与机器学习项目

## 📋 项目概述

这是一个全面的音频处理和机器学习项目，涵盖了从基础的数字信号处理到先进的机器学习音频增强算法的完整实现。项目包含多个独立模块，涉及传统滤波器设计、深度学习降噪、实时音频处理、语音转换等技术。

## 🏗️ 项目结构

### 核心模块分类

```
AudioProcessor/
├── 📁 1. 数字信号处理 (DASP)
│   ├── DASP/                 # 音频数字信号处理仿真代码
│   ├── FIRFilter/           # FIR滤波器实现与仿真
│   ├── FIRSimulation/       # Python FIR滤波器仿真
│   └── IIRFilter/           # IIR滤波器设计与应用
│
├── 📁 2. 音频处理基础
│   ├── Audio/              # 测试音频文件和采样率转换
│   ├── Voice Process/      # 语音信号分析（基音检测等）
│   ├── NoisyPrint/         # 谱减法降噪算法实现
│   └── hubert_onnx/        # Hubert语音特征提取模型
│
├── 📁 3. 机器学习音频增强
│   ├── DeepFilterDemo/     # DeepFilter机器学习降噪Demo
│   ├── gtcrn_onnx_runtime/ # GTCRN低延迟降噪算法ONNX实现
│   ├── GTCRN-Learning/     # GTCRN算法系统性学习教程
│   └── Webrtc_NoisyReduce/ # WebRTC降噪算法实现
│
├── 📁 4. 实时音频处理平台
│   ├── RealTime-Mic-Algorithm-Testing-Platform/  # Qt实时算法测试平台
│   └── WASAPI/             # Windows WASAPI音频驱动开发
│
├── 📁 5. 语音转换与识别
│   ├── RVC/               # 基于VITS的RVC语音转换
│   └── PaddleSpeech/      # PaddleSpeech音频机器学习框架
│
├── 📁 6. 评估与基准测试
│   └── Noise_Reduction_Benchmark/  # 降噪算法基准测试方案
│
└── 📁 7. 文档与学习资源
    └── Document/          # 音频处理理论知识、论文、参考书籍
```

## 🔧 模块详细说明

### 1. 数字信号处理 (DASP)

**DASP/**
- FIR_LowPassFilter.py: FIR低通滤波器实现
- IIR_LowpassFilter.py: IIR低通滤波器实现
- IIR_Level2ButterworthFilter.py: 二阶巴特沃斯滤波器
- resample_rebuild.py: 重采样与信号重建
- test.py: FIR滤波器系数生成测试

**FIRFilter/**
- Main.py: FIR滤波器主程序
- ToolBox.py: 信号处理工具函数

**FIRSimulation/**
- FIRFilter/API.py: FIR滤波器Python API接口
- FIRFilter/HighPass.py: FIR高通滤波器实现

**IIRFilter/**
- ButterWorth高通和低通图例.py: 滤波器可视化
- CalculateCore.cpp: IIR滤波器计算核心(C++)
- CalculateCore.h: IIR滤波器计算核心头文件
- different-level-butterworth.py: 不同阶数巴特沃斯滤波器设计
- magnitude-squared-function.py: 幅度平方函数计算
- images/: 滤波器响应图
- 怎么求解IIR butter-worth-filter.md: IIR巴特沃斯滤波器设计文档
- 设计并应用一个IIR-ButterWorth-Filter示例.md: IIR滤波器应用示例

### 2. 音频处理基础

**Audio/**
- AudioSample-16000hz/: 16kHz采样率测试音频
- AudioSample-48000hz/: 48kHz采样率测试音频
- mp3/: MP3格式测试音频
- voice/: 语音测试音频（包括m4a、wav、pcm格式）
- DownRate.py: 降采样率工具
- UpRate.py: 升采样率工具
- processPCM.py: PCM音频处理工具

**Voice Process/**
- pitch_analyse.py: 基音检测与复倒谱分析
- 基音检测、复倒谱检测.png: 分析结果可视化

**NoisyPrint/**
- Process.py: 带重叠窗的谱减法降噪
- Process_NoneSplit.py: 无分帧的谱减法降噪
- test.py: 测试脚本
- ToolBox.py: 音频处理工具函数
- AudioSource/: 测试音频源文件
- README.md: 项目说明文档

**hubert_onnx/**
- hubert_eval.py: Hubert模型评估脚本
- hubert_export.py: Hubert模型导出到ONNX格式
- test.py: 测试脚本

### 3. 机器学习降噪算法

**DeepFilterDemo/**
- Demo/: C++实现的DeepFilter降噪算法Demo
  - main.cpp: 主程序
  - model/: 预训练模型
  - lib/, include/: 依赖库和头文件
  - CMakeLists.txt: 构建配置
- RealTimeDemo/: 实时DeepFilter降噪演示
- local/: 本地依赖库

**gtcrn_onnx_runtime/**
- api/api.h: API接口定义
- demo/main.cpp: 演示程序
- src/src.cpp: 源代码实现
- STFT/: 短时傅里叶变换实现
- wav_reader/: WAV文件读取器
- model/: ONNX模型文件
- onnx/: ONNX运行时依赖
- main.cpp: 主程序入口
- CMakeLists.txt: 构建配置

**GTCRN-Learning/**
- Chapter1-8/: 系统性的GTCRN学习教程章节
- GTCRN学习提纲.md: 完整学习提纲
- README.md: 项目说明文档

**Webrtc_NoisyReduce/**
- AudioProcessing/: WebRTC音频处理核心模块
- ENC/: 音频编码相关模块
- calculate_FFT_table.m: FFT表计算脚本
- UpRate.py: 采样率提升工具

### 4. 实时音频处理平台

**RealTime-Mic-Algorithm-Testing-Platform/**
- Qt5/: Qt5版本的实时算法测试平台
- Qt6/: Qt6版本的实时算法测试平台
- 支持自定义音频处理算法的快速集成和测试
- 提供音频I/O、可视化、参数调整等完整功能

**WASAPI/**
- AudioCapture/: 音频捕获示例
- AudioRecorder_Demo/: 音频录制演示
- EnumerateDevices/: 音频设备枚举示例
- ReadMe.md: 项目说明文档

### 5. 语音转换与识别

**RVC/**
- assets/: 预训练模型文件
- result/: 转换结果音频
- firstProject.py: RVC语音转换主程序

**PaddleSpeech/**
- Document/: PaddleSpeech相关文档

**hubert_onnx/**
- hubert_eval.py: Hubert模型评估脚本
- hubert_export.py: Hubert模型导出到ONNX格式
- test.py: 测试脚本

### 6. 评估与基准测试

**Noise_Reduction_Benchmark/**
- Objective-BenchMark/BenchMark/: 客观基准测试工具
  - 包含多种评估指标的Python脚本
  - ONNX模型评估
  - 测试数据和配置文件
- ReadMe.md: 基准测试方案说明

### 7. 文档与学习资源

**Document/**
- Book/: 参考书籍（现代语音处理技术及应用）
- FilterDesignInfo/: 滤波器设计参考资料
- Paper/: 学术论文（GTCRN、PerceptNet、RVC等）
- RVC/: RVC相关参数文档
- Voice Signals Process/: 语音信号处理文档
- 前置知识/: 信号与系统基础知识
- 工程开发/: 软件开发指南
- 旧日谈/: 技术历史和经验分享
- 降噪算法参数/: 算法参数配置文档
- 预畸变计算.md: 预畸变计算文档
- 各类技术笔记和开发文档

## 🚀 快速开始

### 环境要求

#### Python环境
```bash
# 推荐使用Python 3.8+
pip install numpy scipy matplotlib
pip install torch onnx onnxruntime
pip install librosa soundfile
```

#### C++环境
- CMake 3.10+
- Qt5/Qt6 (可选，用于GUI应用)
- Visual Studio 2019+ 或 GCC 7+

### 基本使用示例

#### 1. 运行谱减法降噪
```bash
cd NoisyPrint
python Process.py
```

#### 2. 测试FIR滤波器
```bash
cd FIRFilter
python Main.py
```

#### 3. 运行RVC语音转换
```bash
cd RVC
python firstProject.py
```

#### 4. 构建C++项目
```bash
# DeepFilterDemo
cd DeepFilterDemo/Demo
mkdir build && cd build
cmake ..
cmake --build .
```

## 📊 算法性能对比

| 算法类别 | 代表算法 | 延迟 | 计算开销 | 降噪效果 | 适用场景 |
|---------|---------|------|---------|---------|---------|
| 传统滤波 | FIR/IIR | 极低 | 极低 | 基础 | 简单噪声抑制 |
| 谱减法 | NoisyPrint | 低 | 低 | 中等 | 平稳噪声 |
| 机器学习 | DeepFilter | 中等 | 中等 | 良好 | 复杂环境音 |
| 轻量化DL | GTCRN | 低 | 中等 | 优秀 | 实时通信 |
| 工业级 | WebRTC | 低 | 低 | 良好 | 实时通信 |

## 🔬 技术亮点

### 1. 完整的算法演进路径
- 从传统滤波器到深度学习方法的完整实现
- 每个算法都有理论背景和实际代码实现

### 2. 实时处理能力
- 支持毫秒级延迟的实时音频处理
- 提供硬件级别的音频I/O支持

### 3. 工业级实现
- WebRTC等工业标准算法的完整实现
- 注重性能和稳定性的工程优化

### 4. 丰富的学习资源
- 系统性的GTCRN学习教程
- 详细的信号处理理论文档
- 实际工程开发经验分享

## 📈 项目进展

### 已实现功能
- ✅ 传统滤波器设计(FIR/IIR)
- ✅ 谱减法降噪算法
- ✅ DeepFilter机器学习降噪
- ✅ GTCRN轻量化降噪
- ✅ WebRTC降噪算法
- ✅ RVC语音转换
- ✅ 实时音频测试平台

### 计划中功能
- 🔄 更多深度学习模型集成
- 🔄 云端推理支持
- 🔄 移动端部署优化
- 🔄 自动化评估框架

## 🤝 贡献指南

欢迎为项目贡献代码、文档或改进建议！

### 贡献方式
1. 提交Issue报告问题或建议功能
2. Fork项目并提交Pull Request
3. 改进现有算法的实现
4. 添加新的音频处理算法
5. 完善文档和教程内容

### 开发规范
- Python代码遵循PEP8规范
- C++代码遵循Google C++ Style Guide
- 提交代码前请确保通过基础测试
- 新功能请添加相应的文档说明

## 📚 学习资源

### 推荐学习路径
1. **入门阶段**: 学习`Document/`中的信号处理基础
2. **实践阶段**: 尝试`DASP/`和`FIRFilter/`中的滤波器实现
3. **进阶阶段**: 学习`GTCRN-Learning/`中的深度学习降噪
4. **工程实践**: 使用`RealTime-Mic-Algorithm-Testing-Platform/`测试算法

### 参考书籍
- `Document/Book/现代语音处理技术及应用.pdf`
- 滤波器设计相关的学术论文和文档

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交GitHub Issue
- 查看项目文档获取更多信息

---

**最后更新**: 2026年1月26日  
**项目状态**: 活跃开发中