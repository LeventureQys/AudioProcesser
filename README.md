# AudioProcessor - 音频处理与机器学习项目

## 项目简介

这个项目是我在学习音频处理和机器学习过程中积累的代码和实践。涵盖了从传统的数字信号处理到现代的深度学习音频增强算法，包含滤波器设计、噪声消除、语音转换等功能的实现。

## 项目结构

### 主要模块说明

**1. 数字信号处理 (DASP)**
- `DASP/` - 音频信号处理基础算法（FIR/IIR滤波器、重采样等）
- `FIRFilter/` - FIR滤波器实现和测试
- `IIRFilter/` - IIR滤波器设计和计算核心
- `FIRSimulation/` - Python版的滤波器仿真

**2. 音频处理基础**
- `Audio/` - 各种测试音频文件和采样率转换工具
- `Voice Process/` - 语音信号分析，如基音检测
- `NoisyPrint/` - 基于谱减法的噪声消除实现
- `hubert_onnx/` - Hubert语音特征提取模型

**3. 机器学习降噪**
- `DeepFilterDemo/` - DeepFilter降噪算法的C++实现
- `gtcrn_onnx_runtime/` - GTCRN轻量化降噪算法的ONNX版本
- `GTCRN-Learning/` - GTCRN算法学习教程
- `Webrtc_NoisyReduce/` - WebRTC官方降噪算法

**4. 实时音频处理**
- `RealTime-Mic-Algorithm-Testing-Platform/` - Qt实现的实时算法测试平台
- `WASAPI/` - Windows音频驱动开发实践

**5. 语音转换与识别**
- `RVC/` - 基于VITS的语音转换工具
- `PaddleSpeech/` - PaddleSpeech相关代码

**6. 评估工具**
- `Noise_Reduction_Benchmark/` - 降噪算法性能评估框架

**7. 学习资料**
- `Document/` - 信号处理理论、滤波器设计、论文等参考资料

## 快速开始

### 环境要求
- Python 3.8+
- C++环境（CMake 3.10+，可选Qt）

### 安装依赖
```bash
pip install numpy scipy matplotlib torch onnx onnxruntime librosa soundfile
```

### 使用示例

**运行谱减法降噪：**
```bash
cd NoisyPrint
python Process.py
```

**测试FIR滤波器：**
```bash
cd FIRFilter
python Main.py
```

**语音转换示例：**
```bash
cd RVC
python firstProject.py
```

**构建C++项目：**
```bash
cd DeepFilterDemo/Demo
mkdir build && cd build
cmake ..
cmake --build .
```

## 算法对比

| 算法类型 | 代表方法 | 延迟 | 计算量 | 效果 | 适用场景 |
|---------|---------|------|-------|------|---------|
| 传统滤波 | FIR/IIR | 很低 | 很低 | 基础 | 简单噪声 |
| 谱减法 | NoisyPrint | 低 | 低 | 中等 | 平稳噪声 |
| 机器学习 | DeepFilter | 中等 | 中等 | 良好 | 复杂环境 |
| 轻量DL | GTCRN | 低 | 中等 | 优秀 | 实时通信 |
| 工业级 | WebRTC | 低 | 低 | 良好 | 实时通信 |

## 项目特点

1. **完整的算法演进** - 从传统方法到深度学习都有实现
2. **实时处理支持** - 提供低延迟的实时音频处理能力
3. **工业级实现** - 包含WebRTC等成熟方案的实现
4. **学习资源丰富** - 有系统的算法学习教程和理论资料

## 项目进展

**已实现：**
- 传统滤波器设计（FIR/IIR）
- 谱减法噪声消除
- DeepFilter、GTCRN等机器学习降噪
- WebRTC降噪算法
- RVC语音转换
- 实时音频测试平台

**计划中：**
- 更多深度学习模型
- 云端推理支持
- 移动端优化
- 自动化评估系统

## 更新历史

**2026年1月** - 项目文档整理和重构

**2025年11月** - 添加高级音频分析功能（基音检测等）

**2025年9月** - 集成GTCRN轻量化降噪算法

**2025年4月** - 建立降噪算法评估体系

**2025年1月** - 集成WebRTC工业级降噪方案

**2024年12月** - 添加FastASR语音识别

**2024年10月** - 完善传统算法（谱减法）

**2024年7月** - 核心功能开发（DeepFilter、WASAPI、RVC等）

**2024年6月** - 项目初始化

## 贡献指南

欢迎提交改进建议或代码！如果有问题可以通过GitHub Issue反馈。

开发时请注意：
- Python代码遵循PEP8规范
- C++代码使用Google C++风格
- 新功能请添加相应文档

## 学习建议

1. **入门** - 先看`Document/`中的信号处理基础
2. **实践** - 尝试`DASP/`和`FIRFilter/`中的滤波器
3. **进阶** - 学习`GTCRN-Learning/`中的深度学习降噪
4. **工程** - 用`RealTime-Mic-Algorithm-Testing-Platform/`测试算法

## 许可证

MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

*最后更新：2026年1月26日*  
*项目状态：活跃开发中*