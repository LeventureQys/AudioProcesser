# AudioProcessor - 音频处理与机器学习项目

## 项目简介

这个项目是我在学习音频处理和机器学习过程中积累的代码和实践。涵盖了从传统的数字信号处理到现代的深度学习音频增强算法，包含滤波器设计、噪声消除、语音转换等功能的实现。

## 项目结构

```
AudioProcesser/
├── DSP_Filter_Design/          # 数字信号处理滤波器设计
│   ├── FIRFIlter/              # FIR滤波器实现和测试
│   ├── FIRSimulation/          # Python版滤波器仿真
│   └── IIRFilter/              # IIR滤波器设计和计算核心
├── FrameworkLearning/          # 框架学习
│   ├── GTCRN-Learning/         # GTCRN轻量化降噪算法学习教程
│   └── RVC-Learning/           # RVC语音转换学习
├── Document/                   # 学习资料与参考文档
│   ├── DASP/                   # 音频信号处理基础算法资料
│   ├── FilterDesignInfo/       # 滤波器设计参考
│   ├── Book/                   # 参考书籍
│   ├── Paper/                  # 论文
│   ├── RVC/                    # RVC相关资料
│   ├── Voice Signals Process/  # 语音信号处理
│   └── ...                     # 其他理论文档
├── Test_Audio/                 # 测试音频文件
│   ├── AudioSample-16000hz/    # 16kHz采样率音频
│   ├── AudioSample-48000hz/    # 48kHz采样率音频
│   └── ...                     # 其他音频资源及工具脚本
├── Archived_Workshop/          # 归档项目（早期实践代码）
│   ├── DeepFilterDemo/         # DeepFilter降噪算法的C++实现
│   ├── gtcrn_onnx_runtime/     # GTCRN降噪算法的ONNX推理版本
│   ├── Noise_Reduction_Benchmark/ # 降噪算法性能评估框架
│   ├── NoisyPrint/             # 基于谱减法的噪声消除实现
│   └── RealTime-Mic-Algorithm-Testing-Platform/ # Qt实时算法测试平台
├── LICENSE
├── README.md
└── README.en.md
```

### 模块说明

**1. 数字信号处理滤波器设计 (`DSP_Filter_Design/`)**
- `FIRFIlter/` - FIR滤波器实现和测试
- `FIRSimulation/` - Python版的滤波器仿真
- `IIRFilter/` - IIR滤波器设计和计算核心

**2. 框架学习 (`FrameworkLearning/`)**
- `GTCRN-Learning/` - GTCRN轻量化降噪网络结构详解、优化与实践
- `RVC-Learning/` - RVC语音转换框架学习

**3. 学习资料 (`Document/`)**
- 信号处理基础理论（DASP、滤波器设计原理等）
- 论文和参考书籍
- RVC、语音信号处理等专题资料

**4. 测试音频 (`Test_Audio/`)**
- 不同采样率的音频样本（16kHz、48kHz）
- 采样率转换工具脚本

**5. 归档项目 (`Archived_Workshop/`)**

早期实践代码，已归档保留作为参考：
- `DeepFilterDemo/` - DeepFilter降噪算法的C++实现
- `gtcrn_onnx_runtime/` - GTCRN降噪算法的ONNX推理实现
- `Noise_Reduction_Benchmark/` - 降噪算法客观评估框架
- `NoisyPrint/` - 基于谱减法的噪声消除
- `RealTime-Mic-Algorithm-Testing-Platform/` - Qt实现的实时音频算法测试平台

## 算法对比

| 算法类型 | 代表方法 | 延迟 | 计算量 | 效果 | 适用场景 |
|---------|---------|------|-------|------|---------|
| 传统滤波 | FIR/IIR | 很低 | 很低 | 基础 | 简单噪声 |
| 谱减法 | NoisyPrint | 低 | 低 | 中等 | 平稳噪声 |
| 机器学习 | DeepFilter | 中等 | 中等 | 良好 | 复杂环境 |
| 轻量DL | GTCRN | 低 | 中等 | 优秀 | 实时通信 |

## 项目特点

1. **完整的算法演进** - 从传统滤波器到深度学习降噪都有实现
2. **学习资源丰富** - 有系统的算法学习教程和理论资料
3. **工程实践参考** - 归档项目包含可运行的工程代码

## 学习建议

1. **入门** - 先看 `Document/` 中的信号处理基础理论
2. **实践** - 尝试 `DSP_Filter_Design/` 中的滤波器设计
3. **进阶** - 学习 `FrameworkLearning/GTCRN-Learning/` 中的深度学习降噪
4. **参考** - 查阅 `Archived_Workshop/` 中的工程实现代码

## 更新历史

**2026年2月** - 项目结构重整，归档早期项目，精简目录

**2026年1月** - 项目文档整理和重构

**2025年11月** - 添加高级音频分析功能（基音检测等）

**2025年9月** - 集成GTCRN轻量化降噪算法

**2025年4月** - 建立降噪算法评估体系

**2025年1月** - 集成WebRTC工业级降噪方案

**2024年12月** - 添加FastASR语音识别

**2024年10月** - 完善传统算法（谱减法）

**2024年7月** - 核心功能开发（DeepFilter、WASAPI、RVC等）

**2024年6月** - 项目初始化

## 许可证

MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

*最后更新：2026年2月24日*
*项目状态：活跃开发中*