# AudioProcesser - 音频处理学习与实践

一个从传统 DSP 到深度学习的完整音频处理学习项目，涵盖**数字滤波器设计 → 传统降噪算法 → 深度学习降噪 → 语音转换**的全链路工程实践。

[English](README.en.md) | 中文

---

## 全景图

```
AudioProcesser/
│
├── Document/                          # 理论基石（32+ 篇中文文档）
│   ├── 前置知识/                       # 信号与系统、滤波器理论、Attention/Transformer
│   ├── DASP/                          # 数字音频信号处理实例代码
│   ├── FilterDesignInfo/              # 滤波器设计参考（PDF）
│   ├── Voice Signals Process/         # 语音信号分析方法、噪声抑制器设计
│   ├── Paper/                         # 论文
│   └── Book/                          # 参考书籍
│
├── DSP_Filter_Design/                 # 传统 DSP 滤波器实现
│   ├── IIRFilter/                     # Butterworth 滤波器（C++/Python）
│   ├── FIRFIlter/                     # FIR 滤波器 + FFT 可视化
│   ├── FIRSimulation/                 # scipy.signal 仿真
│   └── CalculatorCore/                # IIR 系数计算引擎（Qt/C++）
│
├── FrameworkLearning/                 # 深度学习音频算法学习路径
│   ├── GTCRN-Learning/                # GTCRN 完整学习教程（含 RNNoise 参考实现）
│   ├── ML/                            # 机器学习基础（激活函数、UNet、MNIST）
│   └── RVC-Learning/                  # RVC 语音转换学习
│
├── Archived_Workshop/                 # 工程实践
│   ├── NoisyPrint/                    # 谱减法降噪
│   ├── DeepFilterDemo/                # DeepFilter C++ 实现
│   ├── gtcrn_onnx_runtime/            # GTCRN ONNX 推理（C++17）
│   ├── MeanVC/                        # 流式零样本语音转换
│   ├── Noise_Reduction_Benchmark/     # 降噪算法客观评估
│   ├── RealTime-Mic-Algorithm-Testing-Platform/  # Qt 实时音频测试平台
│   └── ...                            # 更多归档
│
├── Test_Audio/                        # 测试音频 + 采样率转换工具
│
└── third_party/                       # 子模块
    ├── DeepFilterNet/                 # 深度滤波（Rust + Python）
    └── gtcrn/                         # 官方 GTCRN 实现
```

---

## 降噪算法四代演进

| 代次 | 技术 | 项目位置 | 延迟 | 算力 | 效果 |
|------|------|---------|------|------|------|
| 1️⃣ | FIR / IIR 滤波器 | `DSP_Filter_Design/` | 极低 | 极低 | 基础 |
| 2️⃣ | 谱减法 | `Archived_Workshop/NoisyPrint/` | 低 | 低 | 中等 |
| 3️⃣ | DeepFilterNet（CRNN + 复频谱滤波） | `third_party/DeepFilterNet/` | 中 | 中 | 良好 |
| 4️⃣ | GTCRN（分组卷积 + 循环网络） | `third_party/gtcrn/` | 低 | 极低 | 优秀 |

## 语音转换

| 项目 | 技术栈 | 说明 |
|------|--------|------|
| MeanVC | DiT + CFM + WavLM + Vocos | 流式零样本语音转换，2 步生成 |
| RVC-Learning | 学习笔记 | NSF-HiFiGAN 声码器等 |

## 工程实践

| 项目 | 技术栈 | 说明 |
|------|--------|------|
| gtcrn_onnx_runtime | C++17 + ONNX Runtime | GTCRN 本地推理，含自定义 STFT/ISTFT |
| DeepFilterDemo | C++ | DeepFilter 实时降噪 |
| RealTime-Mic | Qt/C++ + WASAPI | 麦克风实时捕获 + 算法插件 |
| NoisyPrint | Python | 谱减法完整实现 + 频谱对比可视化 |
| Noise_Reduction_Benchmark | Python | DNSMOS / NISQA / PESQ / STOI 评估 |

## 第三方子模块

- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** — Rust/Python 深度复频谱降噪
- **[gtcrn](https://github.com/Xiaobin-Rong/gtcrn)** — ICASSP 2024 超轻量实时降噪
- **BenchMark** — MOS 客观评估工具集

---

## 学习路线

```
入门 ──→ Document/前置知识/      信号与系统、滤波器原理、FFT
         │
   实践 ──→ DSP_Filter_Design/     FIR / IIR 滤波器设计与仿真
         │
   进阶 ──→ Document/前置知识/    Attention / Transformer / Conformer 原理
         │
         └──→ FrameworkLearning/  GTCRN 网络结构、RNNoise 参考代码
         │
   深入 ──→ Archived_Workshop/    MeanVC 语音转换、ONNX 推理工程
         │
   部署 ──→ third_party/          DeepFilterNet / GTCRN 官方代码
```

---

## 更新历史

| 时间 | 内容 |
|------|------|
| 2026-05 | MeanVC 学习文档、项目 README 重构 |
| 2026-02 | 项目结构重整，归档早期项目 |
| 2025-11 | 高级音频分析（基音检测、倒谱分析） |
| 2025-09 | 集成 GTCRN 轻量化降噪 |
| 2025-04 | 建立降噪算法评估体系 |
| 2025-01 | WebRTC 工业级降噪方案 |
| 2024-06 | 项目初始化 |

---

## 许可证

[GNU General Public License v3](LICENSE)
