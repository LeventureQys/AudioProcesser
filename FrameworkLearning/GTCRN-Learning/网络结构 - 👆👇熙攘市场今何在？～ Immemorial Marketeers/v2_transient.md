# GTCRN-Light v2 Transient

在 v1 基础上加了瞬态感知损失，对键盘敲击、鼠标点击这类突发噪音效果更好。

## 基本信息

- **Best Epoch**: 71
- **DNSMOS_OVR**: 3.147
- **参数量**: 139,482
- **采样率**: 48kHz

## 网络结构

跟 v1 完全一样，只是损失函数不同：

```
输入 spec (B, 513, T, 2)
    │
    ├─ 可学习频带权重 (513,)
    │
    ▼
ERB_48k.bm(): 513 → 219 频带
    │   低频171直接保留，高频342压缩到48个ERB band
    │
    ▼
SFE_Lite: 特征提取
    │   DWConv(3, 3, k=1×5) → PWConv(3, 3) → BN
    │
    ▼
┌─ Encoder ─────────────────────────────────────┐
│                                               │
│  DSConv (3→32ch, stride=2): 219 → 110        │
│      DWConv → PWConv → BN → PReLU            │
│                                    ← skip1   │
│  DSConv (32→32ch, stride=2): 110 → 55        │
│                                    ← skip2   │
│                                               │
│  GTConvLite × 6 (dilation: 1,2,4,8,4,2)      │
│      每层: DWConv(3×3) → PWConv → BN → PReLU │
│            → TRALite → SEBlock → 残差        │
│                                    ← skip3-8 │
│                                               │
│  SubbandAttention: 频带加权                   │
│      energy → Linear(55,13) → ReLU           │
│            → Linear(13,55) → Sigmoid         │
└───────────────────────────────────────────────┘
    │
    ▼
DPGRNN_Enhanced × 2
    │   pre:  Linear(32 → 32)
    │   intra: 双向GRU×2层 (频率轴, 55步)
    │   post: Linear(64 → 32)
    │   inter: 单向GRU×2层 (时间轴, T步)
    │   post2: Linear(32 → 32)
    │   + LayerNorm + 可学习残差缩放(α,β)
    │
    ▼
┌─ Decoder ─────────────────────────────────────┐
│                                               │
│  GTConvLite × 6 (dilation: 2,4,8,4,2,1)      │
│      + skip connections (逆序)               │
│                                               │
│  DSDeconv (32→32ch): 55 → 110   + skip2      │
│  DSDeconv (32→2ch):  110 → 219  + skip1      │
│                                               │
└───────────────────────────────────────────────┘
    │
    ▼
ERB_48k.bs(): 219 → 513 频带
    │
    ▼
CRM掩码: out = spec * mask (复数乘法)
    │
    ▼
输出 (B, 513, T, 2)
```

## 核心模块说明

### GTConvLite
```
输入 x (B, 32, T, 55)
    │
    ├─────────────────────────┐
    ▼                         │
DWConv2d(32, 32, k=3×3)      │  分组卷积，dilation沿时间轴
    │  padding=(dilation, 1)  │
    ▼                         │
PWConv2d(32, 32, k=1×1)      │
    │                         │
    ▼                         │
BN → PReLU                    │
    │                         │
    ▼                         │
TRALite: 时序注意力           │
    │  energy = x².mean(F)    │
    │  gate = σ(PW(DW(energy)))
    │  out = x * gate         │
    ▼                         │
SEBlock: 通道注意力           │
    │  w = σ(fc2(relu(fc1(   │
    │        x.mean(T,F)))))  │
    │  out = x * w            │
    ▼                         │
    + ←───────────────────────┘ 残差
    │
    ▼
输出 (B, 32, T, 55)
```

### DPGRNN_Enhanced
```
输入 x (B, 32, T, 55)
    │
    ▼
reshape → (B*T, 55, 32)
    │
    ▼
pre: Linear(32 → 32)
    │
    ▼
intra: 双向GRU (沿频率轴)
    │   input_size=32, hidden=32, layers=2
    │   输出 (B*T, 55, 64)
    │
    ▼
post: Linear(64 → 32)
    │
    ▼
reshape → (B, 32, T, 55)
    │
    ├─ y = x + α*h
    ▼
LayerNorm
    │
    ▼
reshape → (B*55, T, 32)
    │
    ▼
pre: Linear(32 → 32)  (共享权重)
    │
    ▼
inter: 单向GRU (沿时间轴)
    │   input_size=32, hidden=32, layers=2
    │
    ▼
post2: Linear(32 → 32)
    │
    ▼
reshape → (B, 32, T, 55)
    │
    ├─ out = y + β*z
    ▼
LayerNorm
    │
    ▼
输出 (B, 32, T, 55)
```

## 配置

```yaml
Model: GTCRN_light_v3_48k_enhanced
width_mult: 2.0
use_two_dpgrnn: true
dpgrnn_layers: 2

Loss: TransientAwareLoss      # 跟v1的区别
  lambda_ri: 30.0
  lambda_mag: 70.0
  compress_factor: 0.3
  transient_weight: 5.0       # 瞬态帧权重放大5倍
  energy_threshold: 2.0
  smooth_window: 3

Augmentation:
  random_gain: [0.5, 2.0]     # -6dB ~ +6dB
  probability: 0.5

Audio:
  n_fft: 1024
  hop_length: 480
  win_length: 1024
```

## 瞬态检测

损失函数会检测能量突变的帧，对这些帧加大惩罚：
1. 算相邻帧能量变化率
2. 超过阈值的标记为瞬态帧
3. 瞬态帧损失乘以 `transient_weight`

## 用法

```python
from models.gtrcn_light_v3_48k_enhanced import GTCRN_light_v3_48k_enhanced

model = GTCRN_light_v3_48k_enhanced(width_mult=2.0, use_two_dpgrnn=True, dpgrnn_layers=2)
ckpt = torch.load('best_model_epoch71_score3.147.tar')
model.load_state_dict(ckpt['model'])
```

## 适用场景

- 办公环境（键盘、鼠标）
- 突发噪音（敲门、拍手）
- 音量变化大的录音

## 训练记录

`full_training_run/` 下有完整训练过程：
- `checkpoints/`: 各epoch权重
- `logs/`: tensorboard日志
- `val_samples/`: 验证样本

## 相关版本

- v0: 原版 GTCRN
- v1: 轻量化基线
- v2 (本版本): 瞬态优化
- v3: 因果流式版
