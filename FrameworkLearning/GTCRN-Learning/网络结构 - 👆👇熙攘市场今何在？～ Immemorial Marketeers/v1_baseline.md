# GTCRN-Light v1 Baseline

基于原版 GTCRN (v0) 的 48kHz 适配与增强版本，用标准 SpecRIMAGLoss 训练。

## 基本信息

- **Best Epoch**: 29
- **DNSMOS_OVR**: 3.1474
- **参数量**: 139,482 (vs v0 ~50K，为适配48kHz增加)
- **采样率**: 48kHz (vs v0 16kHz)

## 相比 v0 的改进

| 改动 | v0 (原版) | v1 (本版本) | 原因 |
|------|-----------|-------------|------|
| 采样率 | 16kHz | 48kHz | 适配高采样率 |
| ERB频带 | 129 | 219 | 覆盖更宽频率 |
| GTConv层数 | 3 | 6 | 增强建模能力 |
| ERB实现 | Linear (可训练) | Buffer (固定) | 便于部署 |
| TRA | RNN/Attention | TRALite (Conv) | 去状态化 |

## 网络结构

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

## 各模块参数

| 模块 | 结构 | 参数量 |
|------|------|--------|
| ERB | 固定滤波器组 | 0 (buffer) |
| SFE | DW(1×5) + PW(1×1) + BN | ~50 |
| DSConv×2 | DW + PW + BN + PReLU | ~3K |
| GTConvLite×12 | DW(3×3) + PW + BN + TRA + SE | ~80K |
| SubbandAttn | Linear(55→13→55) | ~1.5K |
| DPGRNN×2 | GRU×4 + Linear×3 + LN | ~50K |
| DSDeconv×2 | DW + PW + BN + PReLU | ~3K |
| freq_weights | 可学习 | 513 |

## 配置

```yaml
Model: GTCRN_light_v3_48k_enhanced
width_mult: 2.0        # 基础通道16 × 2 = 32
use_two_dpgrnn: true
dpgrnn_layers: 2

Loss: SpecRIMAGLoss_BAK_v2
  lambda_ri: 30.0
  lambda_mag: 70.0
  compress_factor: 0.3

Audio:
  n_fft: 1024
  hop_length: 480
  win_length: 1024
```

## 用法

```python
from models.gtrcn_light_v3_48k_enhanced import GTCRN_light_v3_48k_enhanced

model = GTCRN_light_v3_48k_enhanced(width_mult=2.0, use_two_dpgrnn=True, dpgrnn_layers=2)
ckpt = torch.load('best_model_epoch29_score3.1474.tar')
model.load_state_dict(ckpt['model'])
```

## 相关版本

- v0: 原版 GTCRN
- v1 (本版本): 轻量化基线
- v2: 加了瞬态感知损失
- v3: 因果流式版
