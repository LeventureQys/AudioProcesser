# GTCRN-Light v0 原版 GTCRN

原版 GTCRN 网络，v1 的轻量化改造基于此版本。

## 基本信息

- **来源**: 原版 GTCRN 论文实现
- **参数量**: ~50K (0.05M)
- **MACs**: 0.03G
- **采样率**: 16kHz
- **频点数**: 257 (n_fft=512)
- **ERB频带**: 129

## 网络结构

```
输入 spec (B, 257, T, 2)
    │
    ├─ 拆分: [|S|, Re, Im] → (B, 3, T, 257)
    │
    ▼
ERB.bm(): 257 → 129 频带
    │   低频65直接保留，高频192压缩到64个ERB band
    │   实现: Linear层（可训练）
    │
    ▼
SFE: 子带特征提取
    │   Unfold展开邻频 → 通道膨胀
    │
    ▼
┌─ Encoder ─────────────────────────────────────┐
│                                               │
│  Conv (3→C, stride=2): 129 → 65              │
│      标准Conv2d                               │
│                                    ← skip1   │
│  Conv (C→C, stride=2): 65 → 33               │
│                                    ← skip2   │
│                                               │
│  GTConv × 3 (dilation: 1,2,5)                │
│      每层: Conv(3×3) → BN → PReLU            │
│            → TRA (RNN/Attention门控)          │
│            → 残差                             │
│                                    ← skip3-5 │
└───────────────────────────────────────────────┘
    │
    ▼
DPGRNN × 2
    │   intra: 双向GRU (频率轴, 33步)
    │          input_size=C, hidden=C
    │   inter: 双向GRU (时间轴, T步)
    │          input_size=C, hidden=C
    │
    ▼
┌─ Decoder ─────────────────────────────────────┐
│                                               │
│  GTConv × 3 (dilation: 5,2,1)                │
│      + skip connections (逆序)               │
│                                               │
│  Deconv (C→C): 33 → 65   + skip2             │
│  Deconv (C→3): 65 → 129  + skip1             │
│                                               │
└───────────────────────────────────────────────┘
    │
    ▼
ERB.bs(): 129 → 257 频带
    │   实现: Linear层（可训练）
    │
    ▼
CRM掩码: out = spec * tanh(mask) (复数乘法)
    │
    ▼
输出 (B, 257, T, 2)
```

## 原版特点

### ERB 变换
- 使用可训练的 Linear 层实现
- 参数量较大

### SFE (子带特征提取)
- 使用 Unfold 展开邻近频带
- 通道数膨胀严重

### GTConv
- 标准 Conv2d，参数量大
- TRA 使用 RNN/Attention 实现，状态管理复杂

### DPGRNN
- 隐藏维度等于通道数 C
- intra 和 inter 都是双向 GRU
- 参数量是主要来源

## 问题与改进方向

原版已经很轻量，v1 的改造主要是：
1. **采样率适配**: 16kHz → 48kHz，需要更多频带
2. **工程优化**: ERB 固定化、TRA 去状态化，便于部署
3. **性能提升**: 增加 GTConv 层数和通道数，提升降噪效果

## 相关版本

- v0 (本版本): 原版 GTCRN
- v1: 轻量化改造 + 48kHz 适配
- v2: 瞬态感知损失
- v3: 因果流式版
