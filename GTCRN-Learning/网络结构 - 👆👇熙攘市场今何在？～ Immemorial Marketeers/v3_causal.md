# GTCRN-Light v3 Causal Stream

因果版本，可以做实时流式推理。把离线版改成因果的，牺牲一点性能换取实时能力。

## 基本信息

- **Best Epoch**: 35
- **DNSMOS_OVR**: 2.983
- **参数量**: 145,127
- **采样率**: 48kHz
- **帧延迟**: 10ms (单帧)

## 跟离线版的区别

主要改了这几个地方保证因果性：

1. **GTConvLite → CausalGTConvLite**: 左padding改成 `(kernel-1) * dilation`
2. **TRALite → CausalTRA**: 同样改成因果padding
3. **DPGRNN inter**: 时间轴上的GRU改成单向（频率轴还是双向，不影响因果）
4. **激活函数**: PReLU → SiLU
5. **DSConv/DSDeconv**: 加了中间BN，顺序也调整了

## 性能对比

| 版本 | 参数量 | DNSMOS | 实时 |
|------|--------|--------|------|
| v1 离线 | 139K | 3.15 | × |
| v2 离线 | 139K | 3.15 | × |
| **v3 因果** | **145K** | **2.98** | **√** |

掉了0.17分，换来实时推理能力。

## 网络结构

```
输入 spec (B, 513, T, 2)
    │
    ▼
ERB_48k.bm(): 513 → 219 频带
    │
    ▼
in_conv: Conv2d(2 → 3, k=1×1)
    │
    ▼
┌─ CausalEncoder ───────────────────────────────┐
│                                               │
│  DSConv (3→32ch, stride=2): 219 → 110        │
│      DWConv → BN → SiLU → PWConv → BN → SiLU │
│                                    ← skip1   │
│  DSConv (32→32ch, stride=2): 110 → 55        │
│                                    ← skip2   │
│                                               │
│  CausalGTConvLite × 6 (dilation: 1,2,4,8,4,2)│
│      每层: 因果DWConv(5×5) → BN → SiLU       │
│            → PWConv → BN → SiLU              │
│            → CausalTRA → SE → 残差           │
│                                    ← skip3-8 │
│                                               │
│  SubbandAttention: 频带加权                   │
└───────────────────────────────────────────────┘
    │
    ▼
CausalDPGRNN × 2
    │   pre:  Linear(32 → 32)
    │   intra: 双向GRU×2层 (频率轴, 55步)  ← 双向OK
    │   post: Linear(64 → 32)
    │   inter: 单向GRU×2层 (时间轴, T步)   ← 必须单向!
    │   post2: Linear(32 → 32)
    │   + LayerNorm + 可学习残差缩放(α,β)
    │
    ▼
┌─ CausalDecoder ───────────────────────────────┐
│                                               │
│  CausalGTConvLite × 6 (dilation: 2,4,8,4,2,1)│
│      + skip connections (逆序)               │
│                                               │
│  Fuse: Conv2d(64→32, k=1×1) + skip2          │
│  DSDeconv (32→32ch): 55 → 110                │
│  DSDeconv (32→2ch):  110 → 219  + skip1      │
│                                               │
└───────────────────────────────────────────────┘
    │
    ▼
out_conv: Conv2d(2 → 2, k=1×1)
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

## 因果模块详解

### CausalGTConvLite vs GTConvLite

```
GTConvLite (离线版):
    padding = (dilation, 1)  # 时间轴对称padding
    可以看到前后各 dilation 帧

CausalGTConvLite (因果版):
    pad_t = (kernel-1) * dilation = 4 * dilation
    F.pad(x, (0, 0, pad_t, 0))  # 只在左边(过去)padding
    只能看到过去的帧
```

### CausalTRA vs TRALite

```
TRALite (离线版):
    Conv1d(ch, ch, k=5, padding=2)  # 对称padding
    可以看前后各2帧

CausalTRA (因果版):
    Conv1d(ch, ch, k=5, padding=0)
    F.pad(x, (4, 0))  # 左边padding 4帧
    只能看过去4帧
```

### CausalDPGRNN vs DPGRNN

```
DPGRNN (离线版):
    intra: 双向GRU (频率轴) ← 不影响因果
    inter: 双向GRU (时间轴) ← 非因果!

CausalDPGRNN (因果版):
    intra: 双向GRU (频率轴) ← 保持双向
    inter: 单向GRU (时间轴) ← 改成单向
```

## 流式状态

跑流式推理要维护这些状态：
- GTConv缓存 (12层，不同dilation需要不同长度)
- TRA历史帧 (12层，每层4帧)
- GRU隐藏状态 (2×DPGRNN × 2层 inter)
- Skip连接缓存 (8组)

## 目录结构

```
v3_causal_stream/
├── checkpoints/
│   └── best_model_epoch35_score2.983.tar
├── configs/
│   └── cfg_causal_v2_48k.yaml
├── models/
│   ├── gtcrn_light_v3_48k_causal_v2.py    # 因果模型
│   └── gtcrn_light_v3_48k_causal_train.py # 废弃
├── scripts/
│   ├── inference_causal_stream.py         # 批量推理
│   └── inference_stream_realtime.py       # 流式推理
├── test_samples/
└── C_Stream/                              # C实现
```

## 推理方式

### Python 批量推理
```bash
python scripts/inference_causal_stream.py -i input.wav -o output.wav \
    -c checkpoints/best_model_epoch35_score2.983.tar
```

### Python 流式推理
```bash
python scripts/inference_stream_realtime.py -i input.wav -o output.wav \
    -c checkpoints/best_model_epoch35_score2.983.tar
```

### C 流式推理（部署用）
```bash
cd C_Stream && make
./build/test_audio -w weights/gtcrn_causal_v2.bin -i input.wav -o output.wav
```

## C 实现性能

- 每帧处理: ~2.1ms
- RTF: 0.21 (还有4.7倍余量)
- 跟Python精度差异: ~0.01

## 测试结果 (DNSMOS)

| 文件 | 原始 | Python批量 | Python流式 | C流式 |
|------|------|-----------|-----------|-------|
| 1.wav | 2.21 | 2.85 | 2.69 | 2.69 |
| 2.wav | 2.41 | 3.66 | 3.50 | 3.50 |
| 3.wav | 2.93 | 3.82 | 3.85 | 3.85 |
| 4.wav | 2.65 | 3.78 | 3.79 | 3.78 |
| **平均** | **2.57** | **3.57** | **3.53** | **3.53** |

流式比批量低一点是正常的，因为流式没法看未来帧。

## 训练配置

```yaml
width_mult: 2.0
use_two_dpgrnn: true
dpgrnn_layers: 2
loss: TransientAwareLoss (weight=1.0)
lr: 8e-4 → 5e-6
batch_size: 6
augment: random_gain [0.5, 2.0]
```

## 相关版本

- v0: 原版 GTCRN
- v1: 轻量化基线
- v2: 瞬态优化
- v3 (本版本): 因果流式
