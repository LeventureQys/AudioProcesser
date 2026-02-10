# GTCRN 演进路径

记录 v1 → v2 → v3 的改动和原因。

## 版本概览

| 版本 | 改动点 | 参数量 | DNSMOS | 实时 |
|------|--------|--------|--------|------|
| v1 baseline | 基线 | 139K | 3.15 | × |
| v2 transient | 换损失函数 | 139K | 3.15 | × |
| v3 causal | 因果化改造 | 145K | 2.98 | √ |

## 网络结构 (v1/v2 共用)

```
输入 spec (B, 513, T, 2)
    │
    ├─ 可学习频带权重 (513,)
    │
    ▼
ERB_48k.bm(): 513 → 219
    │   低频171保留，高频342→48 ERB band
    │
    ▼
SFE_Lite: DWConv(1×5) → PWConv → BN
    │
    ▼
┌─ Encoder ─────────────────────────────┐
│  DSConv: 219→110 (stride=2)   ← skip1 │
│  DSConv: 110→55  (stride=2)   ← skip2 │
│  GTConvLite×6 (d=1,2,4,8,4,2) ← skip3-8
│  SubbandAttention                     │
└───────────────────────────────────────┘
    │
    ▼
DPGRNN_Enhanced × 2
    │  intra: 双向GRU (频率轴)
    │  inter: 单向GRU (时间轴)
    │
    ▼
┌─ Decoder ─────────────────────────────┐
│  GTConvLite×6 + skip (逆序)           │
│  DSDeconv: 55→110 + skip2             │
│  DSDeconv: 110→219 + skip1            │
└───────────────────────────────────────┘
    │
    ▼
ERB_48k.bs(): 219 → 513
    │
    ▼
CRM掩码 → 输出
```

### GTConvLite 内部

```
x → DWConv(3×3, dilation) → PWConv → BN → PReLU
  → TRALite (时序注意力)
  → SEBlock (通道注意力)
  → + x (残差)
```

### DPGRNN 内部

```
x (B,C,T,F)
  → reshape (B*T, F, C)
  → Linear → 双向GRU (频率轴) → Linear
  → reshape + LayerNorm
  → reshape (B*F, T, C)
  → Linear → 单向GRU (时间轴) → Linear
  → reshape + LayerNorm
  → 输出
```

---

## v1 → v2: 换损失函数

### 问题

v1 用的是标准 SpecRIMAGLoss，对所有帧一视同仁。但实际听感上，键盘敲击、鼠标点击这类突发噪音处理得不好。DNSMOS 是整段平均，掩盖了这个问题。

### 方案

不改网络，只改损失函数。加了瞬态检测：

```python
# 检测能量突变
energy_diff = |energy[t] - energy[t-1]|
transient = energy_diff > threshold * mean_energy

# 瞬态帧损失放大5倍
loss = Σ weight[t] * frame_loss[t]
weight[t] = 5.0 if transient[t] else 1.0
```

### 结果

- DNSMOS 基本持平 (3.1474 → 3.147)
- 瞬态噪音主观听感明显改善
- 训练时间变长 (29 → 71 epochs)

### 为什么不改网络

能用损失函数解决的问题就不动架构。改架构的代价：
- 要重新验证各模块交互
- 可能引入新bug
- 推理时有额外开销

改损失函数只影响训练，推理零开销。

---

## v2 → v3: 因果化

### 问题

v1/v2 是离线模型，要看完整段音频才能处理。没法用在实时场景（通话、直播）。

延迟分析：
- 非因果模型需要看"未来"帧
- 感受野决定最小延迟，v2大概要200-500ms
- 实时通话要求<50ms

### 方案

把所有"偷看未来"的操作改掉：

| 模块 | v2 (非因果) | v3 (因果) |
|------|-------------|-----------|
| GTConvLite | padding=(d,1) 对称 | pad_t=(k-1)*d 左边 |
| TRALite | Conv1d padding=2 | F.pad(x,(4,0)) |
| DPGRNN inter | 双向GRU | 单向GRU |

频率轴的操作不用改，因为频率轴不涉及时间因果。

### v3 网络结构

```
输入 spec (B, 513, T, 2)
    │
    ▼
ERB_48k.bm(): 513 → 219
    │
    ▼
in_conv: Conv2d(2→3)
    │
    ▼
┌─ CausalEncoder ───────────────────────┐
│  DSConv: 219→110              ← skip1 │
│  DSConv: 110→55               ← skip2 │
│  CausalGTConvLite×6           ← skip3-8
│  SubbandAttention                     │
└───────────────────────────────────────┘
    │
    ▼
CausalDPGRNN × 2
    │  intra: 双向GRU (频率轴) ← 不用改
    │  inter: 单向GRU (时间轴) ← 改成单向
    │
    ▼
┌─ CausalDecoder ───────────────────────┐
│  CausalGTConvLite×6 + skip            │
│  Fuse + DSDeconv: 55→110              │
│  DSDeconv: 110→219 + skip1            │
└───────────────────────────────────────┘
    │
    ▼
out_conv → ERB_48k.bs() → CRM → 输出
```

### 因果模块对比

**GTConvLite → CausalGTConvLite**
```
离线: padding=(dilation, 1)，前后各看dilation帧
因果: F.pad(x, (0,0,pad_t,0))，只看过去pad_t帧
      pad_t = (kernel-1) * dilation
```

**TRALite → CausalTRA**
```
离线: Conv1d(k=5, padding=2)，前后各看2帧
因果: Conv1d(k=5, padding=0) + F.pad(x,(4,0))，只看过去4帧
```

**DPGRNN → CausalDPGRNN**
```
离线: inter用双向GRU，能看整个时间序列
因果: inter改单向GRU，只能看到当前和过去
```

### 其他改动

- 激活函数: PReLU → SiLU
- DSConv: 加了中间BN，顺序调整
- 参数量: 139K → 145K (+4%)

参数量增加是因为单向GRU要增大hidden_size才能保持建模能力。

### 结果

- DNSMOS: 3.15 → 2.98 (-5%)
- 延迟: 10ms (单帧)
- RTF: 0.21 (还有4.7倍余量)

掉了0.17分是预期内的。因果模型看不到未来，信息量必然少于非因果模型。

### 流式状态

实时推理要维护帧间状态：
- GTConv缓存: 12层，不同dilation长度不同
- TRA历史: 12层，每层4帧
- GRU hidden: 2×DPGRNN × 2层
- Skip缓存: 8组

---

## 文件结构

```
archived_models/
├── v1_baseline/
│   ├── original_export/
│   │   └── gtrcn_light_v3_48k_enhanced.py
│   └── best_model_epoch29_score3.1474.tar
│
├── v2_transient/
│   ├── config.yaml
│   ├── best_model_epoch71_score3.147.tar
│   └── full_training_run/
│
└── v3_causal_stream/
    ├── models/
    │   └── gtcrn_light_v3_48k_causal_v2.py
    ├── checkpoints/
    │   └── best_model_epoch35_score2.983.tar
    └── C_Stream/  # C语言实现
```

---

## 选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 离线处理 | v1 | 质量最高 |
| 办公环境 | v2 | 瞬态处理好 |
| 实时通话 | v3 | 低延迟 |
| 嵌入式部署 | v3 + C实现 | 资源占用小 |
