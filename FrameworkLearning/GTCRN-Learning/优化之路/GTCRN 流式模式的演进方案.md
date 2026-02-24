# GTCRN 演进路径

v0 → v1 → v2 → v3 的改动记录。

## 版本概览

```
v0 (原版)     v1 (baseline)    v2 (transient)    v3 (causal)
16kHz/129ERB  48kHz/219ERB     同v1              同v1
~50K参数      139K参数         139K参数          145K参数
              DNSMOS 3.15      DNSMOS 3.15       DNSMOS 2.98
非实时        非实时           非实时            实时 10ms
     │              │                │
     │ 算子级轻量化  │ 换损失函数      │ 因果化
     └──────────────┴────────────────┘
```

---

## v0 → v1: 算子级轻量化

### 背景

原版 GTCRN 是 16kHz 的，要适配 48kHz。顺便做了算子级优化，方便部署。

核心约束：
- 数据流不变：ERB → SFE → Encoder → DPGRNN → Decoder → CRM
- 频轴采样不变：两次 /2
- 时频建模顺序不变：intra(频率) → inter(时间)

### 四个改动点

**1) 卷积 DW-Separable 化**

标准卷积参数太多，拆成 depthwise + pointwise。

```
标准 3x3 Conv (32→32): 9 * 32 * 32 = 9216 参数
DW-Sep: 9*32 + 32*32 = 288 + 1024 = 1312 参数

省了约 1/7
```

应用：DSConv、DSDeconv、GT-ConvLite

**2) TRALite 替代 TRA**

原版 TRA 用 RNN 做时域门控，有状态，不好部署。

改成卷积：
```
energy = mean(x², dim=F)  →  DW-1D(k=3)  →  PW-1D  →  Sigmoid  →  gate
```
无状态，可量化。

**3) DPGRNN 瓶颈化**

RNN 隐藏维是参数大头，加低秩投影：

```
原版: GRU(C, C)               参数 ∝ C²
瓶颈: Linear(C→r) → GRU(r,r) → Linear(r→C)   参数 ∝ r²

r = 0.75C 时，参数约 0.56C²
```

归一化用 LayerNorm(C)，不绑频轴长度。

**4) ERB 固定化**

ERB 三角滤波器组是固定的，没必要训练。

```
原版: nn.Linear(...)  # 计入参数
改后: register_buffer("W_bm", ...)  # 不计参数，导出一致
```

### 频轴闭环

确保上下采样对齐：
```
编码: 219 → 110 → 55
解码: 55 → 110 → 219
```

### 变更对照

| 位置 | v0 | v1 |
|------|-----|-----|
| ERB | Linear (可训练) | Buffer (固定) |
| 卷积 | 标准 Conv | DW-Separable |
| TRA | RNN | Conv (TRALite) |
| DPGRNN | C 维 | C→r→C 瓶颈 |

### 可调参数

```
边缘场景: width_mult=0.75, r=0.5C, use_two_dpgrnn=False
平衡默认: width_mult=1.0, r=0.75C, use_two_dpgrnn=True
高质离线: width_mult=1.25, r=C, use_two_dpgrnn=True
```

---

## v1 → v2: 换损失函数

### 问题

标准 SpecRIMAGLoss 对所有帧权重相同。键盘鼠标这类瞬态噪音处理不好，但 DNSMOS 是整段平均，看不出来。

### 方案

不改网络，只改 loss。检测能量突变的帧，权重放大：

```
energy_diff = |energy[t] - energy[t-1]|
is_transient = energy_diff > threshold * mean_energy

weight = 5.0 if transient else 1.0
loss = Σ weight[t] * frame_loss[t]
```

### 结果

- DNSMOS 持平 (3.147)
- 瞬态噪音主观改善
- 训练时间变长 (29 → 71 epochs)

### 为什么不改网络

能用 loss 解决就不动架构。改 loss 只影响训练，推理零开销。

---

## v2 → v3: 因果化

### 问题

v1/v2 是离线的，要看完整段才能处理。实时场景用不了。

```
非因果: 处理 t 帧要看 [0, ..., t, ..., T-1]  →  延迟 = 整段
因果:   处理 t 帧只看 [0, ..., t]            →  延迟 = 单帧
```

### 方案

把"看未来"的操作都改掉：

**卷积 padding**
```
非因果: padding = (d, d)        前后各看 d 帧
因果:   padding = ((k-1)*d, 0)  只 pad 左边
```

**TRALite**
```
非因果: Conv1d(k=5, padding=2)
因果:   F.pad(x, (4,0)) + Conv1d(k=5, padding=0)
```

**DPGRNN**
- inter: 本来就是单向 GRU，不用改
- intra: 沿频率轴的双向 GRU，不涉及时间因果，不用改

### 其他改动

| 项 | v2 | v3 |
|----|-----|-----|
| 激活 | PReLU | SiLU |
| GTConv kernel | 3×3 | 5×5 |
| BN | DW→PW→BN | DW→BN→PW→BN |
| 输入 | [mag,real,imag] | [real,imag] |
| 参数 | 139K | 145K |

kernel 变大是为了补偿因果化损失的感受野。

### 流式状态

实时推理要维护：
- GTConv 历史帧: 12层，最大缓存 (5-1)*8=32 帧
- TRA 历史能量: 12层，每层 4 帧
- GRU hidden: inter 的 2×DPGRNN×2层
- Skip 缓存: 8 组

### 结果

- DNSMOS: 3.15 → 2.98 (-5%)
- 延迟: 10ms
- RTF: 0.21

掉分是预期内的，因果模型信息量少。

---

## 选型

| 场景 | 版本 | 原因 |
|------|------|------|
| 离线后期 | v1 | 质量最高 |
| 办公录音 | v2 | 瞬态处理好 |
| 实时通话 | v3 | 低延迟 |
| 嵌入式 | v3+C | 资源小 |

---

## 总结

```
v0 → v1: 算子替换，架构语义不变
v1 → v2: 只改 loss，网络不动
v2 → v3: 因果 padding，接受质量损失换实时
```
