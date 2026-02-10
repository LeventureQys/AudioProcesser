# GTCRN 演进路径

记录 v0 → v1 → v2 → v3 的改动和原因。

## 版本概览

| 版本 | 改动点 | 参数量 | DNSMOS | 采样率 | 实时 |
|------|--------|--------|--------|--------|------|
| v0 raw | 原版 GTCRN | ~500K | - | 16kHz | × |
| v1 baseline | 轻量化改造 | 139K | 3.15 | 48kHz | × |
| v2 transient | 换损失函数 | 139K | 3.15 | 48kHz | × |
| v3 causal | 因果化改造 | 145K | 2.98 | 48kHz | √ |

---

## v0 → v1: 轻量化改造

### 问题

原版 GTCRN 存在几个工程问题：
- **参数量大**: 标准卷积 + 全维度 RNN，约 500K 参数
- **计算量高**: 没有深度可分离优化
- **部署困难**: TRA 用 RNN/Attention 实现，状态管理复杂
- **采样率**: 原版针对 16kHz，需要适配 48kHz

### 方案

四大轻量化支柱：

| 模块 | v0 (原版) | v1 (轻量化) | 收益 |
|------|-----------|-------------|------|
| ERB | Linear (可训练) | Buffer (固定滤波器组) | 参数归零 |
| SFE | Unfold 通道展开 | DWConv(1×5) | 低开销 |
| 卷积 | 标准 Conv2d | DW-Separable | 参数/MACs 降 ~8x |
| GTConv | Conv + RNN门控 | GTConvLite (DW + TRALite) | 参数大幅下降 |
| DPGRNN | C 维 RNN | C→r→C 瓶颈 | 参数按 α² 缩减 |

### 具体改动

**1. ERB 固定化**
```
v0: ERB 用 Linear 层实现，参数可训练
v1: ERB 用 register_buffer 固定三角滤波器组，参数归零
```

**2. 卷积 DW-Separable 化**
```
v0: Conv2d(C_in, C_out, k=3×3)
    参数: C_in × C_out × 9

v1: DWConv(C, C, k=3×3, groups=C) + PWConv(C, C, k=1×1)
    参数: C × 9 + C × C ≈ C² (vs C² × 9)
```

**3. TRA 轻量化**
```
v0: RNN/Attention 门控，有隐藏状态
v1: TRALite = DWConv1d(k=5) + PWConv1d + Sigmoid
    零状态，可量化，参数极小
```

**4. DPGRNN 瓶颈化**
```
v0: GRU(input=C, hidden=C)
v1: Linear(C→r) → GRU(input=r, hidden=r) → Linear(r→C)
    r ≈ 0.75C，参数按 α² 缩减
```

**5. 48kHz 适配**
```
v0: n_fft=512, 257频点, 129 ERB
v1: n_fft=1024, 513频点, 219 ERB
    频率分辨率更高，适合高采样率
```

### 结果

- 参数量: ~500K → 139K (-72%)
- 采样率: 16kHz → 48kHz
- 部署友好: 全卷积 + 标准 GRU，易导出 ONNX

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

## 选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 离线处理 | v1 | 质量最高 |
| 办公环境 | v2 | 瞬态处理好 |
| 实时通话 | v3 | 低延迟 |
| 嵌入式部署 | v3 + C实现 | 资源占用小 |
