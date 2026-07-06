# 06｜SFE 与 TRA：两个"点睛"模块

> 这两个模块是论文的原创贡献。它们不属于骨架——没有它们 GTCRN 依然能跑——但是它们让性能从"还行"变成"打过 RNNoise"。看消融实验里 PESQ 从 1.87 涨到 1.94，主要就靠它们俩。

## 1. 先看消融实验：SFE 和 TRA 各贡献多少？

直接看论文 Table 1（DNS3 测试集）：

| SFE | TA | TRA | 参数 | MACs/秒 | SISNR | PESQ | STOI |
|:--:|:--:|:--:|:----:|:----:|:----:|:----:|:----:|
| ✗ | ✗ | ✗ | 13.35K | 33.91M | 9.87 | 1.87 | 0.834 |
| ✗ | ✓ | ✗ | 14.84K | 34.00M | 10.00 | 1.89 | 0.838 |
| ✗ | ✗ | ✓ | 21.65K | 34.47M | 10.25 | 1.91 | 0.840 |
| ✓ | ✗ | ✗ | 15.37K | 39.07M | 10.10 | 1.90 | 0.838 |
| ✓ | ✓ | ✗ | 16.86K | 39.16M | 10.29 | 1.92 | 0.841 |
| **✓** | ✗ | **✓** | **23.67K** | **39.63M** | **10.39** | **1.94** | **0.844** |

读这张表的几个关键点：

1. **裸 GTCRN（无 SFE 无 TRA）也有 13.35K 参数、PESQ 1.87**——比 RNNoise 1.87 已经持平
2. **加 SFE → PESQ +0.03**（1.87→1.90），参数 +2K，算力 +5M
3. **加 TRA → PESQ +0.04**（1.87→1.91），参数 +8K，算力 +0.5M
4. **SFE + TRA → PESQ +0.07**（1.87→1.94），效果叠加，是最优组合
5. **TRA 比 TA（标准时间注意力）好**：21.65K vs 14.84K，但 PESQ 1.91 vs 1.89——TRA 更值

总结：**SFE 是廉价的算力换性能（多 5M MACs，得 +0.03 PESQ），TRA 是昂贵但精准的参数换性能（多 8K 参数，得 +0.04 PESQ）**。两者互补。

## 2. SFE 模块：用一行 `nn.Unfold` 改变特征布局

### 2.1 看代码

[gtcrn.py:64](../../third_party/gtcrn/gtcrn.py#L64)：

```python
class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), 
                                stride=(1, stride), 
                                padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs
```

整个模块就一行核心代码：`nn.Unfold(kernel_size=(1,3))`。

### 2.2 `nn.Unfold` 是做什么的？

`nn.Unfold` 是 PyTorch 里很少被讲到但极其重要的算子。它做的是 **"滑动窗口提取"**——和卷积一样滑窗，但**只提取窗口内的值**，不做内积。

举个例子，输入是 1D 序列 `[a, b, c, d, e]`，`Unfold(k=3, padding=1)`：

```
窗口位置 1: [pad, a, b]
窗口位置 2: [a, b, c]
窗口位置 3: [b, c, d]
窗口位置 4: [c, d, e]
窗口位置 5: [d, e, pad]

输出 shape: (3, 5) — 3 个窗口元素，5 个窗口位置
```

也就是说，**Unfold 把"卷积的输入展开成滑窗矩阵"**——这是手写卷积时最关键的一步：im2col。

### 2.3 SFE 在 GTCRN 里做什么？

输入 `(B, 3, T, 129)`，SFE 之后：

```python
xs = self.unfold(x)  # (B, 3*3, T*129)
xs = xs.reshape(B, 9, T, 129)
```

也就是：**每个频带 f，把它和左右两个邻居 (f-1, f, f+1) 的特征 concat 到通道维**。

视觉化：

```
SFE 之前 (3 通道):
频带 f=10 的特征:  [mag_10, real_10, imag_10]

SFE 之后 (9 通道):
频带 f=10 的特征:  [mag_9, real_9, imag_9,
                    mag_10, real_10, imag_10,
                    mag_11, real_11, imag_11]
```

### 2.4 为什么这样做有效？

这里我们要回到 GT-Conv 的设计。GT-Conv 的核心计算是 `point_conv1`，它是一个 **1×1 卷积**：

```python
self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
```

**1×1 卷积只能看到同一空间位置的特征**——也就是说，对频带 f，1×1 卷积只能看到 f 自己的通道，看不到 f-1 和 f+1 的信息。

如果不用 SFE，1×1 卷积就只能逐频带独立处理，**完全无法利用频域邻居信息**。这对降噪是致命的——因为很多频谱模式（共振峰、谐波）是横跨多个频带的局部模式。

**SFE 提前把邻居信息"打包"塞到通道维，让 1×1 卷积间接看到频域上下文**。

### 2.5 SFE 和 3×3 卷积的等价性

你可能会想：那为什么不直接用 3×3 卷积？不就一样了吗？

数学上确实等价：

```
SFE(k=3) + 1×1 Conv(3C → C')  ≡  3×3 Conv(C → C')
```

但**计算成本不一样**：

- 3×3 卷积：参数 = `C × C' × 9`，每个位置算 `9 × C × C'` MACs
- SFE + 1×1：SFE 不带参数，1×1 卷积参数 = `3C × C'`，每个位置算 `3C × C'` MACs

看起来好像 3×3 卷积参数更多？等等——

**关键差异在于 SFE 后面跟的是 GT-Conv 的 grouped depthwise**！

```
GT-Conv 的完整链路：SFE → 1×1 → depthwise 3×3 → 1×1
等价于：             1×1 → depthwise 3×3 → 1×1（无 SFE）
                     ↑ 但这种链路缺少了"频域邻居信息"！
```

也就是说，**没有 SFE 时，depthwise 3×3 看不到频域邻居（因为 depthwise 在通道维独立，且 padding 只让它看到时间维邻居）**。SFE 提前把频域邻居塞到通道维，让 depthwise 间接看到。

> 这是一个非常巧妙的设计——SFE 不是"在主干上加东西"，而是"**让 depthwise 卷积获得它本来缺失的频域上下文**"，用极小代价补全了一个能力。

### 2.6 算力账：SFE 真的"廉价"吗？

SFE 自己不带参数（Unfold 是纯 reshape），但它会让后续 1×1 卷积的输入通道从 C 变成 3C：

- 1×1 卷积参数：`3C × C'` 是 `C × C'` 的 **3 倍**
- 算力也是 3 倍

所以"廉价"不是"免费"——是相对于 3×3 卷积省了 3 倍计算（同时保留邻居信息）。

## 3. TRA 模块：时序注意力的轻量化实现

TRA 是 GTCRN 真正的原创贡献。我们一步一步看。

### 3.1 设计动机：时间维度上的"重要性"是不均匀的

降噪任务里，并不是每一帧都同等重要：

- **语音活跃帧**（有人在说话）：网络应该精细处理，保留语音细节
- **静音帧**（没人说话，纯噪声）：网络应该大刀阔斧抑制
- **语音起始/结束帧**（瞬态过渡）：最难处理，需要特别关注

**作者希望网络能自动学到"这一帧有多重要"，给重要的帧分配更多关注**。这就是注意力机制的核心思想。

### 3.2 标准注意力 vs 时间注意力 vs TRA

传统 self-attention 计算 query/key/value 之间的相似度，O(T²) 复杂度。对 16K 参数的网络来说太奢侈。

简化的"时间注意力"（TA, Time Attention）：用一个全连接层学出每个时刻的 attention weight，复杂度 O(T)。

**TRA 的创新**：把全连接层换成 GRU，让 attention weight 的计算**带有时间记忆**——这一帧的重要性不仅取决于这一帧本身，还取决于历史帧的状态。

### 3.3 看代码

[gtcrn.py:77](../../third_party/gtcrn/gtcrn.py#L77)：

```python
class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        # 步骤 1: 沿频率维聚合能量
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        
        # 步骤 2: 用 GRU 处理时间序列
        at = self.att_gru(zt.transpose(1,2))[0]  # (B,T,2C)
        
        # 步骤 3: FC 把维度恢复
        at = self.att_fc(at).transpose(1,2)  # (B,C,T)
        
        # 步骤 4: Sigmoid 得到 0-1 的注意力权重
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        # 步骤 5: 应用到原特征
        return x * At
```

### 3.4 逐步拆解 TRA 的 5 个步骤

**步骤 1：能量聚合 `zt = mean(x²)` 沿频率维**

```python
zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T,F) → (B,C,T)
```

为什么用平方再平均？**这是"短时能量"的定义**——`E_t = mean(|X_t|²)` 是 DSP 里描述帧能量的标准方式。

- 平方让正负值都变成正贡献
- 平均消除频率维度，得到"这一帧的总能量"

输出是 `(B, C, T)`——每个通道、每一帧的能量。

> 注意这里没有用 `nn.AvgPool` 或者 `nn.AdaptiveAvgPool`，而是手动 `x.pow(2).mean(-1)`。这是因为 ONNX 和很多嵌入式推理框架对自定义运算的支持更好。

**步骤 2：GRU 处理时间序列**

```python
at = self.att_gru(zt.transpose(1,2))[0]  # (B,T,C) → (B,T,2C)
```

把 `(B, C, T)` 转成 `(B, T, C)`，喂给 GRU。GRU 输入维度 = C = 8 (`in_channels//2`)，输出维度 = 2C = 16。

为什么输出要翻倍？为了**增加表达能力**。注意 GRU 的隐藏维度越大，能记住的状态越复杂。但代价是参数翻倍。

**步骤 3：FC 降维回 C**

```python
at = self.att_fc(at).transpose(1,2)  # (B,T,2C) → (B,C,T)
```

把 GRU 输出从 2C 降回 C，对应原始的通道数。

**步骤 4：Sigmoid 得到 0-1 权重**

```python
at = self.att_act(at)  # 每个值都在 (0, 1) 之间
At = at[..., None]     # 加一个频率维度 (B,C,T,1)，便于广播
```

Sigmoid 把任意实数压到 (0, 1)，作为"乘性权重"。值越接近 1，这一帧越重要；越接近 0，这一帧越被抑制。

**步骤 5：广播相乘应用到原特征**

```python
return x * At
```

`x` 的 shape 是 `(B, C, T, F)`，`At` 的 shape 是 `(B, C, T, 1)`。乘法时 `At` 沿 F 维度广播，**同一帧的所有频带共享同一个权重**。

这是一个**强假设**：注意力在频率维度上是"全局共享"的——也就是说，如果第 50 帧被认为是"重要帧"，那么 f=0 和 f=128 都被同样地重视。

**这个假设合不合理？** 对降噪任务来说基本合理，因为"语音活跃 vs 静音"是一个全频带的属性。但对更精细的任务（比如音乐源分离），可能要做"时频联合注意力"。GTCRN 为了省算力没这么做。

### 3.5 TRA 和 SE-Net 的关系

如果你看过 CV 里的 Squeeze-and-Excitation Network（SE-Net），会发现 TRA 和它非常像：

| SE-Net (CV) | TRA (GTCRN) |
|:----:|:----:|
| 沿 H,W 全局 pooling 得到 channel descriptor | 沿 F 全局 pooling 得到 (C,T) descriptor |
| FC → ReLU → FC → Sigmoid | GRU → FC → Sigmoid |
| 输出 channel attention weight | 输出 (channel, time) attention weight |
| 沿 H,W 广播相乘 | 沿 F 广播相乘 |

**TRA = 把 SE-Net 改成时序版本**——把 FC 换成 GRU（增加时间记忆），把广播维度从空间变成频率。

这是一个非常典型的"借鉴 CV 思想 + 改造为语音任务"的设计。

### 3.6 为什么 GRU 而不是 LSTM？

GRU 比 LSTM 少一个门，参数更少。对于 16 通道的小网络，GRU 已经足够，没必要用 LSTM。

PyTorch 的 GRU 实现还有一个好处：**和很多嵌入式推理引擎（如 CMSIS-NN）天然兼容**。

### 3.7 TRA 的算力账

输入 `(B=1, C=8, T, F=33)`：

- 能量聚合：算力 = T × F × C = T × 264 MACs（很少）
- GRU(8→16)：参数 = 3 × (8×16 + 16×16) = 1152，每帧算力 ≈ 768 MACs
- FC(16→8)：参数 = 128，每帧算力 ≈ 128 MACs

**单个 TRA 大约 1.3K 参数，每帧 1K MACs**。

整个网络 6 个 GT-Conv，每个带一个 TRA，所以 TRA 总参数 ≈ 8K（和 Table 1 一致！）。

## 4. SFE + TRA 在 GT-Conv 中的位置

回到 GT-Conv 的代码 [gtcrn.py:139](../../third_party/gtcrn/gtcrn.py#L139)：

```python
def forward(self, x):
    """x: (B, C, T, F)"""
    x1, x2 = torch.chunk(x, chunks=2, dim=1)

    x1 = self.sfe(x1)                                        # ← SFE 在最前
    h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
    h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
    h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
    h1 = self.point_bn2(self.point_conv2(h1))

    h1 = self.tra(h1)                                        # ← TRA 在最后

    x = self.shuffle(h1, x2)
    return x
```

注意位置：

- **SFE 在最前**：因为它的目的是"为 1×1 卷积提供频域上下文"，所以必须在第一个 1×1 之前
- **TRA 在最后**：因为它的目的是"对处理完的特征做时序加权"，所以放在 conv 链路结束后

如果倒过来（TRA 在前、SFE 在后），不仅不合逻辑，效果也会差——TRA 是基于"已经提取的高级特征"做注意力，输入太原始的话注意力学不到东西。

## 5. 一些工程师容易忽略的细节

### 5.1 SFE 的 `padding=(0, (kernel_size-1)//2)`

```python
self.unfold = nn.Unfold(kernel_size=(1,kernel_size), 
                        stride=(1, stride), 
                        padding=(0, (kernel_size-1)//2))
```

对 k=3，padding = 1。**两边都填 padding！** 不是只填一边。

这意味着 SFE **在频率维上是对称的**——没有因果性问题（因为频率维本来就没有因果）。这一点要和 GT-Conv 的时间维填充区分开：

- SFE：频率维**对称**填充
- GT-Conv depthwise：时间维**只填前**

### 5.2 TRA 的 `mean(x.pow(2))` 不是 `x.abs().mean()`

为什么用平方而不是绝对值？两者都能消除符号：

- **平方**：放大大值的影响，符合"能量"的定义，对突发噪声敏感
- **绝对值**：线性放大，对噪声不那么敏感

降噪任务希望对"能量分布"敏感（区分活跃帧和静默帧），所以用平方。

> 一个小测试：你可以试试改成 `x.abs().mean()`，看看 PESQ 会不会下降。我没跑过这个实验，但根据 Sigmoid 之后又被乘回原特征的设计，估计差别不大——但还是有差。

### 5.3 TRA 没有偏置项处理

注意 `nn.GRU` 和 `nn.Linear` 默认都带 bias。这些 bias 也是参数预算的一部分。

如果你想进一步缩参数，可以在初始化时加 `bias=False`——但这通常会影响训练稳定性，不建议在第一版网络里就做。

## 6. 把 GT-Conv + SFE + TRA 看作一个整体

现在我们可以画一个 GT-Conv 完整数据流图（含 SFE 和 TRA）：

```
              输入 x (B, 16, T, F)
                    │
            chunk dim=1 (分两半)
            ┌───────┴───────┐
            ▼               ▼
        x1 (B,8,T,F)    x2 (B,8,T,F)   ──→ identity
            │
        SFE k=3
            ▼
        x1' (B,24,T,F)
            │
        1×1 Conv(24→16) + BN + PReLU
            ▼
        h1 (B,16,T,F)
            │
        因果 pad (左填 pad_size)
            ▼
        depthwise 3×3 (dilated) + BN + PReLU
            ▼
        h1 (B,16,T,F)
            │
        1×1 Conv(16→8) + BN
            ▼
        h1 (B,8,T,F)
            │
        TRA
            │ ┌─ 能量聚合 (mean square along F)
            │ ├─ GRU(8→16)
            │ ├─ FC(16→8)
            │ ├─ Sigmoid
            │ └─ 广播相乘
            ▼
        h1 (B,8,T,F)
            │
            └───────┬───────┘
                    │
            shuffle (交错合并)
                    ▼
              输出 (B, 16, T, F)
```

整个 GT-Conv 块的**信息流可以理解为**：

1. **拆通道**：节省计算
2. **SFE 注入频域上下文**：为 depthwise 补能力
3. **三明治卷积**：1×1 → depthwise → 1×1，省参数地建模局部时频模式
4. **TRA 应用时序注意力**：让网络聚焦重要帧
5. **Shuffle 合并**：让两路信息交流

## 7. 这一章的设计哲学总结

- **不要追求"通用"，要追求"够用"**：TRA 在频率维度上共享注意力，是个简化，但对 SE 任务够用
- **借鉴成熟模块时要做"翻译"**：SE-Net 来自 CV，TRA 是它的语音域翻译版
- **每个模块解决一个具体瓶颈**：SFE 补 depthwise 的频域视野，TRA 补 grouped 的时间感知
- **算力分配要"精打细算"**：SFE 不带参数但增加 3 倍 conv 算力；TRA 带参数但算力极少。它们恰好互补

下一章我们走到网络末端——**输出阶段的复数掩码 CRM 和混合损失函数**。

---

**上一章**：[05_DPGRNN详解.md](05_DPGRNN详解.md) ｜ **下一章**：[07_输出与损失函数.md](07_输出与损失函数.md)
