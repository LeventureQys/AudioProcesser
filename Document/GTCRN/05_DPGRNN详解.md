# 05｜DPGRNN 详解：双路径 + 分组，RNN 的极致瘦身

> DPGRNN 是 GTCRN 的"灵魂"。如果说 GT-Conv 解决的是局部模式建模，那 DPGRNN 解决的就是**全局时频建模**——也是这个网络真正能"听懂"语音的关键。

## 1. 为什么需要 RNN？卷积不够吗？

我们在第 04 章算过：3 层 GT-Conv 串联的感受野大约 17 帧 ≈ 270ms。这够吗？

对于**短时瞬态噪声**（爆破音、咔哒声）够了。但对于：

- **稳态噪声**（空调、风扇）需要看很长时间才能确认"这是稳定背景"
- **长元音/长辅音的连续判断**（"啊—————"）需要至少 500ms 的上下文
- **语义级噪声**（背景人声）甚至需要几秒的上下文才能区分前景说话人

这些都需要**理论上无限远的时间依赖建模**。CNN 想做到这点只能堆几十层（每层增加 dilation），但这对 23.7K 参数的网络来说不现实。

**RNN 天然适合做这种长程依赖**——它的隐藏状态可以传递任意远，只要训练得当。

## 2. 第一个问题：标准 RNN 在频谱图上怎么用？

假设我们有一个 (B, C, T, F) 的特征图。一个 RNN 应该沿着哪个维度跑？

**选项 A：沿时间维跑（标准用法）**

把 (B, C, T, F) 重塑成 (B*F, T, C)，让 RNN 沿 T 跑。每个频点都有独立的时间序列，RNN 建模"这个频点随时间的演化"。

**缺点**：忽略了频点之间的关系。但语音的谐波结构、共振峰是横跨频域的——只看一个频点的时间序列，永远看不到"这是谐波结构"这个信息。

**选项 B：沿频率维跑**

把 (B, C, T, F) 重塑成 (B*T, F, C)，让 RNN 沿 F 跑。每一帧都被 RNN 扫一遍频率，建模"这一帧的频谱形态"。

**缺点**：完全没有时间依赖。语音的连续性、稳态噪声的持续性都看不到。

**选项 C：同时跑两个 RNN**

A 和 B 都跑！一个建模时间依赖，一个建模频谱结构。**这就是 Dual-Path RNN (DPRNN) 的核心思想。**

## 3. DPRNN 的工作流程

DPRNN 原本是 Luo et al. (2020) 提出来做时域单声道语音分离的，DPCRN 把它搬到时频域。GTCRN 进一步在每个 RNN 上加 grouped 优化。

打开 [gtcrn.py:186](../../third_party/gtcrn/gtcrn.py#L186) 看 `DPGRNN`：

```python
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size       # = 16 (通道数)
        self.width = width                 # = 33 (频率维度)
        self.hidden_size = hidden_size     # = 16

        # Intra-frame RNN: 双向 GRU
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter-frame RNN: 单向 GRU（保证因果）
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
```

两个 RNN，配两个 FC 和两个 LayerNorm。

### 3.1 Intra-frame RNN：建模"一帧的频谱形态"

```python
# gtcrn.py:204-211
## Intra RNN
x = x.permute(0, 2, 3, 1)                                 # (B,T,F,C)
intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T, F, C)
intra_x = self.intra_rnn(intra_x)[0]                      # (B*T, F, C)
intra_x = self.intra_fc(intra_x)
intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)  # (B, T, F, C)
intra_x = self.intra_ln(intra_x)
intra_out = torch.add(x, intra_x)                         # 残差连接
```

关键操作：

1. **把 T 维度合并到 batch**：`(B, T, F, C) → (B*T, F, C)`。每一帧被当成一个独立的"序列"，序列长度 = F = 33。
2. **沿 F 方向跑 RNN**：RNN 看到 f=0 处的频点特征，更新隐藏状态，看到 f=1，再更新……跑完 33 步。
3. **双向 RNN**：因为是"帧内"——同一帧之内，f=10 处的信息可以参考 f=20 处的信息，反之亦然。**频域内没有因果性的概念**（你不能说"高频是未来"），所以可以双向。
4. **残差连接**：`intra_out = x + intra_x`。让 RNN 学"相对原特征的修正"，不要从零重建。

### 3.2 Inter-frame RNN：建模"跨帧的时间依赖"

```python
# gtcrn.py:213-221
## Inter RNN
x = intra_out.permute(0,2,1,3)                             # (B,F,T,C)
inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*F, T, C)
inter_x = self.inter_rnn(inter_x)[0]                       # (B*F, T, C)
inter_x = self.inter_fc(inter_x)
inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)  # (B, F, T, C)
inter_x = inter_x.permute(0,2,1,3)                         # (B, T, F, C)
inter_x = self.inter_ln(inter_x)
inter_out = torch.add(intra_out, inter_x)
```

1. **把 F 维度合并到 batch**：`(B, F, T, C)`。每个频点被当成一个独立的"时间序列"。
2. **沿 T 方向跑 RNN**：对每个频点，RNN 看到 t=0 的特征，更新隐藏状态，看到 t=1……
3. **单向 RNN**：因为有因果性！实时降噪不能看未来。
4. **残差连接**：`inter_out = intra_out + inter_x`。

### 3.3 一个非常关键的观察：两路 RNN 的"非对称"

注意 intra 用 **双向** GRU，inter 用 **单向** GRU。这是 GTCRN 保证因果性的关键。

- **频率方向可以双向**：同一帧内，所有 33 个频点都是同时刻获得的，所以 f=20 可以看 f=10 也可以看 f=30，**不破坏时序因果**。
- **时间方向必须单向**：不能用未来帧。

这种细致的"哪一维可以双向、哪一维必须单向"的考量，是 SE 模型设计里很容易踩坑的地方。看 DCCRN、CRN 等论文，很多都因为某个 BN 或者某个全局 pooling 暗中破坏了因果性。

## 4. 第二个问题：GRNN 是什么？为什么要 "Grouped"？

我们看 [gtcrn.py:156](../../third_party/gtcrn/gtcrn.py#L156) 的 `GRNN`：

```python
class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # 关键：两个独立的 GRU，每个用一半的输入/隐藏维度
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        # ...
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)           # 输入切两半
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)           # 隐藏状态切两半
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)                     # 输出拼起来
        h = torch.cat([h1, h2], dim=-1)
        return y, h
```

### 4.1 标准 GRU 的参数量

`nn.GRU(input_size, hidden_size)` 的参数量是 **3 × (input_size × hidden_size + hidden_size²)**（GRU 有 3 个门：reset、update、new；公式简化版）。

当 `input_size = hidden_size = 16` 时：参数 = 3 × (16×16 + 16×16) = **1536**。

如果 hidden_size = 32：参数 = 3 × (32×32 + 32×32) = **6144**。

参数量是 hidden_size 的**平方**——这就是为什么 RNN 的隐藏维度不能开太大。

### 4.2 Grouped RNN 的瘦身效果

GRNN 把一个 hidden_size=16 的 GRU 拆成两个 hidden_size=8 的 GRU：

- 单个小 GRU：3 × (8×8 + 8×8) = 384
- 两个加起来：768

**对比一下原来的 1536**，**省了一半参数**。

为什么省？因为标准 GRU 里 "input → hidden" 这个矩阵 (16×16) 假设了**所有输入都和所有隐藏维度有交互**。但实际上很多交互是冗余的——拆成 group 后，每个 group 只在自己内部交互，**等价于在权重矩阵上加了 block-diagonal 约束**：

```
原标准 GRU 权重 (16×16):                Grouped GRU 权重:
[● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●]      [● ● ● ● ● ● ● ● . . . . . . . .]
[● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●]      [● ● ● ● ● ● ● ● . . . . . . . .]
[● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●]      [● ● ● ● ● ● ● ● . . . . . . . .]
...                                     ...
                                        [. . . . . . . . ● ● ● ● ● ● ● ●]
                                        [. . . . . . . . ● ● ● ● ● ● ● ●]
```

灰色"."表示置零的权重。

### 4.3 这样做的代价是什么？

代价是 **group 之间没有信息交流**。`y1` 永远只依赖 `x1` 和 `h1`，`y2` 永远只依赖 `x2` 和 `h2`——两组隐藏状态独立演化。

那作者怎么补这个洞？

```python
# gtcrn.py:194-195
self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
self.intra_fc = nn.Linear(hidden_size, hidden_size)
```

**用一个 `nn.Linear(hidden_size, hidden_size)` 在 GRNN 之后做信息交换**！

这个 FC 层是全连接的（不是 grouped），所以它能让 group 1 和 group 2 的信息混起来。这就是论文里说的"implicit feature rearrangement"——本来 ShuffleNet 风格的 GRNN 会用一个显式的 "shuffle" 操作（不可学习），GTCRN 改成"FC 层自己学怎么 shuffle"，这样信息混合方式是数据驱动的。

> 这也是为什么作者在 README 里特别提到："The explicit feature rearrangement layer in the grouped RNN ... can result in an unstreamable model. Therefore, we discard it and implicitly achieve feature rearrangement through the following FC layer in the DPGRNN."
>
> **显式 shuffle 在流式部署时有问题**（涉及非标准 reshape），所以用 FC 替代。这是一个工程驱动的设计调整。

## 5. LayerNorm 而不是 BatchNorm：为什么？

注意 DPGRNN 用的是 `nn.LayerNorm`，而 GT-Conv 用的是 `nn.BatchNorm2d`。这不是随便选的。

### 5.1 BatchNorm 的问题

BatchNorm 在训练时统计 **每个 batch 内** 当前层激活的均值/方差。在 CNN 里很 OK——空间和 batch 维一起统计，样本量足够大。

但 RNN 的特性是**时间维度上每一步状态都不一样**——如果用 BN 沿时间维归一化，会破坏 RNN 的"状态传递"。

而且 SE 模型最终要做流式推理，**流式时 batch size = 1**——BN 在 inference 阶段需要全局统计量（running mean / var），如果训练时和推理时的统计量分布有 mismatch，会出问题。

### 5.2 LayerNorm 的好处

LayerNorm 沿 **特征维度** 做归一化，每一帧、每一个频点独立归一化。

- 不依赖 batch size
- 不破坏时间维度
- 流式部署天然兼容

代码里：

```python
self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
```

shape 是 `(width, hidden_size) = (33, 16)`——也就是对每一帧的 33 × 16 = 528 个特征做归一化。

> **经验法则**：CNN 用 BN，RNN/Transformer 用 LN。GTCRN 在两类层之间分别用合适的归一化，是细致的工程考量。

## 6. 看看实际算账：DPGRNN 多少 MACs？

我们以一层 DPGRNN 为例，输入 `(B=1, C=16, T, F=33)`：

### Intra RNN（双向 GRU）

- 输入序列长 33，每步输入维度 16，hidden_size = 8（双向所以两边各 8）
- GRNN 拆 2 组：每组 input=8, hidden=4，**实际是 2 × 双向 = 4 个 GRU**，每个参数 3 × (8×4 + 4×4) = 144
- 总参数 ≈ 576，加上 input bias 等约 700
- 每帧算力 ≈ 33 步 × 144 MACs × 4 GRU = 19,000 MACs

### Inter RNN（单向 GRU）

- 输入序列长 T，每步输入维度 16，hidden_size = 16（单向）
- GRNN 拆 2 组：每组 input=8, hidden=8
- 参数 ≈ 768
- 每帧算力（每个频点，但所有频点共享 RNN 权重，相当于 33 次重复）≈ 33 × 384 = 12,672 MACs

### FC 层

- `nn.Linear(16, 16)`：256 参数，每帧 33 频点 × 256 = 8,448 MACs

### LayerNorm

- 算力很小，忽略

**单层 DPGRNN 大约 1.5K 参数，每帧 40K MACs**。

两层 DPGRNN 加起来约 **3K 参数，2.5 MMACs/秒**。

加上编解码器的 GT-Conv（~6K 参数 + 8 MMACs/秒），整个网络的主体计算就有 ~20 MMACs/秒，剩下的 13 MMACs/秒在 1×1 卷积、SFE、ERB 等辅助操作上。

## 7. 一个直观的"为什么 DPRNN 有效"的解释

我们用一个具体例子说明 DPRNN 怎么帮到降噪：

**场景**：一个被空调白噪声污染的"啊—————"长元音

**Intra-frame RNN 看到的**：

```
某一时刻 t=50 帧的频谱（33 个 ERB 频带）：
f=0 (低频)：能量高（基频）
f=1-5：能量阶梯下降（谐波）
f=6-15：能量低（谐波之间）
f=16-32 (高频)：相对均匀（白噪声）
```

Intra RNN 沿 f 跑一遍，能学到"有规律的能量分布 = 语音，平坦分布 = 噪声"。它把这个**频谱模式信息**编码到隐藏状态里。

**Inter-frame RNN 看到的**：

```
某个频点 f=2 在 t=0..200 帧的演化：
t=0-10：能量低（无语音）
t=11-100：能量持续高（元音持续）
t=101-200：能量低（元音结束）
```

Inter RNN 沿 t 跑，能学到"能量持续 90 帧高 = 语音，能量平稳持续 200 帧 = 稳态噪声"。它把这个**时间模式信息**编码到隐藏状态里。

**两者结合**：网络能同时识别"频谱上是元音模式 + 时间上是持续 100 帧的稳定激活" = 这是一个语音元音；而"频谱上是平坦分布 + 时间上恒定" = 这是稳态噪声。

**这就是为什么单路 RNN 不够、必须双路**：单维度的信息不足以区分语音和噪声。

## 8. 对比：单路 RNN vs DPRNN

| 方案 | 参数 | 表达能力 | 备注 |
|:----:|:----:|:----:|:----:|
| 单 GRU 沿 T，hidden=64 | ~25K | 强（大隐藏维） | 太大 |
| 单 GRU 沿 T，hidden=16 | ~1.5K | 弱（看不到频谱结构） | 性能差 |
| **DPGRNN，hidden=16** | **~3K** | **强（双路覆盖）** | **GTCRN 的选择** |

**3K 参数达到了 25K 单路 GRU 的效果**——这就是结构创新的力量。

## 9. 设计哲学：用结构换参数

DPGRNN 这个模块完美演绎了一个轻量化网络设计的核心思想：

> **不要靠堆参数提升表达能力，靠结构创新让有限的参数做更多事。**

类似的思想在 CNN 里就是 ResNet（残差结构提升训练稳定性）、DenseNet（密集连接复用特征）、ShuffleNet（分组 + 重排）；在 RNN 里就是 DPRNN、Group RNN、Transformer-XL 等。

**作为工程师，你应该养成的习惯**：当看到一个新模型用了某个特殊结构时，不要只问"这个结构是什么"，要问 **"它替代的是什么？省了什么代价？补了什么洞？"**。

## 10. 小结

- **DPRNN = Intra-frame + Inter-frame 双路径** —— 一路看频谱，一路看时间
- **Grouped RNN** —— 把大 GRU 拆成多个小 GRU，参数减半，用 FC 补信息交流
- **Intra 双向，Inter 单向** —— 因果性的精细控制
- **LayerNorm 而不是 BatchNorm** —— RNN 的归一化选择
- **3K 参数搞定 25K 参数的事** —— 结构创新换计算

下一章我们看 GTCRN 的两个"点睛"模块：**SFE 和 TRA**，它们不是骨架，但是性能从能用到好用的关键。

---

**上一章**：[04_GT-Conv详解.md](04_GT-Conv详解.md) ｜ **下一章**：[06_SFE与TRA详解.md](06_SFE与TRA详解.md)
