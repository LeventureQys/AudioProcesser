# 04｜GT-Conv 详解：ShuffleNetV2 + 时间空洞，省到极致的卷积块

> GT-Conv 是 GTCRN 编码器/解码器的**主力计算单元**。理解它，就理解了 GTCRN 是怎么用极少的参数捕捉时频特征的。

## 1. 先看代码：GT-Conv 长什么样

打开 [gtcrn.py:107](../../third_party/gtcrn/gtcrn.py#L107)：

```python
class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        # 第一个 1x1 卷积：通道升维（in/2 * 3 → hidden）
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        # 中间的 depthwise + dilated 卷积
        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        # 第二个 1x1 卷积：通道降维（hidden → in/2）
        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)  # 时序注意力，第 06 章详解

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, 'b c g t f -> b (c g) t f')
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        # 分支拆分
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        
        # 分支 1：核心处理
        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])  # 因果填充
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)

        # 分支 2：恒等映射
        # ...

        # 合并 + Channel Shuffle
        x = self.shuffle(h1, x2)
        return x
```

这是整篇网络里最复杂的模块，里面集成了 **4 个核心思想**：

1. **ShuffleNetV2 单元**：通道分两半，一半计算一半 identity
2. **Depth-wise Separable Convolution**：把 3×3 拆成"通道独立卷积 + 1×1 通道混合"
3. **Dilated Convolution**：用空洞卷积扩大时间感受野
4. **TRA**：时序注意力（下一章详讲）
5. **SFE + 因果填充**：子带特征 + 因果约束（部分细节）

我们一个一个拆。

## 2. ShuffleNetV2 单元：为什么"一半算一半不算"

### 2.1 设计动机：参数和算力的"对半省"

标准 2D 卷积 `Conv2d(C, C, k=3)` 的参数量是 **C × C × 9**。当 C=16 时，参数 = 2304。如果直接做完整卷积，每个 GT-Conv 块就 2K+ 参数，整个网络 5 个 GT-Conv 就 10K，再加 DPGRNN 和编解码器，肯定超 50K。

**ShuffleNetV2 的关键观察**：在轻量化 CNN 里，**不需要所有通道都过一遍卷积**。把通道分成两半，只让一半参与计算，另一半直接传过去，最后做一次 channel shuffle 让两半信息混合——这样：

- 计算量减半（只算一半通道）
- 参数减半（卷积层输入输出通道都减半）
- 信息不损失（shuffle 让两半交换信息）

### 2.2 代码对照

```python
# gtcrn.py:141 - 拆分
x1, x2 = torch.chunk(x, chunks=2, dim=1)  # 通道维一分为二

# 中间所有卷积都基于 in_channels//2，参数量减半

# gtcrn.py:151 - 合并
x = self.shuffle(h1, x2)
```

注意 `self.shuffle` 不是简单 `concat([h1, x2])`！它做了一次"交错"：

```python
def shuffle(self, x1, x2):
    """x1, x2: (B,C,T,F)"""
    x = torch.stack([x1, x2], dim=1)           # (B, 2, C, T, F)
    x = x.transpose(1, 2).contiguous()          # (B, C, 2, T, F)
    x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B, 2C, T, F)
    return x
```

也就是说，输出通道排列是 `[x1[0], x2[0], x1[1], x2[1], ...]` 而不是 `[x1[0], x1[1], ..., x2[0], x2[1], ...]`。

**为什么要交错？** 因为下一层会再 `chunk(2, dim=1)`——交错排列后，**两个分支被强制混合**，下一次拆分时拿到的"前一半"是 x1 和 x2 的混合，"后一半"也是 x1 和 x2 的混合。这就是 channel shuffle 的本意：**强制信息在层与层之间交换**。

> 没有 shuffle 的话，x1 永远只在自己那半边传，x2 永远不变，相当于两个独立的子网络——那就完全不需要 shuffle 了。

## 3. Depth-wise Separable Convolution：把 3×3 拆成两步

GT-Conv 的核心是 "1×1 → 3×3(depthwise) → 1×1" 的三明治结构。这是 **MobileNet 系列发明的 depthwise separable conv** 的标准形式。

### 3.1 标准卷积 vs Depthwise Separable

| 操作 | 标准 Conv(C_in, C_out, k×k) | Depthwise Separable |
|:----:|:----:|:----:|
| 步骤 1 | — | 1×1 conv: C_in → hidden |
| 步骤 2 | k×k conv: C_in → C_out | k×k depthwise conv: hidden → hidden (groups=hidden) |
| 步骤 3 | — | 1×1 conv: hidden → C_out |
| 参数量 | C_in × C_out × k² | C_in × hidden + hidden × k² + hidden × C_out |
| 算力 | 同上 | 大幅降低（depthwise 只算 hidden×k² 而不是 hidden²×k²） |

**关键：depthwise conv 的 `groups=hidden_channels`**，意味着每个通道独立做 3×3 卷积，通道之间不混合。混合的事情交给前后两个 1×1 conv。

代码里：

```python
# gtcrn.py:121-123
self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=hidden_channels)  # 关键
```

`groups=hidden_channels` 把卷积分成 `hidden_channels` 组，每组 1 个通道——这就是 depthwise。

### 3.2 GTCRN 里的具体数字

`hidden_channels=16`，`in_channels=16`：

- 第一个 1×1 conv：输入 `(in/2 * 3) = 24` 维（SFE 后的 8×3），输出 16 维 → 参数 = 24 × 16 = **384**
- depthwise 3×3：参数 = 16 × 9 = **144**
- 第二个 1×1 conv：16 → 8 → 参数 = 16 × 8 = **128**
- 加上 BN/PReLU 等少量参数

**单个 GT-Conv 块大约 1K 参数**。整个网络 6 个 GT-Conv（编码器 3 个 + 解码器 3 个）才 ~6K 参数。

对比一下：如果用标准 3×3 卷积 `Conv2d(16, 16, 3)`，参数就是 16 × 16 × 9 = 2304，6 个就是 ~14K，**省了一半多**。

## 4. Dilated Convolution：用空洞扩大时间感受野

代码里有个有趣的细节：三层 GT-Conv 的 `dilation` 不同：

```python
# gtcrn.py:234-236
GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
```

dilation 分别是 **1, 2, 5**，**而且只在时间维（第一个数）有 dilation，频率维 dilation 始终是 1**。

### 4.1 什么是 dilation？

空洞卷积（dilated / atrous convolution）就是在卷积核之间插"空格"：

```
普通卷积 kernel=3:      [w0  w1  w2]            感受野 = 3
dilation=2:             [w0  __  w1  __  w2]    感受野 = 5
dilation=5:             [w0 ____ w1 ____ w2]    感受野 = 11
```

**参数量不变（还是 3 个权重），但感受野扩大**。

### 4.2 GTCRN 的时间感受野如何累积

三层 GT-Conv 串联，每层 kernel=3，dilation 分别 1/2/5：

- 第 1 层：感受野 = 3 帧（48ms）
- 第 2 层：每个输出依赖 3 个第 1 层的输出，但因为 dilation=2，跨度变成 5 帧。累积感受野 ≈ 3 + (3-1)×2 = 7 帧（112ms）
- 第 3 层：累积感受野 ≈ 7 + (3-1)×5 = 17 帧 ≈ 270ms

**网络的卷积部分能看到 270ms 的历史**——这覆盖了一个音节的典型长度，对捕捉短时的瞬态噪声（爆破音、咔哒声）足够了。

更长时间的依赖（一句话内的语义结构）交给后面的 DPGRNN。

### 4.3 为什么频率维不用 dilation？

注意 `dilation=(2, 1)` 的第二个 1 是频率维——保持 dilation=1。

原因：**频率维已经被 ERB 压缩 + Encoder 下采样到 33 维**，频率范围很窄；而频率维上的"局部性"很强（共振峰、谐波的邻近相关性），用 dilation 反而会跳过有用的信息。

时间维则不同——语音的时间相关性可以延续到几百毫秒，需要 dilation 来扩展。

> **频率维要密，时间维要宽**——这是 SE 卷积设计的又一条不成文规则。

## 5. 因果填充：保证流式部署

这一行非常关键：

```python
# gtcrn.py:145
h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
```

`pad_size = (kernel_size[0] - 1) * dilation[0]`，对 kernel=3、dilation=5 来说 pad_size = 10。

填充的位置是 `[0, 0, self.pad_size, 0]`——分别对应（频左、频右、时间前、时间后）。**只在"时间前"填充，"时间后"不填**！

### 为什么这样做？

考虑 kernel=3 的 1D 卷积：

```
不填充时，输出 t 依赖于输入 t-1, t, t+1
也就是说，输出在 t 时刻 "看到了未来"（t+1）
```

**这就破坏了因果性**——而 SE 模型要做实时降噪，绝对不能用未来的输入。

```
只填充时间前面：
       输入：[pad][pad][x0][x1][x2][x3]...
       输出 t=0 依赖于 [pad][pad][x0] = 当前 + 历史
       输出 t=1 依赖于 [pad][x0][x1] = 当前 + 历史
       ...
```

这样每个输出只依赖于当前和过去，完美保证因果性。

代码 [gtcrn.py:330](../../third_party/gtcrn/gtcrn.py#L330) 有一段**因果性验证测试**：

```python
"""causality check"""
a = torch.randn(1, 16000)
b = torch.randn(1, 16000)
c = torch.randn(1, 16000)
x1 = torch.cat([a, b], dim=1)   # 前 1 秒是 a，后 1 秒是 b
x2 = torch.cat([a, c], dim=1)   # 前 1 秒是 a，后 1 秒是 c

# 因果模型：前 1 秒的输出对 x1 和 x2 应该完全一样（因为前 1 秒输入相同）
y1 = model(x1)
y2 = model(x2)

print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())  # 应该 ≈ 0
print((y1[16000:] - y2[16000:]).abs().max())              # 应该 != 0
```

这个测试**就是用来验证"网络只看过去、不偷看未来"**的。如果有任何一层卷积或 RNN 双向了，前半段输出就会不一致。

> 工程上，**写完网络后立刻跑因果性测试**是个非常好的习惯。我见过太多论文里"声称是 causal 的"但实际上某个 BN 或 attention 偷偷用了 future frame 的情况。

## 6. PReLU 而不是 ReLU：一个小但重要的选择

注意所有激活函数都是 `nn.PReLU()`，最后输出层才用 `nn.Tanh()`：

```python
self.point_act = nn.PReLU()
self.depth_act = nn.PReLU()
```

**PReLU = Parametric ReLU**：

```
PReLU(x) = max(0, x) + α * min(0, x)
```

其中 `α` 是可学习的参数（每个通道一个）。当 α=0 时就是 ReLU；当 α=0.01 时就是 Leaky ReLU。

**为什么 SE 任务偏好 PReLU？**

1. **负值信息不丢失**：语音 STFT 的实部/虚部是有正负的，纯 ReLU 会把所有负值清零，丢一半信息。
2. **比固定的 LeakyReLU 更灵活**：每个通道自己学一个 α，可以根据特征类型自适应。
3. **参数代价极小**：每个通道就 1 个参数，对 16 通道的 GTCRN 来说，总共增加几十个参数，可以忽略不计。

**为什么最后一层用 tanh？**

因为最后一层输出是 CRM（复数比例掩码），数学上 CRM 的实部/虚部范围理论上是 (-∞, ∞)，但实践中限制在 (-1, 1) 之间会让训练更稳定，避免输出爆炸。

## 7. 编码器/解码器的对称性

GTCRN 的解码器是编码器的**镜像**：

| 编码器 | 解码器 |
|:----:|:----:|
| Conv (1,5) stride=(1,2) → 频率 257→129 | DeConv (1,5) stride=(1,2) → 频率 129→257 |
| Conv (1,5) stride=(1,2), g=2 → 频率 65→33 | DeConv (1,5) stride=(1,2), g=2 → 频率 65→129 |
| GT-Conv dilation=1 | GT-DeConv dilation=1 |
| GT-Conv dilation=2 | GT-DeConv dilation=2 |
| GT-Conv dilation=5 | GT-DeConv dilation=5 |

**注意 dilation 顺序**：编码器里 dilation=1/2/5，解码器里也是 5/2/1（如果按 forward 顺序）。让 dilation 最大的 GT-Conv 处于编码器最深处和解码器最浅处——也就是**最靠近瓶颈的位置**。

这是因为**网络越深，特征越抽象，时间依赖越强**。在深层用 dilation=5 看 17 帧的历史，在浅层用 dilation=1 看 3 帧的历史，符合"特征抽象层级"的规律。

## 8. 看看实际的算力账：单个 GT-Conv 多少 MACs？

我们以编码器最后一个 GT-Conv（dilation=5，输入 (16, T, 33)）为例：

- 输入：(16, T, 33)
- chunk → (8, T, 33)
- SFE k=3 → (24, T, 33)
- 1×1 conv (24, 16): 算力 ≈ T × 33 × 24 × 16 = T × 12,672
- depthwise 3×3 conv (16 groups): 算力 ≈ T × 33 × 16 × 9 = T × 4,752
- 1×1 conv (16, 8): 算力 ≈ T × 33 × 16 × 8 = T × 4,224
- TRA：少量（下一章详细分析）

**单个 GT-Conv 大约 22K MACs/帧**。1 秒 63 帧，约 1.4 MMACs/秒。

整个网络 6 个 GT-Conv 加起来 ≈ 8 MMACs/秒，**占了总算力（33 MMACs/秒）的 25%**。剩下的算力主要在 DPGRNN（下一章）和 1×1 卷积。

## 9. 总结：GT-Conv 的设计哲学

把这一章的关键收束成几条工程信条：

1. **通道分两半，一半计算一半 identity** —— 参数减半，性能不显著下降（前提：channel shuffle）
2. **3×3 卷积 = 1×1 + depthwise + 1×1** —— Depthwise separable 是低算力 CNN 的标配
3. **时间维用 dilation 扩感受野，频率维保持 dense** —— 物理意义匹配
4. **只在时间前侧填充，保因果** —— 不要默认对称填充
5. **PReLU + Tanh 是 SE 任务的"默认激活方案"** —— 不要随便换 ReLU
6. **dilation 越大越靠近瓶颈** —— 抽象特征对应长时依赖

下一章我们看 **DPGRNN**，看 RNN 怎么在这个网络里发挥作用。

---

**上一章**：[03_输入处理与ERB.md](03_输入处理与ERB.md) ｜ **下一章**：[05_DPGRNN详解.md](05_DPGRNN详解.md)
