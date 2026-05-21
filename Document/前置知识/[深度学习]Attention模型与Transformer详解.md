# Attention模型与Transformer详解

# 前言

在上一篇文章中，我们提到了Conformer的核心组件之一就是Self-Attention。如果你对注意力机制和Transformer还不太熟悉，这篇文章就是为你准备的。我们会从最直觉的动机出发，逐步推导到完整的Transformer架构——尽量不使用晦涩的术语，而是用你能感知到的逻辑来串联每一个设计决策。

# 第一部分：为什么需要Attention

## 一个直觉问题

假设你现在要翻译一句话：

> "The cat, which was sitting on the mat, **is** hungry."

当你要翻译动词"is"时，你需要知道它的主语是"cat"，而不是离它更近的"mat"。这是一个**长距离依赖**问题——"cat"和"is"之间隔着好几个词，但它们的语法关系是最紧密的。

在RNN/LSTM时代，模型试图用一个固定长度的**隐藏状态向量**来"记忆"所有过去的信息。这本质上是一个压缩操作：把整句话的上下文压缩成一个固定大小的向量。问题是：

- 压缩一定会丢失信息
- 距离越远，压缩损失越大
- 梯度在反向传播时逐时间步衰减（梯度消失）

## Attention的核心直觉

与其把所有信息压缩到一个向量里，不如**让模型在需要时，自己去"查找"相关的信息**。

这个直觉非常像数据库查询：你有一个查询（Query），想在一堆键值对（Key-Value pairs）中找到最相关的值。Attention机制做的事情正是如此：

1. 你有一个**Query**（查询）："当前我需要什么信息？"
2. 你有一组**Key**（键）：每个输入位置提供一个索引/"标签"
3. 你有一组**Value**（值）：每个输入位置提供实际的内容
4. 你通过比较Query和每个Key的相似度，得到一组权重
5. 用这些权重对Value做加权求和，得到输出

用大白话说：**Attention允许每个输出位置"关注"所有输入位置，并根据相关性决定从每个位置获取多少信息。**

# 第二部分：Scaled Dot-Product Attention

## 一步一步推导

这是最基本的Attention形式，也是最常用的。

**输入：** 一个序列的表示矩阵 $X \in \mathbb{R}^{T \times d}$，$T$ 是序列长度，$d$ 是特征维度。

**Step 1：生成 Q、K、V**

通过三个不同的线性变换，将输入投影到"查询空间"、"键空间"和"值空间"：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$（通常 $d_k = d$）。

**为什么需要三个不同的投影？** 因为"想问什么"（Query）、"有什么可被问"（Key）、"实际内容是什么"（Value）是三类不同的事。分开投影给模型更大的灵活性。

**Step 2：计算相关性分数**

计算每个Query与每个Key的相似度（点积）：

$$S = Q K^T \in \mathbb{R}^{T \times T}$$

其中 $S_{ij}$ 表示第 $i$ 个输出位置对第 $j$ 个输入位置的"关注程度"。

为什么用点积？两个向量的点积本质上是在衡量它们方向的一致性。当两个向量"方向一致"时，它们语义相近，应该获得高关注。

**Step 3：缩放（Scaling）**

$$S' = \frac{S}{\sqrt{d_k}}$$

**为什么除以 $\sqrt{d_k}$？** 这是一个经验性的但极其重要的设计。当 $d_k$ 很大时，点积的数值会非常大，送入softmax后会落入饱和区，梯度接近零。除以 $\sqrt{d_k}$ 相当于对点积做方差归一化（假设Q和K的元素独立同分布，方差为1，则点积的方差为 $d_k$）。

**Step 4：softmax归一化**

$$A = \text{softmax}(S') \quad \text{(沿每一行做softmax)}$$

softmax确保：每一行的权重和为1，且权重集中在少数高相关的Key上。

**Step 5：加权求和**

$$\text{Output} = A V \in \mathbb{R}^{T \times d}$$

每个输出位置得到的是所有Value的加权和，权重由Attention矩阵给出。

## 完整公式

将上述步骤合并，得到Scaled Dot-Product Attention的标准形式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## 用数据流图理解

```
输入 X: (T, d)
    │
    ├── Linear → Q: (T, d_k)
    ├── Linear → K: (T, d_k)
    └── Linear → V: (T, d_k)
         │          │
         │     QK^T / √d_k
         │          │
         │     softmax  → 权重矩阵 A: (T, T)
         │          │
         └──  A × V ──┘
               │
          输出: (T, d_k)
```

## 从信号处理的角度理解

如果你有信号处理的背景，这个操作有一个非常优雅的类比：

$$\text{Output} = \text{softmax}(QK^T / \sqrt{d_k}) \cdot V$$

其中 $\text{softmax}(QK^T / \sqrt{d_k})$ 是一个 $(T \times T)$ 的矩阵。它作用在Value序列上的效果是：**它是一个数据依赖的、非线性的、全局的混合矩阵**。每个输出通道 $j$ 是输入通道 $i$ 的加权混合：

$$\text{Output}_j = \sum_i A_{ji} \cdot V_i$$

这和线性系统中的混合矩阵概念完全一致，只不过这里的混合系数 $A_{ji}$ 不是固定的，而是根据数据内容动态计算的。

# 第三部分：Self-Attention vs Cross-Attention

在理解了基本Attention之后，需要区分两种使用模式：

## Self-Attention（自注意力）

**Q、K、V 来自同一个序列。** 即输入序列的每个位置去"关注"同一个序列的所有其他位置。

这允许序列内部的信息交互——每个词可以看到同一句话中的所有词。这是Transformer的核心机制。

$$\text{SelfAttn}(X) = \text{softmax}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right)(XW_V)$$

## Cross-Attention（交叉注意力）

**Q来自一个序列，K和V来自另一个序列。** 典型场景是Encoder-Decoder结构中的解码器：解码器的Query去"询问"编码器输出的Key和Value。

这实现了两个不同序列之间的信息传递。比如在翻译任务中，每生成一个目标语言词，就去源语言句子中找最相关的信息。

# 第四部分：Multi-Head Attention

## 为什么需要多头

单头Attention的问题在于：它只能捕获一种"关注模式"。但在实际任务中，一个词可能需要从多个维度关注不同的信息。

比如翻译"猫吃了鱼"时：
- 翻译"吃"需要同时关注"猫"（主语-动词关系）
- 也需要关注"鱼"（动词-宾语关系）
- 还可能需要关注时态信息

单头Attention把所有因素混在一起，表达能力受限。多头设计的思路是：**并行运行多个独立的Attention，每个头关注不同的模式，最后拼接起来。**

## 具体操作

```
输入 X: (T, d)
    │
    ├── 给每个头i生成 Q_i, K_i, V_i  (各自投影)
    ├── 每个头独立计算 Attention
    │
    Head_1 = Attention(Q_1, K_1, V_1)  → (T, d_head)
    Head_2 = Attention(Q_2, K_2, V_2)  → (T, d_head)
    ...
    Head_H = Attention(Q_H, K_H, V_H)  → (T, d_head)
    │
    └── Concat[Head_1, ..., Head_H] → (T, H × d_head)
         │
         Linear → (T, d)
```

通常 $d_{head} = d / H$，比如 $d = 512, H = 8, d_{head} = 64$。这样拼接后维度不变，总参数量也和单头Attention相同（因为每个头的Q、K、V投影维度变小了）。

## Multi-Head Attention的完整形式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) W_O$$

$$\text{head}_i = \text{Attention}(QW_{Qi}, KW_{Ki}, VW_{Vi})$$

# 第五部分：Transformer的整体架构

在理解了Attention之后，Transformer的完整架构就水到渠成了。

## 核心认知

> Transformer不做时序处理，它只做信息重排和特征变换。

这是理解Transformer最关键的一点：它没有RNN那样的逐步传播，而是将整个序列**同时处理**。输入序列的每个位置，在Attention层中"一次性"看到所有其他位置。这就是为什么它能并行计算。

## 编码器（Encoder）

```
输入序列 (经过Embedding和位置编码)
    │
    ▼
┌──────────────────────┐
│ N × EncoderLayer     │
│                      │
│  Multi-Head          │
│  Self-Attention   ──残差──→ Add & LayerNorm
│                      │
│  Feed-Forward      ──残差──→ Add & LayerNorm
│                      │
└──────────────────────┘
    │
    ▼
编码输出 (上下文表示)
```

一个EncoderLayer做的事：
1. 输入做LayerNorm → Multi-Head Self-Attention → 加残差 → LayerNorm
2. 上一步的输出做LayerNorm → FFN → 加残差 → LayerNorm

## 解码器（Decoder）

解码器比编码器多一个Cross-Attention层：

```
目标序列 (右移一位，刚开始是<SOS>)
    │
    ▼
┌──────────────────────┐
│ N × DecoderLayer     │
│                      │
│  Masked Multi-Head   │
│  Self-Attention   ──残差──→ Add & LayerNorm
│                      │
│  Multi-Head          │
│  Cross-Attention  ──残差──→ Add & LayerNorm  ← Q来自Decoder, K/V来自Encoder
│                      │
│  Feed-Forward      ──残差──→ Add & LayerNorm
│                      │
└──────────────────────┘
    │
    ▼
Linear → Softmax → 输出概率分布
```

## Masked Self-Attention（因果掩码）

在解码器中，生成第t个词时，不能"偷看"第t+1个及之后的词（因为那些还没生成）。所以需要**掩码**：把Attention矩阵的上三角区域设为 $-\infty$（这样softmax后权重为0）。

```
原始权重矩阵 A:         加上掩码后:
[ a11  a12  a13  a14 ]   [ a11  -∞   -∞   -∞  ]
[ a21  a22  a23  a24 ]   [ a21  a22  -∞   -∞  ]
[ a31  a32  a33  a34 ]   [ a31  a32  a33  -∞  ]
[ a41  a42  a43  a44 ]   [ a41  a42  a43  a44 ]

softmax后权重:
[ 1.0  0    0    0   ]
[ w21  1-w21 0    0   ]
[ w31  w32  1-.. 0   ]
[ w41  w42  w43  w44 ]
```

## Feed-Forward Network (FFN)

每个Attention层之后都跟一个FFN：

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

其中 $W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$。

**为什么需要FFN？** Attention做的是序列中的信息混合（跨位置的线性组合），但缺乏逐位置的非线性变换。FFN是对每个位置独立作用，提供非线性变换和维度扩展-压缩的空间（4倍扩展），增加了模型的表达能力。

从数学角度理解：Attention是**跨位置的混合**，FFN是**逐位置的非线性变换**。两者互补。这种分离是Transformer高效的关键——Attention负责"交流"，FFN负责"思考"。

# 第六部分：位置编码（Positional Encoding）

## 问题的提出

观察Attention公式：$\text{softmax}(QK^T / \sqrt{d_k}) V$

对于任意排列 $\pi$，如果输入序列完全打乱，输出也会被相应地打乱——但Attention权重本身无法区分"谁先谁后"。因为Attention是不包含位置信息的置换等变操作。

这意味着："猫追狗"和"狗追猫"在Attention中本质上没有区别——所有词都在互相看到对方，没有"顺序"的概念。

## 解决方案：加入位置信息

### 正弦位置编码（Original Transformer）

为每个位置 $pos$ 和每个维度 $i$ 计算：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

然后将位置编码直接加到输入Embedding上：$X_{input} = X_{embed} + PE$

**为什么选择正弦/余弦？** 因为存在恒等式：

$$\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$
$$\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$

这意味着对于任意偏移量 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换。这个性质使得模型容易学习**相对位置关系**——"前一个位置"和"当前位置"的关系在不同的绝对位置上是相同的模式。

### 可学习位置编码

直接为每个位置学习一个可训练的嵌入向量。好处是灵活，坏处是无法外推到训练时没见过的序列长度。

### 相对位置编码（用于Conformer/Transformer-XL）

在计算Attention权重时，不定义绝对位置的编码，而是直接建模两个位置之间的**相对距离**。公式变为：

$$A_{ij} = Q_i K_j^T + Q_i R_{i-j}^T$$

其中 $R$ 只依赖于相对距离 $i-j$。这种方法更符合音频/语言的直觉——"相邻"是一个相对概念。

# 第七部分：Layer Normalization (LayerNorm)

## 它做了什么

LayerNorm对一个样本在特征维度上做归一化：

$$\hat{x}_k = \frac{x_k - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_k = \gamma \hat{x}_k + \beta$$

其中 $\mu = \frac{1}{d}\sum x_k$（特征维度的均值），$\sigma^2 = \frac{1}{d}\sum (x_k - \mu)^2$，$\gamma$ 和 $\beta$ 是可学习的缩放和平移参数。

## 为什么需要它

深层网络中，各层的输出分布会不断漂移（内部协变量偏移）。这导致：
- 梯度不稳定
- 训练变慢
- 需要小心初始化

LayerNorm将每一层的输出"拉回"到均值为0、方差为1的分布，然后通过可学习的 $\gamma$ 和 $\beta$ 重新调整。这稳定了深度网络的训练。

# 第八部分：残差连接

Transformer中每个子层（Attention、FFN）都有残差连接：

$$\text{Output} = \text{Input} + \text{Sublayer}(\text{LayerNorm}(\text{Input}))$$

**Post-Norm vs Pre-Norm：** 原始Transformer将LayerNorm放在残差之后（Post-Norm）。后来的研究（包括Conformer）发现放在残差之前（Pre-Norm）训练更稳定，尤其是在深层网络中。因为Pre-Norm让梯度可以通过残差连接"直通"到早期层。

残差连接本质上是为梯度提供了一条"高速公路"，每一层的梯度等于：
$$\text{grad} = \text{grad}_{skip} + \text{grad}_{sublayer}$$

即使子层的梯度很小时，skip连接的梯度也能保证有效的信息传播。

# 第九部分：完整的数据流

以一个简化的Encoder为例，追踪一个batch的数据：

```
输入: 一句话 → Tokenizer → Token IDs: (B, T)

Step 1: Embedding
  Token IDs → Embedding Table → (B, T, D_emb)
  
Step 2: 位置编码
  (B, T, D_emb) + Positional Encoding → (B, T, D_emb)

Step 3: EncoderLayer × N
  
  对于每一层:
  
  ┌─ 输入 x: (B, T, D) ────────────────────────────┐
  │                                                 │
  │  x_norm = LayerNorm(x)                          │
  │  attn_out = MultiHeadAttn(x_norm, x_norm, x_norm) │
  │  attn_out = Dropout(attn_out)                   │
  │  x = x + attn_out          ← 残差连接           │
  │                                                 │
  │  x_norm = LayerNorm(x)                          │
  │  ffn_out = FFN(x_norm)     ← Linear→ReLU→Linear │
  │  ffn_out = Dropout(ffn_out)                     │
  │  x = x + ffn_out           ← 残差连接           │
  │                                                 │
  └─ 输出 x: (B, T, D) ─────────────────────────────┘

Step 4: 输出
  最后一层的输出可以用于各种下游任务：
  - 分类：取第一个token的表示 → Linear → Softmax
  - 序列标注：每个位置 → Linear → Softmax
  - 生成：接Decoder
```

# 第十部分：Transformer的计算复杂度

## Self-Attention的复杂度分析

Attention的核心计算是 $QK^T \in \mathbb{R}^{T \times T}$，复杂度为 $O(T^2 \cdot d)$。

- **与序列长度呈平方关系**：这是Transformer最大的弱点。对于长音频（比如44.1kHz采样，10秒音频 = 441000个采样点），直接使用Transformer的Self-Attention是极其昂贵的。
- 这就是为什么在音频处理中，前面会用卷积下采样大幅降低序列长度，以及为什么Conformer引入了卷积模块来弥补局部采样的不足。

## FFN的复杂度

FFN只做逐位置变换：$O(T \cdot d \cdot d_{ff})$，与序列长度呈**线性**关系。

## 对比

| 模块 | 计算复杂度 | 内存复杂度 |
|------|-----------|-----------|
| Self-Attention | $O(T^2 d)$ | $O(T^2)$ |
| FFN | $O(T d d_{ff})$ | $O(T d)$ |
| Convolution (kernel K) | $O(T d K)$ | $O(T d)$ |

这解释了为什么长序列处理中，Self-Attention是瓶颈。

# 总结：你应该记住的核心要点

1. **Attention的本质**：数据依赖的全局加权求和，让每个位置可以"关注"所有其他位置
2. **Self-Attention vs Cross-Attention**：前者用于序列内部交互，后者用于序列间交互
3. **Multi-Head**：多个并行的注意力头，捕获不同类型的依赖关系
4. **Transformer的核心设计**：Attention负责跨位置信息混合，FFN负责逐位置非线性变换，残差+LayerNorm保证训练稳定性
5. **位置编码**：必不可少的组件，让无序的Attention感知到序列的顺序
6. **复杂度瓶颈**：$O(T^2)$ 是Attention的致命弱点，需要下采样或其他技巧

理解了这些，再去读Conformer就会非常顺畅：Conformer只是在Transformer Block中多插入了一个Convolution Module，让局部建模能力与全局Attention能力互补。本质上的设计语言是完全一致的。
