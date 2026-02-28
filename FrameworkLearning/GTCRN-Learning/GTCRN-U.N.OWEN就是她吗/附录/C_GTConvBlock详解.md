# 附录 C：GTConvBlock 详解

GTConvBlock = **G**rouped **T**emporal **Conv**olution **Block**

GTCRN 的核心特征提取模块，负责在时频域上提取局部特征，同时动态适应不同帧和通道的重要性。

---

## C.1 设计目标

GTConvBlock 需要同时解决四个问题：

1. **提取局部时频特征**：捕捉频谱的局部模式（谐波、共振峰）
2. **扩大感受野**：用空洞卷积看更长的时间范围，不增加参数
3. **省参数**：在极低参数量约束下工作
4. **时序自适应**：动态加权不同帧的重要性

---

## C.2 原版 GTConvBlock（ShuffleNet 风格）

原版 GTCRN（16kHz）的 GTConvBlock 用 ShuffleNet 风格：只处理一半通道，另一半直通，最后 shuffle 混合。

### 结构流程

```
输入 [B, 16, T, F]
    │
    ▼ 分成两半
    ├──────────────────────────────────┐
    ▼                                  ▼
   x1 [B, 8, T, F]              x2 [B, 8, T, F]（直通）
    │                                  │
    ▼ SFE（邻频展开）                   │
  [B, 24, T, F]                        │
    │                                  │
    ▼ PointConv(24→16)                 │
  [B, 16, T, F]                        │
    │                                  │
    ▼ DepthConv(3×3, dilation)         │
  [B, 16, T, F]                        │
    │                                  │
    ▼ PointConv(16→8)                  │
  [B, 8, T, F]                         │
    │                                  │
    ▼ TRA（时序门控）                   │
  h1 [B, 8, T, F]                      │
    │                                  │
    └──────────┬───────────────────────┘
               ▼
         Channel Shuffle
               ▼
        输出 [B, 16, T, F]
```

### 为什么只处理一半通道

在极低参数量（23K）的约束下，不能对所有通道都做完整的卷积。ShuffleNet 的思路是：直通的那一半保留了上一层的信息，处理的那一半提取新特征，shuffle 后两者混合。用一半的计算量达到接近全量处理的效果。

### 为什么 shuffle 而不是直接拼接

shuffle 把两半通道交错排列（h1_0, x2_0, h1_1, x2_1, ...），让下一层的每个卷积核都能同时看到"处理过的"和"直通的"通道。简单拼接会让两半通道在空间上分离，下一层需要更大的卷积核才能融合。

### SFE（子带特征提取）

SFE 把每个频带和左右邻居拼在一起：

```
频率轴:  f0  f1  f2  f3  f4
              ├───┼───┤
              邻频窗口 k=3

对于 f2 位置：SFE 输出 = concat([f1, f2, f3])
8 通道 × 3 邻频 = 24 通道
```

**为什么需要 SFE**：ERB 压缩后，相邻频带之间的关系被割裂。SFE 显式提供邻频上下文，让网络在第一步就能看到局部的频带结构（谐波、共振峰）。

---

## C.3 V1 的 GTConvLite

V1 适配 48kHz，通道数从 16 增加到 32，同时对 GTConvBlock 做了简化。

### 结构流程

```
输入 [B, 32, T, F]
    │
    ├──────────────────────────────┐ 残差
    ▼                              │
DWConv(32, k=3×3, dilation=d)     │  ← 空间特征提取
    ↓                              │
PWConv(32→32, k=1×1)              │  ← 通道混合
    ↓                              │
BN → PReLU                        │
    ↓                              │
TRALite（时间门控）                 │
    ↓                              │
SEBlock（通道门控）                 │
    ↓                              │
    + ←────────────────────────────┘
    ↓
输出 [B, 32, T, F]
```

### 原版 vs V1 的对比

| 组件 | 原版 GTConvBlock | V1 GTConvLite |
|------|-----------------|---------------|
| 通道分割 | 分两半，只处理一半 | 全部处理 |
| SFE | Unfold 展开邻频 | 去掉 |
| 卷积结构 | Point→Depth→Point | DW→PW |
| TRA | GRU 门控 | Conv 门控（TRALite） |
| Shuffle | 有 | 无 |
| 残差 | 无 | 有 |
| SEBlock | 无 | 有 |
| 通道数 | 16 | 32 |
| 层数 | 3 | 6 |

### 为什么去掉 Shuffle

V1 通道数更多（32 vs 16），不需要用 ShuffleNet 的技巧省参数。而且通道数增加后，全量处理的效果更好，不需要牺牲一半通道。

### 为什么去掉 SFE

V1 的深度卷积（DWConv）本身就在频率轴上有 k=3 的感受野，已经能看到邻频信息。SFE 的邻频展开变得冗余。

### 为什么 TRA 改成卷积（TRALite）

原版 TRA 用 GRU 做时序门控，有隐藏状态。问题：
- 流式推理需要维护 GRU 状态，部署复杂
- GRU 不好量化（整数量化对 RNN 不友好）

TRALite 用 1D 卷积替代 GRU：

```
输入 [B, C, T, F]
    ↓ 沿频率轴求均值
能量 [B, C, T]
    ↓ DWConv1d(k=5)
    ↓ PWConv1d
    ↓ Sigmoid
门控 [B, C, T, 1]
    ↓ 乘回输入
输出 [B, C, T, F]
```

无状态，可量化，效果差距不大。

### 为什么加 SEBlock

去掉 Shuffle 后，通道间的信息交互减弱（Shuffle 的作用之一是促进通道交互）。SEBlock 通过学习通道重要性权重来补偿这个损失，同时提供了通道维度的自适应能力。

### 为什么加残差连接

V1 堆叠 6 层（原版 3 层），更深的网络需要残差连接帮助梯度传播，防止梯度消失。

---

## C.4 TRALite vs TRA 详解

### TRA（原版）

```
输入 [B, C, T, F]
    ↓ mean(dim=F)
能量 [B, C, T]
    ↓ GRU(C, C×2)
    ↓ FC(C×2 → C)
    ↓ Sigmoid
门控 [B, C, T, 1]
    ↓ × 输入
输出 [B, C, T, F]
```

优点：GRU 有记忆，能捕捉长程的能量变化趋势
缺点：有状态，流式推理复杂；不好量化

### TRALite（V1）

```
输入 [B, C, T, F]
    ↓ mean(dim=F)
能量 [B, C, T]
    ↓ DWConv1d(k=5, groups=C)
    ↓ PWConv1d
    ↓ Sigmoid
门控 [B, C, T, 1]
    ↓ × 输入
输出 [B, C, T, F]
```

优点：无状态，可量化，流式推理简单
缺点：感受野有限（k=5，只看 5 帧的能量历史）

**V3 的因果化**：非因果版本用对称 padding（左右各 2 帧），因果版本只在左边 pad 4 帧，右边不 pad。

---

## C.5 SEBlock 详解

```
输入 [B, C, T, F]
    ↓ 全局平均池化（时间+频率轴）
[B, C]                ← 每个通道的全局统计
    ↓ FC(C → C/4) → ReLU
    ↓ FC(C/4 → C) → Sigmoid
权重 [B, C]           ← 每个通道的重要性 [0, 1]
    ↓ reshape → [B, C, 1, 1]
    ↓ × 输入
输出 [B, C, T, F]
```

**为什么用两层 FC**：第一层（C→C/4）做降维，强迫网络学习通道间的紧凑关系；第二层（C/4→C）恢复维度，预测每个通道的权重。瓶颈结构减少参数量，同时保持表达能力。

**为什么用 Sigmoid 而不是 Softmax**：Sigmoid 让每个通道独立决定自己的重要性，通道之间不竞争。Softmax 会让通道之间竞争，可能抑制所有通道（当某个通道特别重要时，其他通道都被压低）。

---

## C.6 GTConvBlock 在网络中的位置与分工

```
Encoder:
    DSConv: 219 → 110
    DSConv: 110 → 55
    GTConvLite(d=1)   ← 感受野 3 帧，局部模式
    GTConvLite(d=2)   ← 感受野 5 帧
    GTConvLite(d=4)   ← 感受野 9 帧
    GTConvLite(d=8)   ← 感受野 17 帧，全局模式
    GTConvLite(d=4)   ← 填补网格效应
    GTConvLite(d=2)   ← 细化局部细节
        ↓
    DPGRNN×2（全局时频建模）
        ↓
Decoder:
    GTConvLite(d=2)   ← 镜像，先建立视野
    GTConvLite(d=4)
    GTConvLite(d=8)
    GTConvLite(d=4)
    GTConvLite(d=2)
    GTConvLite(d=1)   ← 最后细化
    DSDeconv: 55 → 110
    DSDeconv: 110 → 219
```

### GTConvBlock 与 DPGRNN 的分工

| 模块 | 范围 | 方式 | 作用 |
|------|------|------|------|
| GTConvBlock | 局部（最多 43 帧） | 卷积 | 提取局部时频模式，动态门控 |
| DPGRNN | 全局（整段/整个频率轴） | RNN | 建模长程时序依赖和全局频率依赖 |

GTConvBlock 通过堆叠扩大感受野，但仍是局部的；DPGRNN 一次看整个频率轴或整段时间，建模全局依赖。两者互补，共同完成特征提取。
