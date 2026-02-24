# 附录 C：GTConvBlock 详解

GTConvBlock = **G**roup **T**emporal **Conv**olution **Block**

GTCRN 的核心特征提取模块，负责在时频域上提取局部特征。

---

## C.1 设计目标

1. **提取局部时频特征**：用卷积捕获频谱的局部模式
2. **扩大感受野**：用空洞卷积看更长的时间范围
3. **省参数**：只处理一半通道，用 ShuffleNet 风格
4. **时序自适应**：用 TRA 门控动态加权不同帧

---

## C.2 原版 GTConvBlock 结构

### 整体流程

```
输入 x: (B, 16, T, F)
    │
    ▼
torch.chunk(x, 2, dim=1)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
   x1: (B, 8, T, F)              x2: (B, 8, T, F)
    │                                  │
    ▼                                  │
  SFE(k=3)                             │
    │ Unfold 展开邻频                   │
    ▼                                  │
  (B, 24, T, F)                        │
    │                                  │
    ▼                                  │
  PointConv1(24→16)                    │
    │ 1×1 卷积压缩通道                  │
    ▼                                  │
  (B, 16, T, F)                        │
    │                                  │
    ▼                                  │
  DepthConv(3×3, dilation)             │
    │ 空洞卷积，扩大感受野               │
    ▼                                  │
  (B, 16, T, F)                        │
    │                                  │
    ▼                                  │
  PointConv2(16→8)                     │
    │ 1×1 卷积恢复通道数                │
    ▼                                  │
  (B, 8, T, F)                         │
    │                                  │
    ▼                                  │
  TRA                                  │
    │ 时序门控                          │
    ▼                                  │
  h1: (B, 8, T, F)                     │
    │                                  │
    └──────────┬───────────────────────┘
               ▼
         shuffle(h1, x2)
               │
               ▼ 交错拼接
        输出: (B, 16, T, F)
```

### 代码对照 (gtcrn.py:107-153)

```python
class GTConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size,
                 stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.pad_size = (kernel_size[0]-1) * dilation[0]

        # 子带特征提取
        self.sfe = SFE(kernel_size=3, stride=1)

        # 第一个 1×1 卷积：扩展通道
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        # 深度卷积：空间特征提取
        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        # 第二个 1×1 卷积：压缩通道
        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)

        # 时序注意力
        self.tra = TRA(in_channels//2)

    def forward(self, x):
        # 分成两半
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        # 只处理 x1
        x1 = self.sfe(x1)                                    # 邻频展开
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))  # 1×1
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0]) # 因果 padding
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))    # 深度卷积
        h1 = self.point_bn2(self.point_conv2(h1))            # 1×1
        h1 = self.tra(h1)                                    # 时序门控

        # Channel Shuffle
        x = self.shuffle(h1, x2)
        return x
```

---

## C.3 各子模块详解

### SFE (Subband Feature Extraction)

把每个频点和它的邻居拼起来，获取局部频率上下文。

```
输入: (B, 8, T, F)

Unfold(kernel_size=(1,3)):
    每个位置取左、中、右 3 个频点
    8 通道 × 3 邻频 = 24 通道

输出: (B, 24, T, F)
```

示意图：
```
频率轴:  f0  f1  f2  f3  f4  f5
              ├───┼───┤
              邻频窗口 k=3

对于 f2 位置，SFE 输出 = concat([f1, f2, f3])
```

为什么这样做：
- 频谱的相邻频点通常相关（谐波、共振峰）
- 显式提供邻频信息，帮助网络学习频率模式

### PointConv (1×1 卷积)

只混合通道，不看空间位置。

```
PointConv1: 24 → 16
    把 SFE 展开的 24 通道压缩到 16

PointConv2: 16 → 8
    把处理后的 16 通道压回 8
```

作用：
- 调整通道数
- 跨通道信息融合
- 类似全连接层，但保持空间结构

### DepthConv (深度卷积)

每个通道独立卷积，不跨通道。

```python
nn.Conv2d(16, 16, kernel_size=(3,3),
          dilation=(d,1), groups=16)  # groups=通道数
```

```
普通卷积 (groups=1):
    所有输入通道 → 混合 → 所有输出通道
    参数: C_in × C_out × k × k

深度卷积 (groups=C):
    每个通道独立卷积
    参数: C × k × k

省参数比例: C_out 倍
```

为什么用深度卷积：
- 参数少
- 配合 PointConv 组成 DW-Separable 结构
- 空间特征和通道混合分离

### Dilation (空洞卷积)

在卷积核内部加入间隔，扩大感受野。

```
dilation=1 (普通):
    █ █ █     感受野 = 3

dilation=2:
    █   █   █     感受野 = 5

dilation=4:
    █       █       █     感受野 = 9
```

原版 GTCRN 的 dilation 序列：[1, 2, 5]

```
Layer 1 (d=1):  看 3 帧
Layer 2 (d=2):  看 5 帧，累积约 7 帧
Layer 3 (d=5):  看 11 帧，累积约 17 帧
```

V1 改成 [1, 2, 4, 8, 4, 2]，先扩大后收缩：
- 扩大：感受野指数增长
- 收缩：填补扩张阶段跳过的位置，避免网格效应

### TRA (Temporal Recurrent Attention)

根据帧能量动态加权，让网络关注重要的时间帧。

```
输入 x: (B, 8, T, F)
    │
    ▼
energy = mean(x², dim=F)    # (B, 8, T) 每帧能量
    │
    ▼
GRU(8, 16)                  # 时序建模
    │
    ▼
FC(16 → 8)                  # 投影回通道数
    │
    ▼
Sigmoid                     # 门控值 [0, 1]
    │
    ▼
gate: (B, 8, T, 1)
    │
    ▼
output = x * gate           # 逐帧加权
```

为什么用 TRA：
- 语音信号能量变化大（有声/无声、辅音/元音）
- 动态关注高能量或变化剧烈的帧
- 抑制静音帧的噪声

### Channel Shuffle

把两半通道交错拼接，促进信息交换。

```python
def shuffle(self, x1, x2):
    # x1, x2: (B, 8, T, F)
    x = torch.stack([x1, x2], dim=1)      # (B, 2, 8, T, F)
    x = x.transpose(1, 2)                  # (B, 8, 2, T, F)
    x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B, 16, T, F)
    return x
```

交错前：
```
通道: [h1_0, h1_1, h1_2, ..., h1_7, x2_0, x2_1, ..., x2_7]
       └──── 处理过的 ────┘  └──── 没处理的 ────┘
```

交错后：
```
通道: [h1_0, x2_0, h1_1, x2_1, h1_2, x2_2, ..., h1_7, x2_7]
       └─交错─┘
```

为什么 shuffle：
- x2 那一半没处理，直接传下去会导致信息不均衡
- shuffle 后下一层的 x1 会包含上一层 x2 的部分
- 多层堆叠后，所有通道都会被处理到

---

## C.4 V1 的 GTConvLite

V1 对 GTConvBlock 做了简化，去掉了 ShuffleNet 结构。

### 结构对比

| 组件 | 原版 GTConvBlock | V1 GTConvLite |
|------|-----------------|---------------|
| 通道分割 | 分两半，只处理一半 | 全部处理 |
| SFE | Unfold 展开 | 去掉 |
| 卷积 | Point→Depth→Point | DW→PW |
| TRA | GRU 门控 | Conv 门控 (TRALite) |
| Shuffle | 有 | 无 |
| 额外 | 无 | 加 SEBlock |

### GTConvLite 结构

```
输入 x: (B, 32, T, 55)
    │
    ├────────────────────────┐ 残差连接
    ▼                        │
DWConv(32, 32, k=3×3)        │
    │ dilation=(d, 1)        │
    │ groups=32              │
    ▼                        │
PWConv(32, 32, k=1×1)        │
    │                        │
    ▼                        │
BatchNorm → PReLU            │
    │                        │
    ▼                        │
TRALite                      │
    │ Conv 门控代替 GRU      │
    ▼                        │
SEBlock                      │
    │ 通道注意力             │
    ▼                        │
    + ←──────────────────────┘
    │
    ▼
输出: (B, 32, T, 55)
```

### 为什么简化

1. **去 Shuffle**：V1 通道数更多 (32 vs 16)，不需要省那么多
2. **去 SFE**：DW 卷积本身就看邻频，SFE 冗余
3. **TRA→TRALite**：GRU 有状态，不好量化和流式
4. **加 SEBlock**：补偿去掉 shuffle 损失的通道交互
5. **加残差**：更深的网络需要残差帮助训练

---

## C.5 TRALite vs TRA

### TRA (原版)

```python
class TRA(nn.Module):
    def __init__(self, channels):
        self.att_gru = nn.GRU(channels, channels*2, 1)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):  # (B, C, T, F)
        zt = torch.mean(x.pow(2), dim=-1)  # (B, C, T)
        at = self.att_gru(zt.transpose(1,2))[0]  # GRU
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        return x * at[..., None]
```

问题：
- GRU 有隐藏状态，流式推理要维护
- GRU 不好量化

### TRALite (V1)

```python
class TRALite(nn.Module):
    def __init__(self, channels):
        self.dw_conv = nn.Conv1d(channels, channels, 5,
                                  padding=2, groups=channels)
        self.pw_conv = nn.Conv1d(channels, channels, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):  # (B, C, T, F)
        energy = x.pow(2).mean(dim=-1)  # (B, C, T)
        gate = self.dw_conv(energy)
        gate = self.pw_conv(gate)
        gate = self.act(gate)
        return x * gate.unsqueeze(-1)
```

优势：
- 无状态，纯卷积
- 好量化
- 流式推理简单

---

## C.6 SEBlock (通道注意力)

V1 新增的模块，学习通道重要性。

```
输入 x: (B, C, T, F)
    │
    ▼
Global Average Pool
    │ mean(dim=(T, F))
    ▼
(B, C)
    │
    ▼
FC(C → C//4) → ReLU
    │
    ▼
FC(C//4 → C) → Sigmoid
    │
    ▼
weights: (B, C)
    │
    ▼
output = x * weights.view(B, C, 1, 1)
```

作用：
- 学习哪些通道更重要
- 自适应加权不同特征
- 补偿去掉 shuffle 后的通道交互

---

## C.7 GTConvBlock 在网络中的位置

```
Encoder:
    Conv: 219 → 110
    Conv: 110 → 55
    GTConvBlock(d=1)  ← 感受野 3
    GTConvBlock(d=2)  ← 感受野 5
    GTConvBlock(d=5)  ← 感受野 11
        ↓
    DPGRNN×2 (瓶颈)
        ↓
Decoder:
    GTConvBlock(d=5)  ← 镜像
    GTConvBlock(d=2)
    GTConvBlock(d=1)
    Deconv: 55 → 110
    Deconv: 110 → 219
```

### 与 DPGRNN 的分工

| 模块 | 范围 | 方式 | 作用 |
|------|------|------|------|
| GTConvBlock | 局部 | 卷积 | 提取局部时频模式 |
| DPGRNN | 全局 | RNN | 建模长程依赖 |

GTConvBlock 通过堆叠扩大感受野，但仍是局部的；DPGRNN 一次看整个频率轴/时间轴。

---

## C.8 总结

GTConvBlock 的核心思想：

1. **DW-Separable 结构**：空间卷积 (DW) + 通道混合 (PW)，省参数
2. **空洞卷积**：不增加参数，扩大感受野
3. **时序门控**：动态关注重要帧
4. **ShuffleNet 风格** (原版)：只处理一半通道，shuffle 交换信息

从原版到 V1 的演变：
- 去掉 shuffle，改用残差 + SEBlock
- TRA 从 GRU 改成 Conv
- 更深 (3层→6层)，更宽 (16通道→32通道)
