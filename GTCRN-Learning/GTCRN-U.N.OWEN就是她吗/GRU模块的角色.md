# DPGRNN：双路径GRU的设计

GTCRN的核心是DPGRNN（Dual-Path Grouped RNN），不是简单的一个GRU，而是两条路径分别处理频率和时间维度。

## 为什么要双路径

语音频谱是2D的：频率轴和时间轴。直接用2D卷积或者把特征拉平喂给RNN都有问题：

- 2D卷积：感受野有限，长程依赖建模弱
- 拉平喂RNN：维度太高，计算量爆炸

双路径的思路是把2D问题拆成两个1D问题：
1. **Intra-path**：沿频率轴跑，建模频率维度的依赖（比如谐波结构）
2. **Inter-path**：沿时间轴跑，建模时间维度的依赖（比如噪声统计变化）

## 具体结构

```python
class DPGRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups=2):
        super().__init__()
        # Intra-path: 沿频率轴，双向
        self.intra_gru = nn.GRU(
            input_size // num_groups,
            hidden_size // num_groups,
            bidirectional=True,
            batch_first=True
        )
        self.intra_fc = nn.Linear(hidden_size, input_size)

        # Inter-path: 沿时间轴
        # V1/V2: bidirectional=True
        # V3: bidirectional=False (因果)
        self.inter_gru = nn.GRU(
            input_size,
            hidden_size,
            bidirectional=True,  # V3改成False
            batch_first=True
        )
        self.inter_fc = nn.Linear(hidden_size * 2, input_size)  # V3是hidden_size

    def forward(self, x):
        # x: [B, T, F, C]
        B, T, F, C = x.shape

        # Intra-path: 每个时间帧独立处理频率轴
        x_intra = x.reshape(B * T, F, C)
        intra_out, _ = self.intra_gru(x_intra)
        intra_out = self.intra_fc(intra_out)
        intra_out = intra_out.reshape(B, T, F, C)
        x = x + intra_out  # 残差

        # Inter-path: 每个频率bin独立处理时间轴
        x_inter = x.permute(0, 2, 1, 3).reshape(B * F, T, C)
        inter_out, _ = self.inter_gru(x_inter)
        inter_out = self.inter_fc(inter_out)
        inter_out = inter_out.reshape(B, F, T, C).permute(0, 2, 1, 3)
        x = x + inter_out  # 残差

        return x
```

## Intra-path：频率轴建模

Intra-path沿频率轴跑，用双向GRU。为什么是双向？因为频率轴没有因果性——低频和高频的关系是对称的。

它主要捕获：
- 谐波结构：基频和各次谐波之间的关系
- 共振峰：相邻频带的能量分布
- 频谱包络：整体的频谱形状

每个时间帧独立处理，所以可以并行。

## Inter-path：时间轴建模

Inter-path沿时间轴跑，这里有因果性的问题：

- **V1/V2（离线）**：双向GRU，能看到整段音频
- **V3（实时）**：单向GRU，只能看历史帧

它主要捕获：
- 噪声统计的慢变化
- 语音的连续性
- 瞬态事件的上下文

每个频率bin独立处理，也可以并行。

## 为什么用两个DPGRNN

GTCRN串联了两个DPGRNN模块。一个不够吗？

实验发现一个DPGRNN的建模能力有限，特别是对复杂噪声场景。两个串联后：
- 第一个DPGRNN做初步的时频建模
- 第二个DPGRNN在此基础上进一步精细化

类似于堆叠多层LSTM的效果，但计算效率更高。

## 分组策略

DPGRNN里用了分组（num_groups=2），把通道分成两组独立处理。好处：
- 参数量减少一半
- 不同组可以学习不同的模式
- 类似于多头注意力的思想

## V3的因果化

V3要实时跑，Inter-path必须改成单向：

```python
# V2
self.inter_gru = nn.GRU(..., bidirectional=True)
self.inter_fc = nn.Linear(hidden_size * 2, input_size)

# V3
self.inter_gru = nn.GRU(..., bidirectional=False)
self.inter_fc = nn.Linear(hidden_size, input_size)
```

Intra-path不用改，因为频率轴没有因果性。

单向GRU的建模能力比双向弱，所以V3把hidden_size从48加到64来补偿，参数量从139K涨到145K。

## 流式推理的状态管理

V3做流式推理时，Inter-path的GRU需要保持hidden state：

```python
class StreamingDPGRNN:
    def __init__(self):
        self.inter_hidden = None  # [num_layers, B*F, hidden]

    def process_frame(self, x):
        # x: [B, 1, F, C] 单帧

        # Intra-path: 无状态，直接算
        intra_out = self.intra_forward(x)

        # Inter-path: 需要维护状态
        inter_out, self.inter_hidden = self.inter_forward(x, self.inter_hidden)

        return intra_out + inter_out
```

Intra-path不需要状态，因为每帧独立处理频率轴。
