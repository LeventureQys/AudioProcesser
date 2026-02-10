# 04. GTConv：门控时序卷积的设计

GTCRN的编码器和解码器各有6层GTConvLite，这是整个网络的特征提取骨干。

## GTConvLite的结构

每层GTConvLite包含四个部分：

```python
class GTConvLite(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        # 1. 深度可分离卷积
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3,
                                   padding=dilation, dilation=dilation,
                                   groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

        # 2. 时间注意力 (TRA)
        self.tra = TemporalRecurrentAttention(channels)

        # 3. SE通道注意力
        self.se = SEBlock(channels)

    def forward(self, x):
        # 深度可分离卷积
        out = self.depthwise(x)
        out = self.pointwise(out)

        # 时间注意力
        out = self.tra(out)

        # 通道注意力
        out = self.se(out)

        # 残差连接
        return out + x
```

## 深度可分离卷积

标准卷积的参数量是 $C_{in} \times C_{out} \times K^2$。深度可分离卷积把它拆成两步：

1. **Depthwise**：每个通道独立卷积，参数量 $C \times K^2$
2. **Pointwise**：1×1卷积混合通道，参数量 $C \times C$

总参数量从 $C^2 \times K^2$ 降到 $C \times (K^2 + C)$，对于C=64、K=3，减少了约8倍。

## 时间注意力（TRA）

TRA是GTCRN的特色模块，用循环的方式计算时间维度的注意力：

```python
class TemporalRecurrentAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        # V3: padding=0, 手动左填充4

    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, F = x.shape

        # 沿时间轴计算注意力
        x_t = x.permute(0, 3, 1, 2).reshape(B * F, C, T)
        attn = torch.sigmoid(self.conv(x_t))
        out = x_t * attn
        out = out.reshape(B, F, C, T).permute(0, 2, 3, 1)

        return out
```

V3的因果版本把padding从对称改成只在左边填充：
```python
# V2: 对称padding
self.conv = nn.Conv1d(ch, ch, kernel_size=5, padding=2)

# V3: 因果padding
self.conv = nn.Conv1d(ch, ch, kernel_size=5, padding=0)
x = F.pad(x, (4, 0))  # 只填充左边
```

## SE通道注意力

Squeeze-and-Excitation模块，让网络学习通道间的重要性：

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, T, F]
        # 全局平均池化
        s = x.mean(dim=(2, 3))  # [B, C]

        # 两层FC学习通道权重
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))

        # 通道加权
        return x * s.unsqueeze(-1).unsqueeze(-1)
```

## Dilation Rate的设计

6层GTConv用了不同的dilation rate：[1, 2, 4, 8, 4, 2]

为什么是这个模式？
- 前4层dilation递增：感受野指数扩大，从局部到全局
- 后2层dilation递减：回到局部细节，帮助重建

这种"先扩后缩"的设计在语音处理里很常见，类似于WaveNet的设计。

感受野计算：
```
dilation=[1,2,4,8,4,2], kernel=3
RF = 1 + sum((k-1)*d) = 1 + 2*(1+2+4+8+4+2) = 43
```

43帧约270ms，足够覆盖大部分语音模式。

## 因果卷积的实现

V3要实时跑，所有卷积都要改成因果的：

```python
# V2: 对称padding
pad = dilation * (kernel - 1) // 2
self.conv = nn.Conv2d(..., padding=(pad, pad))

# V3: 只在左边和上边填充
pad = dilation * (kernel - 1)
self.conv = nn.Conv2d(..., padding=0)
# forward时手动填充
x = F.pad(x, (0, 0, pad, 0))  # 只填充时间维度的左边
```

## 流式推理的状态管理

V3做流式推理时，每层GTConv都需要缓存历史帧：

```python
class StreamingGTConv:
    def __init__(self, num_layers=6, dilations=[1,2,4,8,4,2]):
        # 每层需要缓存 dilation*(kernel-1) 帧
        self.buffers = [
            torch.zeros(B, C, d * 2, F)  # kernel=3, 需要2*d帧
            for d in dilations
        ]

    def process_frame(self, x, layer_idx):
        # 拼接历史帧
        buf = self.buffers[layer_idx]
        x_with_history = torch.cat([buf, x], dim=2)

        # 卷积
        out = self.convs[layer_idx](x_with_history)

        # 更新缓存
        self.buffers[layer_idx] = x_with_history[:, :, 1:, :]

        return out
```

12层GTConv（编码器6层+解码器6层），每层缓存大小不同，总共需要维护不少状态。
