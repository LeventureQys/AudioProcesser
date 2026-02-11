# 附录 A：U-Net 详解

## A.1 什么是 U-Net

U-Net 最初是 2015 年为医学图像分割设计的网络，因其结构像字母 "U" 而得名。

核心思想：**编码器-解码器 + 跳跃连接**

```
输入图像
    │
    ▼
┌─────────┐
│ 编码器   │  逐层下采样，提取高级语义特征
│ (收缩路径)│  分辨率降低，通道数增加
└────┬────┘
     │
     ▼
  瓶颈层     最深层，分辨率最低，语义最抽象
     │
     ▼
┌─────────┐
│ 解码器   │  逐层上采样，恢复空间分辨率
│ (扩展路径)│  分辨率增加，通道数减少
└────┬────┘
     │
     ▼
输出 (与输入同尺寸)
```

---

## A.2 为什么需要跳跃连接

### 问题：下采样会丢失空间细节

```
原图 256×256
    ↓ 下采样
128×128     丢失一半像素
    ↓ 下采样
64×64       继续丢失
    ↓ 下采样
32×32       细节几乎没了
    ↓ 上采样
64×64       只能靠"猜"来恢复细节
    ↓ 上采样
128×128     越猜越离谱
    ↓ 上采样
256×256     边缘模糊，细节丢失
```

### 解决：跳跃连接直接把编码器的细节传给解码器

```
编码器                    解码器
256×256 ─────────────────→ 256×256   细节直接传过来
    ↓                          ↑
128×128 ─────────────────→ 128×128
    ↓                          ↑
64×64 ───────────────────→ 64×64
    ↓                          ↑
32×32 ──────→ 瓶颈 ──────→ 32×32
```

编码器每层的输出保存下来，解码器对应层直接拿来用，不用从头"猜"。

---

## A.3 跳跃连接的实现方式

两种常见方式：

### 拼接 (Concatenation) - 原版 U-Net 用这个

```python
# 解码器某层
x_up = upsample(x)                    # 上采样
x = torch.cat([x_up, skip], dim=1)    # 通道维度拼接
x = conv(x)                           # 卷积融合
```

特点：
- 信息保留完整
- 通道数翻倍，需要更多卷积参数来融合
- 计算量较大

### 相加 (Addition) - GTCRN 用这个

```python
# 解码器某层
x = x + skip    # 直接逐元素相加
x = conv(x)
```

特点：
- 计算量小
- 要求 skip 和 x 尺寸完全一致
- 信息融合是隐式的

---

## A.4 U-Net 的 "U" 形结构

完整结构图：

```
输入                                          输出
  │                                            ↑
  ▼                                            │
┌───┐                                        ┌───┐
│ E1│ ──────────── skip1 ──────────────────→│ D1│
└─┬─┘                                        └─↑─┘
  │ ↓下采样                            上采样↑ │
  ▼                                            │
┌───┐                                        ┌───┐
│ E2│ ──────────── skip2 ──────────────────→│ D2│
└─┬─┘                                        └─↑─┘
  │ ↓下采样                            上采样↑ │
  ▼                                            │
┌───┐                                        ┌───┐
│ E3│ ──────────── skip3 ──────────────────→│ D3│
└─┬─┘                                        └─↑─┘
  │ ↓下采样                            上采样↑ │
  ▼                                            │
┌─────────────────────────────────────────────────┐
│                    瓶颈层                        │
└─────────────────────────────────────────────────┘
```

- 左边：下降的编码器 (Encoder)，逐层下采样
- 右边：上升的解码器 (Decoder)，逐层上采样
- 中间横线：跳跃连接 (Skip Connection)
- 底部：瓶颈层 (Bottleneck)，最深的特征
- 整体像字母 "U"

---

## A.5 编码器的作用

编码器通过下采样逐步提取高级特征：

```
输入: 高分辨率，低级特征 (边缘、纹理)
  │
  ↓ Conv + Pool
第1层: 分辨率/2，开始提取局部模式
  │
  ↓ Conv + Pool
第2层: 分辨率/4，提取更大范围的模式
  │
  ↓ Conv + Pool
第3层: 分辨率/8，提取高级语义 (物体、结构)
  │
  ↓
瓶颈: 分辨率最低，语义最抽象
```

类比：
- 浅层：看到的是"像素"
- 深层：看到的是"物体"

---

## A.6 解码器的作用

解码器通过上采样逐步恢复分辨率：

```
瓶颈: 低分辨率，高级语义
  │
  ↓ Deconv/Upsample + Skip
第1层: 分辨率×2，结合 skip 恢复细节
  │
  ↓ Deconv/Upsample + Skip
第2层: 分辨率×4，继续恢复
  │
  ↓ Deconv/Upsample + Skip
第3层: 分辨率×8，接近原始分辨率
  │
  ↓
输出: 与输入同尺寸
```

关键：每层都用 skip connection 补充细节，不是纯靠"猜"。

---

## A.7 GTCRN 中的 U-Net

GTCRN 把 U-Net 用在语音降噪，输入是 2D 频谱而不是图像：

| 图像 U-Net | GTCRN |
|-----------|-------|
| 2D 图像 (H×W) | 2D 频谱 (T×F) |
| 空间下采样 | 频率轴下采样 |
| 语义特征 | 时频特征 |
| 像素级分割 | 频点级掩码 |

### GTCRN 的 U 形结构

```
频谱 (B, C, T, 219)
    │
    ▼ DSConv, stride=2
(B, 32, T, 110) ────── skip1 ──────→ + ─→ DSDeconv ─→ (B, 2, T, 219)
    │                                      ↑
    ▼ DSConv, stride=2                     │
(B, 32, T, 55) ─────── skip2 ──────→ + ─→ DSDeconv ─→ (B, 32, T, 110)
    │                                      ↑
    ▼ GTConv×6                             │
(B, 32, T, 55) ─────── skip3~8 ────→ + ─→ GTConv×6 ─→ (B, 32, T, 55)
    │                                      ↑
    └──────────→ DPGRNN×2 ─────────────────┘
                 (瓶颈层)
```

### 对应关系

| 编码器层 | 保存为 | 解码器使用 |
|---------|--------|-----------|
| DSConv: 219→110 | skip1 | DSDeconv: 110→219 |
| DSConv: 110→55 | skip2 | DSDeconv: 55→110 |
| GTConv×6 | skip3~8 | GTConv×6 |

---

## A.8 为什么语音降噪适合 U-Net

1. **输入输出同尺寸**
   - 带噪频谱 (B, F, T, 2) → 干净频谱 (B, F, T, 2)
   - 尺寸完全一致，天然适合 U-Net

2. **需要保留细节**
   - 高频细节对音质很重要
   - 下采样会丢失高频，skip connection 能补回来

3. **需要上下文**
   - 降噪需要理解语音内容，区分语音和噪声
   - 深层网络能捕获长程依赖

4. **对称结构**
   - 编码器：提取时频特征
   - 解码器：生成复数掩码
   - 天然对称

---

## A.9 Skip Connection 代码示例

GTCRN 原版代码 (`gtcrn.py`)：

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(9, 16, (1,5), stride=(1,2), ...),   # 129→65
            ConvBlock(16, 16, (1,5), stride=(1,2), ...),  # 65→33
            GTConvBlock(16, 16, ..., dilation=(1,1)),     # 不下采样
            GTConvBlock(16, 16, ..., dilation=(2,1)),
            GTConvBlock(16, 16, ..., dilation=(5,1))
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)      # 保存每层输出作为 skip
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, ..., dilation=(5,1)),     # 对应 encoder 最后一层
            GTConvBlock(16, 16, ..., dilation=(2,1)),
            GTConvBlock(16, 16, ..., dilation=(1,1)),
            ConvBlock(16, 16, ..., use_deconv=True),      # 33→65
            ConvBlock(16, 2, ..., use_deconv=True)        # 65→129
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            # 逆序取 skip: 第 i 层用 en_outs[N-1-i]
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x


class GTCRN(nn.Module):
    def forward(self, spec):
        # ... 前处理 ...

        # 编码，保存所有 skip
        feat, en_outs = self.encoder(feat)

        # 瓶颈层
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        # 解码，使用 skip
        m_feat = self.decoder(feat, en_outs)

        # ... 后处理 ...
```

### 逆序使用的原因

解码器是编码器的"镜像"，所以要逆序取 skip：

```
编码顺序: E0 → E1 → E2 → E3 → E4
保存顺序: [0]  [1]  [2]  [3]  [4]

解码顺序: D0 → D1 → D2 → D3 → D4
使用顺序: [4]  [3]  [2]  [1]  [0]

对应关系:
E4 (最后编码) ←→ D0 (最先解码)
E0 (最先编码) ←→ D4 (最后解码)
```

---

## A.10 U-Net 的变体

### 原版 U-Net (2015)
- 医学图像分割
- 拼接式 skip connection
- 对称的编码器-解码器

### ResUNet
- 加入残差连接
- 每个 block 内部有 shortcut

### Attention U-Net
- skip connection 加入注意力机制
- 自动学习哪些特征更重要

### GTCRN 的改进
- 用相加代替拼接（更轻量）
- 用 DPGRNN 代替普通瓶颈（更强的时频建模）
- 只在频率轴下采样（保留时间分辨率）
