# 附录 B：卷积参数速查

## B.1 参数缩写对照表

| 缩写 | 全称 | 含义 | 常见值 |
|------|------|------|--------|
| k | kernel_size | 卷积核大小 | 1, 3, 5, 7 |
| s | stride | 步长 | 1, 2 |
| p | padding | 填充 | 0, 1, 2 |
| d | dilation | 空洞率 | 1, 2, 4, 8 |
| g | groups | 分组数 | 1, C (depthwise) |

---

## B.2 各参数详解

### kernel_size (k) - 卷积核大小

卷积核覆盖的区域，决定每次"看"多大范围。

```
k=1: 只看当前位置
     █

k=3: 看 3×3 区域
     █ █ █
     █ ● █
     █ █ █

k=5: 看 5×5 区域
     █ █ █ █ █
     █ █ █ █ █
     █ █ ● █ █
     █ █ █ █ █
     █ █ █ █ █
```

2D 卷积可以用不同的高宽：
```
k=(1, 5): 时间轴 1，频率轴 5
          █ █ ● █ █   只看当前时间帧的 5 个频点
```

### stride (s) - 步长

每次移动多少，决定输出尺寸。

```
输入: 8 个位置
      1 2 3 4 5 6 7 8

s=1: 每次移动 1，输出 8 个
     ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

s=2: 每次移动 2，输出 4 个 (下采样)
     ↓   ↓   ↓   ↓

s=4: 每次移动 4，输出 2 个
     ↓       ↓
```

输出尺寸公式：`out = (in + 2*p - k) / s + 1`

### padding (p) - 填充

在输入边缘补零，控制输出尺寸。

```
输入 (5,): [1, 2, 3, 4, 5]

p=0: 不填充
     [1, 2, 3, 4, 5]

p=1: 左右各填 1 个 0
     [0, 1, 2, 3, 4, 5, 0]

p=2: 左右各填 2 个 0
     [0, 0, 1, 2, 3, 4, 5, 0, 0]
```

常用：`p = k // 2` 保持输出尺寸不变 (当 s=1 时)

### dilation (d) - 空洞率

卷积核内部的间隔，扩大感受野但不增加参数。

```
k=3, d=1 (普通卷积):
     █ █ █     感受野 = 3

k=3, d=2 (空洞卷积):
     █   █   █   感受野 = 5

k=3, d=4:
     █       █       █   感受野 = 9
```

感受野公式：`receptive = k + (k-1) * (d-1) = 1 + (k-1) * d`

### groups (g) - 分组数

把输入通道分成 g 组，每组独立卷积。

```
普通卷积 (g=1):
     所有输入通道 → 所有输出通道
     参数: C_in × C_out × k × k

分组卷积 (g=2):
     前半输入通道 → 前半输出通道
     后半输入通道 → 后半输出通道
     参数: C_in × C_out × k × k / g

Depthwise 卷积 (g=C_in):
     每个通道独立卷积
     参数: C_in × k × k
```

---

## B.3 常见组合

### 标准卷积
```python
nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
# 32→64 通道，尺寸不变
```

### 下采样卷积
```python
nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
# 32→64 通道，尺寸 /2
```

### Depthwise 卷积
```python
nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
# 每通道独立卷积，参数少
```

### Pointwise 卷积 (1×1 卷积)
```python
nn.Conv2d(32, 64, kernel_size=1)
# 只混合通道，不看空间
```

### 空洞卷积
```python
nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2)
# 感受野 5，参数量同 k=3
```

---

## B.4 GTCRN 中的实际例子

### DSConv (下采样)
```
Conv(9→16, k=1×5, s=1×2, p=0×2): 129→65

nn.Conv2d(
    in_channels=9,
    out_channels=16,
    kernel_size=(1, 5),   # 时间轴 1，频率轴 5
    stride=(1, 2),        # 时间不变，频率 /2
    padding=(0, 2)        # 频率轴填充保证对齐
)

# 129 → (129 + 2*2 - 5) / 2 + 1 = 65
```

### GTConvBlock 的空洞卷积
```
GTConvBlock(16, d=2)

nn.Conv2d(
    in_channels=16,
    out_channels=16,
    kernel_size=(3, 3),
    dilation=(2, 1),      # 时间轴空洞 2，频率轴不空洞
    padding=(2, 1)        # 保持尺寸
)

# 时间轴感受野: 1 + (3-1)*2 = 5
```

### DW-Separable 卷积
```
# V1 的 DSConv 拆成两步:

# 1. Depthwise: 空间卷积
nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
# 参数: 32 × 3 × 3 = 288

# 2. Pointwise: 通道混合
nn.Conv2d(32, 64, kernel_size=1)
# 参数: 32 × 64 × 1 × 1 = 2048

# 总计: 2336
# 对比标准卷积: 32 × 64 × 3 × 3 = 18432
# 省了约 8 倍
```

---

## B.5 尺寸计算公式

### 标准公式
```
out_size = floor((in_size + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
```

### 简化版 (常用情况)

当 `dilation=1` 时：
```
out_size = floor((in_size + 2*padding - kernel) / stride + 1)
```

当 `stride=1, padding=kernel//2` 时：
```
out_size = in_size  (尺寸不变)
```

当 `stride=2, padding=kernel//2` 时：
```
out_size ≈ in_size / 2  (下采样)
```

### 转置卷积 (上采样)
```
out_size = (in_size - 1) * stride - 2*padding + kernel + output_padding
```

---

## B.6 参数量计算

### 标准卷积
```
params = C_out × C_in × k_h × k_w + C_out (bias)
       ≈ C_out × C_in × k²
```

### Depthwise 卷积
```
params = C × k_h × k_w + C (bias)
       ≈ C × k²
```

### Pointwise 卷积
```
params = C_out × C_in + C_out (bias)
       ≈ C_out × C_in
```

### DW-Separable
```
params = C_in × k² + C_out × C_in
       ≈ C × k² + C² (当 C_in = C_out = C)

对比标准: C² × k²
节省比例: (k² + C) / (C × k²) ≈ 1/C + 1/k²
```
