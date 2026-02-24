# 02. 输入表示：为什么用 ERB

GTCRN把513个FFT频点压缩到219个ERB频带。这个压缩不是随便做的，背后有听觉生理学的支撑。

## 人耳的频率感知

FFT出来的频点是等间距的，比如16kHz采样、1024点FFT，每个频点间隔约15.6Hz。但人耳对频率的感知不是线性的：

- 低频区分辨率高：100Hz和150Hz，人耳能明显区分
- 高频区分辨率低：6000Hz和6050Hz，人耳几乎听不出差别

这意味着FFT的高频部分有很多"冗余"——对人耳来说，好几个相邻频点听起来是一样的。

## ERB的定义

ERB（Equivalent Rectangular Bandwidth）是根据人耳听觉滤波器测量出来的带宽：

$$ERB(f) = 24.7 \times (4.37 \times f/1000 + 1)$$

在100Hz处，ERB约30Hz；在4000Hz处，ERB约460Hz。

GTCRN的ERB变换就是按照这个规律来合并频点：低频保留更多细节，高频合并更多。

## 513 → 219的压缩

GTCRN用的是1024点FFT（513个频点），压缩到219个ERB频带。压缩比约2.3倍。

为什么是219不是更少？因为语音增强对频率分辨率要求比语音识别高。识别任务用40-80个Mel频带就够了，但增强任务需要更精细的频率信息来重建语音。

219是个平衡点：
- 比513少很多，计算量大幅下降
- 比40-80多很多，保留足够的频率细节

## 计算量节省

从513压到219，后续所有操作的计算量都按比例下降：

- GTConv的卷积：频率维度从513降到219
- DPGRNN：频率维度的循环次数减少
- 跳跃连接：特征图尺寸减小

整体计算量大约减少40-50%。

## ERB变换的实现

```python
class ERBTransform(nn.Module):
    def __init__(self, n_fft=1024, n_erb=219, sr=48000):
        super().__init__()
        # 预计算ERB滤波器组 [n_erb, n_fft//2+1]
        filters = self._make_erb_filters(n_fft, n_erb, sr)
        self.register_buffer('filters', filters)

    def _make_erb_filters(self, n_fft, n_erb, sr):
        # 计算每个ERB频带的中心频率和带宽
        # 生成三角形或gammatone滤波器
        # 返回 [n_erb, n_fft//2+1] 的滤波器矩阵
        ...

    def forward(self, stft):
        # stft: [B, T, F, 2]  F=513
        # 矩阵乘法做频带合并
        real = torch.matmul(stft[..., 0], self.filters.T)
        imag = torch.matmul(stft[..., 1], self.filters.T)
        return torch.stack([real, imag], dim=-1)  # [B, T, 219, 2]
```

## 逆ERB变换

输出时需要从219恢复到513：

```python
class IERBTransform(nn.Module):
    def __init__(self, n_fft=1024, n_erb=219):
        super().__init__()
        # 用伪逆矩阵
        erb_filters = self._make_erb_filters(...)
        inv_filters = torch.pinverse(erb_filters)
        self.register_buffer('inv_filters', inv_filters)

    def forward(self, erb):
        # erb: [B, T, 219, 2]
        real = torch.matmul(erb[..., 0], self.inv_filters.T)
        imag = torch.matmul(erb[..., 1], self.inv_filters.T)
        return torch.stack([real, imag], dim=-1)  # [B, T, 513, 2]
```

伪逆恢复会有一点信息损失，但对语音质量影响不大。

## 和Mel的区别

Mel频谱在语音识别领域用得多，ERB在语音增强领域更常见：

- **Mel**：经验公式，主要基于主观听感实验，通常用40-80个频带
- **ERB**：基于听觉滤波器的生理测量，可以用更多频带保留细节

对于语音增强任务，ERB的优势是可以在保持听觉相关性的同时，使用更多的频带（219 vs 80），重建质量更好。
