# 输出层设计：掩码 vs 直接映射

语音增强的输出有两种思路：直接预测干净语音的频谱，或者预测一个掩码去乘带噪频谱。GTCRN选了后者。

## 为什么用掩码

直接映射看起来更直接，但有几个问题：

1. **输出范围不确定**：干净语音的频谱值范围很大，网络不好学
2. **容易过拟合**：网络可能记住训练集的特定模式，泛化差
3. **不稳定**：训练时容易出现极端值

掩码就好控制多了。掩码值在0到1之间（或者稍微超出一点），物理意义明确：0表示完全抑制，1表示完全保留。

## 复数掩码

传统方法只处理幅度谱，相位直接用带噪语音的。但相位对语音质量影响很大，尤其是在低信噪比的时候。

复数掩码同时处理幅度和相位：

$$\hat{S} = M_{complex} \odot Y$$

其中 $M_{complex} = M_r + jM_i$，$\odot$ 是复数乘法。

展开来看：
- 幅度变化：$|M_{complex}|$ 控制幅度缩放
- 相位变化：$\angle M_{complex}$ 控制相位旋转

```python
def complex_multiply(noisy, mask):
    # noisy, mask: [..., 2] 最后维度是(real, imag)
    real = noisy[..., 0] * mask[..., 0] - noisy[..., 1] * mask[..., 1]
    imag = noisy[..., 0] * mask[..., 1] + noisy[..., 1] * mask[..., 0]
    return torch.stack([real, imag], dim=-1)
```

## 掩码约束

掩码不能太离谱，需要约束：

```python
class MaskEstimator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 64, kernel_size=1)  # 32实部+32虚部

    def forward(self, features):
        raw_mask = self.conv(features)
        # 用tanh约束到[-1, 1]范围
        # 实际掩码可以稍微超出[0,1]，允许轻微增强
        return torch.tanh(raw_mask)
```

用tanh而不是sigmoid，是因为复数掩码的实部虚部可以是负的。

## 消融实验

| 输出方式 | PESQ | 训练稳定性 |
|----------|------|-----------|
| 幅度掩码 | 3.25 | 高 |
| 复数掩码（硬约束） | 3.38 | 中 |
| **复数掩码（软约束）** | **3.42** | **高** |
| 直接映射 | 3.35 | 低 |

复数掩码+软约束是最优组合。

## 相位处理的难点

相位比幅度难处理：

1. **周期性**：相位在 $[-\pi, \pi]$ 范围内循环，$\pi$ 和 $-\pi$ 其实是同一个值
2. **高动态**：相位变化很快，相邻帧之间可能差很多
3. **梯度问题**：相位跳变会导致梯度不稳定

GTCRN的做法是不直接预测相位，而是预测复数掩码，让网络隐式地学习相位修正。这样避开了相位的周期性问题。

## 完整输出流程

```python
class GTCRNOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),  # 32 ERB × 2 (实部虚部)
            nn.Tanh()
        )

    def forward(self, features, noisy_erb):
        # features: [B, 256, T, F] 网络提取的特征
        # noisy_erb: [B, 32, T, 2] 带噪ERB复数谱

        # 估计掩码
        mask = self.mask_net(features)  # [B, 64, T, F]
        mask = mask.view(B, 32, 2, T, F).permute(0, 1, 3, 4, 2)  # [B, 32, T, F, 2]

        # 应用掩码
        enhanced = complex_multiply(noisy_erb, mask)
        return enhanced
```

掩码乘完之后还要做逆ERB变换，恢复到257个频点，再做逆STFT得到时域波形。
