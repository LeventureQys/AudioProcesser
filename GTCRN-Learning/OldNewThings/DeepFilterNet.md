# DeepFilterNet：频域深度滤波

官方仓库：[Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

## 概述

DeepFilterNet 是 2022 年德国 Fraunhofer IIS 的 Hendrik Schröter 等人提出的语音增强方法。在 DNS Challenge 2022 上取得了优异成绩，同时计算量控制得当，是性能和效率之间的良好平衡。

核心创新是 **深度滤波 (Deep Filtering)**：不是简单地给每个频点乘一个增益（传统掩码方法），而是预测一个时变的 FIR 滤波器，对相邻几帧做卷积。这样能利用时间上下文，同时隐式地处理相位。

---

## 一、和 RNNoise 的区别

| | RNNoise | DeepFilterNet |
|---|---------|---------------|
| 核心思路 | 频带增益 | 深度滤波 |
| 相位处理 | 不处理 | 隐式处理 |
| 网络 | 纯 GRU | 编码器 + GRU + 双路径解码器 |
| 频率处理 | 32 个 ERB 频带 | 双路径（ERB + 线性） |
| 参数量 | ~100K | ~720K (DeepFilterNet2) |
| PESQ | 2.54 | 2.89 |

DeepFilterNet 用更多参数换来更好效果，但比 DCCRN 那些动辄几百万参数的方法轻量很多。

---

## 二、默认配置参数

DeepFilterNet2 的关键默认参数（来自官方 `config.py` 和 `deepfilternet2.py`）：

### 信号处理参数 (DfParams)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| sr | 48000 | 采样率 |
| fft_size | 960 | FFT 大小 (48kHz × 20ms) |
| hop_size | 480 | 帧移 (48kHz × 10ms) |
| nb_erb | 32 | ERB 频带数 |
| nb_df | 96 | 深度滤波处理的频点数 (0-5kHz) |
| df_order | 5 | 滤波器阶数 |
| df_lookahead | 0 | 前瞻帧数 |

### 模型参数 (ModelParams)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| conv_ch | 16 | 卷积通道数 |
| conv_kernel | (1, 3) | 卷积核大小 |
| conv_kernel_inp | (3, 3) | 输入层卷积核 |
| conv_depthwise | True | 使用深度可分离卷积 |
| emb_hidden_dim | 256 | 嵌入 GRU 隐藏层维度 |
| emb_num_layers | 2 | 嵌入 GRU 层数 |
| df_hidden_dim | 256 | DF 解码器 GRU 隐藏层维度 |
| df_num_layers | 3 | DF 解码器 GRU 层数 |
| df_n_iter | 2 | 深度滤波迭代次数 |
| gru_type | "grouped" | GRU 类型 |
| gru_groups | 1 | GRU 分组数 |
| group_shuffle | True | 组间 shuffle |

---

## 三、ERB 频带

DeepFilterNet 也用了 ERB 频带分组，思路和 RNNoise 一样。

ERB（Equivalent Rectangular Bandwidth）是基于人耳听觉特性的频带划分。人耳在低频处分辨率高，高频处分辨率低，ERB 按这个特性划分。

带宽公式：
$$B_{ERB}(f_c) = 24.7 \times (4.37 \times 10^{-3} f_c + 1)$$

具体数值：
- 100Hz 处，带宽约 35Hz
- 1000Hz 处，带宽约 132Hz
- 8000Hz 处，带宽约 894Hz

为什么用 ERB 不用 Mel？在语音增强任务上，ERB 效果略好一点，而且公式更简洁。

---

## 四、深度滤波 (Deep Filtering)

这是 DeepFilterNet 最核心的创新。

### 传统掩码 vs 深度滤波

**传统掩码方法**：给每个时频点乘一个系数
$$\hat{S}(k, t) = M(k, t) \cdot X(k, t)$$

**深度滤波**：对相邻几帧做卷积
$$\hat{S}(k, t) = \sum_{\tau=0}^{L-1} H(k, t, \tau) \cdot X(k, t-\tau)$$

$H$ 是网络预测的滤波器系数，$L$ 是滤波器长度（DeepFilterNet 用 5）。

直观理解：
```
传统掩码：当前帧 × 系数 = 输出

深度滤波：
  当前帧×H[0] + 前1帧×H[1] + 前2帧×H[2] + 前3帧×H[3] + 前4帧×H[4] = 输出
```

### 为什么能处理相位？

传统掩码只能改变幅度，相位没法动。深度滤波用的是**复数乘法**，$H$ 和 $X$ 都是复数，相乘时相位会叠加。

而且因为用了多帧信息，网络可以学到：语音在相邻帧之间高度相关，噪声相关性低。通过合适的滤波器系数，可以增强语音、抑制噪声，同时隐式修正相位。

### DfOp 实现

官方实现有 6 种 forward 方法，默认用 `real_unfold`：

```python
class DfOp(nn.Module):
    """Deep Filtering Operation

    官方注释: "All forward methods should be mathematically similar.
    DeepFilterNet results are obtained with 'real_unfold'."
    """
    def __init__(self, nb_df, df_order=5, df_lookahead=0):
        self.df_order = df_order      # 滤波器长度，默认 5
        self.nb_df = nb_df            # 处理的频点数，默认 96
        self.df_lookahead = df_lookahead  # 前瞻帧数，默认 0

    def forward_real_unfold(self, spec, coefs, alpha):
        """
        spec:  [B, 1, T, F, 2]  输入频谱 (实部虚部分离)
        coefs: [B, T, O, F, 2]  滤波器系数 (O=df_order)
        alpha: [B, T, 1]        混合系数
        """
        # 取出需要处理的低频部分
        spec_f = spec[..., :self.nb_df, :]

        # 时间轴 padding，获取历史帧
        # pad (df_order - df_lookahead - 1) 帧在前面
        padded = F.pad(spec_f, (0,0,0,0, self.df_order-1-self.df_lookahead, self.df_lookahead))

        # unfold: 把连续的 df_order 帧展开
        # [B, 1, T, F, 2] → [B, 1, T, F, 2, O]
        padded = padded.unfold(dimension=2, size=self.df_order, step=1)

        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # padded: [B, 1, T, F, 2, O]
        # coefs:  [B, T, O, F, 2] → 需要 permute
        coefs = coefs.permute(0, 4, 1, 3, 2)  # [B, 2, T, F, O]

        out_re = padded[..., 0, :] * coefs[:, 0:1] - padded[..., 1, :] * coefs[:, 1:2]
        out_im = padded[..., 0, :] * coefs[:, 1:2] + padded[..., 1, :] * coefs[:, 0:1]

        # 对 df_order 维度求和
        out_re = out_re.sum(dim=-1)  # [B, 1, T, F]
        out_im = out_im.sum(dim=-1)

        spec_f = torch.stack([out_re, out_im], dim=-1)  # [B, 1, T, F, 2]

        # alpha 混合: 原始频谱和滤波后频谱加权
        spec_out = spec.clone()
        spec_out[..., :self.nb_df, :] = (
            spec_f * alpha.unsqueeze(-1) +
            spec[..., :self.nb_df, :] * (1 - alpha.unsqueeze(-1))
        )

        return spec_out
```

关键点：
- `unfold` 把时间轴展开成滑动窗口
- 复数乘法隐式处理相位
- `alpha` 控制滤波强度，范围 [0, 1]

---

## 五、整体架构

### 5.1 架构总览

```
输入: 带噪语音
    │
    ▼ STFT (fft=960, hop=480)
    │
频谱 [B, 1, T, 481, 2]  (481 = fft_size//2 + 1)
    │
    ├────────────────────────────────────┬─────────────────────────────┐
    │                                    │                             │
    ▼                                    ▼                             │
ERB 变换                              取低频部分                        │
    │                                    │                             │
    ▼                                    ▼                             │
feat_erb [B, 1, T, 32]           feat_spec [B, 2, T, 96]              │
(32 个 ERB 频带)                  (96 频点, 实部+虚部)                  │
    │                                    │                             │
    └──────────────┬─────────────────────┘                             │
                   │                                                   │
                   ▼                                                   │
    ┌─────────────────────────────────────┐                           │
    │             Encoder                  │                           │
    │                                      │                           │
    │  ERB 路径 ──────────┬──── DF 路径    │                           │
    │  e0,e1,e2,e3       │     c0,c1      │                           │
    │         └──────────┴─────┘          │                           │
    │                   ↓                  │                           │
    │             emb [B,T,256]            │                           │
    │                   │                  │                           │
    │                   ▼                  │                           │
    │         lsnr [B,T,1] (SNR预测)       │                           │
    └─────────────────────────────────────┘                           │
                   │                                                   │
         ┌─────────┴─────────┐                                        │
         │                   │                                        │
         ▼                   ▼                                        │
┌────────────────┐  ┌────────────────┐                               │
│  ERB Decoder   │  │   DF Decoder   │                               │
│                │  │                │                               │
│ emb + e3,e2,   │  │ emb + c0       │                               │
│      e1,e0     │  │                │                               │
│      ↓         │  │      ↓         │                               │
│ mask [B,1,T,32]│  │ coefs, alpha   │                               │
└────────────────┘  └────────────────┘                               │
         │                   │                                        │
         ▼                   │                                        │
┌────────────────┐           │                                        │
│     Mask       │           │                                        │
│ ERB→线性插值   │           │                                        │
│ 应用到频谱     │←──────────┼────────────────────────────────────────┘
└────────────────┘           │
         │                   │
         ▼                   ▼
┌─────────────────────────────────────────┐
│              DfOp × df_n_iter           │
│                                         │
│   for _ in range(2):                    │
│       spec = df_op(spec, coefs, alpha)  │
└─────────────────────────────────────────┘
                   │
                   ▼
         增强后频谱 [B, 1, T, 481, 2]
                   │
                   ▼ iSTFT
                   │
              干净语音
```

### 5.2 为什么这么设计？

**双路径分工：**

| 路径 | 处理范围 | 分辨率 | 输出 | 作用 |
|------|---------|--------|------|------|
| ERB | 全频段 0-24kHz | 粗 (32 频带) | 实数增益 | 整体能量调整 |
| DF | 低频 0-5kHz | 细 (96 频点) | 复数滤波器 | 精细处理+相位 |

低频是语音的主体，需要精细处理，相位在低频更重要。高频主要是辅音和噪声，粗粒度处理就够了。

---

## 六、Encoder 详解

Encoder 有两条并行路径，最后合并成统一的 embedding。

### 6.1 结构图

```
feat_erb [B, 1, T, 32]              feat_spec [B, 2, T, 96]
    │                                    │
    ▼                                    ▼
erb_conv0: Conv(1→16, 3×3)          df_conv0: Conv(2→16, 3×3)
    │ [B, 16, T, 32]                     │ [B, 16, T, 96]
    ▼                                    │
erb_conv1: Conv(16→16, 1×3, s=1×2)       │
    │ [B, 16, T, 16]  ← 32→16            ▼
    ▼                                df_conv1: Conv(16→16, 1×3, s=1×2)
erb_conv2: Conv(16→16, 1×3, s=1×2)       │ [B, 16, T, 48]  ← 96→48
    │ [B, 16, T, 8]   ← 16→8             │
    ▼                                    ▼
erb_conv3: Conv(16→16, 1×3)          flatten → [B, T, 768]
    │ [B, 16, T, 8]                      │
    │                                    ▼
    ▼                                df_fc_emb: Linear(768→128)
flatten → [B, T, 128]                    │ [B, T, 128]
    │                                    │
    └──────────────┬─────────────────────┘
                   │
                   ▼ Add 或 Concat (取决于 enc_concat)
                   │
            [B, T, 128] 或 [B, T, 256]
                   │
                   ▼
            emb_gru: GroupedGRU(128/256, 256, layers=2)
                   │
                   ▼
            emb [B, T, 256]
                   │
                   ├───────────────────┐
                   │                   ▼
                   │            lsnr_fc: Linear(256→1) + Sigmoid
                   │                   │
                   │                   ▼
                   │            lsnr [B, T, 1]  (本地信噪比估计)
                   │
    输出: e0, e1, e2, e3, emb, c0, lsnr
```

### 6.2 代码实现

```python
class Encoder(nn.Module):
    def __init__(self, p: ModelParams):
        super().__init__()
        # 配置参数
        self.nb_erb = p.nb_erb      # 32
        self.nb_df = p.nb_df        # 96
        conv_ch = p.conv_ch         # 16

        # ERB 路径: 32 → 16 → 8 → 8
        self.erb_conv0 = Conv2dNormAct(
            1, conv_ch,
            kernel_size=p.conv_kernel_inp,  # (3, 3)
            separable=p.conv_depthwise
        )
        self.erb_conv1 = Conv2dNormAct(
            conv_ch, conv_ch,
            kernel_size=p.conv_kernel,      # (1, 3)
            fstride=2                       # 频率轴下采样
        )
        self.erb_conv2 = Conv2dNormAct(conv_ch, conv_ch, p.conv_kernel, fstride=2)
        self.erb_conv3 = Conv2dNormAct(conv_ch, conv_ch, p.conv_kernel, fstride=1)

        # DF 路径: 96 → 48
        self.df_conv0 = Conv2dNormAct(2, conv_ch, p.conv_kernel_inp, separable=p.conv_depthwise)
        self.df_conv1 = Conv2dNormAct(conv_ch, conv_ch, p.conv_kernel, fstride=2)

        # 计算维度
        # ERB: 32 → 16 → 8 → 8, flatten: 16 × 8 = 128
        # DF:  96 → 48, flatten: 16 × 48 = 768 → FC → 128
        emb_in_dim = conv_ch * (self.nb_erb // 4)  # 16 × 8 = 128

        self.df_fc_emb = GroupedLinear(
            conv_ch * (self.nb_df // 2),  # 16 × 48 = 768
            emb_in_dim,                    # 128
            groups=p.lin_groups
        )

        # 合并后的 GRU
        gru_in_dim = emb_in_dim * 2 if p.enc_concat else emb_in_dim
        self.emb_gru = get_gru(
            p.gru_type,
            gru_in_dim,             # 128 或 256
            p.emb_hidden_dim,       # 256
            num_layers=p.emb_num_layers,  # 2
            groups=p.gru_groups,
            shuffle=p.group_shuffle
        )

        # LSNR 预测头
        self.lsnr_fc = nn.Sequential(
            nn.Linear(p.emb_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.lsnr_scale = p.lsnr_max - p.lsnr_min  # 35 - (-15) = 50
        self.lsnr_offset = p.lsnr_min              # -15

    def forward(self, feat_erb, feat_spec):
        """
        feat_erb:  [B, 1, T, 32]   ERB 特征
        feat_spec: [B, 2, T, 96]   DF 特征 (实部+虚部)
        """
        # ERB 路径
        e0 = self.erb_conv0(feat_erb)   # [B, 16, T, 32]
        e1 = self.erb_conv1(e0)          # [B, 16, T, 16]
        e2 = self.erb_conv2(e1)          # [B, 16, T, 8]
        e3 = self.erb_conv3(e2)          # [B, 16, T, 8]

        # DF 路径
        c0 = self.df_conv0(feat_spec)    # [B, 16, T, 96]
        c1 = self.df_conv1(c0)           # [B, 16, T, 48]

        # 展平
        # e3: [B, 16, T, 8] → [B, T, 128]
        emb = e3.permute(0, 2, 1, 3).flatten(2)
        # c1: [B, 16, T, 48] → [B, T, 768] → [B, T, 128]
        cemb = c1.permute(0, 2, 1, 3).flatten(2)
        cemb = self.df_fc_emb(cemb)

        # 合并 (默认是 Add)
        emb = emb + cemb  # [B, T, 128]

        # GRU 时序建模
        emb, _ = self.emb_gru(emb)  # [B, T, 256]

        # LSNR 预测
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr
```

### 6.3 关键点

1. **频率轴下采样**: ERB 路径 32→16→8，压缩 4 倍；DF 路径 96→48，压缩 2 倍
2. **时间轴不变**: 所有卷积的时间步长都是 1
3. **因果卷积**: 只 pad 时间轴的左边，不看未来
4. **LSNR 预测**: 估计本地信噪比，范围 [-15, 35] dB

---

## 七、ERB Decoder 详解

ERB Decoder 是 Encoder ERB 路径的镜像，生成 ERB 频带增益掩码。

### 7.1 结构图

```
emb [B, T, 256]
    │
    ▼
emb_gru: GroupedGRU(256, 256, layers=1)
    │ [B, T, 256]
    ▼
fc_emb: Linear(256→128)
    │ [B, T, 128]
    ▼
reshape → [B, 16, T, 8]
    │
    │    e3 [B, 16, T, 8]
    │        │
    ▼        ▼
    ├── conv3p(e3) ───┐
    │   [B, 16, T, 8] │
    │        +        │
    │        ↓        │
    └──► add ◄────────┘
           │
           ▼
    convt3: Conv(16→16, 1×3)
           │ [B, 16, T, 8]
           │
           │    e2 [B, 16, T, 8]
           │        │
           ▼        ▼
    ├── conv2p(e2) ───┐
           +          │
           ↓          │
    add ◄─────────────┘
           │
           ▼
    convt2: ConvT(16→16, 1×3, s=1×2)  ← 上采样 8→16
           │ [B, 16, T, 16]
           │
           │    e1 [B, 16, T, 16]
           │        │
           ▼        ▼
    ├── conv1p(e1) ───┐
           +          │
           ↓          │
    add ◄─────────────┘
           │
           ▼
    convt1: ConvT(16→16, 1×3, s=1×2)  ← 上采样 16→32
           │ [B, 16, T, 32]
           │
           │    e0 [B, 16, T, 32]
           │        │
           ▼        ▼
    ├── conv0p(e0) ───┐
           +          │
           ↓          │
    add ◄─────────────┘
           │
           ▼
    conv0_out: Conv(16→1, 1×3) + Sigmoid
           │
           ▼
    mask [B, 1, T, 32]  (ERB 增益, 范围 [0,1])
```

### 7.2 代码实现

```python
class ErbDecoder(nn.Module):
    def __init__(self, p: ModelParams):
        super().__init__()
        conv_ch = p.conv_ch  # 16

        # GRU + FC
        self.emb_gru = get_gru(
            p.gru_type,
            p.emb_hidden_dim,       # 256
            p.emb_hidden_dim,       # 256
            num_layers=1
        )
        self.fc_emb = GroupedLinear(
            p.emb_hidden_dim,       # 256
            conv_ch * (p.nb_erb // 4),  # 16 × 8 = 128
            groups=p.lin_groups
        )

        # Pathway 卷积 (1×1, 处理 skip connection)
        self.conv3p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1)
        self.conv2p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1)
        self.conv1p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1)
        self.conv0p = Conv2dNormAct(conv_ch, conv_ch, kernel_size=1)

        # 转置卷积 (上采样)
        self.convt3 = Conv2dNormAct(conv_ch, conv_ch, p.conv_kernel)
        self.convt2 = ConvTranspose2dNormAct(conv_ch, conv_ch, p.conv_kernel, fstride=2)
        self.convt1 = ConvTranspose2dNormAct(conv_ch, conv_ch, p.conv_kernel, fstride=2)

        # 输出层
        self.conv0_out = Conv2dNormAct(
            conv_ch, 1,
            p.conv_kernel,
            activation=nn.Sigmoid()
        )

    def forward(self, emb, e3, e2, e1, e0):
        """
        emb: [B, T, 256]
        e0~e3: Encoder 的 skip connections
        """
        # GRU + FC
        emb, _ = self.emb_gru(emb)              # [B, T, 256]
        emb = self.fc_emb(emb)                   # [B, T, 128]

        # reshape: [B, T, 128] → [B, 16, T, 8]
        emb = emb.view(emb.shape[0], emb.shape[1], -1, self.conv_ch)
        emb = emb.permute(0, 3, 1, 2)            # [B, 16, T, 8]

        # 逐层解码 + skip connection
        x = self.convt3(self.conv3p(e3) + emb)   # [B, 16, T, 8]
        x = self.convt2(self.conv2p(e2) + x)     # [B, 16, T, 16]
        x = self.convt1(self.conv1p(e1) + x)     # [B, 16, T, 32]
        m = self.conv0_out(self.conv0p(e0) + x)  # [B, 1, T, 32]

        return m  # ERB 增益掩码
```

### 7.3 Skip Connection 的作用

| 层级 | Encoder 输出 | Decoder 使用 | 作用 |
|------|-------------|--------------|------|
| e3 | 最深特征 | 第一个 skip | 高级语义 |
| e2 | 中间特征 | 第二个 skip | 中层模式 |
| e1 | 浅层特征 | 第三个 skip | 局部细节 |
| e0 | 最浅特征 | 最后 skip | 精细边界 |

---

## 八、DF Decoder 详解

DF Decoder 生成深度滤波器系数，是网络最关键的部分。

### 8.1 结构图

```
emb [B, T, 256]                     c0 [B, 16, T, 96]
    │                                    │
    ▼                                    ▼
df_gru: GroupedGRU(256, 256, layers=3)  df_convp: Conv(16→10, 1×1)
    │ [B, T, 256]                        │ [B, 10, T, 96]
    │                                    │
    │    ┌─────── df_skip ───────┐       ▼
    │    │ (可选: Linear/None)   │    permute → [B, T, 96, 10]
    │    │                       │       │
    ▼    ▼                       │       │
    ├────┤                       │       │
    │    │                       │       │
    │ df_out: Linear(256→960)    │       │
    │     + Tanh                 │       │
    │    │ [B, T, 960]           │       │
    │    │                       │       │
    │    ▼                       │       │
    │ reshape → [B, T, 96, 10]   │       │
    │    │                       │       │
    │    └───────── + ───────────┘◄──────┘
    │              │
    │              ▼
    │    coefs [B, T, 96, 10]
    │              │
    │              ▼
    │    reshape → [B, T, 5, 96, 2]  (df_order=5, 实部+虚部)
    │
    │
    └───► df_fc_a: Linear(256→1) + Sigmoid
              │
              ▼
         alpha [B, T, 1]  (混合系数)

输出: (coefs, alpha)
```

### 8.2 代码实现

```python
class DfDecoder(nn.Module):
    def __init__(self, p: ModelParams):
        super().__init__()
        self.df_order = p.df_order  # 5
        self.nb_df = p.nb_df        # 96
        self.out_dim = self.nb_df * self.df_order * 2  # 96 × 5 × 2 = 960

        # 卷积 pathway (从 encoder 的 c0 直接连过来)
        # 输出通道 = df_order × 2 = 10
        self.df_convp = Conv2dNormAct(
            p.conv_ch,              # 16
            self.df_order * 2,      # 10
            kernel_size=1
        )

        # 3 层 GRU
        self.df_gru = get_gru(
            p.gru_type,
            p.emb_hidden_dim,       # 256
            p.df_hidden_dim,        # 256
            num_layers=p.df_num_layers,  # 3
            groups=p.gru_groups
        )

        # 可选的 skip connection (默认 None)
        self.df_skip = None
        if p.df_gru_skip == "identity":
            self.df_skip = nn.Identity()
        elif p.df_gru_skip == "grouped":
            self.df_skip = GroupedLinearEinsum(
                p.emb_hidden_dim,
                p.df_hidden_dim,
                groups=p.lin_groups
            )

        # 输出层: 256 → 960
        self.df_out = nn.Sequential(
            GroupedLinear(p.df_hidden_dim, self.out_dim, groups=p.lin_groups),
            nn.Tanh()
        )

        # Alpha 输出: 控制滤波强度
        self.df_fc_a = nn.Sequential(
            nn.Linear(p.df_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, c0):
        """
        emb: [B, T, 256]
        c0:  [B, 16, T, 96]  Encoder 的 DF 路径输出
        """
        # GRU 处理
        c, _ = self.df_gru(emb)  # [B, T, 256]

        # 可选 skip connection
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        # Pathway: 从 encoder 直接连过来
        # c0: [B, 16, T, 96] → [B, 10, T, 96] → [B, T, 96, 10]
        c0 = self.df_convp(c0)
        c0 = c0.permute(0, 2, 3, 1)  # [B, T, 96, 10]

        # Alpha: 控制滤波强度
        alpha = self.df_fc_a(c)  # [B, T, 1]

        # 滤波器系数
        # c: [B, T, 256] → [B, T, 960]
        c = self.df_out(c)
        # reshape: [B, T, 960] → [B, T, 96, 10]
        c = c.view(c.shape[0], c.shape[1], self.nb_df, self.df_order * 2)

        # 加上 pathway skip
        c = c + c0

        # 最终 reshape: [B, T, 96, 10] → [B, T, 5, 96, 2]
        c = c.view(c.shape[0], c.shape[1], self.df_order, self.nb_df, 2)

        return c, alpha
```

### 8.3 输出维度解析

```
滤波器系数 coefs: [B, T, df_order, nb_df, 2]
                  [B, T,    5,      96,   2]

- B: batch size
- T: 时间帧数
- df_order=5: 滤波器长度 (看当前帧 + 前 4 帧)
- nb_df=96: 处理的频点数 (0-5kHz)
- 2: 复数的实部和虚部

alpha: [B, T, 1]
- 混合系数，控制 "原始频谱 vs 滤波后频谱" 的权重
- 范围 [0, 1]，Sigmoid 输出
```

---

## 九、主网络 DfNet

DfNet 把所有模块组装起来。

### 9.1 完整流程

```python
class DfNet(nn.Module):
    def __init__(self, erb_fb, erb_inv_fb, run_df=True):
        super().__init__()
        p = ModelParams()

        self.enc = Encoder(p)
        self.erb_dec = ErbDecoder(p)
        self.df_dec = DfDecoder(p)

        # ERB → 线性频率的掩码应用
        self.mask = Mask(erb_fb, erb_inv_fb, post_filter=p.mask_pf)

        # 深度滤波操作
        self.df_op = DfOp(
            p.nb_df,
            p.df_order,
            p.df_lookahead
        )

        self.df_n_iter = p.df_n_iter  # 迭代次数，默认 2
        self.run_df = run_df

    def forward(self, spec, feat_erb, feat_spec):
        """
        spec:      [B, 1, T, F, 2]   原始频谱 (F=481)
        feat_erb:  [B, 1, T, 32]     ERB 特征
        feat_spec: [B, 2, T, 96]     DF 特征
        """
        # ===== 1. Encoder =====
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)

        # ===== 2. ERB Decoder → 生成掩码 =====
        m = self.erb_dec(emb, e3, e2, e1, e0)  # [B, 1, T, 32]

        # ===== 3. 应用 ERB 掩码 =====
        # 把 32 个 ERB 增益插值到 481 个线性频点，然后乘到频谱上
        spec = self.mask(spec, m)

        # ===== 4. DF Decoder → 生成滤波器系数 =====
        df_coefs, df_alpha = self.df_dec(emb, c0)
        # df_coefs: [B, T, 5, 96, 2]
        # df_alpha: [B, T, 1]

        # ===== 5. 迭代应用深度滤波 =====
        if self.run_df:
            for _ in range(self.df_n_iter):
                spec = self.df_op(spec, df_coefs, df_alpha)

        return spec, m, lsnr, df_alpha
```

### 9.2 处理流程示意

```
时间线:   ────────────────────────────────────►

输入频谱: [噪声+语音]
              │
              ▼
         ┌─────────┐
         │ Encoder │  提取特征
         └────┬────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌─────────┐       ┌─────────┐
│ERB Dec  │       │ DF Dec  │
│  mask   │       │ coefs   │
└────┬────┘       └────┬────┘
     │                  │
     ▼                  │
┌─────────┐             │
│  Mask   │ ◄───────────┘
│ ERB增益 │
└────┬────┘
     │
     ▼
┌───────────────────────────┐
│       DfOp × 2            │  迭代细化
│  (深度滤波, 处理相位)      │
└───────────────────────────┘
              │
              ▼
输出频谱: [干净语音]
```

---

## 十、关键模块详解

### 10.1 Conv2dNormAct (因果卷积)

DeepFilterNet 的所有卷积都是因果的，只 pad 时间轴的左边：

```python
class Conv2dNormAct(nn.Module):
    """因果 2D 卷积

    输入格式: [B, C, T, F] (batch, channel, time, frequency)
    """
    def __init__(self, in_ch, out_ch, kernel_size,
                 fstride=1, separable=False, bias=True):
        super().__init__()

        # kernel_size: (time, freq)
        k_t, k_f = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # 因果 padding: 只 pad 时间轴左边
        # pad 格式: (left_f, right_f, left_t, right_t)
        self.pad = nn.ConstantPad2d((0, 0, k_t - 1, 0), 0.0)

        # Depthwise Separable 卷积
        if separable:
            groups = math.gcd(in_ch, out_ch)
            # Depthwise
            self.conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=(k_t, k_f),
                stride=(1, fstride),
                groups=groups,
                bias=False
            )
            # Pointwise
            self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=bias)
        else:
            groups = 1
            self.conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=(k_t, k_f),
                stride=(1, fstride),
                bias=bias
            )
            self.pw = None

        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, F]
        x = self.pad(x)           # 因果 padding
        x = self.conv(x)
        if self.pw is not None:
            x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x
```

### 10.2 GroupedGRU (分组 GRU)

把输入分成多组，每组独立过 GRU，层间 shuffle 交换信息：

```python
class GroupedGRULayer(nn.Module):
    """单层分组 GRU"""
    def __init__(self, input_size, hidden_size, groups=4):
        super().__init__()
        self.groups = groups
        self.hidden_size = hidden_size

        # 每组的维度
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        group_in = input_size // groups
        group_hidden = hidden_size // groups

        # 每组一个 GRU
        self.grus = nn.ModuleList([
            nn.GRU(group_in, group_hidden, batch_first=True)
            for _ in range(groups)
        ])

    def forward(self, x, h=None):
        # x: [B, T, input_size]
        B, T, _ = x.shape

        # 分组
        x_groups = x.chunk(self.groups, dim=-1)  # list of [B, T, input_size/groups]

        # 每组独立过 GRU
        out_groups = []
        h_groups = []
        for i, gru in enumerate(self.grus):
            h_i = h[:, :, i] if h is not None else None
            out_i, h_i = gru(x_groups[i], h_i)
            out_groups.append(out_i)
            h_groups.append(h_i)

        # 拼接
        out = torch.cat(out_groups, dim=-1)  # [B, T, hidden_size]
        h_out = torch.stack(h_groups, dim=2)

        return out, h_out


class GroupedGRU(nn.Module):
    """多层分组 GRU，层间 shuffle"""
    def __init__(self, input_size, hidden_size, num_layers, groups=4, shuffle=True):
        super().__init__()
        self.num_layers = num_layers
        self.shuffle = shuffle

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(GroupedGRULayer(in_size, hidden_size, groups))

    def forward(self, x, h=None):
        for i, layer in enumerate(self.layers):
            x, h_i = layer(x)

            # 层间 shuffle (除了最后一层)
            if self.shuffle and i < self.num_layers - 1:
                # [B, T, H] → [B, T, groups, H/groups] → [B, T, H/groups, groups] → [B, T, H]
                B, T, H = x.shape
                x = x.view(B, T, -1, self.groups)
                x = x.transpose(2, 3)
                x = x.reshape(B, T, H)

        return x, h
```

Shuffle 的作用：让不同组的信息交换，避免组间隔离。

### 10.3 Mask (ERB → 线性)

把 ERB 增益插值回线性频率：

```python
class Mask(nn.Module):
    def __init__(self, erb_fb, erb_inv_fb, post_filter=False):
        super().__init__()
        # erb_fb:     [F, nb_erb]  线性→ERB 的变换矩阵
        # erb_inv_fb: [nb_erb, F]  ERB→线性 的变换矩阵
        self.register_buffer("erb_fb", erb_fb)
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.post_filter = post_filter

    def forward(self, spec, mask):
        """
        spec: [B, 1, T, F, 2]   频谱 (F=481)
        mask: [B, 1, T, nb_erb] ERB 增益 (nb_erb=32)
        """
        # ERB → 线性频率
        # mask: [B, 1, T, 32] @ [32, 481] → [B, 1, T, 481]
        mask_lin = torch.matmul(mask, self.erb_inv_fb)

        # 应用掩码 (实部和虚部分别乘)
        spec = spec * mask_lin.unsqueeze(-1)

        return spec
```

---

## 十一、损失函数

DeepFilterNet 用了多个损失：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{spec} + \lambda_2 \mathcal{L}_{multi} + \lambda_3 \mathcal{L}_{alpha}$$

### 频谱损失

幅度 L1 + 实部 L1 + 虚部 L1，同时优化幅度和相位：

```python
L_spec = |S| - |Ŝ|_1 + |Re(S) - Re(Ŝ)|_1 + |Im(S) - Im(Ŝ)|_1
```

### 多分辨率 STFT 损失

用不同 FFT 大小（512、1024、2048）算损失，捕获不同时频分辨率：

```python
L_multi = Σ_n L_spec(STFT_n(s), STFT_n(ŝ))
```

### Alpha 损失

对低能量区域更敏感：

```python
L_alpha = ||Ŝ|^α - |S|^α|_1    # α ≈ 0.3
```

为什么不直接用 MSE？MSE 对高能量区域过度关注，低能量的细节容易被忽略。

---

## 十二、计算量和性能

### 版本对比

| 版本 | 参数量 | 特点 |
|------|--------|------|
| DeepFilterNet | ~2M | 原版，效果最好 |
| DeepFilterNet2 | ~720K | 轻量版，性能效率平衡好 |
| DeepFilterNet3 | ~420K | 超轻量，适合嵌入式 |

DeepFilterNet2 的 RTF 在普通 CPU 上约 0.15，比 RNNoise（0.07）慢一倍，但比 DCCRN（0.8）快很多。

延迟方面，帧长 10ms，加上处理时间，总延迟大概 10-20ms，满足实时要求。

### DNS Challenge 2022 结果

| 指标 | DeepFilterNet2 |
|------|----------------|
| PESQ | 2.89 |
| SI-SDR | 17.5 dB |
| DNSMOS | 3.82 |

### 消融实验

| 改动 | PESQ 变化 |
|------|----------|
| 去掉 DF 路径 | -0.17 |
| 去掉 ERB 路径 | -0.21 |
| 滤波器长度 5→1（等价于掩码） | -0.17 |
| ERB 频带 32→16 | -0.11 |

说明双路径设计和深度滤波都很重要。滤波器长度 5 是比较好的选择，再长收益很小。

---

## 十三、与其他方法对比

### vs RNNoise

| | RNNoise | DeepFilterNet |
|---|---------|---------------|
| 定位 | 极致轻量 | 性能效率平衡 |
| 相位 | 不处理 | 隐式处理 |
| 参数量 | 100K | 720K |
| PESQ | 2.54 | 2.89 |
| 适用场景 | 资源极度受限 | 一般实时应用 |

### vs GTCRN

| | GTCRN V1 | DeepFilterNet2 |
|---|---------|----------------|
| 核心思路 | U-Net + 复数掩码 | 双路径 + 深度滤波 |
| 相位处理 | CRM 显式处理 | DF 隐式处理 |
| 参数量 | 139K | 720K |
| DNSMOS | 3.15 | 3.82 |
| 实时支持 | V3 支持 | 支持 |

GTCRN 更轻量，DeepFilterNet 效果更好。

---

## 十四、代码结构

```
DeepFilterNet/
├── df/
│   ├── config.py           # DfParams 配置
│   ├── deepfilternet2.py   # 主模型定义
│   │   ├── ModelParams     # 模型超参数
│   │   ├── Encoder         # 双路径编码器
│   │   ├── ErbDecoder      # ERB 增益解码器
│   │   ├── DfDecoder       # 深度滤波系数解码器
│   │   └── DfNet           # 主网络
│   ├── modules.py          # 基础模块
│   │   ├── Conv2dNormAct   # 因果卷积
│   │   ├── ConvTranspose2dNormAct  # 因果转置卷积
│   │   ├── GroupedGRU      # 分组 GRU
│   │   ├── GroupedLinear   # 分组线性层
│   │   ├── DfOp            # 深度滤波操作
│   │   ├── Mask            # ERB 掩码
│   │   └── ExponentialUnitNorm  # 指数归一化
│   ├── loss.py             # 损失函数
│   └── train.py            # 训练脚本
```

---

## 十五、总结

DeepFilterNet 的核心贡献：

1. **深度滤波**：用时变 FIR 滤波器代替简单掩码，能隐式处理相位
2. **双路径设计**：ERB 处理全频段粗粒度，DF 处理低频精细
3. **分组 GRU + Shuffle**：减少参数量同时保持信息流通
4. **多分辨率损失**：捕获不同时频尺度的信息

DeepFilterNet 是学习语音增强的好材料：比 RNNoise 复杂，但思路清晰；比 DCCRN 轻量，但效果更好。

对于学习顺序：RNNoise（基础）→ DeepFilterNet（改进）→ GTCRN（另一条路线）。
