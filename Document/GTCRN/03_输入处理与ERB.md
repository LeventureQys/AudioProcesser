# 03｜输入处理与 ERB 模块：怎么把 257 个频点压成 129 个还不丢信息？

> 这一章我们要深入第一个"砍"的工具——**ERB 频带合并**。这是 GTCRN 节省算力的最大杠杆，也是把声学知识"硬编码"进网络的典型案例。

## 1. 先看输入：从波形到 STFT，参数怎么定

打开 [infer.py:18](../../third_party/gtcrn/infer.py#L18)：

```python
input = torch.stft(torch.from_numpy(mix), 
                   n_fft=512, 
                   hop_length=256, 
                   win_length=512, 
                   window=torch.hann_window(512).pow(0.5), 
                   return_complex=False)
```

几个关键超参数：

| 参数 | 取值 | 物理意义 |
|:----:|:----:|:----:|
| 采样率 | 16 kHz | 宽带语音标准 |
| `n_fft` | 512 | FFT 长度 |
| `win_length` | 512 | 加窗长度 = 32 ms |
| `hop_length` | 256 | 帧移 = 16 ms |
| 窗函数 | √Hann | 满足完美重构（COLA） |

### 为什么是 32 ms 窗 / 16 ms 帧移？

这是语音处理的"行业默认值"，背后的物理逻辑是：

- **语音的短时平稳性大约在 10-40 ms 之间**。窗太短（比如 8ms），频率分辨率不够，看不清共振峰；窗太长（比如 100ms），把多个音素糊在一起，时间分辨率不够。32ms 是这个 tradeoff 的甜区。
- **帧移 = 窗长 / 2** 是 50% 重叠，配合 √Hann 窗满足 **COLA 条件**（Constant Overlap-Add），保证 ISTFT 重构是无损的（数学上 `Σ window²(n−k·hop) = 1`）。

> 工程提示：如果你要改窗长，必须同步改帧移、改窗函数，否则 ISTFT 会有人工 artifacts。GTCRN 用 √Hann 是为了让"分析窗 × 合成窗 = Hann"，这是 OLA 的标准配方。

### STFT 输出 shape

对于 16kHz / 1秒音频，n_fft=512 / hop=256：

- 帧数 T ≈ 16000 / 256 ≈ 63 帧
- 频点数 F = 512/2 + 1 = **257**（奈奎斯特频率以下）

所以 `(B, 257, T, 2)` 就是这么来的。最后那个 `2` 是 [real, imag]。

## 2. 第一步特征工程：三通道拼接

```python
# gtcrn.py:298-301
spec_real = spec[..., 0].permute(0,2,1)
spec_imag = spec[..., 1].permute(0,2,1)
spec_mag  = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
```

这一步做了什么？把原始的 `(real, imag)` 两通道扩成 `(mag, real, imag)` 三通道。

**为什么要冗余加一个 mag？** 

- **mag 信号噪声比明显**：人耳对响度敏感，幅度谱（spectrogram）是降噪任务最直观的特征。直接给网络一个"被噪声污染"的 mag 比让它自己去 sqrt(real² + imag²) 更省事。
- **real / imag 携带相位信息**：单靠 mag 不够，相位也要修。但相位的"绝对值"对网络没用，所以保留 real/imag 这种笛卡尔形式。
- **冗余特征有助于训练稳定**：给网络多种"角度"的同一信号，让它自己挑哪个最有用——这是 multi-view 思想。

> 这里有一个细节：`+ 1e-12` 是为了避免 sqrt(0) 的梯度爆炸。这种数值技巧在大量算法里都很重要，但论文不会写。

## 3. 核心来了：ERB 频带合并 (Band Merging)

打开 [gtcrn.py:11](../../third_party/gtcrn/gtcrn.py#L11)，看 `ERB` 类。

### 3.1 心理声学常识：什么是 ERB？

ERB（Equivalent Rectangular Bandwidth，等效矩形带宽）是描述人耳"频率分辨率"的尺度。

**人耳对低频敏感、对高频迟钝**——具体地：

- 100 Hz 附近，人耳能分辨大约 30 Hz 的差异
- 1000 Hz 附近，人耳能分辨大约 130 Hz 的差异
- 8000 Hz 附近，人耳能分辨大约 1300 Hz 的差异

也就是说，**线性 Hz 尺度对人耳来说是"过度精细"的**——在高频，相邻几个 Hz 的差异人耳根本听不出来。

ERB 标度（也包括 Bark、Mel 标度）就是把 Hz 重新映射成"人耳感知意义上等间距"的尺度。**在高频区，多个 Hz 的 STFT 频点会被合并成一个 ERB 频带，因为人耳反正分辨不出。**

ERB 的转换公式来自心理声学经典文献（Moore & Glasberg, 1990）：

$$
ERB(f) = 21.4 \log_{10}(0.00437 \cdot f + 1)
$$

代码里的实现完全对应：

```python
# gtcrn.py:22
def hz2erb(self, freq_hz):
    erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
    return erb_f
```

### 3.2 GTCRN 的 ERB 怎么用？

这里作者做了一个**重要的工程取舍**：

> **低频不动，高频才合并。**

代码 [gtcrn.py:280](../../third_party/gtcrn/gtcrn.py#L280)：

```python
self.erb = ERB(65, 64)
```

参数含义：

- `erb_subband_1 = 65`：前 65 个 STFT 频点（0 ~ 2000 Hz）**保持原样**
- `erb_subband_2 = 64`：剩下的 192 个 STFT 频点（2000 ~ 8000 Hz）**合并成 64 个 ERB 频带**

总共输出 65 + 64 = **129 个频带**，比原始 257 砍了将近一半。

### 3.3 为什么低频不合并？

这是论文 Sec 2.1 明确写的：

> "harmonics are more likely to be present in low-frequency bands and rarely occur in high-frequency bands"

**语音的谐波结构主要集中在低频（基频 80-400 Hz，前几个谐波在 1-2 kHz 之内）**。这些谐波之间的精细频率结构（比如基频附近 10 Hz 的差异）对人耳的语音感知至关重要——一旦合并，就会把基频信息搞糊。

而高频的谐波间距已经很大（10次谐波在 800 Hz 处差距是 80 Hz，但在 4000 Hz 处差距已经达到几百 Hz），人耳分辨不出，可以放心合并。

> 这就是把**声学专家知识**直接编码进网络的典型例子。如果让一个完全没有领域知识的人去设计，他可能会均匀地下采样所有频率，那性能就会塌掉。GTCRN 的小巧很大一部分来自这种"先验注入"。

### 3.4 ERB 滤波器组的具体形状：三角窗

我们看 [gtcrn.py:30](../../third_party/gtcrn/gtcrn.py#L30) `erb_filter_banks`：

```python
def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
    low_lim = erb_subband_1/nfft * fs        # = 65/512 * 16000 ≈ 2031 Hz
    erb_low  = self.hz2erb(low_lim)          # ≈ 14.4 ERB
    erb_high = self.hz2erb(high_lim)         # ≈ 25.4 ERB
    erb_points = np.linspace(erb_low, erb_high, erb_subband_2)  # 在 ERB 尺度上均匀取 64 个点
    bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)  # 转回 FFT bin 索引
    
    erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)
    # 构造三角窗
    erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1])) / (bins[1] - bins[0])
    for i in range(erb_subband_2-2):
        erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i]) / (bins[i+1] - bins[i])
        erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i+2])) / (bins[i+2] - bins[i+1])
    erb_filters[-1, bins[-2]:bins[-1]+1] = 1 - erb_filters[-2, bins[-2]:bins[-1]+1]
    return torch.from_numpy(np.abs(erb_filters))
```

这段代码在做什么？**构造 64 个三角形权重函数**，每个三角形覆盖几个 FFT 频点：

```
Filter i:    /\
            /  \
___________/    \___________
   bins[i-1] bins[i] bins[i+1]
```

每个 ERB 频带 = 多个 FFT 频点的加权平均（三角窗加权）。这是 Mel/Bark/ERB 滤波器组的标准形式。

### 3.5 怎么用：BM 和 BS 是一对反操作

```python
# gtcrn.py:51-61
def bm(self, x):
    """Band Merging: (B,C,T,F=257) → (B,C,T,F=129)"""
    x_low = x[..., :self.erb_subband_1]                  # 前 65 点不动
    x_high = self.erb_fc(x[..., self.erb_subband_1:])    # 后 192 点过 ERB 矩阵
    return torch.cat([x_low, x_high], dim=-1)

def bs(self, x_erb):
    """Band Splitting: (B,C,T,F=129) → (B,C,T,F=257)"""
    x_erb_low = x_erb[..., :self.erb_subband_1]
    x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])   # 反变换
    return torch.cat([x_erb_low, x_erb_high], dim=-1)
```

注意 `self.erb_fc` 和 `self.ierb_fc` 用的是 `nn.Linear`，但是：

```python
# gtcrn.py:19-20
self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)
```

**`requires_grad=False`**——也就是说这两个矩阵**不参与训练**！

这就引出了一个有趣的细节：论文里说参数量是 23.7K，但 README 更新成了 48.2K。差的就是这部分 ERB 矩阵的"参数"（虽然不可训练）。论文后来还提到："By replacing the invariant mapping from linear bands to ERB bands in the low-frequency dimension with simple concatenation instead of matrix multiplication, the MACs per second are reduced to 33 MMACs"——意思是低频部分本来可以用一个恒等矩阵走 Linear，但作者后来发现直接 `torch.cat` 拼接更省算力（避免不必要的矩阵乘法）。

> **设计哲学**：把不可训练的、有解析形式的变换写成"不可学习的 Linear"，是把领域知识嵌入网络的优雅方式。后续如果想让 ERB 矩阵可学习（end-to-end 优化），只需要把 `requires_grad` 改成 True。

## 4. SFE 模块：把"邻近频带"塞进通道维

紧接着 ERB 之后是 SFE（Subband Feature Extraction），第 06 章会详细讲。这里先看它在 [gtcrn.py:281](../../third_party/gtcrn/gtcrn.py#L281) 怎么用：

```python
self.sfe = SFE(3, 1)  # kernel_size=3, stride=1
```

`SFE` 的实现 [gtcrn.py:64](../../third_party/gtcrn/gtcrn.py#L64)：

```python
class SFE(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs
```

简单说：用 `nn.Unfold` 做"滑动窗口提取"，把每个频点的左右邻居拼到自己的通道维上。

输入 `(B, 3, T, 129)`，经过 SFE(k=3) → 输出 `(B, 9, T, 129)`：

```
原始第 f 个频带的特征：[c0_mag, c0_real, c0_imag]  (3维)
SFE 后第 f 个频带的特征：[f-1的3个, f的3个, f+1的3个]  (9维)
```

**为什么这样做？** 因为后面的卷积通道数只有 16，且使用了 grouped conv，每个 group 看到的频域上下文非常有限。SFE 提前把"邻居信息"打包好塞到通道维，让后面的 pointwise conv (1×1 conv) 直接就能用到。

详细原理留到第 06 章。

## 5. 一个端到端的算账：ERB 节省了多少算力？

我们做个粗略估算：

**不用 ERB 的情况**：

- 输入 (B, 3, T, **257**)
- 第一层 Conv (3, 16, kernel=(1,5)): 算力 ≈ T × 257 × 5 × 3 × 16 / 2 (因 stride=2) ≈ T × 30,840 MACs

**用了 ERB 的情况**：

- 输入 (B, 9, T, **129**) ← SFE 后通道是 9
- 第一层 Conv (9, 16, kernel=(1,5)): 算力 ≈ T × 129 × 5 × 9 × 16 / 2 ≈ T × 46,440 MACs

哎，看起来反而更高？因为 SFE 把通道从 3 拓到 9 了。

但你要看整个网络——**后面所有的层都因为频率维从 257 砍到 129 而节省了一倍计算**：

- DPGRNN 输入维度从 ~129 降到 33（经过 4 次下采样：129→65→33）。如果不砍频率，最后 DPGRNN 要处理 ~64 维频率，参数量和计算量都翻倍。
- Encoder/Decoder 每一层都因频率维减半而省一半算力。

所以这是一个**头部花点小钱（SFE 通道扩张），换取整体省大钱（后面所有层频率维减半）**的交易。

## 6. ERB 模块的"工程感觉"总结

我们把这一章学到的东西凝成几条工程直觉：

1. **遇到"维度过高、信息冗余"的场景，第一反应应该是去找领域已有的标度（Mel、Bark、ERB）**——不要从零设计下采样策略。
2. **不可训练的领域变换可以写成 nn.Linear 且 `requires_grad=False`**——既享受 GPU 矩阵乘的高效，又不占用训练参数预算。
3. **"砍维度 → 升通道"是一个反复出现的范式**：信息守恒，但表示形式变了。
4. **低频敏感、高频粗略**这种声学先验对语音相关网络极其重要，要主动找机会把这种先验编码进去。

## 7. 一个值得思考的小问题

如果让你把这套 ERB 思路推广到 48 kHz 的全频带降噪（fullband SE），你会怎么改？

提示：

- 频点数会从 257 变成 1025（n_fft=2048 时）
- 低频还是 65 个不动？还是 130 个不动？
- 高频要压缩多少？

这是 DeepFilterNet、ULSE 等全频带模型实际要回答的问题。GTCRN 解决的是宽带（16kHz）的 case，是一个更窄的问题。

---

**上一章**：[02_整体架构.md](02_整体架构.md) ｜ **下一章**：[04_GT-Conv详解.md](04_GT-Conv详解.md)
