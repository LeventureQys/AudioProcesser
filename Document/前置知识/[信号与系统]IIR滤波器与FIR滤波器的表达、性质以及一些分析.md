# 前言

阅读本文需要阅读一些前置知识

[[信号与系统]傅里叶变换、卷积定理、和为什么时域的卷积等于频域相乘。](https://blog.csdn.net/Andius/article/details/139841656?spm=1001.2014.3001.5502)

[[信号与系统]有关滤波器的一些知识背景](https://blog.csdn.net/Andius/article/details/139848523?spm=1001.2014.3001.5502)

[[信号与系统]关于LTI系统的转换方程、拉普拉斯变换和z变换](https://blog.csdn.net/Andius/article/details/139851289?spm=1001.2014.3001.5502)

[[信号与系统]关于双线性变换](https://blog.csdn.net/Andius/article/details/139853924?spm=1001.2014.3001.5502)

## IIR滤波器的数学表达式

IIR（Infinite Impulse Response）滤波器的输出信号 $y[n]$ 可以用输入信号 $x[n]$ 和滤波器系数表示为线性常系数差分方程：

$$
y[n] = -\sum_{k=1}^{N} a_k y[n-k] + \sum_{k=0}^{M} b_k x[n-k]
$$

其中：

- $y[n]$ 是滤波器的输出信号。
- $x[n]$ 是滤波器的输入信号。
- $a_k$ 和 $b_k$ 是滤波器的系数。
- $N$ 是输出信号的反馈项数。
- $M$ 是输入信号的前馈项数。

### 传递函数

IIR滤波器的传递函数 $H(z)$ 是输入信号的Z变换 $X(z)$ 与输出信号的Z变换 $Y(z)$ 之比：

$$
H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}
$$

### 数学性质

1. **因果性 (Causality)：**

   - IIR滤波器通常是因果的，即输出信号在当前时刻只依赖于当前及过去的输入和输出信号。
   - 数学上，如果系统的传递函数 $H(z)$ 在单位圆外是有界的，则该系统是因果的。
2. **稳定性 (Stability)：**

   - IIR滤波器的稳定性取决于系统的极点。如果所有极点都位于单位圆内（即 $|z| < 1$），则系统是稳定的。
   - 数学上，如果传递函数 $H(z)$ 在单位圆内收敛，则系统是稳定的。
3. **频率响应 (Frequency Response)：**

   - IIR滤波器的频率响应 $H(e^{j\omega})$ 是通过将 $z$ 替换为 $e^{j\omega}$ 得到的：

   $$
   H(e^{j\omega}) = \frac{\sum_{k=0}^{M} b_k e^{-j\omega k}}{1 + \sum_{k=1}^{N} a_k e^{-j\omega k}}
   $$

   - 频率响应描述了系统对不同频率成分的响应。
4. **无限冲激响应 (Infinite Impulse Response)：**

   - IIR滤波器的冲激响应 $h[n]$ 是无限长的，即 $h[n]$ 不会在有限时间内变为零。
   - 数学上，如果系统的冲激响应 $h[n]$ 对于所有 $n$ 都不为零，则为IIR滤波器。

### 总结

IIR滤波器通过反馈和前馈项的结合，能够实现复杂的频率响应特性。其数学表达式和性质对于分析和设计滤波器非常重要。IIR滤波器广泛应用于信号处理和通信系统中，因其能用较少的滤波器阶数实现较高的选择性和稳定性。

## FIR滤波器的数学表达式

FIR（Finite Impulse Response）滤波器的输出信号 $y[n]$ 可以用输入信号 $x[n]$ 和滤波器系数表示为线性常系数差分方程：

$$
y[n] = \sum_{k=0}^{M} b_k x[n-k]
$$

其中：

- $y[n]$ 是滤波器的输出信号。
- $x[n]$ 是滤波器的输入信号。
- $b_k$ 是滤波器的系数。
- $M$ 是滤波器的阶数。

### 传递函数

FIR滤波器的传递函数 $H(z)$ 是输入信号的Z变换 $X(z)$ 与输出信号的Z变换 $Y(z)$ 之比：

$$
H(z) = \frac{Y(z)}{X(z)} = \sum_{k=0}^{M} b_k z^{-k}
$$

### 数学性质

1. **因果性 (Causality)：**

   - FIR滤波器通常是因果的，即输出信号在当前时刻只依赖于当前及过去的输入信号。
   - 数学上，如果系统的传递函数 $H(z)$ 在单位圆外是有界的，则该系统是因果的。
2. **稳定性 (Stability)：**

   - FIR滤波器是稳定的，因为其冲激响应是有限长度的，不存在反馈。
3. **线性相位 (Linear Phase)：**

   - FIR滤波器可以设计成具有线性相位响应，即不同频率成分通过滤波器时相位延迟是线性的。这对于避免信号失真非常重要。
4. **频率响应 (Frequency Response)：**

   - FIR滤波器的频率响应 $H(e^{j\omega})$ 是通过将 $z$ 替换为 $e^{j\omega}$ 得到的：

   $$
   H(e^{j\omega}) = \sum_{k=0}^{M} b_k e^{-j\omega k}
   $$

   - 频率响应描述了系统对不同频率成分的响应。
5. **有限冲激响应 (Finite Impulse Response)：**

   - FIR滤波器的冲激响应 $h[n]$ 是有限长的，即在有限时间内变为零。
   - 数学上，如果系统的冲激响应 $h[n]$ 对于 $n > M$ 都为零，则为FIR滤波器。

### 总结

FIR滤波器通过前馈项的组合，能够实现预期的频率响应特性。其数学表达式和性质对于分析和设计滤波器非常重要。FIR滤波器广泛应用于信号处理和通信系统中，因其固有的稳定性和可以实现的线性相位特性，使得它们特别适用于对相位响应有严格要求的应用。

## 一些分析

### 1. IIR滤波器的冲激响应

IIR滤波器的冲激响应 $h[n]$ 是无限长的，这意味着当一个冲激输入（即单位脉冲信号 $\delta[n]$）应用于IIR滤波器时，滤波器的输出会持续无限长的时间。其数学表达式为：

$$
y[n] = -\sum_{k=1}^{N} a_k y[n-k] + \sum_{k=0}^{M} b_k x[n-k]
$$

当输入信号 $x[n] = \delta[n]$ 时，输出信号 $y[n] = h[n]$ 是系统的冲激响应。由于IIR滤波器具有反馈项（即包含前几个输出 $y[n-k]$），这些反馈项会使得冲激响应在理论上永远不会完全衰减至零。

### 2. FIR滤波器的冲激响应

FIR滤波器的冲激响应 $h[n]$ 是有限长的，这意味着当一个冲激输入（即单位脉冲信号 $\delta[n]$）应用于FIR滤波器时，滤波器的输出在有限时间内变为零。其数学表达式为：

$$
y[n] = \sum_{k=0}^{M} b_k x[n-k]
$$

当输入信号 $x[n] = \delta[n]$ 时，输出信号 $y[n] = h[n]$ 是系统的冲激响应。由于FIR滤波器只包含输入信号的前馈项（即没有前几个输出 $y[n-k]$ 的反馈项），冲激响应在有限时间内（即在 $M$ 个采样点之后）会变为零。

### 数学性质

#### IIR滤波器的冲激响应

1. **无限长：** IIR滤波器的冲激响应 $h[n]$ 在理论上是无限长的，因为其反馈结构会导致输出信号持续无限时间。
2. **数学上：**

   $$
   h[n] = \sum_{k=0}^{M} b_k \delta[n-k] - \sum_{k=1}^{N} a_k h[n-k]
   $$

   由于存在反馈项 $a_k h[n-k]$，冲激响应不会在有限时间内变为零。

#### FIR滤波器的冲激响应

1. **有限长：** FIR滤波器的冲激响应 $h[n]$ 是有限长的，因为其前馈结构没有反馈项，导致输出信号在有限时间内变为零。
2. **数学上：**

   $$
   h[n] = \sum_{k=0}^{M} b_k \delta[n-k]
   $$

   由于没有反馈项，冲激响应在 $n > M$ 时会变为零。

### 群延迟（Group Delay）

是指信号中不同频率分量通过滤波器时的相位延迟差异。它表示为：

$$
\tau_g(\omega) = -\frac{d\theta(\omega)}{d\omega}
$$

其中，$\theta(\omega)$ 是滤波器的相位响应。

**相位响应** 是指滤波器对信号不同频率分量引入的相位变化。线性相位响应意味着所有频率分量被等相位延迟处理，保持了信号波形的形状。

#### IIR滤波器

具有反馈结构，其数学形式为：

$$
y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]
$$

由于反馈项（$\sum_{k=1}^{N} a_k y[n-k]$），IIR滤波器的相位响应通常是非线性的。这是因为反馈会引入复杂的极点分布，导致相位响应不是线性的，进而导致群延迟不恒定，形成非线性相位偏移。

#### FIR滤波器

**FIR滤波器** 具有有限冲激响应，其数学形式为：

$$
y[n] = \sum_{k=0}^{M} b_k x[n-k]
$$

FIR滤波器没有反馈项，仅依赖于输入信号的有限个样本。通过适当设计滤波器系数 $b_k$，可以实现线性相位响应，即：

$$
\theta(\omega) = -\tau \omega
$$

其中，$\tau$ 是常数。这意味着群延迟 $\tau_g(\omega)$ 为常数，所有频率分量都具有相同的相位延迟，保持信号的波形不失真。

#### 举个例子

**考虑一个简单的一阶IIR低通滤波器**：

$$
H(z) = \frac{1 - 0.5z^{-1}}{1 - 0.3z^{-1}}
$$

相位响应和群延迟（非线性）如下：

$$
\theta(\omega) \approx -\omega \left( \frac{0.3}{1 - 0.3^2} \right)
$$

$$
\tau_g(\omega) = -\frac{d\theta(\omega)}{d\omega}
$$

**考虑一个简单的三阶FIR低通滤波器，具有对称系数：**

$$
H(z) = 0.25 + 0.5z^{-1} + 0.25z^{-2}
$$

相位响应和群延迟（线性）如下：

$$
\theta(\omega) = -\omega \left( \frac{3}{2} \right)
$$

$$
\tau_g(\omega) = \frac{d\theta(\omega)}{d\omega} = \frac{3}{2}
$$