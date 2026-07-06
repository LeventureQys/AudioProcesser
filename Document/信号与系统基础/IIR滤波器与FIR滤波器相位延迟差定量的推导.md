
# IIR滤波器与FIR滤波器最大的不同：相位延迟

# IIR滤波器相位延迟分析
## 相位响应和延迟

这里讨论一下理想延迟系统的相位延迟。

对于一个给定的系统频率响应$H(e^{jw})$可以表示为

$H(e^{jw}) = |H(e^{jw})|e^{Φ(w)}$

其中$H(e^{jw})$是幅度响应，$Φ(w)$是相位响应。

## 延迟系统的相位响应
对于一个理想的延迟系统，其输出信号是输入信号的延迟版本，即：

$y(n) = x(n-\tau)$

其中$\tau$是延迟时间，对应的频率响应为$H(e^{jw})=e^{-jw\tau}$
这是因为延迟$\tau$样本在时域上相当于在频域上乘以$e^{-jw\tau}$


### 傅里叶变换和频域描述

为了理解延迟系统的频率响应，需要用到离散时间傅里叶变换（DTFT）。DTFT将时域信号转换为频域信号。

- 输入信号$x(n)$的DTFT为：

$$X(e^{jw}) = \sum_{n=-\infty}^{\infty} x(n) e^{-jwn}$$

- 输出信号$y(n)$的DTFT为：

$$Y(e^{jw}) = \sum_{n=-\infty}^{\infty} y(n) e^{-jwn}$$

### 延迟的影响

根据延迟系统的定义：

$$y(n) = x(n - \tau)$$

将这个关系代入到$y(n)$的DTFT公式中：

$$Y(e^{jw}) = \sum_{n=-\infty}^{\infty} x(n - \tau) e^{-jwn}$$

可以通过变量替换来简化计算。令$k = n - \tau$，则$n = k + \tau$：

$$Y(e^{jw}) = \sum_{k=-\infty}^{\infty} x(k) e^{-jw(k + \tau)}$$

分离指数部分：

$$Y(e^{jw}) = \sum_{k=-\infty}^{\infty} x(k) e^{-jwk} e^{-jw\tau}$$

注意到：

$$\sum_{k=-\infty}^{\infty} x(k) e^{-jwk} = X(e^{jw})$$

所以：

$$Y(e^{jw}) = X(e^{jw}) \cdot e^{-jw\tau}$$

### 频率响应

系统的频率响应$H(e^{jw})$定义为输出频域表示与输入频域表示的比值：

$$H(e^{jw}) = \frac{Y(e^{jw})}{X(e^{jw})}$$

将上面的结果代入：

$$H(e^{jw}) = e^{-jw\tau}$$

## 相位响应的推导

我们可以从延迟系统的频率响应H(e^jw)推导出其相位响应:

$H(e^{jw})=e^{-jw\tau}$

从上述式子可以看到，频率响应的相位部分为$Φ(w)=-w\tau$

-----

至此我们知道了系统的延迟是如何表达和推导的，那么我们现在来说一下为什么IIR滤波器和FIR滤波器在相位延迟上会有这么大差别。

## IIR滤波器相位延迟分析

考虑一个IIR滤波器的频率响应函数，应当如下：

一般来说，一个IIR滤波器的输出可以表示为：

$$y(n) = \sum_{k=0}^{N} b_k x(n-k) - \sum_{k=1}^{M} a_k y(n-k)$$

其中，$b_k$和$a_k$是滤波器的系数。

IIR滤波器的频率响应$H(e^{j\omega})$通常表示为：

$$H(e^{j\omega}) = \frac{B(e^{j\omega})}{A(e^{j\omega})}$$

其中，$B(e^{j\omega})$和$A(e^{j\omega})$分别是分子和分母多项式：

$$B(e^{j\omega}) = \sum_{k=0}^{N} b_k e^{-j\omega k}$$
$$A(e^{j\omega}) = 1 + \sum_{k=1}^{M} a_k e^{-j\omega k}$$

相位响应$\phi(\omega)$是频率响应的相位部分：

$$H(e^{j\omega}) = |H(e^{j\omega})| e^{j\phi(\omega)}$$
$$\phi(\omega) = \arg(H(e^{j\omega}))$$

为了定量地分析IIR滤波器的延迟，我们需要计算相位响应的频率导数，即群延迟$\tau_g(\omega)$：

$$\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega}$$

由于IIR滤波器的相位响应不是线性的，所以其群延迟通常是频率的函数，即延迟是频率依赖的。


### 定量推导（纯数学计算）

我们以一个简单的一阶IIR滤波器为例，分析其延迟特性。考虑一个一阶IIR滤波器，其差分方程为：

$$y(n) = x(n) - a y(n-1)$$

其频率响应为：

$$H(e^{j\omega}) = \frac{1}{1 - a e^{-j\omega}}$$

1. **计算频率响应的相位**：

$$H(e^{j\omega}) = \frac{1}{1 - a e^{-j\omega}}$$

我们将其写成极坐标形式：

$$H(e^{j\omega}) = \frac{1}{\sqrt{1 - 2a\cos(\omega) + a^2}} e^{j\phi(\omega)}$$

其中，

$$\phi(\omega) = -\tan^{-1}\left(\frac{a \sin(\omega)}{1 - a \cos(\omega)}\right)$$

2. **计算群延迟**：

$$\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega}$$

$$\phi(\omega) = -\tan^{-1}\left(\frac{a \sin(\omega)}{1 - a \cos(\omega)}\right)$$

利用导数链式法则，

$$\tau_g(\omega) = -\frac{d}{d\omega} \left[-\tan^{-1}\left(\frac{a \sin(\omega)}{1 - a \cos(\omega)}\right)\right]$$

计算导数：

$$\tau_g(\omega) = \frac{a \left(1 - a \cos(\omega)\right)\cos(\omega) + a^2 \sin^2(\omega)}{\left(1 - a \cos(\omega)\right)^2 + a^2 \sin^2(\omega)}$$

简化后得到：

$$\tau_g(\omega) = \frac{a \left(1 - a \cos(\omega) + a \cos^2(\omega)\right)}{1 - 2a \cos(\omega) + a^2}$$

由于公式较为复杂，我们可以直接用数值方法计算和绘制IIR滤波器的群延迟特性。

### 举个例子

我们来搞个示例，这样好懂一点：

考虑一个简单的一阶滤波器

$$H(e^jw)=\frac{1}{1-ae^{-jw}}$$

其相位响应为：

$$ϕ(w)=-arg(1-ae^{-jw})$$

我们可以看到，这个相位响应显然是非线性的，会随着w的不停变化，其变化率也会发生变化，说着说导数的比值会随着w的变化而变化，这显然是我们不想要看到的结果。

# FIR滤波器相位延迟分析

# FIR滤波器的相位延迟推导

FIR（有限脉冲响应）滤波器的延迟特性通常是线性的，这源于其非递归结构和对称系数设计。下面我们详细推导FIR滤波器的相位延迟，并展示如何利用KaTeX进行Markdown文档的编写。

## FIR滤波器的基本形式

一个FIR滤波器的输出可以表示为：

$$ y(n) = \sum_{k=0}^{N} b_k x(n-k) $$

其中，$b_k$ 是滤波器的系数，$N$ 是滤波器的阶数。

## 频率响应和相位响应

FIR滤波器的频率响应 $H(e^{j\omega})$ 可以表示为：

$$ H(e^{j\omega}) = \sum_{k=0}^{N} b_k e^{-j\omega k} $$

相位响应 $ \phi(\omega) $ 是频率响应的相位部分：

$$ H(e^{j\omega}) = |H(e^{j\omega})| e^{j\phi(\omega)} $$
$$\phi(\omega) = \arg(H(e^{j\omega}))$$

## 线性相位的条件

为了实现线性相位，我们通常设计FIR滤波器的系数使其具有对称性或反对称性。对于一个长度为 $N+1$ 的对称FIR滤波器，其系数满足：

$$ b_k = b_{N-k} $$

对于反对称FIR滤波器，其系数满足：

$$ b_k = -b_{N-k} $$

这两种对称性保证了滤波器的相位响应是线性的，即：

$$ \phi(\omega) = -\omega \tau $$

其中，$\tau$ 是一个常数，表示恒定的群延迟。

## 定量推导

考虑一个对称的FIR滤波器，其冲激响应 $h(n)$ 为：

$$ h(n) = h(N-1-n) $$

其频率响应为：

$$ H(e^{j\omega}) = \sum_{k=0}^{N-1} h(k) e^{-j\omega k} $$

由于 $ h(n) $ 的对称性，我们可以将其拆分并合并：

$$ H(e^{j\omega}) = \sum_{k=0}^{(N-1)/2} h(k) \left( e^{-j\omega k} + e^{-j\omega (N-1-k)} \right) $$

利用欧拉公式，我们有：

$$ e^{-j\omega (N-1-k)} = e^{-j\omega (N-1)} e^{j\omega k} $$

合并后得到：

$$ H(e^{j\omega}) = e^{-j\omega (N-1)/2} \sum_{k=0}^{(N-1)/2} h(k) \left( e^{-j\omega (k - (N-1)/2)} + e^{j\omega (k - (N-1)/2)} \right) $$

这表明相位响应是线性的：

$$ \phi(\omega) = -\omega \frac{N-1}{2} $$