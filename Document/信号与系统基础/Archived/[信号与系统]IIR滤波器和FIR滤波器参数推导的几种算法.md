# 前言

最近看这两个滤波器设计嘛，就试着来写一下

主要从双线性变换，z变换，然后举一个例子来进行一下滤波器的设计。

## 例子

我们考虑传递函数如下的low pass filter

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240703110645.png)

# IIR滤波器

## 脉冲响应不变法

考虑这个传递函数$H(s)=\frac{w_0^2}{s^2+\frac{w_0}{Q}s+w_0^2}$

为了更好地将H(s)拆分开来，我们考虑分母的因式分解

假设$H(s)$存在两个根，解一元二次方程，有:

$$
p_{1,2}=-\frac{w_0}{2Q} \pm \sqrt{1-\frac{1}{4Q^2}}
$$

为了方便计算，我们记

$$
\alpha =-\frac{w_0}{2Q}
$$

$$
\beta = w_0\sqrt{1-\frac{1}{4Q^2}}
$$

因此我们知道

$$
p_1=\alpha + j\beta
$$

$$
p_2=\alpha-j\beta
$$

那我们的H(s)此时变成了:

$$
H(s)=\frac{w_0^2}{(s-p_1)(s-p_2)}
$$

我们假设这个式子可以展开，那么我们可以得到H(s)如下：

$$
H(s)=\frac{A}{s-p1} + \frac{B}{s-p_2}
$$

这里A和B是待定的，我们有

$$
\frac{w_0^2}{(s-p_1)(s-p_2)}=\frac{A}{s-p_1}+\frac{B}{s-p_2}
$$

这里我们其实可以计算出来，两边同时乘以分母，就有

$$
A(s-p_2)+B(s-p_1)=w_0^2
$$

求得A和B如下：

$$
A = \frac{w_0^2}{p_1-p_2} \\ B=\frac{w_0^2}{p_2-p_1}
$$

然后我们将H(s)转换到时域上考量，使用你拉布拉斯变换

$$
\mathcal{L}^{-1}\{ \frac{A}{s-p_1}\}=Ae^{p_1t}\\\mathcal{L}^{-1}\{ \frac{A}{s-p_2}\}=Be^{p_2t}
$$

综上，得到

$$
h(t)=Ae^{p_1t}+Be^{p_2t}
$$

带入

$$
p_1=\alpha+\mathcal{j}\beta\\p_2=\alpha-\mathcal{j}\beta
$$

有$h(t)=e^{\alpha t}(A_1e^{\mathcal{j}\beta t}+A_2e^{-\mathcal{j}\beta t})$

其实到这里就已经可以算作阶数了，但是这个可以继续化简减少工作量，我们利用欧拉公式

$$
e^{\mathcal{j}\beta t}=\cos(\beta t) + \mathcal{j}\sin(\beta t) \\
e^{\mathcal{-j}\beta t}=\cos(\beta t) - \mathcal{j}\sin(\beta t)
$$

假定我们在实数域上讨论和考虑，那么最终化简得到

$$
h(t)=2Ae^{\alpha t}cos(\beta t)
$$

这里是直接计算得到脉冲响应函数。

## 双线性变换法

还是考虑这个s域上的传递函数

$$
H(s) = \frac{\omega_0^2}{s^2 + \frac{\omega_0}{Q}s + \omega_0^2}
$$

使用双线性变换法呢，则是直接将H(s)进行z变换，得我们令

$$
s=\frac{2}{T} \frac{1-z^{-1}}{1+z^{-1}}
$$

求解极点，求解特征方程的根，即传递函数分母的零点：

$$
s^2 + \frac{\omega_0}{Q}s + \omega_0^2 = 0
$$

根为：

$$
p_{1,2} = -\frac{\omega_0}{2Q} \pm j \omega_0 \sqrt{1 - \frac{1}{4Q^2}}
$$

设：

$$
\alpha = -\frac{\omega_0}{2Q}
$$

$$
\beta = \omega_0 \sqrt{1 - \frac{1}{4Q^2}}
$$

将 $s$ 替换为双线性变换的表达式：

$$
s = \frac{2}{T} \frac{1 - z^{-1}}{1 + z^{-1}}
$$

代入 $H(s)$ 中：

$$
H\left( \frac{2}{T} \frac{1 - z^{-1}}{1 + z^{-1}} \right) = \frac{\omega_0^2}{\left( \frac{2}{T} \frac{1 - z^{-1}}{1 + z^{-1}} \right)^2 + \frac{\omega_0}{Q} \left( \frac{2}{T} \frac{1 - z^{-1}}{1 + z^{-1}} \right) + \omega_0^2}
$$

将分子分母进行通分并展开，可以得到：

$$
H(z) = \frac{\omega_0^2 \left( 1 + z^{-1} \right)^2}{\left( \frac{2}{T} \right)^2 \left( 1 - z^{-1} \right)^2 + \frac{2 \omega_0}{QT} \left( 1 - z^{-1} \right) \left( 1 + z^{-1} \right) + \omega_0^2 \left( 1 + z^{-1} \right)^2}
$$

最终得到的离散时间传递函数 $H(z)$ 形式为：

$$
H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}
$$

上下一比对，我们最终可以推导出$b_0,b_1,b_2,a_1,a_2$的参数计算公式

$$
b_0 = \frac{\omega_0^2 T^2}{4 + \frac{2\omega_0 T}{Q} + \omega_0^2 T^2}
$$

$$
b_1 = 2b_0
$$

$$
b_2 = b_0
$$

$$
a_1 = \frac{2(\omega_0^2 T^2 - 4)}{4 + \frac{2\omega_0 T}{Q} + \omega_0^2 T^2}
$$

$$
a_2 = \frac{4 - \frac{2\omega_0 T}{Q} + \omega_0^2 T^2}{4 + \frac{2\omega_0 T}{Q} + \omega_0^2 T^2}
$$

# FIR滤波器

还是考虑这个传递函数

$$
H(s) = \frac{\omega_0^2}{s^2 + \frac{\omega_0}{Q}s + \omega_0^2}
$$

为了找到其时域冲激响应 $h(t)$，我们需要对 $H(s)$ 进行拉普拉斯逆变换。首先，重写传递函数：

$$
H(s) = \frac{w_0^2}{(s + \alpha_1)(s + \alpha_2)}
$$

其中，$\alpha_1$ 和 $\alpha_2$ 是特征根，可以通过解方程 $s^2 + \frac{w_0}{Q}s + w_0^2 = 0$ 得到：

$$
\alpha_1, \alpha_2 = \frac{-w_0}{2Q} \pm w_0 \sqrt{\left(\frac{1}{4Q^2} - 1\right)}
$$

然后，采用部分分式分解：

$$
H(s) = \frac{A}{s + \alpha_1} + \frac{B}{s + \alpha_2}
$$

我们可以通过匹配系数法找到 $A$ 和 $B$。

对于每个分式：

$$
\mathcal{L}^{-1}\left\{\frac{A}{s + \alpha_1}\right\} = A e^{-\alpha_1 t}
$$

$$
\mathcal{L}^{-1}\left\{\frac{B}{s + \alpha_2}\right\} = B e^{-\alpha_2 t}
$$

因此，时域冲激响应 $h(t)$ 为：

$$
h(t) = A e^{-\alpha_1 t} + B e^{-\alpha_2 t}
$$

假设采样率为 $f_s$，我们将时域冲激响应 $h(t)$ 离散化得到 $h_d[n]$：

$$
h_d[n] = h(nT) = A e^{-\alpha_1 nT} + B e^{-\alpha_2 nT}
$$

其中 $T = \frac{1}{f_s}$。

选择一个适当的窗函数（如Hamming窗、Hann窗等）$w[n]$，并将其应用到采样后的冲激响应上，以得到最终的FIR滤波器系数。

假设我们使用Hamming窗 $w[n]$：

$$
h_w[n] = h_d[n] \cdot w[n]
$$

其中，Hamming窗的定义为：

$$
w[n] = 0.54 - 0.46 \cos\left(\frac{2 \pi n}{N-1}\right)
$$

最终的FIR滤波器系数为：

$$
h[n] = h_w[n]
$$