# 前言
学一下数字滤波器，顺便把自己学到的东西记录一下



# 窗函数法：时域设计法


考虑一个理想滤波器的频响曲线和冲激响应



![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701194706.png)

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701194722.png)


这里我们先讨论一个fir滤波器与iir滤波器最大的不同，就是所谓相位不同，至于相位究竟哪里不同，这里我觉得需要清晰地讨论一下。

详情见文章，因为这里涉及到很多定量推导，所以文字量比较大，就不参在这篇文章中写明了。

[[信号与系统]IIR滤波器与FIR滤波器相位延迟定量的推导。](https://blog.csdn.net/Andius/article/details/140109464?spm=1001.2014.3001.5502)

对于这样一个时域的时域信号

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701205110.png)

我们知道如果希望这个是线性相位的理想数字滤波器，那么它需要满足几个条件，

1. 这个时域信号得是一个因果的，那么n<0的时候$h_d(n)=0$

2. 需要是堆成的，即$h(n)=h(N-1-n),\tau=(N-1)/2$

那我们就可以让这个时域信号去乘上这个矩形信号，就是：

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701205330.png)

$h_d(n)·R_N(n)$的过程，我们就成为加窗，其中这个$R_N(n)$我们就称为窗函数，这里需要注意的是，我们在这里加窗的操作，实际上还是需要我们去满足这两个条件。

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701205741.png)

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701205755.png)


另外我们在这篇文章中说过[傅里叶变换、卷积定理、和为什么时域的卷积等于频域相乘](https://blog.csdn.net/Andius/article/details/139841656?spm=1001.2014.3001.5501),在这里时域相乘同样的等于频域上的卷积$H_d(w)\ast W_k(w)$

卷积如何造成旁瓣效应这里不过多介绍，因为这里主要讲的是窗函数的定义，以及如何使用窗函数来设计数字滤波器。

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701210002.png)

阻带最大值的w 为 $w_c-\frac{2\pi}{N}$，阻带最低点$w_c+\frac{2\pi}{N}$
我们这里也可以很清晰地看到，阻带频率的宽度为$4\pi/N$

阻带的衰减计算是$R_s=20lg\frac{1+\delta}{\delta}$，这样求出来是一个正值，有时候求出来的log会是一个负的，那就是$$R_s=20lg\frac{\delta}{\delta+1}$$

通带的也类似，则为$$R_s=20lg\frac{1-\delta}{\delta+1}$$


几种常见的窗函数的性能如图所示：

![](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240701210854.png)

当然了，我们当然希望是阻带衰减越大，过渡带宽越窄越好对啵，但是这两个参数不能同时取得，因为如果阻带衰减越大，那么过渡带宽越窄，那么就会导致旁瓣效应增加。

所以在做FIR滤波器的时候，第一个考虑的是阻带衰减，第二个考虑过渡带宽，如果性能够的情况下，肯定是去考虑增加N的数量，这样就可以解决阻带衰减和过渡带宽两难的问题。

