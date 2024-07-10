# 前言

现在不是搞音频嘛，正好自己买了无源音箱，买了套DSP芯片玩一下

## 流程

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/DSP.drawio.png"/>

上图是我们组织一套音频系统的流程，首先我们需要知道各个元件是做什么的

**1. 音源（例如麦克风、音乐播放器等）：**

产生模拟音频信号。

**2. AD 转换器（模数转换器，ADC）：**

将模拟音频信号转换为数字音频信号，以便DSP处理。

**3. DSP（数字信号处理器）：**

接收并处理数字音频信号。处理过程可能包括滤波、均衡、混音、音效处理等。

**4. I²S 接口：**

DSP通过I²S接口将处理后的数字音频信号传输给下游设备。下游设备可能是另一个数字音频设备（如外部DAC、音频放大器等），这些设备再进行后续的处理或转换。

**5. 输出：**

数字音频信号通过下游设备转换为模拟信号，并通过扬声器或耳机播放出来。

一般情况下用户能看到的其实就两部分，输入和输出，输入一般是3.5mm的耳机插口，输出就是喇叭，现在我们把这一套系统拆开来，实际上就应该如上图所示。

## 实物解析

接下来我拿出来我买到的板子，然后一点点说明这些东西应该怎么用，我这里说到的这些设备可能在实际上买到的东西有些区别，但是大差不差。

在我这一套系统中，音源输入不需要使用AD进行转换，而是一个单独的i2s音频输入，所以没有AD，当然了即使有AD，流程也是大差不差的。

1. 输入

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240629183706.png"/>

我这里买的是一块i2s转换输入的信号其实就是一张声卡，从电脑的usb接口引出一个type-B接口，用来进行声音的输出，当电脑上连接上这个type-B接口之后，系统里面会多出这个声卡的信息，也就说明这个声卡连接成功了，此板的供电由type-B的USB接口提供。

买的是PCM2706 USB界面I2S

2. DSP

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240629184153.png"/>

此板是ADAU的1701，这块板子支持界面编程，可以在里面拖各种滤波器，处理器等，性能确实不强，但是至少够用了，作为学习还不错，如果需要更强的性能，可以去买更屌的DSP，或者干脆自己买芯片去设计。
需要注意的是，此块DSP上是有DA的，就是可以直接将针脚接在左下角的OUT口上就可以直接输出音频信号了。

3. DA

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240629184544.png"/>

这是一颗数字信号转模拟信号芯片，可以在这里将DSP输出的i2s信号直接导入到此块芯片上，然后就可以通过这个3.5mm的接口将模拟信号直接播放，或者使用针脚将这些模拟信号输出到外部。

## 接线：

组织这一套系统的接线，我这里有几个流程需要进行：

1. DSP板通电
2. USB模拟连接到DSP板上用于烧录程序
3. i2s音频输入连接到DSP板
4. DSP板音频输出连接到AD

### 1.DSP板通电

只需要给DSP板上的5v针脚通上5v的供电，然后GND接地就可以了。后面所有的GND都代表通电。

### 2. USB模拟连接到DSP板上用于烧录程序

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240629185158.png"/>

USB模拟器上的GND，SCL和SDA针脚分别连接上板子对应的那个针脚。

### 3. i2s音频输入连接到DSP板

这里需要看一下操作手册

<img src="https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/20240629185419.png"/>

可以看到，mp0就是input0针脚，然后两个时钟LRCLK和BCLK分别是mp4和mp5，那么把i2s输入上的DATA SPDIF、LRCLK、BCLK分别接到DSP板上的mp0、mp4和mp5上，这里需要注意的是，这里i2s板上还有一个MCLK需要接到DSP板上，作为主时钟。

这里GND需要并联

### 4. DSP板音频输出连接到DA

有了刚刚i2s输入的前车之鉴，这个的连接就简单了，将对应的BCK、LCK和DIN连接到mp11、mp10和mp6上即可，需要注意的是，DA需要一个3.3v的电压输入，这里可以直接从i2s板上标记有3.3v的输出上接一根线出来连接上，记得并联GND接地线。