# Vol1 Ch12 素材核对清单

> 文档：`Documents/Vol1_Ch12_线性贝叶斯估计量.md`
> OCR 来源：`Temp/chapters_ocr/ch12/ocr_page_321~352.txt`（PDF 第 321~352 页，书内第 306~337 页，映射规则：书内页码 = PDF 页码 − 15）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | 原书编号 | OCR 出处（文件 + 附近内容） |
|---------|---------|---------------------------|
| LMMSE 动机：MMSE 要积分、MAP 要最大化，联合高斯下才好求 | — | ocr_page_321（"MMSE估计量含有多重积分，而MAP估计量含有多维最大值求解"） |
| 线性估计量 $\hat\theta=\sum a_nx[n]+a_N$、贝叶斯 MSE $E[(\theta-\hat\theta)^2]$ | (12.1)(12.2) | ocr_page_321~322（"选择加权系数，来使贝叶斯MSE…最小"） |
| 功率估计反例：$x[0]\sim N(0,\sigma^2)$、$\theta=x^2[0]$，LMMSE=$\sigma^2$、MSE=$2\sigma^4$ | — | ocr_page_322（"最小MSE是…=3σ⁴−2σ⁴+σ⁴=2σ⁴"，原文把 $\sigma$ 误识为 "α/g"） |
| $a_N=E(\theta)-\sum a_nE(x[n])$ | (12.3) | ocr_page_323 |
| $C_{xx}a=C_{x\theta}$ | (12.5) | ocr_page_323 |
| $\hat\theta=E(\theta)+C_{\theta x}C_{xx}^{-1}(x-E(x))$；零均值 $\hat\theta=C_{\theta x}C_{xx}^{-1}x$ | (12.6)(12.7) | ocr_page_323（"与联合高斯的x和θ的MMSE估计量在形式上是相同的"） |
| $\mathrm{Bmse}=C_{\theta\theta}-C_{\theta x}C_{xx}^{-1}C_{x\theta}$ | (12.8) | ocr_page_324 |
| 例 12.1 均匀先验：$\sigma_A^2=A_0^2/3$、$\hat A=\frac{\sigma_A^2}{\sigma_A^2+\sigma^2/N}\bar x$ | (12.9) | ocr_page_324（"$\sigma_A^2=(2A_0)^2/12=A_0^2/3$"；该处原文把 $(2A_0)^2/12$ 写作 "（2A）/12"，据英文原版校订为 $A_0^2/3$） |
| 内积 $(x,y)=E(xy)$、长度 $\|x\|=\sqrt{E(x^2)}$、正交 $E(xy)=0$ | (12.10)(12.11)(12.12) | ocr_page_325 |
| 正交原理 $E[(\theta-\hat\theta)x[n]]=0$ | (12.13)(12.14) | ocr_page_326（"这是重要的正交原理或者投影定理"） |
| 正规方程 $C_{xx}a=C_{x\theta}$ | (12.15)(12.16) | ocr_page_327 |
| 例 12.2 正交矢量估计（投影之和） | — | ocr_page_328 |
| 矢量 LMMSE $\hat\theta=E(\theta)+C_{\theta x}C_{xx}^{-1}(x-E(x))$、$M_\theta=C_{\theta\theta}-C_{\theta x}C_{xx}^{-1}C_{x\theta}$、$Bmse(\theta_i)=[M_\theta]_{ii}$ | (12.20)(12.21)(12.22) | ocr_page_329 |
| 线性变换 $\hat\alpha=A\hat\theta+b$、叠加性 $\hat\alpha=\hat\theta_1+\hat\theta_2$ | (12.23)(12.24) | ocr_page_330 |
| 定理 12.1 贝叶斯高斯-马尔可夫（LMMSE + 误差协方差，不要求高斯） | (12.26)~(12.30) | ocr_page_330（"贝叶斯高斯－马尔可夫定理"） |
| 序贯 DC 电平：$\hat A[N]=\hat A[N-1]+K[N](x[N]-\hat A[N-1])$、$K[N]=\frac{Bmse}{Bmse+\sigma^2}$、$Bmse[N]=(1-K)Bmse[N-1]$ | (12.31)(12.34)(12.35)(12.36) | ocr_page_331~332 |
| 新息 / Gram-Schmidt 正交化 | (12.37)~(12.46) | ocr_page_332~335（"误差矢量…称为新息（innovation）"） |
| 矢量序贯 LMMSE $\hat\theta[n]=\hat\theta[n-1]+K[n](x[n]-h^T[n]\hat\theta[n-1])$ 等 | (12.47)~(12.49) | ocr_page_335 |
| 初始化 $\hat\theta[-1]=E(\theta)$、$M[-1]=C_\theta$；无先验令 $C_\theta\to\infty$ | — | ocr_page_336 |
| 例 12.3 贝叶斯傅里叶（$h^T[0]=[1\ 0]$、$h^T[n]=[\cos2\pi f_0n\ \sin2\pi f_0n]$） | — | ocr_page_336~337 |
| WSS Toeplitz 自相关矩阵 | (12.50) | ocr_page_337 |
| 维纳平滑 $\hat s=R_{ss}(R_{ss}+R_{ww})^{-1}x$、$W=R_{ss}(R_{ss}+R_{ww})^{-1}$、$M_s=(I-W)R_{ss}$ | (12.53)(12.54)(12.55) | ocr_page_338~339 |
| $N=1$：$W=r_{ss}[0]/(r_{ss}[0]+r_{ww}[0])=\eta/(\eta+1)$ | (12.56) | ocr_page_339 |
| 维纳滤波 $\hat s[n]=r^T(R_{ss}+R_{ww})^{-1}x$、维纳-霍夫方程 | (12.57)(12.59) | ocr_page_339~340 |
| 无限维纳滤波器 $\sum h[k]r_{xx}[l-k]=r_{ss}[l]$（$l\ge0$） | (12.60) | ocr_page_340 |
| 非因果平滑频响 $H(f)=P_{ss}/(P_{ss}+P_{ww})$；局部 SNR $\eta(f)=P_{ss}/P_{ww}$、$H=\eta/(\eta+1)$ | (12.61)(12.62) 附近 | ocr_page_341~342（"维纳平滑器对SNR高的数据频谱部分予以加重"） |
| 预测方程 $a=R_{xx}^{-1}r'$、最小 MSE、AR(1) 一步预测 $\hat x[N]=-a[1]x[N-1]$、$l$ 步预测 $(-a[1])^l x[N-1]$ | (12.63)(12.65)(12.66)(12.67) | ocr_page_342~344 |
| AR(1) 预测数值例：$a[1]=-0.95$、$\sigma_u^2=0.1$、$N=11$、$l=1..40$ | — | ocr_page_344（"如果 α[1]=－0.95,=0.1…N=11,1=1,2，…,40"） |

## 2. 据英文原版校订之处及理由

1. **例 12.1 的 $\sigma_A^2$**：OCR 写作"（2A）/12"并给出 $\sigma_A^2=A_0^2/3$，按均匀分布 $U[-A_0,A_0]$ 的方差 $(2A_0)^2/12=A_0^2/3$ 校订（英文原版一致）。
2. **功率估计反例的符号**：OCR 把 $\sigma$ 大量误识为 "α"、"g"，本文按英文原版统一为 $\sigma^2$；MSE $=E[x^4]-2\sigma^2E[x^2]+\sigma^4=3\sigma^4-2\sigma^4+\sigma^4=2\sigma^4$ 与英文原版一致。
3. **全文 $\theta$、$\bar{x}$、$\sigma$**：OCR 分别常误识为 "6"、"元"、"g"，均按 Kay 英文原版（*Fundamentals of Statistical Signal Processing*, Vol I, Prentice Hall）校订。
4. **式（12.61）（12.62）的频域解**：OCR 该页公式严重残缺（"8V-V81"等），本文按英文原版补出 $h[n]\star r_{xx}[n]=r_{ss}[n]$ 与 $H(f)=P_{ss}(f)/(P_{ss}(f)+P_{ww}(f))$。

## 3. 图片清单

| 编号 | 文件 | 脚本 | 碰撞检测 | 状态 |
|------|------|------|---------|------|
| Fig016 | `Documents/figures/Fig016_维纳滤波前后对比.png` | `Temp/scripts/make_fig016.py` | `check_figure` 通过（saved 输出见运行日志） | ✅ 已生成 |

图注要点：AR(1) 信号 $s[n]=0.9s[n-1]+u[n]$、$\sigma_u^2=1$、观测噪声 $\sigma^2=1$、$N=256$、种子 20260916；本现实 MSE 从 1.0651 降到 0.4364，频响 $H(0)=0.9901$、$H(0.5)=0.2169$（低通收缩，符合预期）。

## 4. 未决疑问

- 无重大疑问。原书 §12.7 滤波问题的时变 FIR（式 12.58）与维纳-霍夫方程（12.59）的 OCR 有大量下标错位（"h(n)[k]=an-k" 等），本文按英文原版校订为 $h^{(n)}[k]=a_{n-k}$ 的倒序约定，并在正文以文字表述为主，未逐式转录下标。
