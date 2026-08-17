# Vol2 Ch02 重要 PDF 的总结 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol2_Ch02_重要PDF的总结.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/v2ch02/ocr_page_485~515.txt`（PDF 第 485~515 页，书内第 470~500 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §2.1 引言：性能评估靠解析/数值确定 PDF，否则蒙特卡洛 | `ocr_page_485.txt` L5~10（"检测器性能的评估取决于解析地或者数值地确定数据样本函数的概率密度函数…必须借助蒙特卡洛计算机模拟技术"） |
| 式 (2.1) 高斯标量 PDF、$\mathcal{N}(\mu,\sigma^2)$ | `ocr_page_485.txt` L14~19（"高斯概率密度函数（PDF）（也称为正态PDF）定义为…用 N(μ,σ²) 表示"） |
| 式 (2.2) $\mu=0$ 时矩 $1\cdot3\cdot5\cdots(n-1)\sigma^n$（偶）/0（奇） | `ocr_page_485.txt` L20~23（"1·3.5.(n-1)σ^n n 为偶数 E(x^n)=…n为奇数 (2.2)"） |
| 式 (2.3) $Q(x)$ 右尾、$Q=1-\Phi$、无闭式解 | `ocr_page_485.txt` L31~32 + `ocr_page_486.txt` L3（"称为右尾概率…得不到闭合形式的解"） |
| 式 (2.4) 近似，$x>4$ 相当精确（图 2.2） | `ocr_page_486.txt` L4~6、L36（"一种有时很有用的近似是…(2.4)…当 x>4 时近似是相当精确的"） |
| 正态概率纸：$Q(x)$ 成直线（图 2.3）；混合高斯右尾 $Q(x)/2+Q(x/\sqrt2)/2$（图 2.4） | `ocr_page_486.txt` L36~44（"在正态概率纸上画出 Q(x)…变成了一条直线…混合高斯PDF…右尾概率的函数形式为 Q(x)/2 + Q(x/√2)/2"） |
| $Q^{-1}$ 存在、附录 2C `Qinv.m` | `ocr_page_488.txt` L3~5 |
| 式 (2.5) 多维高斯 PDF、均值矢量、协方差矩阵定义 | `ocr_page_488.txt` L6~20 |
| 式 (2.6) 四阶矩 = 三个二阶矩乘积之和（Isserlis） | `ocr_page_488.txt` L21~23（"E(xi xj xk xl) = E(xi xj)E(xk xl) + E(xi xk)E(xj xl) + E(xi xl)E(xj xk) (2.6)"） |
| §2.2.2 中心 χ²：式 (2.7) PDF、$\Gamma$ 函数定义与性质 | `ocr_page_488.txt` L25~34（"自由度为ν的chi平方PDF定义为…Γ(u)=∫t^(u-1)exp(-t)dt…Γ(1/2)=√π，Γ(n)=(n-1)!"） |
| χ² 来源：$x_i\sim\mathcal{N}(0,1)$ IID，$x=\sum x_i^2$；图 2.5 随 $\nu$ 增大趋高斯；$\nu=1$ 在 $x=0$ 无穷大 | `ocr_page_488.txt` L35~37 |
| 式 (2.8) $E(x)=\nu$、式 (2.9) $\mathrm{var}(x)=2\nu$ | `ocr_page_488.txt` L38~42 |
| $\nu=2$ → 指数 PDF | `ocr_page_488.txt` L43 + `ocr_page_489.txt` L3（"当ν=2时…称为指数PDF"） |
| 式 (2.10) 偶 $\nu$ 右尾、式 (2.11) 奇 $\nu$ 右尾 | `ocr_page_489.txt` L17~34 |
| §2.2.3 非中心 χ²：$x_i\sim\mathcal{N}(\mu_i,1)$、$\lambda=\sum\mu_i^2$；式 (2.12)~(2.15) | `ocr_page_489.txt` L37~41 + `ocr_page_490.txt` L4~20 |
| 式 (2.16) $E(x)=\nu+\lambda$、式 (2.17) $\mathrm{var}(x)=2\nu+4\lambda$ | `ocr_page_491.txt` L6~10（"E(x)=ν+λ (2.16) var(x)=2ν+4λ"） |
| $\lambda=0$ 非中心 χ² 化简为中心 χ²；记号 $\chi'^2_\nu(\lambda)$ | `ocr_page_491.txt` L4~5 |
| §2.2.4 中心 F：$x=(x_1/\nu_1)/(x_2/\nu_2)$；式 (2.17) PDF 含 Beta；式 (2.18) 均值/方差 | `ocr_page_491.txt` L15~38 |
| $F_{\nu_1,\nu_2}$ 分子/分母自由度；图 2.7；$\nu_2\to\infty$ 时 $F\to\chi^2_{\nu_1}/\nu_1$（习题 2.3） | `ocr_page_491.txt` L24~38 |
| §2.2.5 非中心 F：$x_1\sim\chi'^2_{\nu_1}(\lambda)$；式 (2.19) PDF、式 (2.20) 均值/方差；Patnaik 1949 | `ocr_page_492.txt` L26~44 |
| §2.2.6 瑞利：$x=\sqrt{x_1^2+x_2^2}$，$x_1,x_2\sim\mathcal{N}(0,\sigma^2)$；式 (2.21) PDF | `ocr_page_492.txt` L46~50 |
| 式 (2.22) $E(x)=\sigma\sqrt{\pi/2}$（OCR 作 "√(πσ²/2)" 残片）、方差 $(2-\pi/2)\sigma^2$ | `ocr_page_493.txt` L4~7（OCR 残缺，据英文原版校订，见 §2） |
| 式 (2.23) $\Pr\{x>\gamma\}=\exp(-\gamma^2/(2\sigma^2))$；瑞利 $x^2\sim\chi^2_2$ | `ocr_page_493.txt` L8~19（"如果是瑞利随机变量，那么 x²=u，其中 u~χ²₂…Qχ²(x)=exp(-x/2)…于是得到(2.23)式"） |
| §2.2.7 莱斯：$x_1\sim\mathcal{N}(\mu_1,\sigma^2)$、$x_2\sim\mathcal{N}(\mu_2,\sigma^2)$；式 (2.24) PDF 含 $I_0$；$\alpha^2=\mu_1^2+\mu_2^2$ | `ocr_page_493.txt` L32~34 + `ocr_page_494.txt` L3~9 |
| $\alpha^2=0$ 莱斯化简为瑞利；矩用合流超几何函数（Rice 1948 / McDonough & Whalen 1995） | `ocr_page_494.txt` L9~10 |
| 式 (2.26) $\Pr\{x>\gamma\}=Q_{\chi'^2_2(\lambda)}(\gamma^2/\sigma^2)$，$\lambda=(\mu_1^2+\mu_2^2)/\sigma^2$ | `ocr_page_494.txt` L11~20 |
| §2.3 高斯二次型：$\mathbf{x}^T\mathbf{A}\mathbf{x}$，$\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\mathbf{C})$；三条特殊情况 | `ocr_page_494.txt` L36~41 + `ocr_page_495.txt` L4~10 |
| 式 (2.27) $A=C^{-1},\mu=0 \Rightarrow \chi^2_n$；式 (2.28) $A=C^{-1},\mu\neq0 \Rightarrow \chi'^2_n(\lambda),\lambda=\mu^TC^{-1}\mu$；式 (2.29) $A$ 幂等秩 $r$, $C=I$, $\mu=0 \Rightarrow \chi^2_r$ | `ocr_page_494.txt` L39~41、`ocr_page_495.txt` L4~10 |
| §2.4 渐近高斯 PDF：零均值 WSS 高斯过程，$C=R$ Toeplitz；式 (2.30) $\lambda_i=P_{xx}(f_i)$、$f_i=i/N$ | `ocr_page_495.txt` L16~51 |
| 相关时间定义（数据长度远大于相关时间才适用） | `ocr_page_495.txt` L51~53 |
| 式 (2.32) 特征分解 $R=\sum\lambda_i v_i v_i^H$；式 (2.33) $\det(R)=\prod P_{xx}(f_i)$；式 (2.34) $R^{-1}$ | `ocr_page_496.txt` L47~51 + `ocr_page_497.txt` L6~16 |
| 式 (2.35) 渐近对数似然（离散）、周期图 $I(f)$、式 (2.36) 积分形式 | `ocr_page_497.txt` L20~66 |
| §2.5 蒙特卡洛：$T=\frac1N\sum x[n]$，$x[n]\sim\mathcal{N}(0,\sigma^2)$ IID；式 (2.37) $\Pr\{T>\gamma\}=Q(\gamma/\sqrt{\sigma^2/N})$ | `ocr_page_498.txt` L4~12 |
| 蒙特卡洛步骤（`x=sqrt(var)*randn(N,1)`；$T=\frac1N\sum x[n]$；$M$ 现实；$\hat P=M_c/M$） | `ocr_page_498.txt` L15~23 |
| 式 (2.38) $M\ge[Q^{-1}(\alpha/2)]^2(1-P)/(\varepsilon^2 P)$ | `ocr_page_498.txt` L27~33 |
| 例：$P=0.16$、95%、$\varepsilon=0.01$ → $M\approx2\times10^5$；重要采样 Mitchell 1981 | `ocr_page_499.txt` L5~10 |
| 图 2.10：$N=10,\sigma^2=10$，$M=1000$ 与 $M=10000$（附录 2E `montecarlo.m`） | `ocr_page_499.txt` L11~16 + `ocr_page_515.txt` L8~12（`var=10; N=10; M=1000`） |
| 附录 2A 蒙特卡洛次数推导：$\hat P$ 近似高斯，$E(\hat P)=P$，$\mathrm{var}(\hat P)=P(1-P)/M$ | `ocr_page_505.txt` L4~27 |
| 附录 2C `Q.m` 验证用例 $Q([0,1,2])=[0.5,0.1587,0.0228]$、`Qinv.m` $=[0,0.9998,1.9991]$ | `ocr_page_509.txt` L12、L25~26 |
| 附录 2D `Qchipr2.m` 验证用例：$(1,2,0.5)\to0.7772$、$(5,6,10)\to0.5063$、$(8,10,15)\to0.6161$ | `ocr_page_513.txt` L3~8 |
| 页码：书内 = PDF − 15 | `ocr_page_486.txt` L1（顶栏"471"）、`ocr_page_500.txt` L1（顶栏"485"），与目录"第2章起于470、第3章起于501"一致 |

## 2. 据英文原版校订处

1. **瑞利均值（式 2.22）**：OCR 页 493 L4~7 作 "To2 E(α) (2.22) var(r) 右尾概率…"，均值项残缺（"To2"为 $\sqrt{\pi\sigma^2/2}$ 的误认）。据 Kay 英文原版，瑞利 $E[x]=\sigma\sqrt{\pi/2}$、$\mathrm{var}[x]=(2-\pi/2)\sigma^2$。理由：瑞利是 $\chi^2_2$ 的平方根，$\chi^2_2$ 均值 2、方差 4，平方根后均值/方差为上述值，且与式 (2.23) 右尾 $\exp(-\gamma^2/(2\sigma^2))$ 的 $\sigma^2$ 口径自洽。
2. **中心 F 均值/方差（式 2.18）**：OCR 页 491 L29~38 残缺（"E(r) 12/(ν2-2)…var(r)=2ν2²(ν1+ν2-2)/(ν1(ν2-2)²(ν2-4))"）。据英文原版校订为 $E[x]=\nu_2/(\nu_2-2)\ (\nu_2>2)$、$\mathrm{var}[x]=2\nu_2^2(\nu_1+\nu_2-2)/\big(\nu_1(\nu_2-2)^2(\nu_2-4)\big)\ (\nu_2>4)$，正文已用。
3. **非中心 F 均值/方差（式 2.20）**：OCR 页 492 L33~41 残缺（"ν2(ν1+λ)/(ν1(ν2-2))…2(ν2/ν1)²[((ν1+λ)²+(ν1+2λ)(ν2-2))/((ν2-2)²(ν2-4))]"）。据英文原版校订，正文 §4 已用，并保留"$\nu_2>2$/$\nu_2>4$"的矩存在条件。
4. **式 (2.37) 的统计量口径**：OCR 页 498 正文残缺，但附录 2E `montecarlo.m` 明确 `T(i)=mean(x)`、`Ptrue=Q(gamma/(sqrt(var/N)))`，故 (2.37) 是**样本均值**统计量 $T=\frac1N\sum x[n]$ 的右尾 $Q(\gamma/\sqrt{\sigma^2/N})$，不是平方和。正文 §8 与备忘 6 已按此口径转述。
5. **莱斯右尾的非中心参量**：OCR 页 494 L19 作 "其中 λ=(μ₁²+μ₂²)/σ²"，与式 (2.26) 的 $Q_{\chi'^2_2(\lambda)}(\gamma^2/\sigma^2)$ 一致，无需校订。
6. **指数分布的参数**：OCR 只给"ν=2 时 χ² 即指数 PDF，$p(x)=\tfrac12 e^{-x/2}$"，未给均值/方差。本文速查表写 $E=2,\mathrm{var}=4$（由 $\chi^2_2$ 的 $\nu=2$ 直接得），属推导自算，已在表中注明来源。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig022 | `figures/Fig022_重要分布速查.png` | `Temp/scripts/make_fig022.py` | ✅ 通过（`plotutil.check_figure`，含 (a) 形状 + (b) 关系树两子图） |

- 图意：上 (a) 五条分布曲线（高斯/χ²₂/χ²₆/瑞利/莱斯），下 (b) 分布关系树（高斯 → 平方和/包络 → χ²/非中心χ²/瑞利/莱斯）。
- 运行命令 `py -3.14 make_fig022.py`（workdir=`Temp/scripts`）已见到 `saved ...` 输出。生成时修过两处：① 初版关系树放在半宽子图内，莱斯文本框越出 Axes 边界且与瑞利文本重叠，改为上下两行全宽布局；② 树箭头由 `annotate`（空文本包围盒含箭头 bbox，与框文本相交触发碰撞）改为 `FancyArrowPatch`（`add_patch`，不产生文字包围盒，同 `make_fig005.py` 的做法）。修复后无碰撞告警。
- 曲线为确定性函数（scipy.special.gamma/iv 计算），不涉及随机种子。

## 4. 未决疑问

1. **式 (2.4) 的精确系数**：OCR 页 486 L4~5 只保留 "exp" 与 "(2.4)"，未识别出完整近似式。本文按 Kay 英文原版与习题 2.2 的分部积分提示，写为 $Q(x)\approx\frac{1}{x\sqrt{2\pi}}\exp(-x^2/2)$（一阶近似）。若要含 $1-1/x^2$ 修正项的高阶形式，需对照英文原版 (2.4) 再核。
2. **式 (2.22) 是否含方差**：OCR 页 493 把均值标 (2.22)、方差紧随其后、右尾标 (2.23)。英文原版是否把均值与方差合并在同一式号下，本文未逐一比对页码，按"均值 (2.22)、方差同处给出、右尾 (2.23)"转述，不影响结论。
3. **§2.2.4 的 F 分布 PDF（式 2.17）具体形式**：OCR 页 491 L19~23 残缺（仅 "(1+x)^{...}" 与 Beta 函数关系可见）。正文未抄写 F 分布 PDF 的具体闭式，只给构造与矩，规避了残缺处；如需完整 PDF，参考 Johnson & Kotz 或英文原版。
