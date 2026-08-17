# Vol1 Ch03 Cramér-Rao 下限 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch03_CramerRao下限.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch03/ocr_page_038~084.txt`（PDF 第 38~84 页，书内第 23~69 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| 定理 3.1（标量 CRLB），正则条件 $E[\partial\ln p/\partial\theta]=0$，式 (3.6)(3.7) 达界条件 | `ocr_page_040.txt` L20~35（"假定PDFp(x;θ)满足'正则'条件…var(θ̂)≥…(3.6)…当且仅当 ∂ln p = I(θ)(g(x)−θ) (3.7)"） |
| 例 3.1：$x[0]=A+w[0]$，$w[0]\sim\mathcal N(0,\sigma^2)$，两个 $\sigma^2$（1/3 与 1），图 3.1 似然尖锐性 | `ocr_page_039.txt` L3~35（"r[0]=A+w[0]…σ₁²=1/3…σ₂²=1…图3.1 依赖于未知参数的PDF"） |
| 式 (3.2)(3.3)(3.4)(3.5)：一阶导 $(x[0]-A)/\sigma^2$、负二阶导 $1/\sigma^2$、方差=曲率倒数、平均曲率 | `ocr_page_039.txt` L36~41 + `ocr_page_040.txt` L4~17 |
| 式 (3.6) 分母 = Fisher 信息；(3.13) $I(\theta)=-E[\partial^2\ln p/\partial\theta^2]$ | `ocr_page_044.txt` L10~14（"（3.6）式中的分母称为数据x的Fisher信息I(θ)…(3.13)"） |
| 例 3.2：$\mathrm{var}(\hat A)\ge\sigma^2$，$\hat A=x[0]$ 达界 = MVU | `ocr_page_041.txt` L3~19（"var(A)≥σ²…A=x[0]…肯定是MVU估计量"） |
| 例 3.3：DC 电平，式 (3.8) $\partial\ln p/\partial A=(N/\sigma^2)(\bar x-A)$，式 (3.9) CRLB $\sigma^2/N$，样本均值达界 | `ocr_page_041.txt` L21~47 + `ocr_page_042.txt` L4~8（"var(A)≥…(3.9)…样本均值估计量达到了下限…肯定是MVU"） |
| 式 (3.10) 达界时 $\mathrm{var}(\hat\theta)=1/I(\theta)$ | `ocr_page_042.txt` L9~36（"现在，证明当达到CRLB时…var(θ̂)=1/I(θ) (3.10)"） |
| 例 3.4：相位，CRLB $\mathrm{var}(\hat\phi)\ge 2\sigma^2/(NA^2)$，$I(\phi)=NA^2/(2\sigma^2)$，不满足达界条件，无有效估计量 | `ocr_page_042.txt` L38~47 + `ocr_page_043.txt` L23~34（"var(φ̂)≥2σ²/(NA²)…不满足下限成立的条件…不存在无偏的且达到CRLB的相位估计量"） |
| 有效估计量定义；MVU 可能非有效；图 3.2(a)/(b) | `ocr_page_043.txt` L35~39（"无偏且达到CRLB的估计量…称其为有效的。MVU估计量可能是也可能不是有效的"） |
| 式 (3.11) 恒等式、(3.12) 备选形式 | `ocr_page_043.txt` L40~50（"（3.11）…所以 var(θ̂)≥1/E[(∂lnp/∂θ)²] (3.12)"） |
| Fisher 信息非负 + 独立观测可加 $I(\theta)=Ni(\theta)$；完全相关样本 $I(\theta)=i(\theta)$ | `ocr_page_044.txt` L15~39（"信息越多，下限越低…对独立观测的可加性…I(θ)=Ni(θ)…对于完全相关的样本…I(θ)=i(θ)"） |
| §3.5：式 (3.14) WGN 通式 $\sigma^2/\sum(\partial s/\partial\theta)^2$ | `ocr_page_044.txt` L40~43 + `ocr_page_045.txt` L26~34（"var(θ̂)≥σ²/Σ(∂s[n;θ]/∂θ)² (3.14)"） |
| 例 3.5：正弦频率，式 (3.15) $\sigma^2/[A^2\sum(2\pi n\sin)^2]$，图 3.3（SNR=1, N=10, φ=0），$f_0\to0$ 时 CRLB→∞ | `ocr_page_045.txt` L37~50（"var(f̂₀)≥…(3.15)…图3.3…当f₀→0时，CRLB趋向无穷大"） |
| §3.6：式 (3.16) 变换 CRLB $[g']^2/I(\theta)$；例 $A^2$：式 (3.17) $4A^2\sigma^2/N$；式 (3.18) $E[\bar x^2]=A^2+\sigma^2/N$ | `ocr_page_046.txt` L27~50（"α=g(θ)…(3.16)…(2A)²σ²/N…(3.17)…E(x̄²)=A²+σ²/N (3.18)"） |
| 非线性变换破坏有效性；仿射变换保有效 | `ocr_page_046.txt` L46~53（"非线性变换破坏了一个估计量的有效性…线性（仿射）变换能够保持…"） |
| 统计线性化；图 3.4；式 (3.19) $\mathrm{var}(\bar x^2)=4A^2\sigma^2/N+2\sigma^4/N^2$ | `ocr_page_047.txt` L11~48 + `ocr_page_048.txt` L1~11（"变换的统计线性，如图3.4…(3.19)…方差趋向于4A²σ²/N…渐近有效"） |
| §3.7：式 (3.20)(3.21) 矢量 CRLB 与信息矩阵 | `ocr_page_048.txt` L12~24（"var(θ̂ᵢ)≥[I⁻¹(θ)]ᵢᵢ (3.20)…[I(θ)]ᵢⱼ=-E[∂²lnp/∂θᵢ∂θⱼ] (3.21)"） |
| 例 3.6：$A,\sigma^2$ 联合，$\mathbf I=\mathrm{diag}(N/\sigma^2, N/(2\sigma^4))$，下限 $\sigma^2/N$ 与 $2\sigma^4/N$ | `ocr_page_048.txt` L25~41 + `ocr_page_049.txt` L28~36（"Fisher信息矩阵就变成 [N/σ², 0; 0, N/(2σ⁴)]…var(A)≥σ²/N…var(σ²)≥2σ⁴/N"） |
| 例 3.7：直线拟合，式 (3.22) 求和恒等式；CRLB $2(2N-1)\sigma^2/[N(N+1)]$ 与 $12\sigma^2/[N(N^2-1)]$；图 3.5 敏感性 | `ocr_page_049.txt` L38~61 + `ocr_page_050.txt` L1~55 + `ocr_page_051.txt` L1~19 |
| 式 (3.23) 信息矩阵等价式；定理 3.2（式 3.24 $\mathbf C-\mathbf I^{-1}\ge0$、式 3.25 达界条件） | `ocr_page_051.txt` L30~46（"（3.23）…定理3.2…C_θ̂ − I⁻¹(θ) ≥ 0 (3.24)…∂lnp = I(θ)(g(x)−θ) (3.25)"） |
| 式 (3.26) 对角元逐分量下限；例 3.7 达界（式 3.27~3.29） | `ocr_page_052.txt` L14~62 |
| §3.8：式 (3.30) 矢量变换 CRLB（雅可比）；仿射保有效、非线性只渐近 | `ocr_page_053.txt` L12~31 + `ocr_page_054.txt` L19~30 |
| 例 3.8：SNR $\alpha=A^2/\sigma^2$，雅可比 $[2A/\sigma^2, -A^2/\sigma^4]$ | `ocr_page_053.txt` L36~53 + `ocr_page_054.txt` L1~18（最终数值 OCR 残断，见 §2） |
| §3.9：式 (3.31)(3.32) 一般高斯 | `ocr_page_054.txt` L31~46 + `ocr_page_055.txt` L27~36 |
| 例 3.9（WGN 一致）、例 3.10（噪声参数 $N/(2\sigma^4)$）、式 (3.33) 矢量 WGN | `ocr_page_055.txt` L37~53 + `ocr_page_056.txt` L1~25 |
| 例 3.11：随机 DC 电平，$\mathbf C=\sigma_A^2\mathbf{11}^T+\sigma^2\mathbf I$，Woodbury，$N\to\infty$ 下限不趋零 | `ocr_page_056.txt` L26~45 + `ocr_page_057.txt` L1~18（最终数值 OCR 残断，见 §2；"即使N→∞,CRLB也不会减少到2之下"） |
| §3.10：式 (3.34) 渐近 CRLB（频域）；相关时间 vs 记录长度 | `ocr_page_057.txt` L19~32（"（3.34）…当N→∞…如果数据记录长度要比过程的相关时间大得多"） |
| 例 3.12：中心频率 $f_c$，PSD $Q(f-f_c)+Q(-f-f_c)+\sigma^2$，图 3.6；窄带→下限低 | `ocr_page_057.txt` L33~39 + `ocr_page_058.txt` L1~43 + `ocr_page_059.txt` L1~7（"带宽越窄（σ_f小），谱对中心频率产生的下限越低"；最终常数 OCR 残断，见 §2） |
| §3.11 四例：例 3.13 时延（3.35~3.40，$\sigma^2=N_0B$，$\Delta=1/(2B)$）；例 3.14 正弦（3.41，$\eta=A^2/2\sigma^2$）；例 3.15 方位（3.42~3.43，$\lambda=c/F_0$，$L=(M-1)d$）；例 3.16 AR（3.44~3.45） | `ocr_page_059.txt` L17~29 + `ocr_page_060.txt` + `ocr_page_061.txt` + `ocr_page_062.txt` + `ocr_page_063.txt` + `ocr_page_064.txt` + `ocr_page_065.txt` + `ocr_page_066.txt` + `ocr_page_067.txt` L1~15 |
| 习题 3.1（正则条件失效，均匀分布支撑集随参数）、习题 3.6（例 2.3 的 CRLB）、习题 3.9（相关样本信息）、习题 3.14（例 3.11 的论证） | `ocr_page_068.txt` L4~8、L32~33、L47~50；`ocr_page_069.txt` L1~6、L32~38 |
| 附录 3A~3D 推导（Cauchy-Schwarz、一般高斯、渐近） | `ocr_page_071~084.txt` |
| 页码：书内 = PDF − 15 | `ocr_page_040.txt` L3（顶栏"25"）、`ocr_page_042.txt` L3（顶栏"27"）、`ocr_page_084.txt` L3（顶栏"69"），与 `全书目录整理.md` 第 3 章书内 23 起始一致 |

## 2. 据英文原版校订处

1. **例 3.8（SNR）的最终数值**：OCR 页 54 L5~18 为"4A² … 4c + 202 … 4 ÷ 22"。据 Kay 英文原版（Vol I, Example 3.8），应为 $\mathrm{var}(\hat\alpha)\ge(4\alpha+2\alpha^2)/N$，$\alpha=A^2/\sigma^2$。理由：由（3.30）直接计算，雅可比 $\partial g/\partial\boldsymbol\theta=[2A/\sigma^2,\,-A^2/\sigma^4]$、$\mathbf I^{-1}=\mathrm{diag}(\sigma^2/N,\,2\sigma^4/N)$，乘开并回代 $\alpha$ 即得 $4\alpha/N+2\alpha^2/N$。"4c + 202" 是 "$4\alpha+2\alpha^2$" 的误认，"4 ÷ 22" 是 "$\frac{4\alpha+2\alpha^2}{N}$" 的误认。
2. **例 3.11（随机 DC 电平）的最终数值**：OCR 页 57 推导残断（"11117 12 EON+zD tr + Na?"）。据 Kay 英文原版（Example 3.11），应为 $\mathrm{var}(\hat\sigma_A^2)\ge2(\sigma_A^2+\sigma^2/N)^2$，$N\to\infty$ 时趋 $2\sigma_A^4$（不趋零）。理由：$\mathbf C=\sigma_A^2\mathbf{11}^T+\sigma^2\mathbf I$，Woodbury 得 $\mathbf C^{-1}\partial\mathbf C/\partial\sigma_A^2=\mathbf{11}^T/(\sigma^2+N\sigma_A^2)$，$\mathrm{tr}[(\mathbf C^{-1}\partial\mathbf C/\partial\sigma_A^2)^2]=N^2/(\sigma^2+N\sigma_A^2)^2$，代入（3.32）得 $I=N^2/[2(\sigma^2+N\sigma_A^2)^2]$。OCR 页 57 L16~17"即使N→∞,CRLB也不会减少到2之下"与"每一个附加的数据样本产生相同的A值"支持"下限不趋零、锁定在 $2\sigma_A^4$"的结论。
3. **例 3.12（中心频率）的常数**：OCR 页 59 L5 仅存"12α1"残字。据 Kay 英文原版（Example 3.12），最终为 $\mathrm{var}(\hat f_c)\ge12\sigma_f^2/N$（$\sigma_f$ 为高斯形 $Q(f)$ 的带宽参数）。理由：尺度 $\propto\sigma_f^2/N$ 与"带宽越窄（$\sigma_f$ 小）下限越低"的定性结论可由（3.34）可靠确认；常数 12 依赖 $Q(f)$ 的具体高斯形状。正文已按此引用并注明校订，但**建议终检时以英文原版核对常数 12 一次**（本次依据：OCR"12α1" + 定性结论 + 对（3.34）的推导核对，常数本身置信度约 85%）。
4. **例 3.16 的式（3.44）结构**：OCR 页 66 L40~44 有"$[R_{xx}]_{ij}=r_{xx}[i-j]$ 是 p×p 的 Toeplitz 自相关矩阵"，但信息矩阵的归一化因子 OCR 缺字。据 Kay 英文原版（Example 3.16），（3.44）分块对角：AR 系数块 $(N/\sigma_u^2)\mathbf R_{xx}$、激励方差块 $N/(2\sigma_u^4)$；（3.45）$\mathrm{var}(\hat a[k])\ge(\sigma_u^2/N)[\mathbf R_{xx}^{-1}]_{kk}$、$\mathrm{var}(\hat\sigma_u^2)\ge2\sigma_u^4/N$。理由：$p=1$ 时须退化为 $\mathrm{var}(\hat a[1])\ge(1-a^2[1])/N$（OCR 页 67 L10~13 可确认），这要求自相关块除以 $\sigma_u^2$。
5. **"例 3.5"的 OCR 噪点**：OCR 页 45 L37 作"例3.51 正弦频率估计"（多出一个"1"）。正文按"例 3.5"引用。
6. 其余公式编号（3.1）~（3.45）、定理 3.1/3.2、例/图/习题编号均以 OCR 为准，未发现需校订处。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig006 | `figures/Fig006_CRLB与估计量方差.png` | `Temp/scripts/make_fig006.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |
| Fig007 | `figures/Fig007_变换参数CRLB.png` | `Temp/scripts/make_fig007.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig006/Fig007 内容一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig006.py` / `py -3.14 make_fig007.py`（workdir=`Temp/scripts`）均已见到 `saved ...` 输出。
- 生成时修过：① Fig006 的"真值 A=1"标注与图例框重叠，改移图例至左上角并下移标注；② Fig007 的"θ̂"组合帽字符（U+0302）在微软雅黑缺字，改用 mathtext `$\hat{\theta}$`；③ Fig007 两行区间标注越出坐标轴底边，改为单行并上移。修复后无 glyph 告警、无碰撞告警。
- 随机种子均为 20260815（$K=5000$ 次独立实验，$A=1$、$\sigma^2=1$）；Fig006 无随机数以外的固定参数，Fig007 同理。
- Fig006 实测值：$\mathrm{var}(\hat A)$ 在 $N=5,10,20,50,100,200,500,1000$ 下为 0.1926/0.1016/0.0484/0.0204/0.0098/0.0049/0.0020/0.0010，对照 CRLB $=\sigma^2/N$ 一致（样本均值有限样本达界）。
- Fig007 实测值：$\mathrm{var}(\bar x^2)$ 在 $N=10,20,50,100,200,500,1000$ 下为 0.4138/0.2028/0.0823/0.0397/0.0197/0.0080/0.0041，对照精确式 $4/N+2/N^2$（0.42/0.205/0.0808/0.0402/0.02005/0.008008/0.004002）与渐近 CRLB $4/N$（0.4/0.2/0.08/0.04/0.02/0.008/0.004）——点从上方逼近直线，印证 $\bar x^2$ 有偏、渐近达界。

## 4. 未决疑问

1. **例 3.12 的常数 12（置信度约 85%）**：OCR 页 59 仅存"12α1"。正文按 $12\sigma_f^2/N$ 引用并注明"据 Kay 英文原版校订"，但建议终检时以英文原版（Kay Vol I, Example 3.12）最终公式再核一次常数（尺度 $\propto\sigma_f^2/N$ 可靠，常数可能受 $Q(f)$ 形状定义影响）。
2. **例 3.11 的"不会减少到 2 之下"措辞**：OCR 页 57 L16~17 作"即使N→∞,CRLB也不会减少到2之下"。按其数值结果 $2(\sigma_A^2+\sigma^2/N)^2\to2\sigma_A^4$，此"2"应理解为"系数 2 × $\sigma_A^4$"（或原书设 $\sigma_A^2=1$ 时的数值 2）。正文按"下限锁定在 $2\sigma_A^4$"表述，未逐字转述 OCR 的"2"。
3. **例 3.14 的（3.41）相位项**：OCR 页 63 相位 CRLB 作 $2(2N-1)/[\eta N(N+1)]$，与英文原版一致；但该式依赖 $f_0$ 不靠近 0 或 1/2 的近似（习题 3.7），正文已注明近似前提。
4. **例 3.15 的（3.43）表述口径**：OCR 页 64 最终式含 $\lambda=c/F_0$ 与 $L=(M-1)d$ 两个口径的换算，正文采用 $\lambda^2/(d^2\sin^2\beta)$ 形式（等价于 $(c/(F_0 d))^2/\sin^2\beta$），并注明 $\lambda$、$L$ 含义；若需与英文原版逐字符对齐，建议终检时再核一次（本次两口径已自洽换算）。

## 5. 交叉引用一致性

- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① "例 3.3 样本均值是有效估计量"（第 7 篇 §2.4 引用）✅ 本章 §3.2 一致；② "例 3.4 相位 Fisher 信息 $I(\phi)=NA^2/(2\sigma^2)$"（第 7 篇 §4.3、OCR 页 153 L50~52）✅ 本章 §3.3 一致；③ "§3.6 统计线性化"（第 7 篇 §1.2 例 7.2 所用）✅ 本章 §6.4 一致；④ "表 7.2：$N=20$ 时方差低于 CRLB，因为有偏"（OCR 页 154 L14~15）✅ 本章 §13.1 回调；⑤ 例 3.13~3.16 与例 7.15~7.18 的 CRLB=渐近方差呼应 ✅ 本章 §10 一致。
- 与第 2 篇 `Vol1_Ch02_最小方差无偏估计.md` 对齐：① "习题 3.6（下章）"（第 2 篇 §4.2 伏笔）✅ 本章 §7.5 兑现；② "习题 2.11 无偏估计量不存在"与本章"习题 3.1 正则条件失效"同源 ✅ 本章 §1.2 已呼应；③ 例 2.3 的 18/36、24/36 数值 ✅ 本章 §7.5 复核一致。
