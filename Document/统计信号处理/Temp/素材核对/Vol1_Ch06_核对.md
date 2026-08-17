# Vol1 Ch06 最佳线性无偏估计量 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch06_最佳线性无偏估计量.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch06/ocr_page_125~143.txt`（PDF 第 125~143 页，书内第 110~128 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §6.1 引言：MVU 即使存在也无法求出；准最佳损失无法确定；BLUE 只需一、二阶矩 | `ocr_page_125.txt` L4~14（"MU估计量即使存在也光法求出…永远也不能确定我们可能失去多少性能…利用PDF的一、二阶矩的知识就可以确定BLUE"） |
| §6.2 小结：BLUE 由 (6.5)、方差 (6.6)；矢量 (6.16)(6.17)；高斯时 BLUE 也是 MVU | `ocr_page_125.txt` L16~20 |
| 式 (6.1) 线性估计量 $\hat\theta=\sum a_n x[n]$；BLUE = 无偏 + 最小方差；"线性类中最佳" | `ocr_page_125.txt` L22~32（"BLUE限定估计量与数据呈线性的…只有MVU估计量刚好是线性时，BLUE才是最佳的"） |
| 例 5.8 均匀噪声：MVU 非线性（max+min 的中点），BLUE=样本均值，准最佳，图 6.1(b) | `ocr_page_126.txt` L7~11（"均匀分布的噪声均值的估计问题（参见例5.8），求得的MVU估计量是 max x[n]…/2N…非线性…BLUE是样本均值…准最佳的"） |
| WGN 功率估计：MVU=(1/N)Σx²[n] 非线性；线性估计量期望恒为 0，连无偏都做不到；换数据 y=x² | `ocr_page_126.txt` L35~45 + `ocr_page_127.txt` L1~13 |
| 式 (6.2) 无偏约束 $E(\hat\theta)=\sum a_n E(x[n])=\theta$；式 (6.3) 方差 $=\mathbf a^T\mathbf C\mathbf a$ | `ocr_page_127.txt` L15~34 |
| 式 (6.4) 均值线性 $E(x[n])=s[n]\theta$；反例 $E(x[n])=\cos\theta$ 无解；$x[n]=s[n]\theta+w[n]$ | `ocr_page_127.txt` L35~45 |
| 约束 $\mathbf a^T\mathbf s=1$（$\mathbf s=[s[0]\cdots s[N-1]]^T$）；式 (6.5)(6.6) BLUE 与最小方差 | `ocr_page_128.txt` L5~26 |
| 确定 BLUE 只需 1. s（或成比例的均值）2. 协方差 C，即前二阶矩，不是整个 PDF | `ocr_page_128.txt` L33~36 |
| 例 6.1 白噪声 DC：$s[n]=1$，$\mathbf s=\mathbf 1$，BLUE=样本均值，var=σ²/N，与 PDF 无关 | `ocr_page_128.txt` L37~43 + `ocr_page_129.txt` L4~11 |
| 例 6.2 不相关噪声 DC：式 (6.7)(6.8) 加权平均与方差；"方差小的样本赋予重的 BLUE 加权" | `ocr_page_129.txt` L12~25 + `ocr_page_130.txt` L1~9 |
| $\mathbf C^{-1}$ 起着预白化作用，呼应例 4.4；高斯噪声下 BLUE 与 MVU 相同是一般结论 | `ocr_page_130.txt` L10~15 |
| 式 (6.9) 矢量线性估计量、$\hat{\boldsymbol\theta}=\mathbf A\mathbf x$；式 (6.11) $E(\hat\theta)=\mathbf A E(x)=\theta$；式 (6.12) $E(x)=\mathbf H\theta$ | `ocr_page_130.txt` L16~41 |
| 式 (6.13) $\mathbf A\mathbf H=\mathbf I$；式 (6.14) $\mathbf a_i^T\mathbf h_j=\delta_{ij}$；式 (6.15) 方差 $\mathbf a_i^T\mathbf C\mathbf a_i$ | `ocr_page_131.txt` L3~14 |
| 式 (6.16)(6.17) 矢量 BLUE 与协方差；与一般线性模型 (4.25) MVU 形式相同 | `ocr_page_131.txt` L16~30 |
| 定理 6.1（高斯–马尔可夫）：式 (6.18) $x=H\theta+w$、式 (6.19) BLUE、式 (6.20) 最小方差、式 (6.21) 协方差；w 的 PDF 任意 | `ocr_page_131.txt` L31~38 + `ocr_page_132.txt` L1~11 |
| §6.6 引言：光干涉仪自相关函数非高斯、精确 PDF 难求 → 促使应用 BLUE | `ocr_page_132.txt` L13~21 |
| 例 6.3 源定位：式 (6.22) $t_i=T_0+R_i/c+\varepsilon_i$；式 (6.23) $R_i=\sqrt{(x_s-x_i)^2+(y_s-y_i)^2}$；噪声 PDF 不作假定 | `ocr_page_132.txt` L32~42 |
| 式 (6.24) 一阶泰勒 $R_i\approx R_{ni}+\cos\alpha_i\delta x_s+\sin\alpha_i\delta y_s$；$\cos\alpha_i=(x_n-x_i)/R_{ni}$、$\sin\alpha_i=(y_n-y_i)/R_{ni}$ | `ocr_page_133.txt` L23~43 |
| 式 (6.25) $\tau_i=T_0+(\cos\alpha_i/c)\delta x_s+(\sin\alpha_i/c)\delta y_s+\varepsilon_i$（$\tau_i=t_i-R_{ni}/c$）；避免时钟同步 → TDOA | `ocr_page_133.txt` L44~45 + `ocr_page_134.txt` L4~11 |
| 式 (6.26) TDOA 差分线性模型 | `ocr_page_134.txt` L12~19 |
| 噪声 $\mathbf w=\mathbf A\boldsymbol\varepsilon$（A 为 (N−1)×N 差分矩阵），$\mathbf C=\sigma^2\mathbf A\mathbf A^T$ | `ocr_page_134.txt` L31~39 |
| 式 (6.27) BLUE、式 (6.28) 最小方差、式 (6.29) 协方差 $\mathbf C_{\hat\theta}=\sigma^2[H^T(AA^T)^{-1}H]^{-1}$ | `ocr_page_135.txt` L4~14 |
| 三天线线阵（图 6.4）：H 碎片 "−cosα / 1−sinα / −cosα / −(1−sinα)"；协方差碎片 "2cos²α / 3/2 / (1−sinα)²"；"希望 α 小…加大天线间隔…基线加大…短距离或小 α 精度最好" | `ocr_page_135.txt` L15~30 |
| 习题 6.1~6.16（含 6.5 对数变换、6.6 仿射估计量、6.13 加权最小二乘、6.16 误用协方差） | `ocr_page_135.txt` L36~45 + `ocr_page_136~139.txt` |
| 附录 6A 标量 BLUE 推导（拉格朗日乘子 $J=\mathbf a^T\mathbf C\mathbf a+\lambda(\mathbf a^T\mathbf s-1)$，梯度 $2\mathbf C\mathbf a+\lambda\mathbf s$，$\mathbf a_{\mathrm{opt}}=\mathbf C^{-1}\mathbf s/(\mathbf s^T\mathbf C^{-1}\mathbf s)$，全局最小验证 $(\mathbf a-\mathbf a_{\mathrm{opt}})^T\mathbf C(\mathbf a-\mathbf a_{\mathrm{opt}})\ge0$） | `ocr_page_140.txt` L1~24 + `ocr_page_141.txt` |
| 附录 6B 矢量 BLUE 推导（逐分量拉格朗日，$\mathbf a_{i\mathrm{opt}}=\mathbf C^{-1}\mathbf H(\mathbf H^T\mathbf C^{-1}\mathbf H)^{-1}\mathbf e_i$） | `ocr_page_142.txt` + `ocr_page_143.txt` |
| 页码：书内 = PDF − 15 | `ocr_page_126.txt` L2（"111"）、`ocr_page_127.txt` L1（"112"）、`ocr_page_135.txt` L1（"120"）、`ocr_page_143.txt` L1（"128"），与 `全书目录整理.md` 第 6 章书内 110 起始一致 |

## 2. 据英文原版校订处

1. **例 6.3 三天线线阵的最终协方差矩阵（本扫描 OCR 仅存碎片，已全文推导校订）。** OCR 页 135 L17~30 残存 "−cosα / 1−sinα / −cosα / −(1−sinα)"（对应 H 矩阵）与 "2cos²α / 3/2 / (1−sinα)²"（对应最终协方差）。据 Kay 英文原版（Vol I, Example 6.3）重新推导：

   三天线相对标称源的方向角取 $\alpha$、$90^\circ$、$180^\circ-\alpha$（对称线阵），TDOA 差分矩阵 $\mathbf A=\begin{bmatrix}-1&1&0\\0&-1&1\end{bmatrix}$，信号矩阵
   $$
   \mathbf H=\frac1c\begin{bmatrix}-\cos\alpha & 1-\sin\alpha\\ -\cos\alpha & -(1-\sin\alpha)\end{bmatrix}
   $$
   与 OCR 碎片逐项吻合。$\mathbf A\mathbf A^T=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$，$(\mathbf A\mathbf A^T)^{-1}=\frac13\begin{bmatrix}2&1\\1&2\end{bmatrix}$。记 $a=\cos\alpha$、$b=1-\sin\alpha$，算得
   $$
   \mathbf H^T(\mathbf A\mathbf A^T)^{-1}\mathbf H=\frac1{c^2}\begin{bmatrix}2\cos^2\alpha & 0\\ 0 & \frac23(1-\sin\alpha)^2\end{bmatrix}
   $$
   代入 (6.29) 得 $\mathbf C_{\hat\theta}=\sigma^2[\mathbf H^T(\mathbf A\mathbf A^T)^{-1}\mathbf H]^{-1}=\sigma^2c^2\,\mathrm{diag}\!\bigl(\tfrac{1}{2\cos^2\alpha},\;\tfrac{3}{2(1-\sin\alpha)^2}\bigr)$，即
   $$
   \mathrm{var}(\widehat{\delta x_s})=\frac{\sigma^2c^2}{2\cos^2\alpha},\qquad \mathrm{var}(\widehat{\delta y_s})=\frac{3\sigma^2c^2}{2(1-\sin\alpha)^2}
   $$
   与 OCR 碎片 "2cos²α""3/2""(1−sinα)²" 完全对应。**定性结论**（"希望 α 小 → 加大天线间隔 → 基线加大；短距离或小 α 精度最好"）与 OCR 页 135 L28~30 逐字一致；$\alpha\to90^\circ$ 时两方差 $\to\infty$（天线向源前方收缩、基线失效），$\alpha\to0$ 时趋于有限最小值 $\sigma^2c^2/2$ 与 $3\sigma^2c^2/2$，为本系列补充推论的直接结果。

2. **例 5.8 的 OCR 残字**：OCR 页 126 L8 作 "max r[n] / 2N"，实为均匀噪声均值 MVU "$\frac12(\max_n x[n]+\min_n x[n])$"（非线性中点）的碎片（图 6.1(b) 另存 "max[r!]" 字样）。正文按第 5 章例 5.8 的标准结果表述为"最大值与最小值的中点"，未逐字转述残字，也未编造其方差数值。

3. **其余公式编号（6.1）~（6.29）、定理 6.1、例 6.1/6.2/6.3、图 6.1~6.4、习题 6.1~6.16 均以 OCR 为准**，未发现需进一步校订处。OCR 中 "高斯－马尔可夫" 因 OCR 分词写作 "高斯－马 尔可夫"（页 131 L31），正文统一作 "高斯–马尔可夫"。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig010 | `figures/Fig010_BLUE几何解释.png` | `Temp/scripts/make_fig010.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig010 内容（"BLUE 的约束与最优线性组合示意"）一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig010.py`（workdir=`Temp/scripts`）已见 `saved ...` 输出。
- 生成时修过：① 子图 (b) 的 "BLUE 权重 $a_{opt}\propto C^{-1}s$" 标注越出 Axes 右边界，改扩 xlim 至 (−1.95, 2.75) 并将标注改为两行、字号 9；② 椭圆角度原用 `V[:,0]`（小特征值特征向量，−45°），改回 `V[:,1]`（大特征值特征向量，45°），确保 1σ 椭圆长轴沿相关方向。修复后无 glyph 告警、无碰撞告警。
- 子图 (b) 数值（本系列推导，非随机）：$\mathbf C=\begin{bmatrix}1&0.6\\0.6&1\end{bmatrix}$（$\rho=0.6$），$\mathbf s=(1,0.4)$，$\mathbf a_{\mathrm{opt}}=\mathbf C^{-1}\mathbf s/(\mathbf s^T\mathbf C^{-1}\mathbf s)=(1.1176,-0.2941)$，$\mathrm{var}=1/(\mathbf s^T\mathbf C^{-1}\mathbf s)=0.9412$；脚本运行时打印值 `a_opt = [ 1.1176 -0.2941] var = 0.9412` 与手算一致。

## 4. 未决疑问

1. **三天线线阵的角度口径**：本文按"三天线相对标称源方向角为 $\alpha$、$90^\circ$、$180^\circ-\alpha$"推导，得到与 OCR 碎片吻合的对角协方差矩阵。原书图 6.4 的具体几何（三天线沿一线摆放、源在正侧方的具体夹角标注）在扫描件中不可辨，本文的 $\mathbf H$ 矩阵靠"碎片吻合 + 推导自洽"锁定，**建议终检时以英文原版图 6.4 再核一次角度定义**（最终方差形式 $1/(2\cos^2\alpha)$ 与 $3/(2(1-\sin\alpha)^2)$ 置信度高，角度 $\alpha$ 的具体指向置信度约 90%）。
2. **附录 6A/6B 的印刷页码缺失**：OCR 页 140（附录 6A）与页 142（附录 6B）扫描件均无印刷页码（按相邻页应为书内 125、127）。正文在"实现备忘"中已注明，未在正文引用这两页的页码。
3. **例 5.8 均匀噪声 MVU 的方差数值**：本章仅在 §2.2 引用"例 5.8 证明性能差别很大"（OCR 页 126 L11 "性能上的差别是很大的"），未引用具体方差数值（该数值属于第 5 章例 5.8，本扫描第 6 章 OCR 中无此数字），避免跨章编造。

## 5. 交叉引用一致性

- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① 第 7 篇 §1.1 "BLUE 路线：适用前提是噪声协方差矩阵已知…本题噪声方差就是待估参数 A…标准 BLUE 框架不适用" ✅ 本章 §7.1 已明确 "C 必须已知，C 依赖未知参数时 BLUE 不适用" 并回调例 7.1；② 第 7 篇引用 "第 6 章 BLUE" 在全书地图中的定位 ✅ 本章写在前头一致。
- 与第 1 篇 `Vol1_Ch01_引言.md` 对齐：第 1 篇 §4 地图把第 6 章定位为 "只知道噪声的一、二阶矩（不必是高斯）时，最优的线性估计" ✅ 本章口径一致。
- 与第 4 章对齐：本章 §4.2 "高斯线性模型 MVU 即 (HᵀC⁻¹H)⁻¹HᵀC⁻¹x，恰为线性 → BLUE=MVU" 与原书 (4.25) 及 OCR 页 131 L26~30 一致 ✅。
- 与第 8 章对齐：本章 §7.5 "矢量 BLUE 与加权最小二乘同形（习题 6.13）" 与 OCR 页 138 L19~21 一致，为第 8 篇最小二乘埋下伏笔 ✅。
