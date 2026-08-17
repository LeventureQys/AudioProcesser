# Vol1 Ch04 线性模型 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch04_线性模型.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch04/ocr_page_085~099.txt`（PDF 第 85~99 页，书内第 70~84 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §4.1 定位（"MVU 估计量的确定一般来说是一项很困难的任务…线性模型…"） | `ocr_page_085.txt` L5~9 |
| §4.2 小结：线性模型由（4.8）定义、MVU（4.9）、协方差（4.10）、一般线性模型（4.25）（4.26）、已知信号分量（4.31）（4.32） | `ocr_page_085.txt` L10~16 |
| 直线拟合例 3.7 延续，矩阵形式 x=Hθ+w（4.1），H 为 N×2 观测矩阵 | `ocr_page_085.txt` L18~27 |
| 噪声 W~N(0,σ²I) 称线性模型；达界条件（4.2） | `ocr_page_086.txt` L5~13 |
| 推导：∂ln p/∂θ=(Hᵀx−HᵀHθ)/σ²（用对称矩阵恒等式 ∂θᵀAθ/∂θ=2Aθ） | `ocr_page_086.txt` L14~31 |
| θ̂=(HᵀH)⁻¹Hᵀx（4.5）、I(θ)=HᵀH/σ²（4.6）、C=σ²(HᵀH)⁻¹（4.7）；MVU 有效、达 CRLB | `ocr_page_086.txt` L32~45 |
| H 列线性独立 ⟺ HᵀH 可逆（习题 4.2）；列相关 → 参数不可识别（图 4.1，x[n]=A+B） | `ocr_page_087.txt` L4~22 |
| 定理 4.1：x=Hθ+w（4.8）、θ̂=(HᵀH)⁻¹Hᵀx（4.9）、C=σ²(HᵀH)⁻¹（4.10） | `ocr_page_087.txt` L23~33 |
| θ̂~N(θ, σ²(HᵀH)⁻¹)（4.11）高斯 PDF，性能完全确定 | `ocr_page_087.txt` L34~35 + `ocr_page_088.txt` L3~4 |
| 例 4.1 曲线拟合：二次 x(t_n)=θ₁+θ₂t_n+θ₃t_n²+w，Vandermonde 矩阵；（p−1）阶多项式一般式 | `ocr_page_088.txt` L11~38 + `ocr_page_089.txt` L1~17 |
| 例 4.2 傅里叶分析：（4.12）正余弦谐波、f_k=k/N、p=2M、M<N/2、H 列正交（DFT，习题 4.5） | `ocr_page_089.txt` L18~47 |
| 例 4.2 正交性 → HᵀH 对角 → â_k,b̂_k 估计（4.14）、均值（4.15）、协方差（4.16）对角、各分量独立 | `ocr_page_090.txt` L1~54 + `ocr_page_091.txt` L1~11 |
| 例 4.3 系统辨识：TDL/FIR（4.17）、H 为 Toeplitz [u[i−j]]（4.18）、[HᵀH]_ij（4.19）、大 N 自相关（4.20） | `ocr_page_091.txt` L12~35 + `ocr_page_092.txt` L1~14 |
| 例 4.3 PRN 使 HᵀH≈N r_uu[0] I 对角 → 各权重独立（4.21）、ĥ_i（4.22）互相关、近似 MVU（4.23） | `ocr_page_092.txt` L15~43 + `ocr_page_093.txt` L1~50 + `ocr_page_094.txt` L1~14 |
| 一般线性模型：w~N(0,C)、白化 C⁻¹=DᵀD（4.24） | `ocr_page_094.txt` L17~33 |
| θ̂=(HᵀC⁻¹H)⁻¹HᵀC⁻¹x（4.25）、C=(HᵀC⁻¹H)⁻¹（4.26） | `ocr_page_094.txt` L34~41 + `ocr_page_095.txt` L1~6 |
| 例 4.4 色噪声 DC 电平：Â=1ᵀC⁻¹x/(1ᵀC⁻¹1)、var=1/(1ᵀC⁻¹1)、预白化解释（4.27） | `ocr_page_095.txt` L8~32 |
| 已知信号分量 x=Hθ+s+w：θ̂=(HᵀH)⁻¹Hᵀ(x−s)（4.28）、C=σ²(HᵀH)⁻¹（4.29） | `ocr_page_095.txt` L33~40 + `ocr_page_096.txt` L1~3 |
| 例 4.5 白噪声 DC 电平+指数信号：s=[1 r … r^{N−1}]、Â=(1/N)Σ(x[n]−r^n)、var=σ²/N | `ocr_page_096.txt` L5~13 |
| 定理 4.2：x=Hθ+s+w（4.30）、θ̂=(HᵀC⁻¹H)⁻¹HᵀC⁻¹(x−s)（4.31）、C=(HᵀC⁻¹H)⁻¹（4.32）、有效达 CRLB | `ocr_page_096.txt` L16~29 |
| 习题 4.2（列独立 ⟺ 可逆）、4.3（病态 H=[1 1;1 1+ε]）、4.13（随机 H，协方差 σ²E[(HᵀH)⁻¹]）、4.14（衰落） | `ocr_page_097.txt` L8~18、L25~32、L33~41 + `ocr_page_098.txt` L33~41 + `ocr_page_099.txt` L1~8 |
| 页码：书内 = PDF − 15 | `ocr_page_086.txt` L2（顶栏"71"）、`ocr_page_099.txt` L2（顶栏"84"），与 `全书目录整理.md` 第 4 章书内 70 起始一致 |

## 2. 据英文原版校订处

1. **式（4.1）~（4.3）的矩阵/梯度**：OCR 页 85 L23~27 为"M + H = X""[α[0] r[1] . . .]""[A B]′"等残字，页 86 L14~31 的梯度被认成"[8HH+ H× -*x] / 202""ObTe / 0TAG / 2A0"等。据 Kay 英文原版（Vol I, §4.3）应为 $\mathbf{x}=\mathbf{H}\boldsymbol\theta+\mathbf{w}$ 与 $\partial\ln p/\partial\boldsymbol\theta=(\mathbf{H}^T\mathbf{x}-\mathbf{H}^T\mathbf{H}\boldsymbol\theta)/\sigma^2$。理由：矩阵求导恒等式 $\partial(\boldsymbol\theta^T\mathbf{A}\boldsymbol\theta)/\partial\boldsymbol\theta=2\mathbf{A}\boldsymbol\theta$（$\mathbf{A}$ 对称）代入对数似然即可，且与达界条件（4.2）自洽。
2. **例 4.2 的 $\mathbf{H}^T\mathbf{H}=(N/2)\mathbf{I}$ 与 $\mathbf{C}_{\hat{\boldsymbol\theta}}=(2\sigma^2/N)\mathbf{I}$**：OCR 页 90 L33~37 为"120 / N-2 / HTH / ++."、页 91 L4~7 为"C α(HTH)-I / (学) / (4.16)"。据 Kay 英文原版应为 $\mathbf{H}^T\mathbf{H}=(N/2)\mathbf{I}_{2M}$、$\mathbf{C}_{\hat{\boldsymbol\theta}}=(2\sigma^2/N)\mathbf{I}_{2M}$。理由：① 式（4.14）估计量的 $2/N$ 因子（OCR 页 90 L38~45 的"2-N … [n] cos"）正是 $\hat{\boldsymbol\theta}=(2/N)\mathbf{H}^T\mathbf{x}$，这要求 $\mathbf{H}^T\mathbf{H}=(N/2)\mathbf{I}$；② "协方差对角、幅度估计独立"的定性结论（OCR 页 91 L8）与对角协方差一致；③ DFT 正交性（$\sum\cos^2(2\pi kn/N)=N/2$，$0<k<N/2$，习题 4.5）。
3. **式（4.25）（4.31）的矩阵**：OCR 页 94 L38 为"(H"" H)-H"x -()"、页 96 L22 为"(s -- X)r-DrHt-(Hr-O1H)"。据 Kay 英文原版应为 $\hat{\boldsymbol\theta}=(\mathbf{H}^T\mathbf{C}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{C}^{-1}\mathbf{x}$ 与 $\hat{\boldsymbol\theta}=(\mathbf{H}^T\mathbf{C}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{C}^{-1}(\mathbf{x}-\mathbf{s})$。理由：白化 $\mathbf{C}^{-1}=\mathbf{D}^T\mathbf{D}$ 代入定理 4.1 的结果 $(\mathbf{H}'^T\mathbf{H}')^{-1}\mathbf{H}'^T\mathbf{x}'$（$\mathbf{H}'=\mathbf{D}\mathbf{H}$、$\mathbf{x}'=\mathbf{D}\mathbf{x}$），回代 $\mathbf{D}^T\mathbf{D}=\mathbf{C}^{-1}$ 即得。
4. 其余公式编号（4.1）~（4.32）、定理 4.1/4.2、例 4.1~4.5、图 4.1~4.3、习题 4.1~4.14 均以 OCR 为准，未发现需校订处。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig008 | `figures/Fig008_线性模型数据流.png` | `Temp/scripts/make_fig008.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig008 内容（"线性模型 x=Hθ+w 的结构与例子"）一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig008.py`（workdir=`Temp/scripts`）已见 `saved ...` 输出。
- 生成时修过：① 初版用 `annotate("", ...)` 画箭头，空字符串标注产生空包围盒与框内文字"误撞"，改为 `FancyArrowPatch`（patch 不产生文本包围盒）；② 右图"矩阵 H"标签与列头"1/n"重叠，上移标签；③ 结论行文字过宽、左缘越出坐标轴，拆成两行并居中。修复后无碰撞告警。
- Fig008 为静态结构图，无随机数（无需种子）。右图 H 矩阵仅示意 N=6 行，实际 N 任意（N>p=2）。

## 4. 未决疑问

1. **例 4.2 的 $N/2$ 常数（置信度约 90%）**：$\mathbf{H}^T\mathbf{H}=(N/2)\mathbf{I}$ 由式（4.14）的 $2/N$ 因子 + DFT 正交性 + "协方差对角"定性结论共同推断，自洽且可靠；但 OCR 页 90 的直接证据残断。**建议终检时以英文原版（Kay Vol I, Example 4.2）再核一次常数因子 $N/2$**（本次依据充分，属例行复核）。
2. **例 4.3 的 Cauchy-Schwarz 论证细节**：OCR 页 92 L19~43 的推导被认成"eTdTT-'e / 5T52 / 2=1 / (5T52)3≤TES52"等残字。正文只保留结论（"使方差最小的充要条件是 HᵀH 对角"）与 PRN 结论，未展开推导，不影响结论正确性；推导细节建议以英文原版为准。

## 5. 交叉引用一致性

- 与第 3 篇 `Vol1_Ch03_CramerRao下限.md` 对齐：① "例 3.7 直线拟合的达界套路在第 4 章系统化"（第 3 篇 §7.4 伏笔）✅ 本章 §1.2 兑现；② "例 3.3 样本均值方差 σ²/N" ✅ 本章例 4.4 白噪声退化一致；③ "例 3.6 的 CRLB 2σ⁴/N" 将在第 5 篇例 5.11 回调 ✅（本章 §9 提及）。
- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① "定理 7.5（式 7.46）θ̂=(HᵀC⁻¹H)⁻¹HᵀC⁻¹x 正是第 4 章证明过的 MVU"（第 7 篇 §6.4）✅ 本章 §3.3、§6.5 作为引用所指正文一致；② "与第 8 章最小二乘加权形式同形"（第 7 篇 §6.4 顺带一提）✅ 本章 §3.3 呼应。
- 与第 6 篇（BLUE，未写）对齐：本章 §9.2 预告"只知一、二阶矩时 BLUE 接管，高斯下与本章解重合"，待第 6 篇兑现。
