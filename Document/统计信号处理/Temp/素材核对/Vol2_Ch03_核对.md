# Vol2_Ch03 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch03_统计判决理论I.md`
> OCR 来源：`Temp/chapters_ocr/v2ch03/ocr_page_516~540.txt`（PDF 516~540，书内 501~525）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| 定理 3.1（式 3.3）似然比 $L(\mathbf{x})=p(\mathbf{x};H_1)/p(\mathbf{x};H_0)>\gamma$ | ocr_page_519.txt L41~50（"定理 3.1 (Neyman-Pearson)…L(x) = … (3.3)…门限由…求出"） | 一致 |
| 式 (3.2) $P_{FA}=\int_{R_1}p(\mathbf{x};H_0)d\mathbf{x}=\alpha$ | ocr_page_519.txt L3~5（"PFA = p(x; Ho)dx = (3.2)"） | 一致 |
| 例 3.1：$P_{FA}=10^{-3}\Rightarrow\gamma'=3,\ P_D=0.023$ | ocr_page_520.txt L7~36（"我们要求虚警概率为PFA=10-3…所以"=3…Pp=…=0.023"） | 一致 |
| 例 3.1：$P_{FA}=0.5\Rightarrow\gamma'=0$ | ocr_page_521.txt L4~6（"如果Pr=0.5…由此可得=0"） | 一致（$P_D$ 数值 OCR 残缺，文中未引用） |
| 例 3.2 式 (3.5)：$\bar{x}>\sigma^2\ln\gamma/(NA)+A/2$ | ocr_page_521.txt L44~48（"x[n] > in+ / NA …(3.5)"） | 一致（OCR 符号残缺，据上下文校订） |
| 式 (3.8)：$P_D=Q(Q^{-1}(P_{FA})-\sqrt{NA^2/\sigma^2})$ | ocr_page_522.txt L31~37 | 一致 |
| ENR = $NA^2/\sigma^2$（信号能量噪声比） | ocr_page_522.txt L38~39（"NA"/α2 称为信号能量噪声比 ENR"） | 一致 |
| 式 (3.9) 偏移系数 $d^2$ | ocr_page_524.txt L3~7 | 一致 |
| 式 (3.10) $P_D=Q(Q^{-1}(P_{FA})-\sqrt{d^2})$ | ocr_page_524.txt L14 | 一致 |
| ROC：45° 线 = 掷硬币；$d^2\to\infty$ 理想、$d^2\to0$ 回 45° 线；凹函数（习题 3.13） | ocr_page_527.txt L5~11、L33~35；ocr_page_536.txt L23~29 | 一致 |
| 式 (3.12) 错误概率 $P_e$ | ocr_page_529.txt L26~29 | 一致 |
| 式 (3.13) 门限 $P(H_0)/P(H_1)$；式 (3.14) ML | ocr_page_529.txt L35~45 | 一致 |
| 例 3.5 式 (3.15) $P_e=Q(\sqrt{NA^2/(4\sigma^2)})$ | ocr_page_530.txt L40~43（"错误概率随NA/…单调递减"附近） | 一致 |
| 式 (3.16) MAP | ocr_page_531.txt L5~8 | 一致 |
| 图 3.10 先验 $P(H_0)=P(H_1)=1/2$ vs $1/4$ vs $3/4$ | ocr_page_531.txt L10~24 | 一致 |
| 式 (3.17) 贝叶斯风险；式 (3.18) 门限 | ocr_page_531.txt L34~36；ocr_page_532.txt L5~9 | 一致 |
| 式 (3.21)(3.22)(3.24) 多元 MAP/ML | ocr_page_532.txt L25~40；ocr_page_533.txt L3~6 | 一致 |
| 例 3.6 判决 $\bar{x}<-A/2$ / 之间 / $>A/2$ | ocr_page_533.txt L47~50 | 一致 |
| 式 (3.25) $P_e=\tfrac43 Q(\sqrt{NA^2/(4\sigma^2)})$ | ocr_page_534.txt L17~32（OCR 残缺，见 §2 校订） | 校订 |
| 习题 3.20 $(2M-2)/M$ 因子 | ocr_page_537.txt L24~30（"2M --- 2 … NA2 … Pe"） | 校订 |
| 例 3.3 方差变化：统计量 $\sum x^2[n]$、$N=1$ 时 $\lvert x[0]\rvert>\gamma'$ | ocr_page_524.txt L17~35；ocr_page_525.txt L1~15 | 一致 |
| 例 3.4 非高斯无充分统计量（第 10 章） | ocr_page_525.txt L37~38；ocr_page_526.txt L32~33 | 一致 |
| 表 3.1 统计术语对照（显著性水平=虚警、势=检测率） | ocr_page_519.txt L20~40 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **式 (3.25)**：OCR 页 534 该式残缺（只剩"NA2 (3.25) 4g2"），据例 3.6 的对称性推导与习题 3.20 的 $(2M-2)/M$ 通式校订为 $P_e=\tfrac43 Q(\sqrt{NA^2/(4\sigma^2)})$。理由：$M=3$ 时 $(2M-2)/M=4/3$，且与正文"P_e 比二元假设检验时增加了"（二元为 $Q(\sqrt{NA^2/(4\sigma^2)})$）自洽。
2. **例 3.1 的 $P_{FA}=0.5$ 情形**：OCR 页 521 只给出了 $\gamma'=0$，$P_D$ 数值缺失，文中**未引用**该 $P_D$，宁可略去。
3. **式 (3.3) 的门限记号**：OCR 中门限 $\gamma$（似然比）与 $\gamma'$（统计量）在例 3.1 中混写，本文按原书逻辑区分二者（$\gamma'=\ln\gamma+\tfrac12$），并在"实现备忘"第 1 条说明。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig023 | `Documents/figures/Fig023_ROC曲线.png` | `Temp/scripts/make_fig023.py` | ✅ 通过（check_figure strict=True） |
| Fig024 | `Documents/figures/Fig024_NP准则示意.png` | `Temp/scripts/make_fig024.py` | ✅ 通过（check_figure strict=True） |

Fig023 曲线用公式（3.10）$P_D=Q(Q^{-1}(P_{FA})-\sqrt{d^2})$，$d^2\in\{0,0.5,1,4,9\}$；Fig024 用例 3.1 的 $\mathcal{N}(0,1)$ vs $\mathcal{N}(1,1)$，门限 $\gamma'=1/2$（$P_{FA}=Q(0.5)\approx0.31$）与 $\gamma'=3$（$P_{FA}=Q(3)\approx10^{-3}$，$P_D=Q(2)\approx0.023$）。两图中 $Q(0.5)$、$Q(2)$、$Q(3)$ 均为模型自算派生值，正文已标注。

## 4. 未决疑问

- 无实质性未决。例 3.1 中 $P_{FA}=0.5$ 时的 $P_D$ 数值（OCR 残缺）已按任务书"拿不准宁可略去"处理，未写入正文。
