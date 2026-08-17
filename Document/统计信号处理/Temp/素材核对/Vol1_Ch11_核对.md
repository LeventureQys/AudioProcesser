# Vol1 Ch11 一般贝叶斯估计量 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch11_一般贝叶斯估计量.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch11/ocr_page_292~320.txt`（PDF 第 292~320 页，书内第 277~305 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §11.1 引言（贝叶斯风险、MMSE/MAP、误差椭圆、解卷积/维纳） | `ocr_page_292.txt` L3~11 |
| §11.2 小结（(11.1) 风险、(11.2) 绝对误差→中值、(11.3) 成功-失败→众数、(11.10)(11.12)(11.23)(11.24)(11.26)(11.29)(11.30)、定理 11.1） | `ocr_page_292.txt` L12~24 |
| §11.3 风险函数 (11.1) $R=E[C(\varepsilon)]$、$\varepsilon=\hat\theta-\theta$ | `ocr_page_292.txt` L25~32 |
| 图 11.1 三种代价：二次型、(11.2) 绝对误差、(11.3) 成功-失败 | `ocr_page_293.txt` L1~21 |
| 绝对误差 → 后验中值（Leibnitz 求导）；成功-失败 → 后验众数（MAP） | `ocr_page_293.txt` L22~33 + `ocr_page_294.txt` L1~36 |
| 图 11.2（一般后验三者不同；高斯后验三者重合） | `ocr_page_294.txt` L37~40 + `ocr_page_295.txt` L1~13 |
| §11.4 (11.5) 边缘后验、(11.7)(11.10) 矢量 MMSE | `ocr_page_295.txt` L14~41 + `ocr_page_296.txt` L1~20 |
| (11.12)(11.13) 最小贝叶斯 MSE = 后验协方差对角元平均 | `ocr_page_296.txt` L21~41 |
| 例 11.1 贝叶斯傅里叶分析（$M{=}1$、例 4.2、瑞利衰落、先验 $\mathcal N(0,\sigma_A^2I)$） | `ocr_page_297.txt` L1~32 |
| MMSE 与经典只差比例因子；$\sigma_A^2\gg 2\sigma^2/N$ 时相同；后验协方差与 x 无关 | `ocr_page_297.txt` L33~41 + `ocr_page_298.txt` L1~15 |
| (11.14) 无先验 → 经典 MVU | `ocr_page_298.txt` L16~23 |
| (11.15)(11.16) 仿射变换可交换；(11.17) 独立数据集叠加 | `ocr_page_298.txt` L24~40 + `ocr_page_299.txt` L1~20 |
| §11.5 (11.18)(11.19) MAP 定义 | `ocr_page_299.txt` L21~36 |
| 例 11.2 指数 PDF：MAP $\hat\theta=N/(\sum x[n]+\lambda)$；$\lambda\to0$ → $1/\bar x$；图 11.3 | `ocr_page_300.txt` L1~35 |
| 例 11.3 均匀先验：MAP 为截断 (11.20)；图 11.4；MAP 无积分只求最大 | `ocr_page_301.txt` L1~32 |
| 矢量 MAP (11.21)(11.22) 标量边缘 vs (11.23)(11.24) 联合峰；图 11.5 | `ocr_page_302.txt` L1~41 |
| 例 11.4 未知方差（共轭先验 $p(A\mid\sigma^2)=\mathcal N(\mu_A,\sigma^2)$、逆伽马 $p(\sigma^2)$） | `ocr_page_303.txt` L11~43 + `ocr_page_304.txt` L1~44 |
| 例 11.4 结果 $N\to\infty$ 退化为贝叶斯 MLE | `ocr_page_305.txt` L1~24 |
| 例 11.5 变换 $\beta=1/\theta$：MAP 非线性变换不可互换、先验带雅可比 | `ocr_page_305.txt` L25~43 + `ocr_page_306.txt` L1~34 |
| §11.6 (11.26) $\varepsilon\sim\mathcal N(0,\mathrm{Bmse}(\theta))$；图 11.6 | `ocr_page_306.txt` L35~48 + `ocr_page_307.txt` L1~13 |
| 例 11.6 DC 电平高斯先验：误差高斯、$N\to\infty$ 一致 | `ocr_page_307.txt` L14~22 |
| 矢量误差协方差 (11.27)~(11.30)；$\mathbf M_\theta$ 对角元 = 最小 Bmse | `ocr_page_307.txt` L23~26 + `ocr_page_308.txt` L1~37 |
| 例 11.7 误差椭圆 (11.31)、$\chi^2_2$、$c^2=-2\ln(1-P)$、图 11.7 | `ocr_page_308.txt` L37~41 + `ocr_page_309.txt` L1~32 + `ocr_page_310.txt` L1~12 |
| 定理 11.1 (11.32)~(11.36) | `ocr_page_310.txt` L13~29 |
| §11.7 解卷积 (11.37)、图 11.8、盲解卷积"相当难" | `ocr_page_310.txt` L30~37 + `ocr_page_311.txt` L1~32 |
| (11.38) 离散化 $x=Hs+w$、$H$ 下三角；(11.39) MMSE | `ocr_page_312.txt` L1~47 |
| (11.40) 维纳滤波器 $\hat s=C_s(C_s+\sigma^2I)^{-1}x$；标量收缩 $\eta/(\eta+1)$ | `ocr_page_313.txt` L1~16 |
| AR(1) 例子、图 11.9 PSD、图 11.10 滤波前后、低通 | `ocr_page_313.txt` L16~30 + `ocr_page_314.txt` L1~41 |
| 页码：书内 = PDF − 15 | `ocr_page_293.txt` L2（顶栏"278"）、`ocr_page_320.txt` L2（顶栏"305"），与目录"第 11 章起于 277"一致 |

## 2. 据英文原版校订处

1. **符号系统性校订**：OCR 把 $\theta$ 大量误识为 "6"、$\bar x$ 误识为 "元"、$\sigma$ 误识为 "g"、"众数/中值"偶有错位。全文按 Kay 英文原版（Vol I）校订符号，公式编号不变。
2. **例 11.4 的 $\sigma^2$ MAP 公式**：OCR 页 304 L29~44、页 305 L1~10 严重残缺（"N+5 / N+5 / +2X"等残字）。据英文原版，先验 $p(\sigma^2)\propto(\sigma^2)^{-2}e^{-\lambda/\sigma^2}$（逆伽马形状参数 $\alpha{=}1$），得 $\hat\sigma^2=\bigl[\sum(x[n]-\hat A)^2+(\hat A-\mu_A)^2+2\lambda\bigr]/(N+5)$。理由：分母 $N{+}5=N{+}3{+}2\alpha$（$\alpha{=}1$），分子 $2\lambda$ 来自 $\lambda/\sigma^2$ 项求导。$\hat A=(N\bar x+\mu_A)/(N+1)$ 由 $\sigma_A^2=\sigma^2$ 时最小化 $Q(A)=(A-\mu_A)^2+\sum(x[n]-A)^2$ 直接得出，与 OCR 页 304 L3~8（"MA!I / +N"）一致。
3. **例 11.7 误差圆半径 0.83/1.52/2.15**：OCR 页 310 L5~8 显示 "P = 0.5 / 0.9 / 0.99 / 0.83/1.52/2.15"。据英文原版，这组数是误差圆的**半径（半轴）**而非 $c$ 本身（因 $\mathbf M_\theta=\tfrac12 I$，半径 $=c/\sqrt2$）。正文已明确标注"半径（半轴）"。
4. **例 11.2 的 MAP**：OCR 页 300 L22~27 为残字 "NE->"。据英文原版 $g(\theta)=N\ln\theta-\theta\sum x[n]+\ln\lambda-\lambda\theta$，求导得 $\hat\theta=N/(\sum x[n]+\lambda)$。
5. **例 11.5 的 MAP**：OCR 页 306 L31~32 为残字 "N+2 / Y+IN"。据英文原版 $\hat\beta=(\sum x[n]+\lambda)/(N+2)$（先验变换 $p_\beta(\beta)=p_\theta(1/\beta)/\beta^2$，指数 $-(\sum x[n]+\lambda)/\beta-(N+2)\ln\beta$）。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig015 | `figures/Fig015_MMSE与MAP对比.png` | `Temp/scripts/make_fig015.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig015（"同一后验下 MMSE（条件均值）与 MAP（峰）对比"）一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig015.py`（workdir=`Temp/scripts`）已见 `saved ...` 输出，且打印 `mode=0.527 median=1.0 mean=1.377`，与对数正态闭式值（众数 $e^{-\sigma^2}$、中值 $e^{\mu}$、均值 $e^{\mu+\sigma^2/2}$，$\mu{=}0,\sigma{=}0.8$）一致。
- Fig015 为自建数值实验（对数正态 $\mu{=}0,\sigma{=}0.8$；高斯 $\mathcal N(1,0.04)$），与原书无冲突。
- 复核方式说明：本会话模型不支持图像输入（`read_image` 报"model does not declare image input"），未做人工读图复核；已用 `plotutil.check_figure` 程序化碰撞检测通过。

## 4. 未决疑问

1. **例 11.4 先验 $p(\sigma^2)$ 的精确形状参数**：OCR 页 303 L24~30 只认出 "$p(A,\sigma^2)=p(A\mid\sigma^2)p(\sigma^2)$ … 逆伽马 PDF 的一个特例"，无法从 OCR 直接确认形状参数 $\alpha{=}1$。正文给出的 $\hat\sigma^2$（分母 $N{+}5$）依赖 $\alpha{=}1$ 这一取值，已注明"据英文原版校订"。**建议终检以英文原版（Vol I, Example 11.4）复核先验形式与最终公式一次**。
2. **例 11.3 图 11.4 的三子图**：OCR 页 301 L34~45 排版混乱（(a)(b)(c) 与 $\bar x$ 区间对应关系需仔细辨认）。正文只取结论 (11.20) 的分段函数（$\bar x<-A_0\to-A_0$、$|\bar x|\le A_0\to\bar x$、$\bar x>A_0\to A_0$），未转录图 11.4 的图形细节。**建议终检核对图 11.4 的三个子图与 (11.20) 三段的一一对应**。

## 5. 交叉引用一致性

- 与第 10 篇 `Vol1_Ch10_贝叶斯原理.md` 对齐：① 后验 PDF、贝叶斯线性模型（定理 10.3、(10.28)~(10.33)）✅ 本章 §2.2/§4.2 直接复用；② "§6.2 多余参数积分消除" ✅ 本章 §2.1 的矢量 MMSE（其余分量当多余参数）；③ "$H$ 不必满秩 → 参数数 $\ge$ 数据数"（第 10 篇 §5.1 埋）✅ 本章 §5.2（$H=I$，维纳滤波）兑现。
- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① "低 SNR 下 MLE 被野值带跑，贝叶斯用先验压制野值（第 11 篇兑现）"（第 7 篇 §4.3/§10）✅ 本章 §5.3 兑现（收缩因子 $\eta/(\eta+1)$ 压低噪声尖峰，代价是信号也被平滑）；② "MLE 的不变性（定理 7.2）" ✅ 本章 §3.3 与 MAP 的非不变性对照；③ "MAP = MLE + 先验修正项" ✅ 本章 §3.1。
- 伏笔：本章 §5.2 埋"维纳滤波器完整性质第 12 章"，指向 `Vol1_Ch12_线性贝叶斯估计量.md`（真实存在）。
