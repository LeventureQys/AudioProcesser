# Vol1 Ch10 贝叶斯原理 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch10_贝叶斯原理.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch10/ocr_page_268~291.txt`（PDF 第 268~291 页，书内第 253~276 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §10.1 引言（经典"确定未知常量"→ 贝叶斯"随机变量现实"；两个出场理由） | `ocr_page_268.txt` L5~17 |
| §10.2 小结（(10.2)(10.5)(10.11)(10.14)、定理 10.2/10.3、(10.28) 均值第 11 章证明为矢量最小 MSE） | `ocr_page_268.txt` L18~27 |
| §10.3 截断样本均值 (10.1)、图 10.1（有偏但 MSE 更小） | `ocr_page_269.txt` L1~37 |
| 重建数据模型：先验 $U[-A_0,A_0]$、图 10.2 | `ocr_page_269.txt` L38~40 + `ocr_page_270.txt` L1~18 |
| 贝叶斯 MSE (10.2) $E[(A-\hat A)^2]$、误差 $A-\hat A$、期望对 $p(\mathbf x,A)$ | `ocr_page_270.txt` L5~18 |
| (10.3)(10.4) 经典 vs 贝叶斯 MSE；蒙特卡洛口径差别；"苹果和桔子" | `ocr_page_270.txt` L19~33 |
| (10.5) $\hat A=E(A\mid x)$（后验均值）；图 10.3 先验 vs 后验 | `ocr_page_271.txt` L1~34 |
| (10.6) 贝叶斯规则 $p(A\mid x)=p(x\mid A)p(A)/\int$ | `ocr_page_272.txt` L1~11 |
| (10.7)(10.8) 截断高斯后验 | `ocr_page_272.txt` L12~55 |
| (10.9) MMSE 无闭式；$A_0\gg\sigma^2/N$ 时近似 $\bar x$；图 10.4 数据淹没先验 | `ocr_page_273.txt` L1~31 |
| (10.10) 一般 MMSE $\hat\theta=E(\theta\mid x)=\int\theta p(\theta\mid x)d\theta$ | `ocr_page_273.txt` L33~42 |
| 先验选择警告（"除非先验建立在物理约束上，否则用经典"） | `ocr_page_274.txt` L1~6 |
| §10.4 例 10.1 高斯先验 $A\sim\mathcal N(\mu_A,\sigma_A^2)$ | `ocr_page_274.txt` L21~40 |
| (10.11) $\hat A=\alpha\bar x+(1-\alpha)\mu_A$，$\alpha$ 加权因子 $0<\alpha<1$ | `ocr_page_276.txt` L14~20 |
| (10.12) 后验方差随 $N$ 减小；图 10.5 | `ocr_page_276.txt` L24~31 |
| 无先验 $\sigma_A^2\to\infty$ → $\hat A\to\bar x$ | `ocr_page_277.txt` L6~7 |
| (10.13) $\mathrm{Bmse}=\int\mathrm{var}(A\mid x)p(x)dx$；(10.14) 且 $<\sigma^2/N$ | `ocr_page_277.txt` L8~24 |
| 高斯先验再生性 + 电压表比喻 | `ocr_page_277.txt` L25~33 |
| §10.5 (10.15) 二维高斯；图 10.6 椭圆等值线、横截面 | `ocr_page_278.txt` L9~61 |
| 定理 10.1 (10.16)(10.17) | `ocr_page_279.txt` L4~27 |
| (10.18) $\mathrm{var}(y\mid x)=\mathrm{var}(y)(1-\rho^2)$；(10.19) $\rho$ 定义 | `ocr_page_279.txt` L31~42 |
| (10.20)(10.21) MMSE 归一化形式 $\hat y_n=\rho x_n$；图 10.7 | `ocr_page_279.txt` L42~58 + `ocr_page_280.txt` L1~9 |
| (10.22) $\mathrm{Bmse}(\hat y)=\mathrm{var}(y)(1-\rho^2)$ | `ocr_page_280.txt` L10~15 |
| 定理 10.2 (10.24)(10.25) | `ocr_page_280.txt` L16~34 |
| §10.6 贝叶斯线性模型 (10.26)；$C_{xx}=HC_\theta H^T+C_w$、$C_{x\theta}=C_\theta H^T$ | `ocr_page_281.txt` L1~36 |
| 定理 10.3 (10.28)(10.29) | `ocr_page_281.txt` L37~39 + `ocr_page_282.txt` L1~13 |
| "与经典相比 H 不必满秩" | `ocr_page_282.txt` L13~14 |
| 例 10.2 Woodbury 恒等式 (10.30)(10.31) 序贯型 | `ocr_page_282.txt` L14~40 |
| (10.32)(10.33)(10.34) 替代形式、"信息"相加 | `ocr_page_283.txt` L24~35 |
| §10.7 多余参数 (10.35)(10.36)(10.37)(10.38) | `ocr_page_283.txt` L36~42 + `ocr_page_284.txt` L1~23 |
| 例 10.3 比例协方差矩阵、逆伽马先验 (10.39) | `ocr_page_284.txt` L24~46 + `ocr_page_285.txt` L1~9 |
| §10.8 确定性参数的贝叶斯估计；(10.40) mse 表达式；图 10.8 | `ocr_page_285.txt` L10~40 + `ocr_page_286.txt` L1~10 |
| 无信息先验 $\sigma_A^2\to\infty$；习题 10.15~10.17（熵/互信息） | `ocr_page_286.txt` L11~23 |
| 页码：书内 = PDF − 15 | `ocr_page_269.txt` L2（顶栏"254"）、`ocr_page_291.txt` L2（顶栏"276"），与目录"第 10 章起于 253"一致 |

## 2. 据英文原版校订处

1. **符号系统性校订**：OCR 把 $\theta$ 大量误识为 "6"、$\bar x$ 误识为 "元"、$\sigma$ 误识为 "g"、$\mu$ 误识为 "u" 等。全文按 Kay 英文原版（*Fundamentals of Statistical Signal Processing*, Vol I, Prentice Hall）校订符号，公式编号不变。
2. **式 (10.11) 的 $\alpha$**：OCR 页 276 L16~20 只有残字 "G?HA = α+ (1 -α)μA" 与 "0α<1"。据英文原版 $\alpha=\sigma_A^2/(\sigma_A^2+\sigma^2/N)$。正文另给出等价"精度加权"口径 $\hat A=\frac{(N/\sigma^2)\bar x+(1/\sigma_A^2)\mu_A}{N/\sigma^2+1/\sigma_A^2}$（推导补充，非原书公式）。
3. **式 (10.12)(10.14) 后验方差与最小 Bmse**：OCR 页 276 L25~26、页 277 L18~23 均为残字。据英文原版 $\mathrm{var}(A\mid x)=\sigma^2\sigma_A^2/(N\sigma_A^2+\sigma^2)=1/(1/\sigma_A^2+N/\sigma^2)$，$\mathrm{Bmse}(\hat A)$ 同值且 $<\sigma^2/N$。正文已核对该值与式 (10.40) 对 $A$ 求平均一致（$\alpha^2\sigma^2/N+(1-\alpha)^2\sigma_A^2=\sigma^2\sigma_A^2/(N\sigma_A^2+\sigma^2)$）。
4. **式 (10.40)**：OCR 页 285 L27~29 为 "mse(A) = α2var(a) + [αA +(1 - α)±μA - A)2"。据英文原版为 $\mathrm{mse}(\hat A)=\alpha^2(\sigma^2/N)+(1-\alpha)^2(\mu_A-A)^2$（其中 $b=(1-\alpha)(\mu_A-A)$）。
5. **例 10.3 的逆伽马先验 (10.39) 与积分结果**：OCR 页 284 L28~34、页 285 L6~9 严重残缺，正文只定性描述"逆伽马先验的特例"与"学生 t 型重尾形式"，未转录具体系数。见"未决疑问"。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig014 | `figures/Fig014_高斯先验与后验.png` | `Temp/scripts/make_fig014.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig014（"先验×似然→后验（高斯-高斯共轭）"）一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig014.py`（workdir=`Temp/scripts`）已见 `saved ...` 输出，且打印 `post_mean=0.5 post_var=0.0312 alpha(N=16)=0.5`，与正文 §3 的推导一致。
- Fig014 为自建数值实验（示意值 $\mu_A{=}0$、$\sigma_A^2{=}1/16$、$\sigma^2{=}1$、$N{=}16$、$\bar x{=}1$），与原书无冲突；三条曲线各自归一化到峰值 1（乘积关系在正文图注说明）。
- 复核方式说明：本会话模型不支持图像输入（`read_image` 报"model does not declare image input"），未做人工读图复核；已用 `plotutil.check_figure` 程序化碰撞检测通过（文本/图例越界与两两相交检测）。

## 4. 未决疑问

1. **例 10.3 的 (10.39) 先验与积出 $p(\mathbf x\mid\theta)$ 的完整形式**：OCR 页 284 L28~34（"我们指定2的先验PDF，>exp(-) α2 >0 … 逆伽马(ama)PDF的一个特殊情况"）与页 285 L6~9（"r(号+1) … (A +xTC-1(0)x)"）残缺。正文只保留定性结论（逆伽马先验 → 学生 t 型重尾 $p(\mathbf x\mid\theta)$），未转录具体系数与 $\Gamma$ 项。**建议终检以英文原版（Vol I, Example 10.3）复核一次**，不影响正文结论。

## 5. 交叉引用一致性

- 与第 1 篇 `Vol1_Ch01_引言.md` 对齐：① "道琼斯先验 [2800,3200]、$p(\mathbf x;\theta)$ vs $p(\mathbf x\mid\theta)$ 第 10 章展开"（第 1 篇 §2.3 明说）✅ 本章"写在前头"与 §1.1 兑现；② "估计量是随机变量、好坏用统计量" ✅ 本章 §2 的贝叶斯 MSE 是其贝叶斯版。
- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① "贝叶斯路线用先验压制野值（第 11 篇兑现）"（第 7 篇 §4.3/§10）✅ 本章 §6.2 埋下"先验改善精度"的种子，第 11 篇 §5.3 兑现；② MLE 与贝叶斯的分野（第 7 篇 §2.1）✅ 本章 §1 的分号/竖线记号线。
- 伏笔：本章 §5.1 埋"$H$ 不必满秩 → 参数数 $\ge$ 数据数"，第 11 篇 §5（维纳滤波 $H=I$）兑现。
