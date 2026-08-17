# Vol1 Ch09 矩方法 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch09_矩方法.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch09/ocr_page_253~267.txt`（PDF 第 253~267 页，书内第 238~252 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §9.1 定位（"容易确定和实现简单""不是最佳""数据记录足够长时有用""通常一致""可作 MLE Newton-Raphson 起始"） | `ocr_page_253.txt` L2~7 |
| §9.2 小结（(9.11) 一般形式、(9.15)(9.16) 近似均值方差、(9.18)(9.19) 高 SNR） | `ocr_page_253.txt` L8~13 |
| §9.3 高斯混合 PDF $p(x;\varepsilon)=(1-\varepsilon)\phi_1+\varepsilon\phi_2$；(9.1) $E[x^2]=(1-\varepsilon)\sigma_1^2+\varepsilon\sigma_2^2$；(9.3) $\hat\varepsilon=(\frac1N\sum x^2-\sigma_1^2)/(\sigma_2^2-\sigma_1^2)$；(9.4) 方差；一致性（习题 7.5） | `ocr_page_253.txt` L14~29 + `ocr_page_254.txt` L1~38 |
| 标量总结：(9.5) $\mu_k=h(\theta)$；(9.6) 样本矩；(9.7) $\hat\theta=h^{-1}(\hat\mu_k)$ | `ocr_page_254.txt` L38~39 + `ocr_page_255.txt` L1~13 |
| 例 9.1 WGN 中 DC 电平：$\mu_1=A$，$\hat A=$ 样本均值，$h$ 恒等变换 | `ocr_page_255.txt` L14~22 |
| 例 9.2 指数 PDF：$\mu_1=1/\lambda$，$\hat\lambda=1/$ 样本均值 | `ocr_page_255.txt` L23~38 |
| §9.4 矢量：(9.8)~(9.11) $\hat\theta=h^{-1}(\hat\mu)$；前 $p$ 阶矩可能不足；尽量用最低阶矩（方差随阶数增大）；需互相关矩 | `ocr_page_256.txt` L1~31 |
| 例 9.3 高斯混合三参数：偶数阶矩 $E[x^2],E[x^4]=3(1-\varepsilon)\sigma_1^4+3\varepsilon\sigma_2^4,E[x^6]=15(1-\varepsilon)\sigma_1^6+15\varepsilon\sigma_2^6$；$\nu=\sigma_1^2\sigma_2^2$ 巧解（Rider 1961） | `ocr_page_256.txt` L32~36 + `ocr_page_257.txt` L1~16 |
| §9.5 统计评价：(9.13) $\hat\theta=g(T)$；(9.14) 一阶泰勒；(9.15) $E[\hat\theta]=g(\mu)$；(9.16) 近似方差（$\mathbf C_T$ 协方差） | `ocr_page_257.txt` L17~31 + `ocr_page_258.txt` L1~27 |
| 例 9.4 指数 PDF（继续）：$\hat\lambda$ 也是 MLE，渐近 $\mathcal N(\lambda,\lambda^2/N)$（(7.8) 式） | `ocr_page_258.txt` L28~38 + `ocr_page_259.txt` L1~34 |
| 泰勒前提：数据记录足够大 / 高 SNR；$N$ 多大要蒙特卡洛确定 | `ocr_page_259.txt` L36~41 |
| 高 SNR：(9.17) 展开；(9.18) $E[\hat\theta]=h(0)=g(s(\theta))=\theta$；(9.19) 近似方差 | `ocr_page_260.txt` L1~28 |
| 例 9.5 白噪声中指数信号：$x[n]=r^n+w[n]$，估计量 $\hat r=\frac{x[1]+x[2]}{x[0]+x[1]}$（无噪声时 $=\frac{r+r^2}{1+r}=r$） | `ocr_page_260.txt` L29~49 + `ocr_page_261.txt` L1~19 |
| §9.6 频率估计：相位随机 $\phi\sim\mathcal U[0,2\pi]$ 使信号 WSS；ACF $r_{ss}[k]=\frac{A^2}{2}\cos 2\pi f_0 k$；$A=\sqrt2$ 时 $r_{xx}[1]=\cos 2\pi f_0$；(9.20) $\hat f_0=\frac1{2\pi}\arccos(\hat r_{xx}[1])$ | `ocr_page_261.txt` L20~43 + `ocr_page_262.txt` L1~14 |
| 低 SNR 时 $\arccos$ 参数超 1 → 估计无意义 | `ocr_page_262.txt` L14~17 |
| (9.21) 近似均值 $E[\hat f_0]\approx f_0$；(9.22) 近似方差 $\propto 1/N^2$（CRLB $\propto 1/N^3$，(3.41) 式）；$f_0$ 贴 0 或 1/2 方差暴增（图 9.1 $\arccos$ 斜率） | `ocr_page_262.txt` L18~41 + `ocr_page_263.txt` L1~44 |
| 图 9.2 蒙特卡洛：$A=\sqrt2$、SNR$=A^2/2\sigma^2=1/\sigma^2$、$f_0=0.2$、$\phi=0$、$N=50$、1000 现实 | `ocr_page_263.txt` L43~44 + `ocr_page_264.txt` L1~9 |
| 习题 9.5（ARMA 的 AR 参数矩方法 = Yule-Walker）；习题 9.11（确定相位仍一致）；习题 9.12（幅度未知推广） | `ocr_page_265.txt` L23~27 + `ocr_page_266.txt` L23~36 |
| 页码：书内 = PDF − 15 | `ocr_page_254.txt` L2（顶栏"239"）、`ocr_page_267.txt` L2（顶栏"252"），与目录"第 9 章起于 238"一致 |

## 2. 据英文原版校订处

1. **式 (9.20) 的归一化 $1/(N-1)$**：OCR 页 262 L9~13 被认成"对于=1,ACF 的一个合理的估计量是 N-2 … 2r(n]r[n + 1]"。据 Kay 英文原版（Vol I, §9.6）为 $\hat r_{xx}[1]=\frac{1}{N-1}\sum_{n=0}^{N-2}x[n]x[n+1]$。理由：无噪声且 $A=\sqrt2$ 时 $\frac{1}{N-1}\sum s[n]s[n+1]\to\cos 2\pi f_0$（习题 9.11 同款收敛式），归一化 $1/(N-1)$ 使 (9.20) 的 $\arccos$ 自变量正确落在 $[-1,1]$。
2. **例 9.5 的估计量 $\hat r=(x[1]+x[2])/(x[0]+x[1])$**：OCR 页 260 L36~38 被认成"r2 +w[2] +r +w[1] / r +w[1] + 1 + w[0]"。据 Kay 英文原版（Vol I, Example 9.5）应为 $\hat r=\frac{x[1]+x[2]}{x[0]+x[1]}$（对应 $x[n]=r^n+w[n]$，无噪声时 $=\frac{r+r^2}{1+r}=r$）。理由：分子是"两个相邻样本之和"、分母是"前两个样本之和"，回代无噪声信号得 $r$，与 (9.18) 的 $h(0)=g(s(\theta))=\theta$ 自洽。
3. 其余公式编号（9.1）~（9.22）、例 9.1~9.5、图 9.1/9.2 均以 OCR 为准，未发现需校订处。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig013 | `figures/Fig013_矩方法示意.png` | `Temp/scripts/make_fig013.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig013（"样本矩匹配理论矩求参数"）一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig013.py`（workdir=`Temp/scripts`）已见 `saved ...` 输出。
- Fig013 右图为自建数值实验（种子 20260816，真值 $\varepsilon=0.3$、$N=2000$、$\sigma_1^2=1$、$\sigma_2^2=4$），实测 $\hat\mu_2\approx1.97$、$\hat\varepsilon\approx0.323$（理论 $\mu_2=1.9$，与真值 0.3 吻合）。

## 4. 未决疑问

1. **式 (9.22) 的完整求和表达式**：OCR 页 263 L22~31 被认成"s?[1] + 4 cos? 2π fn s?[n] + s′[N -- 2]"等残字。正文给出简化结构 $\frac{\sigma^2}{(2\pi)^2(N-1)^2\sin^2 2\pi f_0}[s^2[0]+4\cos^2 2\pi f_0\sum_{n=1}^{N-2}s^2[n]+s^2[N-2]]$。**建议终检以英文原版（Vol I, §9.6）核对一次具体系数与求和上下限**（本次定性结论"方差 $\propto1/N^2$、贴边暴增"可靠，系数属例行复核）。
2. **例 9.3 的 $\nu=\sigma_1^2\sigma_2^2$ 求解细节**：OCR 页 257 L6~15 只认出"μ6 - 5μ4μ2 / 5μ4 - 15μ2"等残字。正文只保留结论（Rider 1961 巧解），未展开"解 $u^2-\sqrt{u^2-4\nu}$ 形式"的具体代数。**建议以英文原版（Vol I, Example 9.3）复核该非线性方程组的解法**（不影响正文结论）。

## 5. 交叉引用一致性

- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① "例 7.2 的矩估计雏形（第 9 章）"（第 7 篇 §1.2 明说）✅ 本章 §1.3 兑现（$\hat A=\sqrt{\hat u+\tfrac14}-\tfrac12$ 是矩方法特例）；② "例 7.2 的一致性逻辑" ✅ 本章 §3.1（大数定律 + 连续映射）；③ AR 参数渐近 MLE ≈ Yule-Walker（第 7 篇例 7.18）✅ 本章 §5.3 三家会师。
- 与第 8 篇 `Vol1_Ch08_最小二乘估计.md` 对齐：① 例 8.12 的修正 Yule-Walker（LS 解）✅ 本章 §5.3 与矩方法视角合流；② "无概率假设"的谱系（LS → 矩方法）✅ 本章"写在前头"。
- 伏笔：矩方法"可作 MLE 的 Newton-Raphson 起始估计"（§9.1）✅ 本章"写在前头"埋下，接第 7 篇 §5.3 数值迭代。
