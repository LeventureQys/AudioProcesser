# Vol1 Ch08 最小二乘估计 —— 素材核对清单

> 交付文档：`Document/统计信号处理/Documents/Vol1_Ch08_最小二乘估计.md`
> OCR 素材：`Document/统计信号处理/Temp/chapters_ocr/ch08/ocr_page_197~252.txt`（PDF 第 197~252 页，书内第 182~237 页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | OCR 出处（文件 / 行号附近内容） |
|---------|-------------------------------|
| §8.1 定位（"追溯到 1795 年高斯研究行星运动""无任何概率假设，只需信号模型""不是最佳、性能无法评价"） | `ocr_page_197.txt` L5~12 |
| §8.2 小结（(8.1)~(8.17)、(8.28)~(8.31)、(8.46)~(8.48)、(8.50)(8.52)、(8.61)(8.62)） | `ocr_page_197.txt` L13~27 |
| (8.1) $J(\theta)=\sum(x[n]-s[n])^2$，观测区间 $n=0..N-1$ | `ocr_page_198.txt` L7~11 |
| 例 8.1 $s[n]=A$，LSE = 样本均值；非零均值噪声 → 估计 $A+E(w)$；模型 $x=A+Bn+w$ → 有偏 | `ocr_page_198.txt` L27~34 + `ocr_page_199.txt` L1~15 |
| 例 8.2 $s[n]=\cos 2\pi f_0 n$ 频率非线性；例 8.3 $s[n]=A\cos 2\pi f_0 n$（$f_0$ 已知）线性；可分离问题 | `ocr_page_199.txt` L16~38 + `ocr_page_200.txt` L1~8 |
| (8.2) $s[n]=\theta h[n]$；(8.4) $\hat\theta=\sum xh/\sum h^2$；(8.6) $J_{\min}$ | `ocr_page_200.txt` L11~53 |
| (8.7) $0\le J_{\min}\le\sum x^2[n]$（习题 8.2） | `ocr_page_201.txt` L10~16 |
| (8.8) $s=H\theta$（$H$ 为 $N\times p$ 满秩，$N>p$）；(8.9)；(8.10) $\hat\theta=(H^TH)^{-1}H^Tx$；标准方程 $H^TH\hat\theta=H^Tx$ | `ocr_page_201.txt` L17~39 |
| "同形不同义"：BLUE 要求 $E(x)=H\theta$、$C_x=\alpha I$；有效估计还要求高斯 | `ocr_page_201.txt` L40~45 |
| (8.11)~(8.13) $J_{\min}$（等幂矩阵 $A^2=A$） | `ocr_page_201.txt` L45 + `ocr_page_202.txt` L1~14 |
| (8.14) 加权 LS（$W$ 正定）；(8.15) $\hat A=\sum(x/\sigma_n^2)/\sum(1/\sigma_n^2)$ 是 BLUE（$W=C^{-1}$）；(8.16)(8.17) | `ocr_page_202.txt` L15~38 |
| §8.5 几何：$J=\|x-H\theta\|^2$（8.19）；正交原理（8.21）$e^TH=0^T$；"第 12 章也出现类似正交原理" | `ocr_page_202.txt` L39~43 + `ocr_page_203.txt` + `ocr_page_204.txt` L1~32 + `ocr_page_205.txt` L1~3 |
| 例 8.4 傅里叶分析 $s[n]=a\cos 2\pi f_0n+b\sin 2\pi f_0n$ | `ocr_page_203.txt` L6~25 |
| 例 8.5 $f_0=k/N$（$k=1..N/2-1$）时 $h_1^Th_2=0$、$H^TH=(N/2)I$、$\hat\theta=(2/N)H^Tx$ | `ocr_page_205.txt` L26~35 + `ocr_page_206.txt` L1~19 |
| 投影矩阵 $P=H(H^TH)^{-1}H^T$：对称、等幂、奇异（秩 $p$）；$P^\perp=I-P$；$J_{\min}=\|P^\perp x\|^2$ | `ocr_page_206.txt` L20~40 |
| §8.6 按阶递推：图 8.5/8.6 选阶；真值 $s(t)=1+0.03t$、$\sigma^2=0.1$、$N\sigma^2=10$；对称区间 $[-M,M]$ 使列正交；Gram-Schmidt | `ocr_page_207.txt` L6~53 + `ocr_page_208.txt` L1~53 + `ocr_page_209.txt` L25~48 + `ocr_page_210.txt` L1~15 |
| (8.25)(8.26) $\hat\theta_k$、$J_{\min,k}$；(8.27) $H_{k+1}=[H_k\ h_{k+1}]$；(8.28)~(8.31) 按阶递推；$P_k^\perp=I-H_k(H_k^TH_k)^{-1}H_k^T$；$D_k=(H_k^TH_k)^{-1}$ | `ocr_page_210.txt` L16~49 + `ocr_page_211.txt` L1~8 |
| 例 8.6 直线拟合（结果与 (8.24) 一致） | `ocr_page_211.txt` L9~44 + `ocr_page_212.txt` L1~78 |
| 按阶递推几点看法：新列正交→前 $k$ 分量不变；$P_k^\perp x$ 是残差；监视 $h_{k+1}^TP_k^\perp h_{k+1}$ 剔除小值列；$J_{\min}$ 单调降；(8.32)(8.33) $r_{k+1}^2$ 相关系数平方；(8.34) 递推投影矩阵 | `ocr_page_212.txt` L82 + `ocr_page_213.txt` + `ocr_page_214.txt` L1~28 |
| §8.7 序贯：(8.36) $\hat A[N]=\hat A[N-1]+\frac{1}{N+1}(x[N]-\hat A[N-1])$；(8.37) 加权；(8.38) $K[N]=\frac{var}{var+\sigma_N^2}$；(8.39) $var=(1-K)var$；(8.40)~(8.42) | `ocr_page_214.txt` L29~40 + `ocr_page_215.txt` + `ocr_page_216.txt` + `ocr_page_217.txt` L1~24 |
| 图 8.9 蒙特卡洛（$A=10$、$\sigma^2=1$）；$var(\hat A[N])=\sigma^2/(N+1)$ | `ocr_page_217.txt` L24~32 |
| (8.43) $J_{\min}$ 序贯 | `ocr_page_217.txt` L33~36 |
| 矢量序贯：$C$ 对角才能序贯；(8.44)(8.45)；(8.46)~(8.48)；无需求逆；初始化 $n\ge p$ 或 $\hat\theta[-1]=0$、$\Sigma[-1]=\alpha I$（$\alpha$ 大） | `ocr_page_217.txt` L37~39 + `ocr_page_218.txt` + `ocr_page_219.txt` L1~42 |
| 例 8.7 傅里叶序贯（$p=2$，批初始化 $x[0],x[1]$） | `ocr_page_220.txt` L7~30 |
| (8.49) $J_{\min}$ 序贯 | `ocr_page_221.txt` L10~12 |
| §8.8 约束 LS：(8.50) $A\theta=b$（$r<p$ 满秩）；拉格朗日乘子；(8.52) $\hat\theta_c=\hat\theta-(H^TH)^{-1}A^T[A(H^TH)^{-1}A^T]^{-1}(A\hat\theta-b)$ | `ocr_page_221.txt` L13~38 + `ocr_page_222.txt` L1~12 |
| 例 8.8 约束信号 $\theta_1=\theta_2$ → 取平均 $(x[0]+x[1])/2$；几何：约束信号估计 = 无约束估计在约束子空间上的投影 | `ocr_page_222.txt` L13~31 + `ocr_page_223.txt` L1~31 |
| §8.9 非线性 LS：$x\sim N(s(\theta),\sigma^2I)$ 时 LSE = MLE；参数变换（例 8.9 正弦 $A,\phi\to\alpha_1,\alpha_2$）；参数分离 $s=H(\alpha)\beta$（例 8.10 阻尼指数 $s=A_1r_1^n+A_2r_2^n$） | `ocr_page_224.txt` L1~35 + `ocr_page_225.txt` + `ocr_page_226.txt` |
| 必要条件 (8.55)~(8.60)；Newton-Raphson (8.61)；Gauss-Newton (8.62)；线性模型 $s=H\theta$ 一步收敛 | `ocr_page_227.txt` + `ocr_page_228.txt` + `ocr_page_229.txt` L1~13 |
| 例 8.11 滤波器设计 → LS Prony（$f_c=0.1$、$n_0=25$、$N=50$、$p=q=10$；Parks and Burrus 1987）；$h_d[n]$ 是"数据"、$h[n]$ 是"信号" | `ocr_page_229.txt` L14~41 + `ocr_page_230.txt` + `ocr_page_231.txt` + `ocr_page_232.txt` L1~37 |
| 例 8.12 ARMA 的 AR 参数 → 修正 Yule-Walker (8.63)(8.64)；用估计 ACF 当数据、$H$ 随机 | `ocr_page_232.txt` L38~46 + `ocr_page_233.txt` + `ocr_page_234.txt` + `ocr_page_235.txt` L1~32 |
| 例 8.13 ANC：$x[n]=10\cos(2\pi(0.1)n+\pi/4)$、参考 $r[n]=\cos(2\pi(0.1)n)$、$p=2$、$\lambda=0.99$；稳态 $h[0]=16.8$、$h[1]=-12.0$ | `ocr_page_235.txt` L33~47 + `ocr_page_236.txt` + `ocr_page_237.txt` L1~45 + `ocr_page_238.txt` L1~9 |
| 例 8.14 锁相环：$s[n]=\cos(2\pi f_0n+\phi)$，对称区间 $[-M,M]$，Gauss-Newton 线性化，高 SNR 收敛到真值（Proakis 1983；Stoica et al. 1989） | `ocr_page_239.txt` + `ocr_page_240.txt` L1~22 |
| 附录 8A/8B/8C（分块矩阵求逆、Woodbury 恒等式） | `ocr_page_246~252.txt` |
| 页码：书内 = PDF − 15 | `ocr_page_198.txt` L2（顶栏"183"）、`ocr_page_240.txt` L2（顶栏"225"），与目录"第 8 章起于 182"一致 |

## 2. 据英文原版校订处

1. **式 (8.28) 按阶递推的矩阵形式**：OCR 页 210 L26~49 的矩阵被认成"ht++P+hk+!"、"D,HTh++h+H,D"等残字。据 Kay 英文原版（Vol I, §8.6）校订为分块形式 $\hat{\boldsymbol\theta}_{k+1}=\begin{bmatrix}\hat{\boldsymbol\theta}_k\\0\end{bmatrix}+\begin{bmatrix}-\mathbf{D}_k\mathbf{H}_k^T\mathbf{h}_{k+1}\\1\end{bmatrix}\frac{\mathbf{h}_{k+1}^T\mathbf{P}_k^\perp\mathbf{x}}{\mathbf{h}_{k+1}^T\mathbf{P}_k^\perp\mathbf{h}_{k+1}}$。理由：① 由 $(\mathbf{H}_{k+1}^T\mathbf{H}_{k+1})^{-1}$ 的分块矩阵求逆（Schur 补 $\mathbf{h}_{k+1}^T\mathbf{P}_k^\perp\mathbf{h}_{k+1}$）直接导出；② 与 (8.30) 的 $D_{k+1}$ 递推、(8.31) 的 $J_{\min}$ 递推自洽；③ 例 8.6 直线拟合代回后得到 $B_2=\frac{\mathbf{h}_2^T\mathbf{P}_1^\perp\mathbf{x}}{\mathbf{h}_2^T\mathbf{P}_1^\perp\mathbf{h}_2}=\frac{12}{N(N^2-1)}\bigl(\sum nx[n]-\bar x\sum n\bigr)$，与 (8.24) 一致。
2. **式 (8.61) Newton-Raphson / (8.62) Gauss-Newton**：OCR 页 227~229 被认成"+++1 +(HH)-1HI(x -HO)"等残字。据 Kay 英文原版（Vol I, §8.9）校订为正文形式（(8.61) 含二阶导数 $\mathbf{G}_n$，Gauss-Newton (8.62) 省略 $\mathbf{G}_n$）。理由：正文"线性模型一步收敛"的定性结论（OCR 页 228 L22~26 "即一步达到收敛"）与此形式一致。
3. **例 8.13 的初始化 $\boldsymbol\Sigma[-1]=10^5\mathbf{I}$**：OCR 页 237 L44 显示"[-1]    105"，应为 $10^5$（上标丢失）。据 Kay 英文原版（Vol I, Example 8.13）为 $\boldsymbol\Sigma[-1]=10^5\mathbf{I}$。理由：这是序贯 LS 常用的"大方差弱先验"初始化，且与 §5.3 的"$\alpha$ 取大"口径一致。
4. **例 8.9 的逆变换 $\hat{\phi}=\arctan(-\hat\alpha_2/\hat\alpha_1)$**：OCR 页 225 L23~28 只认出"arctan"字样，参数关系据英文原版（$\alpha_2=-A\sin\phi\Rightarrow \phi=\arctan(-\alpha_2/\alpha_1)$）校订。
5. 其余公式编号（8.1）~（8.66）、例 8.1~8.14、图 8.1~8.17 均以 OCR 为准，未发现需校订处。

## 3. 图片清单

| 编号 | 文件名 | 生成脚本 | 碰撞检测 |
|------|--------|----------|---------|
| Fig011 | `figures/Fig011_最小二乘正交原理.png` | `Temp/scripts/make_fig011.py` | ✅ 通过（`plotutil.check_figure`） |
| Fig012 | `figures/Fig012_序贯最小二乘.png` | `Temp/scripts/make_fig012.py` | ✅ 通过（`plotutil.check_figure`，含两子图 (a)/(b)） |

- 图意与 `图片编号登记.md` 登记的 Fig011（"最小二乘的几何正交原理"）、Fig012（"序贯更新：新数据到来时估计量的修正"）一致，未超出预分配编号，未新建编号。
- 运行命令 `py -3.14 make_fig011.py`、`py -3.14 make_fig012.py`（workdir=`Temp/scripts`）均见 `saved ...` 输出。
- Fig011 为静态几何示意图，无随机数；Fig012 右图为自建蒙特卡洛，种子 20260815，真值 $A=10$、$\sigma^2=1$，最终 $\hat A[N-1]\approx10.03$（印证收敛）。

## 4. 未决疑问

1. **式 (8.24) 的 $A_2$、$B_2$ 完整表达式**：OCR 页 208 L13~31 被认成"2(2N - 1)/N(N + 1)""12/N(N + 1)"等残字。正文只保留"例 8.6 结果与 (8.24) 一致"的结论，未展开 (8.24) 的完整系数。**建议终检以英文原版（Vol I, §8.6）核对一次 $A_2=\frac{2(2N-1)}{N(N+1)}\sum x[n]-\frac{6}{N(N+1)}\sum n x[n]$ 与 $B_2=\frac{12}{N(N^2-1)}\sum n x[n]$ 的具体系数。**
2. **式 (8.28) 的完整矩阵形式**：OCR 页 210 残断严重，正文按英文原版校订为简洁形式。若后续要严格复现按阶递推算法，需以英文原版附录 8A 为准（本次结论正确，属例行复核）。

## 5. 交叉引用一致性

- 与第 1 篇 `Vol1_Ch01_引言.md` 对齐：① "1795 年高斯用最小二乘预测行星运动（第 8 章正式登场）"（第 1 篇 §1.2 伏笔）✅ 本章"写在前头"§1 兑现；② "指数上的平方和正是日后最小二乘的原型（第 8 章揭晓）"（第 1 篇 §2.2）✅ 本章 §1.2 兑现。
- 与第 7 篇 `Vol1_Ch07_最大似然估计.md` 对齐：① "MLE 和 LS 在高斯线性模型上殊途同归、差异在概率假设（第 8 篇对照）"（第 7 篇 §6.4 顺带一提）✅ 本章 §2.1 兑现；② 非线性 LS 的 Newton-Raphson/Gauss-Newton 回调第 7 篇 §5.3 ✅ 本章 §7.2。
- 与第 4 篇（线性模型 MVU）、第 6 篇（BLUE）对齐：本章 §2.1 的"同形不同义"表格与两章结论一致（第 4 篇 §3、第 6 篇均给出 $(H^TH)^{-1}H^Tx$）。
- 伏笔：① 正交原理"第 12 章随机参数估计也出现"（OCR 页 205 明说）✅ 本章 §3.2 埋下，待第 12 篇兑现；② 序贯 LS 是"第 13 章卡尔曼的确定性前身" ✅ 本章 §5.3 埋下，待第 13 篇兑现。
