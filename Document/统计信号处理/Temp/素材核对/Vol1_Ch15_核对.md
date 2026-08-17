# Vol1 Ch15 素材核对清单

> 文档：`Documents/Vol1_Ch15_复数据与复参数的扩展.md`
> OCR 来源：`Temp/chapters_ocr/ch15/ocr_page_412~469.txt`（PDF 第 412~469 页，书内第 397~454 页，映射规则：书内页码 = PDF 页码 − 15；PDF 469 为空页）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | 原书编号 | OCR 出处（文件 + 附近内容） |
|---------|---------|---------------------------|
| 本章定位：不展开新原理，只处理复数据和复参数的代数运算 | — | ocr_page_412（"我们并不展开任何新的原理，只是为了处理复数据和复参数的代数运算"） |
| 复包络傅里叶关系 $S(F)=\tilde{S}(F-F_0)+\tilde{S}^*(-(F+F_0))$ | (15.1) | ocr_page_413（"S(F) = S(F - Fo) + S"(-(F + Fo) (15.1)"） |
| 窄带表示 $s(t)=2\mathrm{Re}[\tilde{s}(t)e^{j2\pi F_0 t}]$、$s(t)=2\tilde{s}_R\cos-2\tilde{s}_I\sin$ | (15.2)(15.3) | ocr_page_413（"s(t) = 2Re [3(t)exp(32πFat)) (15.2)…s(t) = 2s(t) cos 2πFot - 2sr(t) sin 2πFot (15.3)"） |
| 例 15.1 正弦复包络 $\tilde{s}(t)=\sum A_i e^{j\phi_i}e^{j2\pi(F_i-F_0)t}$；采样后 $\tilde{x}[n]=\sum A_i e^{j2\pi f_i n}$ | — | ocr_page_414~415（"3(t) = exp(j:)exp[j2(F; - Fo)t)…3[n] = A; exp(32 fin)"） |
| 例 15.2 复幅度 LS：复导数路线两步得 $\hat{A}=\sum\tilde{x}\tilde{s}^*/\sum|\tilde{s}|^2$ | — | ocr_page_416~417（"Z[n]s [n] …结果与实数情况类似：我们可以使用复变量来简化最小化过程"） |
| 复均值 $E(\tilde{z})=E(u)+jE(v)$ | (15.7) | ocr_page_418（"E(α) = E(u) + jE(u) (15.7)"） |
| 复方差 $\mathrm{var}=E|\tilde{z}-E(\tilde{z})|^2=E|\tilde{z}|^2-|E(\tilde{z})|^2$ | (15.9)(15.10) | ocr_page_418（"var() = E (1 - E()12) (15.9)…var() = E(2) - [E()[2 (15.10)"） |
| 协方差矩阵 Hermitian、半正定 | (15.13) | ocr_page_418~419（"协方差矩阵是Hermitian矩阵…半正定矩阵"） |
| 标量复高斯 PDF $\frac{1}{\pi\sigma^2}e^{-|\tilde{z}-\tilde\mu|^2/\sigma^2}$，$u,v$ 独立、各 $\mathcal{N}(\mu/2,\sigma^2/2)$ | (15.16) | ocr_page_420（"P() = T02 (15.16)…假定其实部和虚部分别服从N（2/2)和N（，/2）"） |
| 实协方差矩阵特殊形式 $\begin{bmatrix}\mathbf{A}&-\mathbf{B}\\\mathbf{B}&\mathbf{A}\end{bmatrix}$，A 对称、B 斜对称 | (15.19) | ocr_page_421（"这是一个实4×4一般协方差矩阵的特殊情况 -B (15.19) 2BA…A为对称的,B为反对称的,即B"=-B"） |
| $n=2$ 时协方差满足 $\mathrm{cov}(u_1,u_2)=\mathrm{cov}(v_1,v_2)$、$\mathrm{cov}(u_1,v_2)=-\mathrm{cov}(v_1,u_2)$ | (15.20) | ocr_page_421（"cov(u1, u2) = COv(U1,U2) (15.20) cov(u1, u2) - cov(uz, Ur)"） |
| 定理 15.1 复多维高斯 PDF、$\mathbf{C}_{\tilde x}=\mathbf{A}+j\mathbf{B}$ | (15.22) | ocr_page_423（"定理15.1(复多维高斯PDF）…C = 2(Cnu + jCru)…p(x) = (15.22)"） |
| 性质 4 仿射变换 $\mathbf{y}\sim\mathcal{CN}(\mathbf{A}\boldsymbol\mu+\mathbf{b},\mathbf{A}\mathbf{C}\mathbf{A}^H)$ | (15.23) | ocr_page_424（"y ~ CN(Ai + b, ACA) (15.23)"） |
| 性质 6 四阶矩 + $E(\tilde x_i\tilde x_j)=0$（伪方差为零的约束来源） | (15.24) | ocr_page_424（"E(24) E(2)E(4) + E(4)E(2) (15.24)…E（x;²）=E（）=0…正是这个约束条件导致了假定的实协方差矩阵的特殊形式"） |
| 例 15.3 WGN 的 DFT 系数为独立同分布复高斯；$\sigma^2$ 估计归一化 $c=N(N/2-1)/2$ | — | ocr_page_424~426（"DFT系数是独立且同分布的…c应该选择为N(N/2-1)/2"） |
| Hermitian 型 $Q=\tilde{\mathbf{x}}^H\mathbf{A}\tilde{\mathbf{x}}$：$E(Q)=\mathrm{tr}(\mathbf{A}\mathbf{C})$、$\mathrm{var}(Q)=\mathrm{tr}(\mathbf{A}\mathbf{C}\mathbf{A}\mathbf{C})$ | (15.29)(15.30) | ocr_page_427~428（"E(xHAx) = tr(ACa) (15.29) var(xHAx) = tr(ACACa) (15.30)"） |
| 复 WSS：ACF $r_{\tilde x\tilde x}[k]=E(\tilde{x}^*[n]\tilde{x}[n+k])$；CCF 约束 $r_{uu}=r_{vv}$、$r_{uv}=-r_{vu}$；PSD $P_{uv}=-P_{vu}$（纯虚互 PSD） | (15.31)~(15.36) | ocr_page_428~429（"r[周] = E(*[n][n + ]) (15.31)…ruu[K] = u[周] (15.33)…Puu(f) = Pou(f) (15.34)…ri[] 2ruu[k] + 2jrur[] (15.35)…P:(f) = 2(Puu(f) + jPut(f) (15.36)"） |
| 例 15.5 带通高斯噪声复包络采样后 $\tilde{x}[n]\sim\mathcal{CN}(0,\sigma^2)$ CWGN | — | ocr_page_429~431（"x[n[为复白高斯噪声(CWGN)…[n] ~ CN(0,α")"） |
| 复导数定义 $\partial J/\partial\tilde z=\frac12(\partial J/\partial\alpha-j\partial J/\partial\beta)$；$\partial\tilde z^*/\partial\tilde z=0$、$\partial|\tilde z|^2/\partial\tilde z^*=\tilde z$ | (15.40)(15.42)(15.43) | ocr_page_431~432（"实标量函数对复参数的复导数定义为 (15.40)…(15.42)…(15.43)"） |
| 线性/Hermitian 求导 $\partial(\mathbf{b}^H\boldsymbol\theta)/\partial\boldsymbol\theta=\mathbf{b}^*$、$\partial(\boldsymbol\theta^H\mathbf{A}\boldsymbol\theta)/\partial\boldsymbol\theta=(\mathbf{A}\boldsymbol\theta)^*$；"实数不是特例" | (15.44)(15.45)(15.46) | ocr_page_433（"ATA*=(AO)* (15.46)…如果为实的，那么我们应该有aJ/a0=2A6…这样实数的情况就不是特殊情况了"） |
| 例 15.6 Hermitian 型最小化 → $\hat\theta=(\mathbf{H}^H\mathbf{C}^{-1}\mathbf{H})^{-1}\mathbf{H}^H\mathbf{C}^{-1}\tilde{\mathbf{x}}$ | (15.50) | ocr_page_434（"=(H"C-1H)-1HHC-1i (15.50)"） |
| 约束最小化 $\mathbf{a}_{\mathrm{opt}}=\mathbf{W}^{-1}\mathbf{B}^H(\mathbf{B}\mathbf{W}^{-1}\mathbf{B}^H)^{-1}\mathbf{b}$；例 15.7 复均值 BLUE | (15.51) | ocr_page_436（"(15.51)…A 的 BLUE为 1TC-1元 / 1TC-11 在形式上与实数情况相同（参见例6.2）"） |
| 复 CRLB（迹项 + $2\mathrm{Re}$ 均值项） | (15.52) | ocr_page_437（"[1(] = tr … + 2Re (15.52)"） |
| 复参数达界条件 $\partial\ln p/\partial\tilde\theta^*=\mathbf{I}(\tilde\theta)(\hat{\tilde\theta}-\tilde\theta)$、$\mathbf{C}_{\hat\theta}=\mathbf{I}^{-1}(\tilde\theta)$ | (15.54)(15.56) | ocr_page_439~440（"0ln p(;0) = I(0)(8 - 8) (15.54)…Cs = I-1(0) (15.56)"） |
| 例 15.9 复经典线性模型 $\hat{\tilde\theta}=(\mathbf{H}^H\mathbf{C}^{-1}\mathbf{H})^{-1}\mathbf{H}^H\mathbf{C}^{-1}\tilde{\mathbf{x}}$、$\mathbf{C}_{\hat\theta}=(\mathbf{H}^H\mathbf{C}^{-1}\mathbf{H})^{-1}$；也是 MLE、加权 LSE、非高斯时 BLUE | (15.58)(15.59) | ocr_page_440~441（"= (HHC-1H)-1H"C-1÷ (15.58)…C =(HHC-1H)-1 (15.59)…若w不是复高斯随机失量，那么自可以证明为此问题的BLUE(参见习题15.20)"） |
| MLE 方程（迹项 + $2\mathrm{Re}$ 均值项） | (15.60) | ocr_page_442（"ln p(x; ≤) 8C+(6) + (x -i(5)"C(E) -1(E)(x - u(6) … +2Re [(x -u($))"C='(5) (15.60)"） |
| 例 15.10 复正弦相位 MLE $\hat\phi=\arctan(\mathrm{Im}X(f_0)/\mathrm{Re}X(f_0))$；"可与例 7.6 比较" | — | ocr_page_442~443（"@ = arctan Im(X (fo)) / Re(X(fo)) 可以将其与例7.6的结果进行比较"） |
| 复贝叶斯后验均值/协方差 | (15.61)(15.62) | ocr_page_443（"E(0x) E(0) + CC (x - E(x)) (15.61) Cee - CeClCt (15.62)"） |
| 复贝叶斯线性模型 MMSE、最小贝叶斯 MSE | (15.63)~(15.67) | ocr_page_443~444（"= g+CeoHH (HCecHH+Ca)-1 (-Hμs) (15.64)…[(C + HHC"H)-1] (15.67)"） |
| 渐近复高斯 PDF $-N\ln\pi-N\int[\ln P+I/P]df$；对照实版本 $\frac{N}{2}\to N$ | (15.68) | ocr_page_445（"Inp(;) -Nln - N I(f) (15.68) P(f)"） |
| 渐近 CRLB | (15.69) | ocr_page_445（"0in P(f;E) oln Pr(f:E) [I()]j = N (15.69)"） |
| DFT 系数渐近 $X(f_k)\xrightarrow{a}\mathcal{CN}(0,NP(f_k))$ 渐近独立 | (15.70)(15.71) | ocr_page_446~447（"E[X(ft)X*(f)] ~ NPr-(fx)Okl…X(fk) & CN(0, NP(f)) (15.71)"） |
| 例 15.12 周期图 $\mathrm{var}(\hat P)\approx P^2$，非一致 | — | ocr_page_448（"var(P(fk)) = P21(fx) 它不随 N的增加而减少…周期图不是一致的"） |
| 例 15.13 复正弦 CRLB：$\mathrm{var}(\hat A)\ge\sigma^2/(2N)$、$\mathrm{var}(\hat f_0)\ge 6\sigma^2/[(2\pi)^2A^2N(N^2-1)]$、$\mathrm{var}(\hat\phi)\ge(2N-1)/[\eta N(N+1)]$；复对实 f0/φ 半、A 四分之一 | (15.72) | ocr_page_450（"var(A) 2N…var(fo) (2)2mN(N2 - 1)…2(2N - 1) var(0) A'N(N + 1)…对于和，复情况的限是实情况的二分之-一;,对于A,则是四分之一"） |
| 复正弦 MLE：周期图峰值 + $\hat{\tilde A}=\frac1N\sum\tilde{x}e^{-j2\pi\hat f_0 n}$；"MLE 是精确的，并不像实的情况要求 N 很大" | (15.73)(15.74) | ocr_page_451（"A = A: [n] exp(-j2 fon) (15.73)…注意MLE是精确的，并不像在实的情况时要求的那样假定N很大"） |
| 例 15.14 MVDR 波束形成 $\hat{\tilde s}=\mathbf{e}^H\mathbf{C}^{-1}\tilde{\mathbf{x}}/(\mathbf{e}^H\mathbf{C}^{-1}\mathbf{e})$；$\mathbf{C}=\sigma^2\mathbf{I}$ 退化为常规波束形成 | (15.75) | ocr_page_453~454（"波束形成器的输出端为 eC-1x(t) (15.75)…如果C="I…这就是所谓的常规波束形成器"） |

## 2. 据英文原版校订之处及理由

1. **（15.72）复正弦 CRLB**：OCR 该页公式残损为 `var(A) 2N`、`var(fo) (2)2mN(N2-1)`、`2(2N-1) var(0) A'N(N+1)`，无法可靠读出分子分母；按 Kay 英文原版 §15.10 补全为 $\sigma^2/(2N)$、$6\sigma^2/[(2\pi)^2A^2N(N^2-1)]$、$(2N-1)/[\eta N(N+1)]$，并与 OCR 后文"f0 和 φ 为实情形的二分之一、A 为四分之一"交叉自洽（实版本例 3.14 为 $2\sigma^2/N$、$12\sigma^2/[(2\pi)^2A^2N(N^2-1)]$、$2(2N-1)/[\eta N(N+1)]$）。
2. **例 15.8（随机正弦的 $\sigma_A^2$ 估计）**：OCR 该页估计量代数式严重残损（`[这He[2 ... N2` 等），无法可靠转录；正文只保留"有效但方差不随 $N$ 递减、非一致，对比习题 3.14"的结构结论，不转录具体公式。
3. **全文复共轭 / 共轭转置 / 矩阵分式**：OCR 大量把 `*` 丢成空、把 $\mathbf{H}^H$ 识成 `H"`、把 $\mathbf{C}^{-1}$ 识成 `C-1`，均按 Kay 英文原版校订；（15.16）（15.19）（15.22）（15.46）（15.50）（15.52）（15.58）（15.64）~（15.67）（15.68）均经校订。
4. **"伪方差"措辞**：OCR 原文未直接使用"伪方差"一词，而是用 "$E(\tilde x_i\tilde x_j)=0$" 与"这个约束条件导致了假定的实协方差矩阵的特殊形式"表述；本文正文以"伪方差（pseudo-variance）"作为标准术语并同步给出 OCR 的等价表述 $E[(\tilde z-\mu)^2]=0$，未改变其数学含义。

## 3. 图片清单

| 编号 | 文件 | 脚本 | 碰撞检测 | 状态 |
|------|------|------|---------|------|
| Fig020 | `Documents/figures/Fig020_复高斯与实数化对应.png` | `Temp/scripts/make_fig020.py` | `check_figure` 通过 | ✅ 已生成 |

图注要点：Fig020 为两面板散点对比（种子 20261515）：(a) 实部虚部独立等方差（$N(0,0.5)$）→ 等概率线是圆（循环对称、伪方差=0，可写成 $\mathcal{CN}$）；(b) $\mathrm{var}(u)=0.5,\mathrm{var}(v)=2$ 不等 → 等概率线是椭圆（伪方差≠0，不能写成 $\mathcal{CN}$）。生成过程无缺字形警告。

## 4. 未决疑问

- 例 15.8 的 $\sigma_A^2$ 估计量精确代数式无法从 OCR 可靠读出（本文已按规则略去并注明），复现者请以 Kay 英文原版 Example 15.8 为准。
- 例 15.14 图 15.6 的数值（$\sigma^2=1$、$M=10$、信号 $\beta_s=90^\circ$、$p=10$ 等）OCR 只可辨 `对于 p=10,α2=1,M=10,以及在 3,=90(所以f =0)处的信号`，本文未转录图 15.6 的具体曲线数据，只引其"干扰与信号同向时不能衰减"的结论。
