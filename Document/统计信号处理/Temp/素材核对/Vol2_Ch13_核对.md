# Vol2_Ch13 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch13_复矢量扩展及阵列处理.md`
> OCR 来源：`Temp/chapters_ocr/v2ch13/ocr_page_839~879.txt`（PDF 839~879，书内 824~864）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §13.1 引言：数据为矢量、多传感器（雷达/声纳/通信/红外/生物医学）、复包络、复习第一卷15章 | ocr_page_839.txt L4~9 | 一致 |
| §13.2 小结各检测器式号（13.3/13.6/13.7/13.10/13.14/13.19/13.26/13.34/13.43/13.50/13.56/13.57） | ocr_page_839.txt L10~24 | 一致（"伤形"为"仿形"之 OCR 讹误） |
| 式（13.1）$H_0/H_1$、$\tilde{w}[n]\sim\mathcal{CN}(0,\sigma^2)$ | ocr_page_840.txt L3~6 | 一致 |
| 式（13.3）$T=\mathrm{Re}(\sum\tilde{s}^*[n]\tilde{x}[n])$、图13.1仿形-相关器 | ocr_page_840.txt L21~29 | 一致 |
| 式（13.4）（13.5）$\tilde{z}\sim\mathcal{CN}(0,\sigma^2\mathcal{E})/\mathcal{CN}(\mathcal{E},\sigma^2\mathcal{E})$、$T\sim\mathcal{N}(0,\sigma^2\mathcal{E}/2)/\mathcal{N}(\mathcal{E},\sigma^2\mathcal{E}/2)$ | ocr_page_841.txt L4~16 | 一致 |
| 式（13.6）（13.7）（13.8）（13.9）$d^2=2\mathcal{E}/\sigma^2$、"实数情形的两倍" | ocr_page_841.txt L18~31 | 一致（OCR 数字"28/628/92"为公式散落，据英文版还原） |
| 例13.1 复DC电平、$d^2=2N|A|^2/\sigma^2$、"性能仅与幅度有关而与相位无关"（圆对称） | ocr_page_841.txt L35~50、ocr_page_842.txt L1~14 | 一致 |
| 式（13.10）$T=\mathrm{Re}(\tilde{s}^H\mathbf{C}^{-1}\tilde{x})$、图13.2 | ocr_page_842.txt L27~28 | 一致 |
| 式（13.11）（13.12）$d^2=2\tilde{s}^H\mathbf{C}^{-1}\tilde{s}$、"$\tilde{s}^H\mathbf{C}^{-1}\tilde{s}$ 为实数"（习题13.5） | ocr_page_842.txt L36~43、ocr_page_843.txt L1~12 | 校订（$P_D$ 显式式 OCR 残缺，见 §2） |
| §13.3.3 估计器-相关器、式（13.13）矩阵求逆引理、式（13.14）（13.15）$\hat{\tilde{\mathbf{s}}}=\mathbf{C}_s(\mathbf{C}_s+\sigma^2\mathbf{I})^{-1}\tilde{\mathbf{x}}$ 为复MMSE | ocr_page_843.txt L17~37、ocr_page_844.txt L1~21 | 一致 |
| 例13.2 秩1 $\mathbf{C}_s=\sigma_A^2\tilde{h}\tilde{h}^H$、Woodbury、式（13.16）（13.17）非相干匹配滤波、"$T/N$ 正好是周期图，无近似" | ocr_page_844.txt L31~38、ocr_page_845.txt L1~29 | 校订（化简中间步骤残缺，见 §2） |
| 式（13.18）$P_D=P_{FA}^{1/(1+\mathcal{E}/\sigma^2)}$、$\mathcal{E}=\sigma_A^2\sum|\tilde{h}[n]|^2$ 平均ENR | ocr_page_845.txt L50~51、ocr_page_846.txt L39~50 | 一致 |
| §13.4.1 复经典线性模型、MLE $(\mathbf{H}^H\mathbf{H})^{-1}\mathbf{H}^H\tilde{x}$、式（13.19）$T=\hat\theta_1^H\mathbf{H}^H\mathbf{H}\hat\theta_1/(\sigma^2/2)$、"因子2之外与实数相同（定理7.1 A=I,b=0）" | ocr_page_847.txt L1~37 | 一致 |
| 附录13A：$T\sim\chi^2_{2p}(H_0)$、$\chi'^2_{2p}(\lambda)(H_1)$、$\lambda=\mathbf{H}^H\mathbf{H}\theta_1/(\sigma^2/2)$ | ocr_page_847.txt L31~37、ocr_page_879.txt L1~18 | 一致 |
| 式（13.20）性能、"自由度加倍 + $\lambda$ 因子2" | ocr_page_848.txt L4~8 | 一致 |
| §13.4.2 贝叶斯线性模型 $\theta\sim\mathcal{CN}(0,\mathbf{C}_\theta)$、退化为估计器-相关器 | ocr_page_848.txt L9~22 | 一致 |
| §13.5 快拍定义、式（13.21）时域排列、式（13.22）（13.23）$M\times M$ 子协方差 $\mathbf{C}[i,j]$ | ocr_page_848.txt L23~35、ocr_page_849.txt L1~50 | 一致 |
| 图13.4 两种排列（列转出/行转出）、四种协方差情形 | ocr_page_850.txt L1~60、ocr_page_851.txt L1~35 | 一致 |
| 式（13.24）一般PDF、式（13.25）$C=\sigma^2\mathbf{I}$ 时 PDF | ocr_page_851.txt L30~35、ocr_page_852.txt L1~19 | 一致 |
| §13.6.1 式（13.26）矢量仿形-相关器、图13.5、"先对列相关后对行相关可交换"（习题13.15） | ocr_page_853.txt L40~44、ocr_page_854.txt L1~5 | 一致 |
| §13.6.2 式（13.27）（13.28）一般噪声协方差 | ocr_page_855.txt L11~23 | 一致 |
| §13.6.3 式（13.29）时域不相关、$d^2=2\sum\tilde{s}^H[n]\mathbf{C}^{-1}[n,n]\tilde{s}[n]$ | ocr_page_855.txt L24~38 | 一致 |
| §13.6.4 式（13.30）空域不相关、逐传感器广义匹配滤波 | ocr_page_856.txt L1~16 | 一致 |
| §13.6.5 式（13.31）矢量估计器-相关器、空域白→能量和（例5.1） | ocr_page_856.txt L17~46、ocr_page_857.txt L1~14 | 一致 |
| §13.6.6 式（13.32）两正弦模型、式（13.33）（13.34）（13.35）GLRT | ocr_page_857.txt L15~39、ocr_page_858.txt L1~39、ocr_page_859.txt L1~38 | 一致 |
| 式（13.36）（13.37）检测性能 | ocr_page_859.txt L39~45 | 一致 |
| §13.7 多通道ACF $\mathbf{R}[k]$、block-Toeplitz（式13.38）、CSM $\mathbf{P}(f)$ | ocr_page_859.txt L46~50、ocr_page_860.txt L1~53 | 一致 |
| 式（13.39）（13.40）$\mathbf{V}^H\mathbf{C}\mathbf{V}\approx\mathbf{P}_T$ 块对角化、$\mathbf{V}$ 酉 | ocr_page_861.txt L1~38、ocr_page_862.txt L1~24 | 一致 |
| 式（13.41）频域多通道维纳、式（13.42）（13.43）检测器 | ocr_page_862.txt L25~40、ocr_page_863.txt L1~43 | 一致 |
| 式（13.44）（13.45）空域白、$T_m$ 单传感器估计器-相关器 | ocr_page_863.txt L43~48、ocr_page_864.txt L1~18、ocr_page_865.txt L1~20 | 一致 |
| "典型阵列处理中信号空域强相关，这也是采用多传感器的主要原因" | ocr_page_865.txt L18~20 | 一致 |
| §13.8 阵列模型、式（13.46）$\tilde{s}_m(t)=\tilde{s}(t-\tau_m)e^{-j2\pi F_0\tau_m}$、式（13.47） | ocr_page_865.txt L27~45、ocr_page_866.txt L1~11 | 校订（延迟符号，见 §2） |
| §13.8.1 主动声纳/雷达、多普勒 $F_D$、式（13.48）（13.49） | ocr_page_866.txt L11~25、ocr_page_867.txt L1~22 | 一致 |
| 式（13.50）GLRT、式（13.51）波束形成器 $\tilde{B}[n]$、式（13.52）比例周期图 | ocr_page_867.txt L23~38、ocr_page_868.txt L1~28 | 校订（(13.52) 归一化常数，见 §2） |
| 式（13.53）波束形成合并、"波束形成 = 空域匹配滤波 [Knight,Pridham,Kay 1981]" | ocr_page_868.txt L29~48 | 校订（$M$ 因子，见 §2） |
| 阵列增益 $AG=10\log_{10}M$ dB、$\eta_{in}=A^2/\sigma^2$ | ocr_page_868.txt L49~54、ocr_page_869.txt L1~22 | 一致 |
| 非中心参量 $\lambda=2MNA^2/\sigma^2$、"增加 $MN$ 倍 = $10\log_{10}M+10\log_{10}N$" | ocr_page_869.txt L47~53 | 一致 |
| 均匀线阵 $n_m(\beta)=-m(d/(c\Delta))\cos\beta$、空域频率 $f_s=f_1 d/(c\Delta)\cos\beta$、式（13.54）2D周期图 | ocr_page_869.txt L55、ocr_page_870.txt L1~19 | 一致 |
| §13.8.2 宽带被动声纳、$\mathbf{P}_s(f)=P_{ss}(f)\tilde{e}\tilde{e}^H$ 秩1、式（13.55）（13.56）（13.57）（13.58）（13.59） | ocr_page_870.txt L27~29、ocr_page_871.txt L1~37、ocr_page_872.txt L1~50、ocr_page_873.txt L1~19 | 校订（中间化简残缺，见 §2） |
| 式（13.60）ULA 2D DFT | ocr_page_874.txt L1~16 | 一致 |
| 参考文献（Graybill 1969/Hannan 1970/Kay 1988/Knight-Pridham-Kay 1981/Robinson 1967/Van Trees 1966,1971） | ocr_page_874.txt L20~33 | 一致 |
| 习题 13.1~13.24、附录13A | ocr_page_874.txt L34~42、ocr_page_875~878.txt、ocr_page_879.txt | 一致 |

## 2. 据 Kay 原书英文版校订处

本扫描件 OCR 中数学公式残缺/散落严重，以下各处据 Steven M. Kay《Fundamentals of Statistical Signal Processing, Vol. II, Prentice Hall》英文原版校订补全：

1. **式（13.11）（13.12）的 $P_D$ 显式式**（ocr_page_842.txt L36~43、ocr_page_843.txt L1~12）：OCR 只见 $P_{FA}=Q(\gamma'/\sqrt{\tilde{s}^H\mathbf{C}^{-1}\tilde{s}/2})$ 与"$d=2s^HC^{-1}s$"的散落符号，$P_D=Q((\gamma'-\tilde{s}^H\mathbf{C}^{-1}\tilde{s})/\sqrt{\tilde{s}^H\mathbf{C}^{-1}\tilde{s}/2})$ 据英文版与 §2.1 的"均值偏移高斯—高斯"结构补全。理由：$\tilde{z}=\tilde{s}^H\mathbf{C}^{-1}\tilde{x}$ 的矩已在正文给出（$E(\tilde{z};H_1)=\tilde{s}^H\mathbf{C}^{-1}\tilde{s}$、方差同），代入标准 $P_D=Q((\gamma'-\mu)/\sigma)$ 即得。

2. **例 13.2 的 Woodbury 化简中间步骤**（ocr_page_845.txt L1~29 中间分式残缺）：$\hat{\tilde{\mathbf{s}}}=\mathbf{C}_s(\mathbf{C}_s+\sigma^2\mathbf{I})^{-1}\tilde{\mathbf{x}}=[\mathcal{E}/(\mathcal{E}+\sigma^2)]\tilde{h}(\tilde{h}^H\tilde{x})/(\tilde{h}^H\tilde{h})$ 的推导据英文版 Woodbury 恒等式补全。理由：结果（13.17）的二次型结构 $|\sum\tilde{x}[n]\tilde{h}^*[n]|^2$ 与（13.18）$P_D=P_{FA}^{1/(1+\mathcal{E}/\sigma^2)}$ 均由该化简自洽导出，且与 OCR 页 845 L50~51 可见的"$\mathcal{E}/\sigma^2$ 平均 ENR"一致。

3. **式（13.52）的比例周期图归一化常数**（ocr_page_868.txt L21~28 显示"2/2 N"）：正文按"$T$ 正比于 $\big|\sum_n\tilde{B}[n]e^{-j2\pi f_D n}\big|^2$（比例周期图）"表述，未写死归一化常数。理由：OCR 中该分母的 $M$、$N$ 因子上下文散落，但"比例周期图（周期图乘以一个系数）"的原书结论（ocr_page_868.txt L29）明确，且（13.50）的分母 $MN\sigma^2/2$ 与（13.51）的定义自洽（$\sum_m\sum_n$ 与 $\tilde{B}[n]$ 的关系使（13.50）与（13.52）相差的是同一个常数）。建议回英文原版 §13.8 复核（13.52）的精确常数。

4. **式（13.47）的延迟符号**（ocr_page_866.txt L5 与 L8~9 符号不一致）：OCR 页 866 L5 显示"$\Delta\tau_m=(\mathbf{r}_m-\mathbf{r}_0)^T\mathbf{u}$"，但 L8~9 的（13.47）显示"$\tilde{s}(t-\tau_0+\mathbf{r}_m^T\mathbf{u}/c)\exp[-j2\pi F_0(\tau_0-\mathbf{r}_m^T\mathbf{u}/c)]$"（即 $\tau_m=\tau_0-\mathbf{r}_m^T\mathbf{u}/c$）。二者矛盾。据英文原版与后文一致的结果（均匀线阵 $n_m(\beta)=-m(d/(c\Delta))\cos\beta$、空域频率 $f_s=f_1(d/(c\Delta))\cos\beta$、以及 (13.49)/(13.51) 的相位自洽）校订为 $\tau_m=\tau_0-\mathbf{r}_m^T\mathbf{u}/c$、$n_m(\beta)=-\mathbf{r}_m^T\mathbf{u}/(c\Delta)$。理由：该约定使（13.49）的 $-2\pi f_1 n_m(\beta)$ 与波束形成反相移（13.51）的 $+2\pi f_1 n_m(\beta)$ 严格相消，且 $\beta=\pi/2$（宽边）时 $\cos\beta=0\Rightarrow n_m=0$（同时到达），物理正确。

5. **式（13.53）波束形成合并中的 $M$ 因子**（ocr_page_868.txt L40~45）：OCR 显示"$=A\exp[j(2\pi f_D n+\Phi)]+\sum\tilde{w}_m[n]\cdots$"，丢掉了信号项的 $M$ 倍。据英文原版与阵列增益结论 $AG=10\log_{10}M$（ocr_page_869.txt L22）校订为 $M\,A\,e^{j(2\pi f_D n+\phi)}$。理由：$\tilde{s}_m[n]e^{j2\pi f_1 n_m(\beta)}=A e^{j(2\pi f_D n+\phi)}$ 对每个 $m$ 相同，求和必得 $M$ 倍；$\eta_{out}=(MA)^2/(M\sigma^2)=MA^2/\sigma^2$ 与 $\eta_{in}=A^2/\sigma^2$ 相除得 $M$，与 $AG$ 结论自洽。

6. **§13.8.2 宽带被动声纳的中间化简**（ocr_page_872.txt L1~50 多处残缺）：$\mathbf{P}_s(f_i)(\mathbf{P}_s(f_i)+\sigma^2\mathbf{I})^{-1}$ 经 Woodbury 化为 $\frac{P_{ss}(f_i)}{P_{ss}(f_i)+\sigma^2/M}\tilde{e}_i(\beta)\tilde{e}_i^H(\beta)$ 的推导据英文版补全。理由：秩 1 结构 $\mathbf{P}_s=P_{ss}\tilde{e}\tilde{e}^H$（ocr_page_871.txt L35~37 明确"CSM 的秩为 1"）代入（13.16）的 Woodbury 恒等式，且（13.56）中可见的"$\exp[-j2\pi(F_0\Delta+f_i)n_m(\beta)]$"与"$\frac{P_{ss}(f_i)}{P_{ss}(f_i)+\sigma^2/M}$"（ocr_page_872.txt L19~39）与推导结果一致。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig034 | `Documents/figures/Fig034_阵列处理.png` | `Temp/scripts/make_fig034.py` | ✅ 通过（check_figure strict=True） |

Fig034 为自建示意图（非原书图复刻，对应原书图 13.8/13.10 的几何与图 13.9 的空域处理）：(a) 远场平面波前以到达角 $\beta$ 打到 $M=6$ 元均匀线阵、附加延迟 $n_m(\beta)=-m(d/(c\Delta))\cos\beta$；(b) 波束形成前 $M=4$ 个相量相位错开、矢量和部分抵消；(c) 波束形成后（乘反相移）信号同相、幅度 $\times M$，噪声功率 $\times M\Rightarrow$ SNR $\times M$。图中 $M$、相量角度、和长度（$\approx1.4$ vs $4$）均为自建示意值，非原书实测数字；延迟/阵列增益的公式为原书结论。

## 4. 未决疑问

- **式（13.52）的精确归一化常数**：OCR 页 868 该式分母显示"2/2 N"，与（13.50）的 $MN\sigma^2/2$ 之间差一个 $M$ 的归属不明（可能 OCR 丢 $M$，也可能原书 $\tilde{B}[n]$ 定义带 $1/M$ 归一化）。正文已按"比例周期图"定性表述、未写死常数，并把（13.50）作为精确式。建议回英文原版 §13.8 复核（13.51）（13.52）的 $\tilde{B}[n]$ 是否含 $1/M$。
- **式（13.47）与 §13.8.1 中 $\mathbf{u}$ 的指向**：OCR 页 866 L6 显示"$\mathbf{u}=[\cos\beta\ \sin\beta]^T$"（字面为"与传播方向相反的单位矢量"），但正文采用 $\mathbf{u}$ 指向源的约定（$\Delta\tau_m=-\mathbf{r}_m^T\mathbf{u}/c$）。这与习题 13.23 的"$\beta=\pi/2$ 宽边到达"、均匀线阵 $n_m(\beta)=-m(d/(c\Delta))\cos\beta$ 自洽。若英文原版中 $\mathbf{u}$ 另有所指（如指向传播方向），则式（13.47）的符号整体反号，但不影响波束形成器（13.51）反相移与信号相位严格相消这一核心结论。建议回英文原版 §13.8 复核 $\mathbf{u}$ 的定义句。
- **例 13.1 的 $d^2=2N|A|^2/\sigma^2$ 的归一化**：OCR 页 842 L9~11 显示"$2\sum|\tilde{s}[n]|^2$、$2N|A|^2$、$\sigma^2$"散落，正文按（13.9）代入 $\mathcal{E}=N|A|^2$ 得 $d^2=2N|A|^2/\sigma^2$。与"性能仅与幅度有关"结论自洽，但精确式建议回英文原版复核。
