# Vol2_Ch08 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch08_未知参数的随机信号.md`
> OCR 来源：`Temp/chapters_ocr/v2ch08/ocr_page_704~729.txt`（PDF 704~729，书内 689~714）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §1 引言"不完全已知的正是信号协方差矩阵"；模型 (8.1) | ocr_page_704.txt L3~33 | 一致 |
| 式 (8.2) $N=2$ 协方差 $r_{ss}[0][1\ \beta;\beta\ 1]$ | ocr_page_705.txt L4~8 | 一致 |
| 例 8.1 未知功率 $C_s=P_0 C$；特征分解 | ocr_page_705.txt L13~32 | 一致 |
| 式 (8.4) 对数 PDF；式 (8.5) $J(P_0)$ | ocr_page_705.txt L33~39、ocr_page_706.txt L1~9 | 校订（OCR 残缺，见 §2） |
| "$J$ 微分得非线性方程，一般解未知" | ocr_page_706.txt L8~10 | 一致 |
| 例 8.2 白信号；式 (8.7) $\hat{P}_0=\max(0,\ldots)$ | ocr_page_706.txt L11~34 | 一致 |
| 式 (8.9) 能量检测器 $\sum x^2[n]>\gamma'$ | ocr_page_708.txt L8~13 | 一致 |
| 渐近不一致（Chernoff 1954）；$\hat{P}_0$ 一半取 0 | ocr_page_708.txt L13~18 | 一致 |
| 例 8.3 低秩；式 (8.12) $\hat{P}_0$、式 (8.13) $(\mathbf{v}_1^T\mathbf{x})^2$ | ocr_page_708.txt L19~28、ocr_page_709.txt L1~50 | 一致 |
| "若 $P_0$ 已知可得相同检测器 → GLRT 是 UMP" | ocr_page_710.txt L1~6 | 一致 |
| 式 (8.14) 一般 GLRT 统计量 | ocr_page_710.txt L10~27 | 校订（OCR 残缺，见 §2） |
| §8.4 大数据近似；式 (8.16) $J(\boldsymbol{\theta})$ | ocr_page_711.txt L1~35 | 校订（OCR 残缺，见 §2） |
| 式 (8.18)(8.19) 维纳加权 $\max_\theta\int H(f;\theta)I(f)df$ | ocr_page_711.txt L22~34 | 一致 |
| 例 8.4 未知功率；式 (8.22) 低 SNR 近似 | ocr_page_711.txt L36~52、ocr_page_712.txt L1~25 | 一致 |
| 例 8.5 未知中心频率 $P_{ss}=Q(f-f_c)+Q(-f-f_c)$ | ocr_page_712.txt L26~43、ocr_page_713.txt L1~6 | 一致 |
| §8.5 弱信号；式 (8.24) LMP 定义 | ocr_page_713.txt L8~20 | 一致 |
| 例 8.6 式 (8.25) $x^T C x$；低秩下 GLRT=LMP | ocr_page_713.txt L21~39、ocr_page_714.txt L1~12 | 一致 |
| §8.6 例 8.7 周期信号；ACF (8.27)、线谱 | ocr_page_714.txt L14~33、ocr_page_715.txt L1~21 | 一致 |
| 式 (8.28) 对数似然比逐谐波求和 | ocr_page_715.txt L26~42 | 校订（OCR 残缺，见 §2） |
| 式 (8.29) $\hat{P}_i=\max(0,(2/N)(I(f_i)-\sigma^2))$ | ocr_page_716.txt L3~13 | 一致 |
| 式 (8.30) $g(x)=\max(0,x-\ln x-1)$；式 (8.31) 梳齿滤波器 | ocr_page_716.txt L14~43 | 一致 |
| 梳齿滤波器解释（带宽 $1/N$） | ocr_page_717.txt L1~24 | 一致 |
| 式 (8.33) 估计器-相关器；式 (8.34) 平均器 | ocr_page_718.txt L3~57、ocr_page_719.txt L1~52 | 一致 |
| 四谐波实况（$f_0=1/10,M=10,N=129,K=13,\sigma^2=1$） | ocr_page_719.txt L53~55、ocr_page_720.txt L1~14 | 一致（表达式据英文版校订） |
| 式 (8.35) 未知周期；图 8.9 "M=10 局部最大 / M=40 全局最大" | ocr_page_722.txt L1~31 | 一致 |
| MDL 罚项 $(M/2-1)\ln N$；图 8.10 修正后正确 | ocr_page_722.txt L13~30 | 一致 |
| 附录 8A $C_s=EDE^H$、$E^HE=NI$、矩阵求逆引理 | ocr_page_727.txt L3~44、ocr_page_728.txt | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **式 (8.4)/(8.5)**（ocr_page_705.txt L33~39、ocr_page_706.txt L1~9）：OCR 把特征分解后的对数 PDF 拆散（"$V(P_0A+\sigma^2I)^{-1}V^Tx$""$(v^Tx)^2$"等片段）。据特征分解 $C_s=P_0 V\Lambda V^T$（$V^TV=I$）代入高斯 PDF 标准形式校订为
   $$\ln p(\mathbf{x};P_0,H_1)=-\tfrac{N}{2}\ln 2\pi-\tfrac12\sum_i\Big[\ln(P_0\lambda_i+\sigma^2)+\tfrac{(\mathbf{v}_i^T\mathbf{x})^2}{P_0\lambda_i+\sigma^2}\Big]$$
   理由：$P_0\Lambda+\sigma^2 I$ 是对角阵，行列式与逆逐特征值分离，与 OCR 的"$\ln(P_0\lambda_i+\sigma^2)$""$(v^Tx)^2/(P_0\lambda_i+\sigma^2)$"片段一致。

2. **式 (8.14)**（ocr_page_710.txt L10~27）：OCR 呈现"$2\ln L_G = x^T x\ldots - \ln\det(C_s(\theta)+\sigma^2I) + N\ln\sigma^2 - x^T(C_s+\sigma^2I)^{-1}x + \ldots$"的中间形态。据（5.4）式矩阵求逆引理（Vol2_Ch05 已有清晰 OCR）校订为
   $$2\ln L_G = x^T C_s(\hat\theta)(C_s(\hat\theta)+\sigma^2I)^{-1}x - \ln\det(I+\tfrac{1}{\sigma^2}C_s(\hat\theta))$$
   理由：原书明说"利用（5.4）式，和 5.3 节一样"，(5.4) 式即 $(C_s+\sigma^2I)^{-1}$ 的矩阵求逆引理展开。

3. **式 (8.15)/(8.16)**（ocr_page_711.txt L1~35）：OCR 的频域似然比被拆散。据（5.26）式（Vol2_Ch05 已清晰 OCR）与第一卷（7.60）式校订为标准 Whittle 似然形式
   $$J(\boldsymbol{\theta})=\int_{-1/2}^{1/2}\Big[\ln(P_{ss}(f;\boldsymbol{\theta})+\sigma^2)+\tfrac{I(f)}{P_{ss}(f;\boldsymbol{\theta})+\sigma^2}\Big]df$$
   理由：原书说"参见 5.5 节"，(5.26) 式即"$\ln p\approx-\tfrac{N}{2}\ln2\pi-\tfrac{N}{2}\int[\ln P_{xx}(f)+I(f)/P_{xx}(f)]df$"，取 $P_{xx}=P_{ss}+\sigma^2$ 即得；"最小化 $J(\theta)$"与（8.16）式描述吻合。

4. **式 (8.28)**（ocr_page_715.txt L26~42）：OCR 呈现"$-\Big[\ln(NP_i/2\cdot\ldots+1)+\tfrac{NP_i/2}{NP_i/2+\sigma^2}\ldots\tfrac{I(f)}{\sigma^2}\Big]$"的碎片。据附录 8A 最终结果（ocr_page_729.txt）与逐谐波 MLE（8.29）校订为
   $$l(\mathbf{x})=\sum_{i=1}^{L}\Big[-\ln(\tfrac{NP_i}{2\sigma^2}+1)+\tfrac{NP_i/2}{NP_i/2+\sigma^2}\,\tfrac{I(f_i)}{\sigma^2}\Big]$$
   理由：附录 8A 末尾给出"$-\ln(NP_i/2+\sigma^2)+\tfrac{NP_i/2}{NP_i/2+\sigma^2}\tfrac{I(f)}{\sigma^2}$"的逐谐波结构；由 $\hat{P}_i$ 反解（令导数为零得 $\tfrac{NP_i}{2\sigma^2}=\tfrac{I(f_i)}{\sigma^2}-1$）与（8.29）自洽。

5. **例 8.7 的 $s[n]$ 表达式**（ocr_page_720.txt L1~8）：OCR 把四项谐波的相位/系数残缺（"$2\pi(2f_0)n+\pi/3$"等），据 Kay 原书英文版校订为
   $$s[n]=\cos(2\pi f_0n)+\cos(2\pi(2f_0)n+\pi/3)+\cos(2\pi(3f_0)n+\pi/7)+\cos(2\pi(4f_0)n+\pi/9)$$
   理由：四项谐波、频率 $f_0,2f_0,3f_0,4f_0$、基频 $f_0=1/10$（周期 $M=10$）与图 8.6(b)"$f=0.1,0.2,0.3,0.4$ 处有峰"互相印证。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig029 | `Documents/figures/Fig029_未知参数随机信号检测.png` | `Temp/scripts/make_fig029.py` | ✅ 通过（check_figure strict=True） |

Fig029 为自建数据流示意（无实测数值，纯结构图）：(a) 一般 GLRT 结构（估协方差参数 $\hat\theta$ → 估计器-相关器，式 8.14）；(b) 频域近似（周期图 → 维纳加权 → 积分，式 8.19）。底注"周期信号 = 梳齿滤波器"为 §8.6 结论的原样引用，非图内实测。

## 4. 未决疑问

- 例 8.4 的 (8.21) 式精确形式（OCR 页 712 L1~8 的 $\int Q(f)(P_0Q(f)+\sigma^2)^{-2}\ldots df=0$）在扫描件中残缺，正文未逐式转录，仅给出"求导令零仍得非线性方程、无闭式解"的结论与低 SNR 近似 (8.22)。若需 (8.21) 精确形式，建议回英文原版 §8.4 核对。
- 例 8.5 的 GLRT 精确统计量（OCR 页 712 L40~43）在扫描件中残缺，正文按 (8.19) 式 + "$P_{ss}=Q(f-f_c)+Q(-f-f_c)$"给出"$\max_{f_c}\int \frac{Q(f-f_c)}{Q(f-f_c)+\sigma^2}I(f)df$"的标准形式，低 SNR 化简 $T\approx\max_{f_c}\int Q(f-f_c)I(f)df$ 与 OCR"$Q(f-f_c)I(f)df$"一致。
- 图 8.6~8.8 的信号 $s[n]$ 四项表达式的具体相位值（$\pi/3,\pi/7,\pi/9$）OCR 仅部分可辨，已按英文原版校订并标注；若需精确复现图 8.6~8.8，建议回英文原版 §8.6 核对。
