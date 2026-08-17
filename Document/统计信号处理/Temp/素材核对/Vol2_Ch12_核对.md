# Vol2_Ch12 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch12_模型变化检测.md`
> OCR 来源：`Temp/chapters_ocr/v2ch12/ocr_page_812~838.txt`（PDF 812~838，书内 797~823）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §1.1 语音/机器/声速/温度的变化，主要讨论突变 | ocr_page_812.txt L3~10 | 一致 |
| 例 12.1 假设（12.1）（12.2）、PDF（12.3） | ocr_page_812.txt L24~40、ocr_page_813.txt L1~42 | 一致 |
| 式（12.4）$\ln L$、式（12.5）$T=\frac1{N-n_0}\sum(x[n]-A_0)$ | ocr_page_813.txt L43~49、ocr_page_814.txt L1~21 | 一致 |
| $T(\mathbf{x})\sim\mathcal{N}(0,\sigma^2/(N-n_0))$（$H_0$）/$\mathcal{N}(\Delta A,\sigma^2/(N-n_0))$（$H_1$）；式（12.6）（12.7） | ocr_page_814.txt L22~35 | 一致 |
| 延迟"$N-n_0=30$ 个样本"（$P_{FA}=0.001,P_D=0.99$） | ocr_page_814.txt L36~38 | 派生（$\Delta A^2/\sigma^2=1$ 条件 OCR 未显式写，见 §2） |
| 图 12.2：$A_0=1,\Delta A=1,n_0=50,\sigma^2=1$；$P_{FA}=10^{-6}$→18 样本、$10^{-3}$→4 样本 | ocr_page_814.txt L39~43、ocr_page_815.txt L1~9 | 一致 |
| 例 12.2 方差 $1\to4$ 跳变、能量检测器 $T=\sum x^2[n]$ | ocr_page_816.txt L4~27、ocr_page_817.txt L21~26 | 一致 |
| §12.3 末"只需知道为正就足够，UMP"（习题 12.4） | ocr_page_817.txt L27~29 | 一致 |
| 例 12.3 MLE（$\hat{A},\hat{A}_1,\hat{A}_2$）、式（12.8） | ocr_page_817.txt L32~52、ocr_page_818.txt L1~37 | 一致 |
| 式（12.9）$\chi_1^2$ / $\chi_1^{\prime2}(\lambda)$、式（12.10）$\lambda$、最佳在 $n_0=N/2$ | ocr_page_818.txt L38~49、ocr_page_819.txt L1~10 | 一致 |
| 例 12.4 扫描取最大、"渐近 GLRT 统计量在这里并不成立" | ocr_page_819.txt L12~40、ocr_page_820.txt L1~4 | 一致 |
| 式（12.11）电平与时刻都未知 | ocr_page_820.txt L5~19 | 一致 |
| 图 12.4：$n_0=20,n_1=50,n_2=65$，$A=1,4,2,6$，$\sigma^2=1$；复杂度 $N^3/6$、一般 $O(N^M)$ | ocr_page_820.txt L20~27 | 一致 |
| 例 12.5 分段常数、式（12.12）$J(\mathbf{A},\mathbf{n})$ | ocr_page_821.txt L4~31 | 一致 |
| DP 最短路径 A→D、马尔可夫性质 | ocr_page_821.txt L32~50 | 一致 |
| 式（12.13）$\Delta_i$、式（12.14）$I_k[L]$ 递推、式（12.15）段误差 | ocr_page_822.txt L30~34、ocr_page_823.txt L1~52 | 一致 |
| 数值例子结果 $\hat{n}_0=20,\hat{n}_1=49,\hat{n}_2=65$ | ocr_page_824.txt L19~21 | 一致 |
| 序贯最小二乘递归公式、初值 $\hat{A}[0,0]=x[0],J_{\min}[0,0]=0$ | ocr_page_824.txt L22~37 | 一致 |
| 附录 12A 通用分段 DP（12A.1）、$\Delta_i=-\ln p_i$ | ocr_page_835.txt、ocr_page_836.txt L1~13 | 一致 |
| 附录 12B dp.m（`randn('seed',0)`、$A=[1;4;2;6]$、索引 ±1） | ocr_page_837.txt、ocr_page_838.txt | 一致 |
| 机动检测：标称位置、加速度项、式（12.16）、$P_{FA}=\exp(-\gamma'/2)$ | ocr_page_825.txt L6~30、ocr_page_826.txt L1~43、ocr_page_827.txt L1~8 | 校订（式 12.16 后显式 OCR 残缺，见 §2） |
| 机动检测例：$(0,0)$、速度 $(1,1)$、$\Delta=1$、$n_0=50$、$(a_x,a_y)=(0.03,0.05)$、$M=20$、门限 100→$n=65$（延迟 35 样本） | ocr_page_827.txt L22~30 | 一致 |
| PSD 检测：式（12.17）AR(1) PSD；式（12.19）（12.20）（12.21） | ocr_page_827.txt L32~40、ocr_page_829.txt、ocr_page_830.txt L1~20 | 校订（OCR 残缺，见 §2） |
| 式（12.22）GLRT、"第一项功率变化、第二项 PSD 形状变化" | ocr_page_830.txt L21~46 | 校订（OCR 残缺，见 §2） |
| PSD 例：$\sigma^2=5\to a[1]=-0.9,\sigma_u^2=1$，功率变到 $1/(1-a^2[1])=5.3$ | ocr_page_830.txt L47~49、ocr_page_831.txt L1~4 | 一致 |
| 图 12.10 峰值 $\hat{n}_0=51$；图 12.11 $\hat{a}[1]\approx-0.8$（真值 $-0.9$） | ocr_page_831.txt L5~10 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **延迟 30 样本的 $\Delta A^2/\sigma^2$ 取值**（ocr_page_814.txt L36~38）：OCR 只写"若跳变量与噪声功率之比为 $\Delta A^2/\sigma^2$，要求 $P_{FA}=0.001,P_D=0.99$，那么延迟最小为 $N-n_0=30$ 个样本"，未显式写 $\Delta A^2/\sigma^2$ 的数值。据（12.6）（12.7）反解：$d^2=(N-n_0)\Delta A^2/\sigma^2$，$Q^{-1}(0.001)=3.0902$、$Q^{-1}(0.99)=2.3263$，得 $d^2\approx29.3$，故需 $\Delta A^2/\sigma^2=1$ 才得 $N-n_0\approx30$。正文已注明"此数为由 (12.6)(12.7) 反解的派生值"。

2. **机动检测的显式统计量**（ocr_page_826.txt L23~43 残缺）：式（12.16）为 $T(\mathbf{x})=\hat{\boldsymbol{\theta}}^T\mathbf{H}^T\mathbf{H}\hat{\boldsymbol{\theta}}/\sigma^2$，但展开式 OCR 残缺。据 Kay 原书英文版 §12.6.1 与"$\mathbf{H}^T\mathbf{H}=\mathrm{diag}(\mathbf{h}^T\mathbf{h},\mathbf{h}^T\mathbf{h})$"校订为正文给出的"加速度匹配滤波"形式。理由：$\hat{\boldsymbol{\theta}}=(\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\boldsymbol{\varepsilon}$，$(\mathbf{H}^T\boldsymbol{\varepsilon})$ 的各分量正是 $\sum(n-n_0)^2\Delta^2\varepsilon[n]/2$，与 OCR 页 826 可见的"$\sum(n-n_0)^2\Delta^2$"（分子）与"$\sum(n-n_0)^4\Delta^4$"（分母）结构自洽。

3. **（12.17）~（12.22）PSD 检测**（ocr_page 827~830 多处残缺）：AR(1) PSD、$\hat{a}[1]$、$\hat\sigma_2^2$、$L_G(\mathbf{x})$ 均据 Kay 原书英文版 §12.6.2 校订。理由：(12.19) 是"负的一阶自相关归一化"（AR(1) 中 $x[n]=-a[1]x[n-1]+u[n]$ 的 MLE），(12.21) 由 (12.19) 代入 (12.20) 得，二者与 OCR 页 830 可见的"$\sum x[n]x[n-1]$、$\sum x^2[n-1]$、$(1-\hat{a}^2[1])$"字样一致；(12.22) 的"第一项功率变化、第二项谱形变化"与 OCR 页 830 L42~46 的说明一致，且两项都 >1 由习题 12.16 保证。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig033 | `Documents/figures/Fig033_模型变化检测.png` | `Temp/scripts/make_fig033.py` | ✅ 通过（check_figure strict=True） |

Fig033 为自建示意图（随机种子 20260821）：(a) 变点信号与分段均值（$A_1=1\to A_2=4$，$n_0=50$，$\sigma^2=1$，$N=100$，自建数据）；(b) GLRT 统计量（式 12.11）随候选 $n_0$ 扫描、峰值逼近真值；(c) 动态规划最短路径示意（边权为**示意值**，非原书图 12.5 的逐点复刻，对应"每中间节点只留一条最短入边"的 DP 原理）。数据与边权均为自建示意，非原书实测数字。

## 4. 未决疑问

- 式（12.4）中间步骤 OCR（ocr_page_814.txt L6~9）呈现"$-2\Delta A(x[n]-A_0)+\Delta A^2$"与"$2\sigma^2$"的分式结构可辨，正文据此转录；但（12.4）到（12.5）的"除以 $N-n_0$ 取平均"一步在原书是"把与数据无关的项并入门限"，正文已按此表述，未逐式复现原书可能的中间形式。
- 例 12.4 的判决统计量 $T(\mathbf{x})=\max_{n_0}\bigl[\sum(x[n]-A_0)-\frac{\Delta A}{2}(N-n_0)\bigr]$ 是正文把 (12.4) 代入"$\max_{n_0}\ln L(\mathbf{x};n_0)$"后、约去正常数 $\Delta A/\sigma^2$ 得到的形式；OCR 页 819 L38~52 显示的分式结构可辨，但 $\Delta A/2$ 的系数位置 OCR 略模糊，正文按 $\ln L=(\Delta A/\sigma^2)\sum(x-A_0)-(\Delta A^2/(2\sigma^2))(N-n_0)$ 化简，与 (12.4) 自洽。建议回英文原版 §12.4 复核该系数的精确写法。
- 图 12.2(b) 的 18/4 样本延迟是"该现实里首次过门限"的样本数（现实依赖），非期望值；正文已明确标注，避免读者误当通用公式。
