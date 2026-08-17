# Vol2_Ch07 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch07_具有未知参数的确定性信号.md`
> OCR 来源：`Temp/chapters_ocr/v2ch07/ocr_page_661~703.txt`（PDF 661~703，书内 646~688）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §1 引言"最重要是不完全已知信号"、通用模型 $s[n]=A\cos(2\pi f_0n+\Phi)$ | ocr_page_661.txt L5~27 | 一致 |
| §1.2 GLRT vs 贝叶斯；"GLRT 只要求最大值而非积分"；UMP 通常不存在 | ocr_page_661.txt L28~35、ocr_page_662.txt L1~15 | 一致 |
| 式 (7.1) $P_D=Q(Q^{-1}(P_{FA})-\sqrt{d^2})$，$d^2=E/\sigma^2$ | ocr_page_662.txt L16~22 | 一致 |
| §2 完全未知信号 → 能量检测器；式 (7.3) $\sum x^2[n]$；式 (7.4) 估计器-相关器 $\hat{s}[n]=x[n]$ | ocr_page_662.txt L23~39、ocr_page_663.txt L1~20 | 一致 |
| 式 (7.5) $d^2_{ED}$、式 (7.6) $d^2_{MF}$ | ocr_page_663.txt L21~31（偏移系数，附录 7A 印证） | 一致 |
| 处理增益 $10\log_{10}N$ vs $5\log_{10}N$；"N=1000 时损耗 11.5 dB" | ocr_page_663.txt L31~49、ocr_page_664.txt L1~15 | 一致 |
| 图 7.1 输入 SNR 需求；MF/ED 两种 $\eta$ 表达式 | ocr_page_664.txt L4~9 | 一致 |
| §3 未知幅度模型；UMP 存在当符号已知（式 7.9/7.10） | ocr_page_664.txt L43~49、ocr_page_665.txt L1~34 | 一致 |
| 式 (7.11) $\hat{A}=\sum x[n]s[n]/\sum s^2[n]$ | ocr_page_666.txt L8~15 | 一致 |
| 式 (7.13) 相关器平方 $T=(\sum x[n]s[n])^2/\sum s^2[n]$ | ocr_page_666.txt L40~53 | 一致 |
| 式 (7.16) 双 $Q$ 和 + "低 $P_{FA}$ 约 0.5 dB" | ocr_page_667.txt L24~42、ocr_page_668.txt L4~9 | 一致 |
| 图 7.3 GLRT vs 透视 | ocr_page_668.txt L9~36 | 一致 |
| 式 (7.18) 贝叶斯"相关器+平方相关器" | ocr_page_668.txt L39~55、ocr_page_669.txt L1~16 | 校订（OCR 残缺，见 §2） |
| §4 未知到达时间；式 (7.19)(7.21) 扫描相关取最大 | ocr_page_669.txt L20~37、ocr_page_670.txt L39~53 | 一致 |
| 式 (7.22) 频域；式 (7.23) 幅度也未知 | ocr_page_671.txt L12~26 | 一致 |
| 性能"N-M+1 个相关高斯最大值"难算 | ocr_page_670.txt L55~57 | 一致 |
| §5 正弦四档阶梯（表） | ocr_page_671.txt L28~49 | 一致 |
| 式 (7.24) 相关器平方（$f_0,\Phi$ 已知） | ocr_page_672.txt L6~23 | 一致 |
| 式 (7.25) 周期图；"正交/非相干匹配滤波器" | ocr_page_675.txt L30~51 | 一致 |
| 式 (7.26) $P_{FA}=\exp(-\gamma'/\sigma^2)$；式 (7.28) $\lambda=NA^2/(2\sigma^2)$ | ocr_page_676.txt L23~38 | 一致 |
| "图 7.6(b) 比 (a) 衰减小于 1 dB" | ocr_page_676.txt L38~41 | 一致 |
| 式 (7.30) 周期图峰值；"FFT 是窄带检测基本组成" | ocr_page_677.txt L30~37 | 一致 |
| 式 (7.31) 频率搜索修正（$N/2-1$ 单元） | ocr_page_677.txt L37~44 | 一致 |
| 式 (7.33) 谱图峰值；图 7.7 实况（$A=1,f_0=0.25,M=128,n_0=128,N=512,\sigma^2=0.5$、$\hat f_0=0.25$、$\hat n_0=141$） | ocr_page_678.txt L21~45、ocr_page_679.txt L1~6 | 一致 |
| §6 经典线性模型 (7.34)；定理 7.1（式 7.35） | ocr_page_680.txt L4~23、ocr_page_681.txt L26~36、ocr_page_682.txt L1~11 | 一致 |
| 例 7.3 → 与 (7.16) 一致 | ocr_page_683.txt L1~17 | 一致 |
| 例 7.4 DC 偏移补偿；"相关前减样本均值" | ocr_page_683.txt L18~31、ocr_page_684.txt、ocr_page_685.txt L1~61 | 一致 |
| §7.2 例 7.5 声纳/雷达；式 (7.36)(7.37)(7.38) | ocr_page_686.txt L5~47、ocr_page_687.txt、ocr_page_688.txt L1~36 | 一致 |
| 设计算例：5000 码 / 5000 ft/s / 6 秒 / 2000 样本/s / 12000 样本 / $L=6000$ / $N\approx350$ | ocr_page_689.txt L8~22 | 一致 |
| 例 7.6 干扰；式 (7.39)(7.40) DFT 单元作废 | ocr_page_690.txt L3~26、ocr_page_691.txt、ocr_page_692.txt L1~56、ocr_page_693.txt L1~25 | 一致 |
| 附录 7A $d^2=(E/\sigma^2)^2/(2N)$ | ocr_page_700.txt、ocr_page_701.txt L8~14 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **式 (7.18) 贝叶斯检测器**（ocr_page_668.txt L39~55、ocr_page_669.txt L1~16）：OCR 呈现"$T(x) = \times^T (HC_eH^T + C)^{-1} H\mu_\theta + \ldots$"与"$HA/(2\sigma^2(\sigma^2+\sigma_A^2 s^T s))$"等残缺字样。据（5.30）式 + Woodbury 恒等式校订为
   $$T(\mathbf{x})=\frac{\mu_A}{\sigma^2+\sigma_A^2\mathbf{s}^T\mathbf{s}}\mathbf{x}^T\mathbf{s}+\frac{\sigma_A^2}{2\sigma^2(\sigma^2+\sigma_A^2\mathbf{s}^T\mathbf{s})}(\mathbf{x}^T\mathbf{s})^2$$
   理由：原书明说"由（5.30）式……相关器（$x^Ts$）和平方相关器的组合"，(5.30) 式在 Vol2_Ch05 已有清晰 OCR；令 $\boldsymbol{\mu}_s=\mathbf{s}\mu_A$、$\mathbf{C}_s=\sigma_A^2\mathbf{s}\mathbf{s}^T$、$\mathbf{C}_w=\sigma^2\mathbf{I}$ 代入 (5.30) 即得上式。若 $\mu_A=0$ 退化为平方相关器，与习题 7.12 一致。

2. **式 (7.5) $d^2_{ED}$ 的写法**：OCR 页 663 的"$Q_{ED}$ 2N"残缺，据附录 7A（ocr_page_700~701）校订为 $d^2_{ED}=(E/\sigma^2)^2/(2N)$。理由：附录 7A 末尾明确给出"$d^2 = (E/\sigma^2)^2/(2N)$，由于 $\lambda=E/\sigma^2$"。

3. **式 (7.12)→(7.13) 中间"除以 $\sum s^2[n]$"**：OCR 页 666 的 (7.12)~(7.14) 排版残缺（分母 $\sum s^2[n]$ 被 OCR 拆散），据 (7.11) 代入 (7.8) 的标准化简校订。理由：(7.13) 的分子分母结构与 (7.16) 的 $d^2=A^2\sum s^2[n]/\sigma^2$ 自洽。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig028 | `Documents/figures/Fig028_能量检测器.png` | `Temp/scripts/make_fig028.py` | ✅ 通过（check_figure strict=True） |

Fig028 为自建数据流示意（无实测数值，纯结构图）：(a) 相关器平方（未知幅度 GLRT，式 7.13）；(b) 能量检测器（式 7.3）。底注的"0.5 dB / 11.5 dB"为正文结论的原样引用，非图内实测。

## 4. 未决疑问

- 式 (7.7)（OCR 页 663 L37~43 的"$10\log_{10}\ldots=3-10\log_{10}2$"）在扫描件中残缺，正文未逐式转录，仅采用其结论"处理增益 $10\log_{10}N$ vs $5\log_{10}N$"与"N=1000→11.5 dB"。若需 (7.7) 精确形式（损失 $=10\log_{10}(2/\eta)$ 的 DC 情形），建议回英文原版 §7.3 核对。
- 式 (7.31) 的精确非中心卡方自变量（OCR 页 677 L37~44 呈现"$Q_{\chi^2}\ldots 2\ln\frac{1}{1-(1-P_{FA})^{1/(N/2-1)}}$"的部分残缺），正文按标准频率搜索修正形式表述，与 (7.37) 的一阶近似 $P_{FA}\approx L\,P_{FA}(\text{单元})$ 自洽。
- 例 7.5 的"约 $N=350$"是原书由 (7.38) 反解得到的近似值（OCR 页 689 L22 明确"约是 N=350"），非精确闭式解；正文照录并注明"约"。
