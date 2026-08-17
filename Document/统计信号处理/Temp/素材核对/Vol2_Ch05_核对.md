# Vol2_Ch05 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch05_随机信号.md`
> OCR 来源：`Temp/chapters_ocr/v2ch05/ocr_page_578~612.txt`（PDF 578~612，书内 563~597）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §1.1 语音例："声意波形"、协方差已知 | ocr_page_578.txt L5~9 | 一致（"声意"为"声音/语音"OCR 讹误，见 §2） |
| 式 (5.1) $T(\mathbf{x})=\sum x^2[n]$ 能量检测器 | ocr_page_578.txt L28~34、ocr_page_579.txt L29~33 | 一致 |
| 式 (5.2)(5.3) $P_{FA}=Q_{\chi^2_N}(\gamma'/\sigma^2)$、$P_D=Q_{\chi^2_N}(\gamma'/(\sigma_s^2+\sigma^2))$ | ocr_page_579.txt L38~44、ocr_page_580.txt L3~9 | 一致（右尾记号据第 2 章整理） |
| 图 5.1 能量检测器性能（N=25，SNR −20~10 dB） | ocr_page_580.txt L16~40 | 一致 |
| SNR 定义 $\sigma_s^2/\sigma^2$，$P_D$ 随其单调递增 | ocr_page_580.txt L10~16 | 一致 |
| 式 (5.5)(5.6) 估计器-相关器 $T=\mathbf{x}^T\mathbf{C}_s(\mathbf{C}_s+\sigma^2\mathbf{I})^{-1}\mathbf{x}$、$\hat{\mathbf{s}}=\mathbf{C}_s(\mathbf{C}_s+\sigma^2\mathbf{I})^{-1}\mathbf{x}$ | ocr_page_581.txt L18~29 | 一致 |
| 式 (5.7) MMSE $\hat{\boldsymbol{\theta}}=\mathbf{C}_{\theta x}\mathbf{C}_{xx}^{-1}\mathbf{x}$（维纳滤波） | ocr_page_581.txt L31~37 | 一致 |
| 图 5.2 估计器-相关器 | ocr_page_582.txt L6~13 | 一致 |
| 例 5.3 相关信号 $N=2$、相关系数 $\rho$ | ocr_page_582.txt L30~31、ocr_page_583.txt L1~31 | 一致 |
| 式 (5.9) 标准形 $\sum \lambda_{s_n}/(\lambda_{s_n}+\sigma^2)y^2[n]$ | ocr_page_584.txt L16~21 | 一致 |
| 图 5.3 标准形式；图 5.4 PDF 等值线 | ocr_page_584.txt L36~40、ocr_page_585.txt L19~25 | 一致 |
| 式 (5.10)(5.11) 特征函数逆的性能积分 | ocr_page_585.txt L31~44、ocr_page_586.txt L1~5 | 一致（被积式 OCR 残缺，据附录 5A 校订） |
| 例 5.4 成对特征值（N=2M，块对角） | ocr_page_586.txt L7~41、ocr_page_587.txt L1~36 | 一致 |
| 复数据下任意 $\mathbf{C}_s$ 性能可用 (5.14)(5.15) 求 | ocr_page_587.txt L38~40 | 一致 |
| 式 (5.16)(5.17) 有色噪声 $T=\mathbf{x}^T\mathbf{C}_w^{-1}\hat{\mathbf{s}}$、$\hat{\mathbf{s}}=\mathbf{C}_s(\mathbf{C}_s+\mathbf{C}_w)^{-1}\mathbf{x}$ | ocr_page_587.txt L41~45、ocr_page_588.txt L1~5 | 一致 |
| §5.4 贝叶斯线性模型 $\mathbf{x}=\mathbf{H}\boldsymbol{\theta}+\mathbf{w}$ | ocr_page_588.txt L17~23 | 一致 |
| 式 (5.18) $T=\mathbf{x}^T\mathbf{H}\mathbf{C}_\theta\mathbf{H}^T(\mathbf{H}\mathbf{C}_\theta\mathbf{H}^T+\sigma^2\mathbf{I})^{-1}\mathbf{x}$ | ocr_page_588.txt L30~32 | 一致 |
| 例 5.5 瑞利衰落正弦 $s[n]=A\cos(2\pi f_0 n+\Phi)$、$[a\ b]^T\sim\mathcal{N}(0,\sigma_A^2\mathbf{I})$ | ocr_page_588.txt L35~41、ocr_page_589.txt L1~22 | 一致 |
| ACF $r_{ss}[k]=\sigma_A^2\cos 2\pi f_0 k$、瑞利幅度 | ocr_page_590.txt L1~13 | 一致 |
| $\mathbf{H}^T\mathbf{H}\approx(N/2)\mathbf{I}$（$N$ 大、$0<f_0<1/2$） | ocr_page_590.txt L40~56 | 一致 |
| 式 (5.20) 周期图 $T'=\frac1N|\sum x[n]e^{-j2\pi f_0 n}|^2$ | ocr_page_591.txt L14~35 | 一致 |
| 图 5.6 正交/非相干匹配滤波器、周期图检测器 | ocr_page_592.txt L26~40 | 一致 |
| 式 (5.21)(5.22) $P_{FA}=\exp(-\gamma'/\sigma^2)$、$P_D=\exp(-\gamma'/(N\sigma_A^2/2+\sigma^2))$ | ocr_page_592.txt L15~23、ocr_page_593.txt L4~8 | 一致 |
| 式 (5.23) $P_D=P_{FA}^{1/(1+\bar\eta)}$，$\bar\eta=N\sigma_A^2/(2\sigma^2)$ | ocr_page_593.txt L10~15 | 一致 |
| 图 5.7 检测性能增长缓慢；图 5.8 瑞利幅度 PDF | ocr_page_593.txt L16~46 | 一致 |
| 例 5.6 非相干 FSK；式 (5.24) $I(f_1)>I(f_0)$ | ocr_page_594.txt L3~34 | 一致 |
| 式 (5.25) $P_e=1/(2+\bar\eta)$ | ocr_page_596.txt L9~15 | 一致（积分结果，OCR 分式残缺，见 §2） |
| 图 5.10 衰落 vs 无衰落性能对比 | ocr_page_596.txt L17~43 | 一致 |
| §5.5 式 (5.26) 渐近对数 PDF | ocr_page_597.txt L4~16 | 一致 |
| 式 (5.27) $T=N\int P_{ss}(f)/(P_{ss}(f)+\sigma^2)I(f)df$ | ocr_page_597.txt L36~41 | 一致 |
| 式 (5.28)(5.29) $H(f)=P_{ss}/(P_{ss}+\sigma^2)$、频域相关 | ocr_page_597.txt L42~53、ocr_page_598.txt L1~8 | 一致 |
| §5.6 式 (5.30) 一般高斯 $T=\mathbf{x}^T(\mathbf{C}_s+\mathbf{C}_w)^{-1}\boldsymbol{\mu}_s+\tfrac12\mathbf{x}^T\mathbf{C}_w^{-1}\mathbf{C}_s(\mathbf{C}_s+\mathbf{C}_w)^{-1}\mathbf{x}$ | ocr_page_598.txt L27~33 | 一致 |
| (5.30) 两特例：匹配滤波 / 估计器-相关器 | ocr_page_599.txt L3~9 | 一致 |
| 例 5.7 平均器 + 能量检测器 | ocr_page_599.txt L11~24 | 一致 |
| §5.7 式 (5.31) TDL；式 (5.32) $\mathbf{h}\sim\mathcal{N}(0,\mathbf{C}_h)$ | ocr_page_600.txt L4~7、L20~27 | 一致 |
| 式 (5.33) $T=\mathbf{x}^T\mathbf{H}\mathbf{C}_h\mathbf{H}^T(\mathbf{H}\mathbf{C}_h\mathbf{H}^T+\sigma^2\mathbf{I})^{-1}\mathbf{x}$ | ocr_page_601.txt L22~24 | 一致 |
| 式 (5.34) $\mathbf{H}^T\mathbf{H}\approx\mathcal{E}\mathbf{I}$（PRN） | ocr_page_601.txt L31~33 | 一致 |
| 式 (5.35) 非相干多路径组合器 $\sum \sigma_{h_k}^2/(\sigma_{h_k}^2+\sigma^2/\mathcal{E})z^2[k]$ | ocr_page_602.txt L19~24 | 一致 |
| 图 5.12 组合器；图 5.13 延迟散射函数 | ocr_page_603.txt L10~28、L31~34 | 一致 |
| §7.3 $p=4$ 例（$\sigma_{h_0}^2=\sigma_{h_1}^2=1/6,\sigma_{h_2}^2=\sigma_{h_3}^2=1/3$）、平均能量 $E=\mathcal{E}$ | ocr_page_603.txt L30~40 | 一致 |
| 式 (5.37)(5.38) 检测性能 | ocr_page_604.txt L4~34 | 略（OCR 中间分式残缺，见 §4） |
| "$P_D=0.95$ 时信噪比小 4.3 dB"（分集增益） | ocr_page_605.txt L3~6 | 一致（OCR "4.3lB" 为 "4.3 dB"） |
| 图 5.14 随机 TDL vs 瑞利衰落 | ocr_page_605.txt L7~29 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **"声意波形"**（ocr_page_578.txt L6）：OCR 把"声音"或"语音"误识为"声意"，按文义（讨论语音信号）校订为"声音/语音波形"。理由：英文原版此处为 speech waveform 语境。
2. **式 (5.25) $P_e=1/(2+\bar\eta)$**：ocr_page_596.txt 的积分与分式残缺（仅存 "2+η"、"Nσ_A²/2+σ²"），据标准结果（非相干 FSK 经瑞利衰落的经典错误概率 $P_e=1/(2+\bar\eta)$，$\bar\eta$ 为平均 SNR）与 OCR 残留的 "2+η" 校订。理由：该式是"$H_0$ 下周期图指数分布、$H_1$ 下另一指数分布"积分的标准闭式，与 OCR 页 596 的积分结构吻合。
3. **式 (5.10)(5.11) 的性能积分**：ocr_page_585~586 的特征函数/反傅里叶表达式大量残缺，据附录 5A（ocr_page_611~612）校订为 $\alpha_n=\lambda_{s_n}\sigma^2/(\lambda_{s_n}+\sigma^2)$、被积函数 $\prod(1-2j\alpha_n\omega)^{-1/2}$（$P_{FA}$）与 $\prod(1-2j\lambda_{s_n}\omega)^{-1/2}$（$P_D$）的反傅里叶。理由：附录 5A 明确给出 $T(\mathbf{x})=\sum\alpha_n z^2[n]$（$z[n]\sim\chi^2_1$）与 $\chi^2_1$ 的特征函数 $(1-2j\omega)^{-1/2}$，反推即得。
4. **"4.3 dB"**（ocr_page_605.txt L5）：OCR 呈现 "4.3lB"，"l" 为 "d" 的误识，校订为 "4.3 dB"。
5. **例 5.1 中"称为噪声实际上是误称"**（ocr_page_578.txt L29）：原文说把白高斯信号 $s[n]$ 称为"噪声"是误称（因它是信号而非噪声）。正文按此口径转述，未引具体数字。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig026 | `Documents/figures/Fig026_估计器相关器数据流.png` | `Temp/scripts/make_fig026.py` | ✅ 通过（check_figure strict=True） |

Fig026 为自建数据流示意（无实测数值，纯结构图）：(a) 直接形式"数据→维纳滤波→相关→判决"；(b) 标准形"去相关→加权能量→判决"。参数 $\lambda_{s_n}$、$\sigma^2$ 均为示意符号，未引实测数值。

## 4. 未决疑问

- 式 (5.37)(5.38)（TDL 例子的 $P_{FA}$/ $P_D$ 闭式）OCR 页 604 中间分式（$A_0$、$B_0$ 及部分分式展开）大量残缺，仅"$P_D=2e^{-3\bar\eta/2}-e^{-3\bar\eta}$"的化简结果与"$4.3$ dB"结论可辨；但该 $P_D$ 形式在 $\bar\eta\to0$ 时趋 1、不含门限 $\gamma'$，疑似为固定 $P_{FA}$（图 5.14 的 $10^{-5}$）下的特定表达，OCR 不足以可靠还原其完整口径。**正文因此未逐式转录 (5.37)(5.38)，只保留定性的"4.3 dB 分集增益"结论。** 若需精确公式，建议回英文原版 §5.7 核对。
