# Vol2_Ch09 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch09_未知噪声参数.md`
> OCR 来源：`Temp/chapters_ocr/v2ch09/ocr_page_730~764.txt`（PDF 730~764，书内 715~749）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §9.1 引言"噪声 PDF 除少数参数外已知""双复合假设检验" | ocr_page_730.txt L5~10 | 一致 |
| 门限反解 $P_{FA}=\Pr\{T(x)>\gamma;H_0\}$ | ocr_page_730.txt L24~30 | 一致 |
| 式 (9.1) 样本均值统计量；式 (9.2) $\gamma'=\sqrt{\sigma^2/N}Q^{-1}(P_{FA})$ | ocr_page_731.txt L1~10 | 一致（OCR 中 (9.2) 式排版散，按语义拼合） |
| "即估即用 estimate and plug"；信号引入偏差 | ocr_page_731.txt L11~18 | 一致 |
| 参考数据 $w_R[n]$、$\hat\sigma_R^2$；统计量学生 t；CFAR 定义 | ocr_page_731.txt L18~33 | 一致 |
| "并不知道它是否是使 $P_D$ 最大的最佳检验统计量" | ocr_page_731.txt L31~33 | 一致 |
| 式 (9.3) 模型；"没有最佳解的复合假设检验问题" | ocr_page_731.txt L34~40、ocr_page_732.txt L1~6 | 一致 |
| 例 9.1 MLE $\hat\sigma_0^2=\Sigma x^2/N$、$\hat\sigma_1^2=\Sigma(x-A)^2/N$ | ocr_page_732.txt L8~19 | 一致 |
| "GLRT 是在两种假设下估计 $\sigma^2$；不同于即估即用" | ocr_page_732.txt L11~12 | 一致 |
| "在 $H_0$ 下 PDF 与 $\sigma^2$ 有关，不能建立门限；GLRT 不是 CFAR 甚至不是渐近 CFAR" | ocr_page_733.txt L9~12 | 一致 |
| $A$ 也未知时 GLRT 渐近 CFAR（例 6.5/6.7，$\chi_1^2$、$\lambda=NA^2/\sigma^2$） | ocr_page_733.txt L13~27 | 一致 |
| §9.4.1 已知信号 $\hat\sigma_1^2=\Sigma(x-s)^2/N$；"惟一的差别是归一化因子"；非 CFAR；补救（习题 9.9） | ocr_page_734.txt L14~33、ocr_page_735.txt L1~31 | 一致 |
| §9.4.2 随机信号 (9.7) $J(\sigma^2)$；"不能用解析方法求解"；弱信号近似 (9.9)(9.10) | ocr_page_735.txt L32~59、ocr_page_736.txt L1~33 | 一致 |
| "渐近 PDF 不能由 6.5 节标准形式给出" | ocr_page_736.txt L33~34 | 一致 |
| 定理 9.1（式 9.12~9.15）：模型、F 分布、非中心参量 | ocr_page_737.txt L1~36 | 一致 |
| "由于 $P_{FA}$ 不依赖于 $\sigma^2$，检验得到的是 CFAR" | ocr_page_737.txt L32~33 | 一致 |
| 例 9.2 $F_{1,N-1}$；$\lambda=NA^2/\sigma^2$ | ocr_page_738.txt L1~33 | 一致 |
| 例 9.3 检验 $\theta=0$；(9.16) 信号/噪声子空间；$E[(P_H w)(P_H^\perp w)^T]=0$ | ocr_page_738.txt L34~43、ocr_page_739.txt L1~23 | 一致 |
| 例 9.4 Fisher 信息奇异（$A=0$ 估不出 $\alpha$）；"Rao 检验限定为除幅度外都已知" | ocr_page_739.txt L24~43、ocr_page_740.txt L1~16 | 一致 |
| §9.5 AR(1) 模型 (9.17)；ACF (9.18)；PSD (9.19) | ocr_page_740.txt L28~39、ocr_page_741.txt L1~6 | 一致 |
| 图 9.3 $a[1]=-0.9$、$\sigma_u^2=2$（低频有色） | ocr_page_741.txt L7~65 | 一致（图内容，正文未逐点转录） |
| §9.5.1 渐近似然；AR 参数 MLE (9.20)(9.21)；GLRT (9.22) | ocr_page_742.txt L1~40 | 一致 |
| "预白化"解释（$A_0(f)$ 输出抬高 $\hat\sigma$ 估计） | ocr_page_743.txt L14~21 | 一致 |
| §9.5.2 Rao 检验 (9.26)~(9.31)；"归一化的非相干广义匹配滤波器" | ocr_page_744.txt L1~36、ocr_page_745.txt L1~38 | 一致 |
| 块对角 Fisher 信息 → 非中心参量不减少 (9.35) | ocr_page_745.txt L49~54、ocr_page_746.txt L1~19 | 一致 |
| 定理 9.2（式 9.36）：一般线性模型 Rao 检验；渐近性能与噪声参数已知相同 | ocr_page_746.txt L30~39、ocr_page_747.txt L1~14 | 一致 |
| §9.6 LFM 信号 $p(t)=\cos[2\pi(F_0 t+\frac12 kt^2)]$；宽带高分辨率；噪声颜色重要 | ocr_page_747.txt L15~30 | 一致 |
| 线性模型 (9.37)；Rao 检验 (9.38)；渐近 (9.39) | ocr_page_747.txt L31~39、ocr_page_748.txt L1~26 | 一致 |
| 近似统计量 (9.40)（预白化 + 相关 + 平方 + 归一化）；图 9.6 | ocr_page_748.txt L27~49、ocr_page_749.txt L1~11 | 一致 |
| 数值算例 $A=0.5,f_0=0.05,m=0.0015,\Phi=0,N=100$，扫频 $0.05\to0.2$，$a[1]=0.95,\sigma_u^2=1$ | ocr_page_749.txt L31~35 | 一致 |
| 图 9.8 短数据 $N=100$ 蒙特卡洛与渐近吻合；图 9.9 Rao 优于匹配滤波器；匹配滤波器"由于 CFAR 缺乏需归一化" | ocr_page_751.txt L12~23 | 一致 |
| 非中心参量 (9.41) | ocr_page_748.txt L49~50、ocr_page_749.txt L4~9 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **式 (9.4)/(9.5)**（ocr_page_732.txt L33~41、ocr_page_733.txt L1~12）：OCR 中 (9.4) 的分子分母碎片（"$\bar{x}-A/2$""$2A$""$\Sigma(x[n]-A)^2$"）与 (9.5) 的渐近系数（"$2\sigma^2+NA^2$""$2\sigma^2\sqrt{N\sigma^2}$"）残缺。正文据 GLRT 单调等价统计量校订：
   $$T(\mathbf{x}) = \frac{\bar{x}-A/2}{\frac1N\sum_{n=0}^{N-1}(x[n]-A)^2}$$
   理由：$2\ln L_G = N\ln(\Sigma x^2/\Sigma(x-A)^2)$，且 $\Sigma x^2 = \Sigma(x-A)^2 + 2AN(\bar{x}-A/2)$，故 $2\ln L_G$ 单调等价于 $(\bar{x}-A/2)/[\frac1N\Sigma(x-A)^2]$；与 OCR 的"$\bar{x}-A/2$"分子、"$\Sigma(x-A)^2$"分母、"$2A$"因子（来自 $2A(\bar{x}-A/2)$）吻合。(9.5) 的精确系数正文未逐式转录，仅给出"$H_0$ 下 PDF 含 $\sigma^2$ → 非 CFAR"的结论（该结论 OCR 明确给出，无争议）。

2. **式 (9.16) 的投影矩阵记号**（ocr_page_739.txt L1~23）：OCR 中 $P_H$、$P_H^\perp$ 与范数符号有缺字，正文据正交投影矩阵标准定义校订为 $P_H=\mathbf{H}(\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T$、$P_H^\perp=\mathbf{I}-P_H$、$T=\frac{N-p}{p}\|P_H\mathbf{x}\|^2/\|P_H^\perp\mathbf{x}\|^2$。理由：与 OCR"将矢量投影到 H 列的正交投影矩阵"及"计算这两个矢量平方长度之比"一致。

3. **式 (9.31)/(9.32) 的矩阵形式**（ocr_page_745.txt L33~48）：OCR 中该两式排版严重残缺，正文据"归一化非相干广义匹配滤波器"的标准形式校订为 $T_R=(\mathbf{s}^T\mathbf{C}^{-1}\mathbf{x})^2/(\mathbf{s}^T\mathbf{C}^{-1}\mathbf{s})$ 及时域等价形式。理由：与 OCR 文字"$(\mathbf{s}^T\mathbf{C}^{-1}(\theta_{w0})\mathbf{x})^2$""$\mathbf{s}^T\mathbf{C}^{-1}(\theta_{w0})\mathbf{s}$"片段及"归一化非相干广义匹配滤波器"一致。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig030 | `Documents/figures/Fig030_CFAR门限自适应.png` | `Temp/scripts/make_fig030.py` | ✅ 通过（check_figure strict=True） |

Fig030 为自建示意（固定门限 vs 自适应门限的 $P_{FA}$ 对比，$\sigma^2=1/4$、$P_{FA}$ 目标 0.10）。图中 $P_{FA}=0.10$ 与 $0.26$ 是 $Q(1.28)=0.10$、$Q(1.28/2)=Q(0.64)=0.26$ 的自算派生值，仅作示意，非 OCR 原书数值；已在图注中说明"自建示意、对应式 9.2"。

## 4. 未决疑问

- 式 (9.4)(9.5) 的精确归一化系数在扫描件中残缺（ocr_page_732 L33~41、ocr_page_733 L1~12），正文按 GLRT 单调等价统计量校订了 (9.4)，(9.5) 只给结论未逐式转录。若需 (9.4)(9.5) 的逐系数精确形式，建议回英文原版 §9.3 核对。
- 式 (9.8)（§9.4.2 随机信号的 $\sigma^2$ MLE 方程）在 ocr_page_736.txt L1~8 残缺，正文未逐式转录，仅给出"非线性方程无解析解"的结论与弱信号近似 (9.9)(9.10)。
- 式 (9.25)（§9.5.1 的 $H_0$ 下 AR 参数估计的时域形式）在 ocr_page_743.txt L5~14 残缺，正文按 (9.24) 的对称形式（用 $\mathbf{x}$ 代替 $\mathbf{w}$）转述，未逐式转录 (9.25)。
