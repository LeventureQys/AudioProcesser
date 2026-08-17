# Vol2_Ch04 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch04_确定信号.md`
> OCR 来源：`Temp/chapters_ocr/v2ch04/ocr_page_541~577.txt`（PDF 541~577，书内 526~562）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| 式 (4.1) $H_0/H_1$ 已知信号 + WGN | ocr_page_541.txt L27~31 | 一致 |
| 式 (4.3) $T(\mathbf{x})=\sum x[n]s[n]>\gamma'$（仿形-相关器） | ocr_page_542.txt L58~64 | 一致 |
| 式 (4.5) $h[n]=s[N-1-n]$（镜像） | ocr_page_543.txt L30~33 | 一致 |
| 例 4.2 衰减指数 $s[n]=r^n,\ 0<r<1$ | ocr_page_543.txt L12~15 | 一致 |
| 式 (4.7) $H(f)=S^*(f)e^{-j2\pi f(N-1)}$；无噪声输出 = 能量 | ocr_page_544.txt L30~33；ocr_page_545.txt L4~6 | 一致 |
| 式 (4.10) 输出 SNR；Cauchy-Schwarz → 最大 SNR $=E/\sigma^2$ | ocr_page_545.txt L19~32；ocr_page_546.txt L3~14 | 一致 |
| 图 4.4 $T\sim\mathcal{N}(0,\sigma^2E)$（H0）/ $\mathcal{N}(E,\sigma^2E)$（H1） | ocr_page_547.txt L4~7 | 一致 |
| 式 (4.14) $P_D=Q(Q^{-1}(P_{FA})-\sqrt{E/\sigma^2})$ | ocr_page_547.txt L29~33 | 一致 |
| 图 4.5 与图 3.5 相同；波形无关（图 4.6） | ocr_page_548.txt L3~9 | 一致 |
| 处理增益 $PG=10\log_{10}N$ dB | ocr_page_549.txt L3~5 | 一致 |
| $P_{FA}=10^{-3}$：$P_D=0.5$ 需 ENR≈10dB，$P_D=0.95$ 需 +4dB（$N\times2.5$） | ocr_page_549.txt L8~12 | 一致 |
| 式 (4.15) $d^2=E/\sigma^2$ = 输出 SNR | ocr_page_549.txt L13~18 | 一致 |
| 式 (4.16) $T(\mathbf{x})=\mathbf{x}^T\mathbf{C}^{-1}\mathbf{s}$（广义匹配滤波器） | ocr_page_550.txt L12~15 | 一致 |
| 例 4.3 不等方差加权 $1/\sigma_n^2$；预白化 | ocr_page_550.txt L25~42 | 一致 |
| 式 (4.17) 频域大 N 近似 $\int X(f)S^*(f)/P_{ww}(f)df$ | ocr_page_551.txt L18~23 | 一致 |
| 式 (4.18) $P_D=Q(Q^{-1}(P_{FA})-\sqrt{\mathbf{s}^T\mathbf{C}^{-1}\mathbf{s}})$ | ocr_page_552.txt L17~20 | 一致 |
| 例 4.5 选 $\mathbf{C}$ 最小特征值特征矢量；正相关→差分 $[1,-1]$ | ocr_page_553.txt L26~35；ocr_page_554.txt L3~14 | 一致 |
| 式 (4.19) $d^2=\int\lvert S(f)\rvert^2/P_{ww}(f)df$ | ocr_page_554.txt L16~20 | 一致 |
| 式 (4.20) 最小距离 $D_i^2=\sum(x[n]-s_i[n])^2$ | ocr_page_555.txt L30~34 | 一致 |
| 式 (4.21) $T_i(\mathbf{x})=\sum x[n]s_i[n]-\tfrac12\sum s_i^2[n]$ | ocr_page_556.txt L19~27 | 一致 |
| 式 (4.23) $P_e=Q(\sqrt{\lVert s_1-s_0\rVert^2/(4\sigma^2)})$ | ocr_page_558.txt L13~17（OCR 乱码，见 §2 校订） | 校订 |
| 式 (4.24) $\rho_s=s_1^T s_0/[\tfrac12(s_1^T s_1+s_0^T s_0)]$；式 (4.25) $P_e=Q(\sqrt{\bar{\mathcal{E}}(1-\rho_s)/(2\sigma^2)})$ | ocr_page_558.txt L24~34 | 一致 |
| 例 4.7 PSK：$\rho_s=-1$，$P_e=Q(\sqrt{E/\sigma^2})$，$10^{-9}$ 需约 15dB | ocr_page_559.txt L10~12 | 一致 |
| 例 4.8 FSK：$\rho_s\approx0$，$P_e=Q(\sqrt{E/(2\sigma^2)})$，能量 2 倍（+3dB） | ocr_page_560.txt L6~9 | 一致 |
| 式 (4.26) $T_i(\mathbf{x})=\sum x[n]s_i[n]$ 选最大 | ocr_page_560.txt L15~19 | 一致 |
| 式 (4.28) M 元正交等能量 $P_e$（积分式） | ocr_page_561.txt L41~51（OCR 乱码，见 §2 校订） | 校订 |
| 式 (4.29) $T(\mathbf{x})=\mathbf{x}^T\mathbf{C}^{-1}\mathbf{H}\boldsymbol{\theta}_1$ | ocr_page_563.txt L16~18 | 一致 |
| 式 (4.30) $\hat{\boldsymbol{\theta}}^T\mathbf{C}_{\hat{\theta}}^{-1}\boldsymbol{\theta}_1$；例 4.9 正弦 | ocr_page_564.txt L11~28 | 一致 |
| 式 (4.31) $P_D=Q(Q^{-1}(P_{FA})-\sqrt{\boldsymbol{\theta}_1^T\mathbf{C}_{\hat{\theta}}^{-1}\boldsymbol{\theta}_1})$ | ocr_page_565.txt L7~9 | 一致 |
| 例 4.10 门限 $PT/(\Delta\sigma^2)=2\ln 2$；信道容量 $C=P/(2\sigma^2\Delta\ln 2)$ | ocr_page_565.txt L39~40；ocr_page_566.txt L1~5、L23~29 | 一致 |
| 例 4.11 四灰度 50×50 图像，$\sigma^2=0.5$，3×3/5×5 窗口 | ocr_page_567.txt L27~34；ocr_page_568.txt L3~6 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **式 (4.23)**：OCR 页 558 乱码（"2/s1 So2 /α2/s1 - sol/2"），据标准推导（$T=T_1-T_0$，$E(T|H_0)=-\tfrac12\lVert s_1-s_0\rVert^2$，$\mathrm{var}=\sigma^2\lVert s_1-s_0\rVert^2$）校订为 $P_e=Q(\sqrt{\lVert s_1-s_0\rVert^2/(4\sigma^2)})$。理由：代入 PSK（$\lVert s_1-s_0\rVert^2=4E$）得 $Q(\sqrt{E/\sigma^2})$，与例 4.7 正文一致。
2. **式 (4.28)**：OCR 页 561 的积分式残缺，据 Kay 原书英文版校订为 $P_e = 1-\int_{-\infty}^{\infty}\Phi^{M-1}(u)\,\phi(u-\sqrt{E/\sigma^2})\,du$（$\phi$ 为标准正态 PDF、$\Phi$ 为其 CDF）。理由：该式是"$T_0\sim\mathcal{N}(E,\sigma^2E)$、其余 $T_i\sim\mathcal{N}(0,\sigma^2E)$ 且独立"下 $\Pr\{\text{所有 }T_i<T_0\}$ 的标准积分形式，OCR 页 561 L41~51 残留的 $\Phi^{M-1}$、$\exp$ 结构与此吻合。
3. **式 (4.27) 的 $i\neq0$ 均值**：OCR 页 561 L28~31 的"$N(=8,\sigma^2E), i\neq0$"中"=8"为 OCR 误识，校订为 $N(0,\sigma^2E)$（正交信号下非本假设统计量均值为 0）。
4. **例 4.7 的"$10^{-9}$"指数**：OCR 页 559 L12 显示"对于典型的 10-"的误差率"（指数在扫描件中残缺），据 Kay 原书英文版校订为 $10^{-9}$；文中以"$10^{-9}$ 量级"的软化口径表述，并保留"约需 15 dB 平均 ENR"（OCR"15 dB"清晰可辨）。
5. **式 (4.24) 的分母**：OCR 页 558 的"I(sT's1 + st'so)"中"I"为"½"的误识，校订为 $\rho_s = s_1^T s_0/[\tfrac12(s_1^T s_1+s_0^T s_0)]$；与习题 4.20"$|\rho_s|\le1$"的要求自洽。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig025 | `Documents/figures/Fig025_匹配滤波器输出SNR.png` | `Temp/scripts/make_fig025.py` | ✅ 通过（check_figure strict=True） |

Fig025 为自建时域示意：信号 $s=[2,1,4,3]$（$N=4$，非对称以突出"镜像"），$E=\sum s^2[n]=30$，$h=s$ 反转 $=[3,4,1,2]$，输出 $y=s*h$ 在 $n=3$ 达峰 $30$。$E=30$ 为自建演示值，正文已标注"自建演示"。

## 4. 未决疑问

- 无实质性未决。式 (4.23)/(4.28) 的 OCR 乱码已按英文原版校订并注明；OCR 页 561 中"正交信号要求 $N\ge M$"一句（正文 §5.3 的图 4.15 说明）OCR 呈现为"对于 M 个正交信号，要求 N≥M"，与标准结论一致，正文按"信号数不能超过维数"的口径转述，未引具体数字。
