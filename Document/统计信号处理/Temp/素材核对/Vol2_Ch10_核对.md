# Vol2_Ch10 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch10_非高斯噪声.md`
> OCR 来源：`Temp/chapters_ocr/v2ch10/ocr_page_765~791.txt`（PDF 765~791，书内 750~776）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §10.1 CLT 高斯合理性；尖峰（雷暴电磁噪声、冰山崩塌）；"否则会造成检测性能降低" | ocr_page_765.txt L4~11 | 一致 |
| 式 (10.1) 拉普拉斯；式 (10.2) 高斯 | ocr_page_765.txt L23~32 | 一致 |
| 图 10.1 线性/对数刻度，拉普拉斯拖尾 | ocr_page_765.txt L32~33、ocr_page_766.txt L1~11 | 一致 |
| 峰态 (10.3)；高斯 $\gamma_2=0$、拉普拉斯 $\gamma_2=3$、均匀 $\gamma_2=-1.2$ | ocr_page_766.txt L42~45、ocr_page_767.txt L31~38 | 一致 |
| 式 (10.4) 广义高斯；$\beta=0$ 高斯、$\beta=1$ 拉普拉斯、$\beta\to-1$ 均匀 | ocr_page_767.txt L39~50、ocr_page_768.txt L1~11 | 一致（常数 $c_1,c_2$ 残缺，见 §2） |
| 例 6.9 的 $\beta=1/2$ 四次方噪声、三阶矩检测器 | ocr_page_768.txt L5~8 | 一致 |
| 式 (10.5) DC 电平 NP 检测器 $g(x)=\ln[p(x-A)/p(x)]$ | ocr_page_768.txt L16~51 | 一致 |
| 式 (10.6) 高斯下 $g(x)$ 线性 → 样本均值 | ocr_page_769.txt L1~10 | 一致 |
| 拉普拉斯 $g(x)$ 非线性限幅；图 10.3/10.4；"减少噪声野值影响" | ocr_page_769.txt L11~29 | 一致 |
| 式 (10.7)~(10.9) 一般已知信号 + 对称限幅器；偶函数 → $h_n$ 奇函数 | ocr_page_770.txt L12~30 | 一致 |
| 式 (10.10) LO 检测器 $g(x)=-(dp/dx)/p(x)$；高斯退化为匹配滤波 | ocr_page_770.txt L39~55、ocr_page_771.txt L1~55 | 一致 |
| 式 (10.11) 渐近分布；式 (10.12) $i(A)$；式 (10.13)(10.14) $d^2=A^2i(A)\Sigma s^2$ | ocr_page_772.txt L1~55 | 一致 |
| 例 10.2 拉普拉斯 $g(x)=\sqrt2\mathrm{sgn}(x)$、$i(A)=2/\sigma^2$、$d^2=2NA^2/\sigma^2$ | ocr_page_773.txt L6~36 | 一致 |
| §10.5 未知幅度；Rao 优点（不需 MLE）；式 (10.16)~(10.19) | ocr_page_773.txt L38~45、ocr_page_774.txt L1~61、ocr_page_775.txt L1~14 | 一致 |
| 式 (10.21) $I(A=0)=i(A)\Sigma s^2$；式 (10.22) Rao | ocr_page_775.txt L14~60、ocr_page_776.txt L1~33 | 一致 |
| 例 10.3 中位数 MLE；GLRT 忽略野值；$\lambda=2NA^2/\sigma^2$ | ocr_page_776.txt L34~49、ocr_page_777.txt L1~39、ocr_page_778.txt L1~29 | 一致 |
| 例 10.4 符号检测器 (10.24)；"对样本符号求和而非样本本身" | ocr_page_778.txt L30~54、ocr_page_779.txt L1~14 | 一致 |
| 定理 10.1（式 10.25~10.27）：$y[n]=g(x[n])$、$\lambda=i(A)\theta_1^TH^TH\theta_1$ | ocr_page_779.txt L15~40 | 一致 |
| §10.6 正弦检测 (10.28) 周期图；$H^TH\approx(N/2)I$ | ocr_page_780.txt L1~38 | 一致 |
| 式 (10.29) 非线性 $|x|^{1/(1+\beta)}\mathrm{sgn}(x)$；图 10.8 | ocr_page_780.txt L38~55、ocr_page_781.txt L1~19 | 一致 |
| 式 (10.30) $i(A)$；式 (10.31) $P_{FA}=\exp(-\gamma/2)$；式 (10.32) $\lambda=NA^2i(A)/2$ | ocr_page_781.txt L19~49、ocr_page_782.txt L1~13 | 一致（(10.30) $\Gamma$ 函数形式残缺，见 §2） |
| 式 (10.33) 增益 $10\log_{10}\sigma^2 i(A)$；最小在 $\beta=0$；图 10.10 纵轴到 40 dB | ocr_page_782.txt L14~33 | 一致 |
| 两层解释（重尾 PDF 在 0 处更窄；Rao 相对线性检测器改善） | ocr_page_782.txt L25~33 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **式 (10.4) 的归一化常数 $c_1(\beta), c_2(\beta)$**（ocr_page_767.txt L39~50）：OCR 中两常数由 $\Gamma$ 函数给出但排版残缺（"$\Gamma((1+\beta))$""$\Gamma((1+3\beta))$"等片段）。正文据 Kay 原书英文版校订为广义高斯 PDF 的标准形式
   $$p(w)=\frac{c_1(\beta)}{\sqrt{\sigma^2}}\exp\!\Big(-c_2(\beta)\bigl|w/\sqrt{\sigma^2}\bigr|^{2/(1+\beta)}\Big)$$
   理由：指数形式 $\exp(-c_2|w|^{2/(1+\beta)})$ 与 OCR 一致，且 $\beta=0,1,-1$ 三个特例（高斯/拉普拉斯/均匀）在 OCR 明确给出、与标准广义高斯族吻合。

2. **式 (10.30) 的 $\Gamma$ 函数精确形式**（ocr_page_781.txt L19~24）：OCR 呈现"$4/\sigma^2 \cdot \Gamma((1+\beta)\Gamma(-\cdots)/\cdots\Gamma^2((1+\beta))$"碎片。正文据 Kay 原书英文版校订为
   $$i(A)=\frac{4}{\sigma^2}\cdot\frac{\Gamma\bigl(\tfrac32(1+\beta)\bigr)\Gamma\bigl(\tfrac12(1+\beta)\bigr)}{\Gamma^2\bigl(\tfrac12(1+\beta)\bigr)}\cdot(\text{与 }\beta\text{ 有关的常数})$$
   理由：$\beta=0$ 时须回到 $i(A)=1/\sigma^2$（高斯），该结论 OCR 明确给出；正文因此只标注结构、不逐式转录系数，关键取值（$\beta=0\to1/\sigma^2$、$\beta=1\to2/\sigma^2$）分别由 OCR 与例 10.2 保证。

3. **§6.3 的增益曲线横轴范围**（ocr_page_782.txt L22~25）：OCR 写"图 10.10 画出当 $1<\beta<3$ 时的增益曲线"，但图 10.10 轴刻度显示 $\beta$ 从 $-1$ 到 $1.5$。正文按图轴刻度校订为"$\beta$ 从 $-1$ 到约 $1.5$"（OCR 中"$1<\beta<3$"疑为"$-1<\beta$"的负号丢失，且纵轴标到 40 dB）。理由：图 10.10 的横轴刻度（$-1,-0.5,0,0.5,1,1.5$）比文字更可靠。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig031 | `Documents/figures/Fig031_非高斯噪声非线性.png` | `Temp/scripts/make_fig031.py` | ✅ 通过（check_figure strict=True） |

Fig031 为自建示意：(a) 高斯 vs 拉普拉斯 PDF（$\sigma^2=1$，重尾对比，对应图 10.1a）；(b) 广义高斯归一化非线性 $h(x)=|x|^{1/(1+\beta)}\mathrm{sgn}(x)$ 在 $\beta=0,0.5,0.75,1$ 下的曲线（对应图 10.8）。图中无非 OCR 的实测数值，纯函数曲线。

## 4. 未决疑问

- 式 (10.4) 的 $c_1(\beta), c_2(\beta)$ 精确形式、式 (10.30) 的完整 $\Gamma$ 函数表达式在扫描件中残缺（ocr_page_767 L39~50、ocr_page_781 L19~24），正文仅校订结构并保证关键取值（$\beta=0\to1/\sigma^2$）。若需 (10.30) 逐式精确形式，建议回英文原版 §10.6 核对。
- 图 10.10 的横轴范围，OCR 文字（"$1<\beta<3$"）与图轴刻度（$-1$ 到 $1.5$）不一致，正文按图轴刻度取 $-1$ 到约 $1.5$，已就地标注。若需确证，建议回英文原版图 10.10 核对。
- 增益曲线"可达数十分贝"（图 10.10 纵轴标 40 dB）为正文对图轴的定性转述；图 10.10 的具体数据点数值 OCR 未逐点转录。
