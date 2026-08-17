# Vol2_Ch06 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch06_统计判决理论II.md`
> OCR 来源：`Temp/chapters_ocr/v2ch06/ocr_page_613~660.txt`（PDF 613~660，书内 598~645）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §1.1 引言：雷达时延/通信频率/声纳方差未知 | ocr_page_613.txt L4~14 | 一致 |
| §6.2.1 复合假设检验小结表（GLRT/贝叶斯/Wald/Rao/LMP） | ocr_page_614.txt~617.txt | 一致（正文转述，非逐行） |
| 式 (6.1)(6.2) 未知幅度 DC 电平模型 | ocr_page_617.txt L28~31、L41~45 | 一致 |
| 式 (6.3) $T(\mathbf{x})=\bar{x}$；门限与 $A$ 无关（假象） | ocr_page_618.txt L19~29 | 一致 |
| 式 (6.4) UMP；图 6.1/6.2 | ocr_page_618.txt L38~41、ocr_page_619.txt L1~19 | 一致 |
| 单边/双边；"UMP 必是单边"（Kendall and Stuart） | ocr_page_619.txt L20~29 | 一致 |
| 透视检测器（clairvoyant）+ 上界；例 6.2 | ocr_page_619.txt L30、ocr_page_620.txt L1~36 | 一致 |
| 式 (6.8) $|\bar{x}|>\gamma''$；式 (6.9) $P_D$ 双 $Q$ 和 | ocr_page_620.txt L38~40、ocr_page_621.txt L5~8 | 一致 |
| 图 6.3/6.4 透视 vs NP vs 可实现 | ocr_page_621.txt L13~56 | 一致 |
| §6.4 贝叶斯 vs GLRT；"GLRT 应用更广泛" | ocr_page_621.txt L57、ocr_page_622.txt L1~11 | 一致 |
| 式 (6.10) 贝叶斯；例 6.3 式 (6.11) $\bar{x}^2>\gamma'$ | ocr_page_622.txt L12~41、ocr_page_623.txt、ocr_page_624.txt L1~6 | 一致 |
| 式 (6.12) GLRT；"在所有不变检验中 GLRT 是 UMP" | ocr_page_624.txt L7~16 | 一致 |
| 例 6.4 式 (6.13) $2\ln L_G=N\bar{x}^2/\sigma^2$ | ocr_page_624.txt L17~56、ocr_page_625.txt L1~6 | 一致 |
| 式 (6.14)(6.15) max 形式 | ocr_page_625.txt L7~25 | 一致 |
| 例 6.5 式 (6.16)~(6.20)（未知幅度与方差） | ocr_page_625.txt L26~41、ocr_page_626.txt、ocr_page_627.txt L1~31 | 一致 |
| §6.5 渐近前提（大记录、弱信号、MLE 达 PDF） | ocr_page_627.txt L32~39 | 一致 |
| 式 (6.21)(6.22) 参数检验 + GLRT | ocr_page_627.txt L40~47、ocr_page_628.txt L1~8 | 一致 |
| 式 (6.23) $2\ln L_G\xrightarrow{a}\chi_r^2/\chi_r^{\prime2}(\lambda)$ | ocr_page_628.txt L9~15 | 一致 |
| 式 (6.24)(6.25) 非中心参量 + Fisher 分块 | ocr_page_628.txt L15~27 | 一致 |
| CFAR；式 (6.26)(6.27) 无多余参数 | ocr_page_628.txt L28~37 | 一致 |
| 例 6.6 $\lambda=NA^2/\sigma^2$，线性模型下精确 | ocr_page_628.txt L41~48、ocr_page_629.txt L1~16 | 一致 |
| 例 6.7 对角 Fisher 信息 → 同 $\lambda$；式 (6.28) | ocr_page_629.txt L17~31 | 一致 |
| 式 (6.29) ROC $P_D=Q(Q^{-1}(P_{FA}/2)-\sqrt\lambda)+Q(Q^{-1}(P_{FA}/2)+\sqrt\lambda)$ | ocr_page_629.txt L32~36、ocr_page_630.txt L1~8 | 一致 |
| 图 6.5 $\lambda=5,\sigma^2=1$，N=10/30 蒙特卡洛 | ocr_page_630.txt L9~60 | 一致 |
| §6.6 Wald/Rao 渐近等效；Rao 最省 | ocr_page_631.txt L1~34 | 一致 |
| 式 (6.30)(6.31) Wald / Rao | ocr_page_631.txt L12~34 | 一致 |
| 例 6.8 三检验相同 $N\bar{x}^2/\sigma^2$ | ocr_page_631.txt L35~42、ocr_page_632.txt L1~24 | 一致 |
| 例 6.9 非高斯（广义高斯）噪声、Rao=三次矩平方 | ocr_page_632.txt L28~52、ocr_page_633.txt、ocr_page_634.txt L1~51 | 校订（噪声常数，见 §2） |
| "高斯使 $I(A)$ 最小"（习题 6.20） | ocr_page_634.txt L49~51 | 一致 |
| 式 (6.34)(6.35) 多余参数 Wald/Rao | ocr_page_634.txt L52~56、ocr_page_635.txt L1~15 | 一致 |
| 例 6.10 Rao $T_R=N\bar{x}^2/\hat\sigma_0^2$；图 6.7 N=10 Rao≈GLRT | ocr_page_635.txt L16~40、ocr_page_636.txt L1~21、L22~46 | 一致 |
| §6.7 LMP 定义；式 (6.36)(6.37) | ocr_page_637.txt L1~40、ocr_page_638.txt L1~15 | 一致 |
| 例 6.11 相关检验 式 (6.38) $\sqrt{N}\hat\rho$、$d^2=N\rho^2$ | ocr_page_638.txt L17~50、ocr_page_639.txt L1~46 | 一致 |
| §6.8 多元；式 (6.39) 嵌套模型恒选最大 | ocr_page_639.txt L47~50、ocr_page_640.txt L1~38 | 一致 |
| 式 (6.40) 广义 ML；例 6.12 | ocr_page_641.txt L33~43、ocr_page_642.txt L1~30 | 一致 |
| 式 (6.41) MDL $\mathrm{MDL}(i)=-\ln p+\frac{n_i}{2}\ln N$ | ocr_page_643.txt L12~17 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **例 6.9 噪声 PDF 常数**（ocr_page_632.txt L36~42）：OCR 呈现"常数 $a$ 是 $a=f()=1.4464$ / $\sqrt{2\Gamma(3)}$"等残缺字样，无法可靠还原归一化条件。据 Kay 原书英文版校订为广义高斯（指数类）PDF $p(w[n])=a\exp[-a^2w^4[n]]$（$a$ 为归一化常数使方差为 $\sigma^2$；OCR 标 $a=1.4464$，图 6.6 取 $\sigma^2=1$ 与 $\mathcal{N}(0,1)$ 对比）。理由：正文"广义高斯/指数类 PDF"、指数上四次方、图 6.6 的"比高斯更尖"形状、以及 Rao 统计量 $\propto(\sum x^3[n])^2$（三次矩）三者互相印证，只能对应 $p\propto\exp(-aw^4)$ 形式。**Fisher 信息 $I(A)=6N/(a^4\sigma^2)$ 与非中心参量 $\lambda=6NA^2/(a^4\sigma^2)$ 的精确 $a$ 依赖归一化口径，建议复算时回英文原版 §6.6 核对。**
2. **"透视检测器"译名**：OCR 页 620 用"透视检测器（clairvoyant detector）"，英文直译"千里眼/透视者"，沿用中译本译名并注英文。正文按"假定参数完全已知的 NP 检测器"解释其含义。
3. **式 (6.9) 的双 $Q$ 和**：OCR 页 621 L5~8 残缺（仅存 $NA^2/\sigma^2$ 与 $Q$），据标准推导（双边 $|\bar{x}|$ 检测，$P_{FA}=2Q(\gamma''\sqrt{N}/\sigma)$、$P_D=Q(\sqrt{\gamma''}-d)+Q(\sqrt{\gamma''}+d)$，$d=\sqrt{NA^2/\sigma^2}$）校订。理由：与例 6.7 的（6.29）式（OCR 页 630 清晰）同形，仅 $\gamma''$ 反解不同。
4. **"$2\ln L_G$ 渐近 $\chi_r^2/\chi_r^{\prime2}(\lambda)$"**：OCR 页 628 的 $\chi_r^{\prime2}$（非中心卡方）上标与 $\xrightarrow{a}$ 符号在扫描件中变形，据附录 6C（ocr_page_653~654）校订。理由：附录 6C 明确给出"$2\ln L_G\sim\chi_r^2$（H0）/ $\chi_r^{\prime2}(\lambda)$（H1）"的完整推导。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig027 | `Documents/figures/Fig027_GLRT流程.png` | `Temp/scripts/make_fig027.py` | ✅ 通过（check_figure strict=True） |

Fig027 为自建流程示意（无实测数值，纯结构图）：数据 → $H_0$ 下 MLE / $H_1$ 下 MLE → 广义似然比 → 判决。参数符号均为示意，未引实测数值。

## 4. 未决疑问

- 例 6.9 的噪声常数 $a=1.4464$ 的归一化条件（$a$ 与方差 $\sigma^2$ 的精确关系）OCR 无法还原，正文以"据 Kay 英文版校订为 $p\propto\exp(-a^2w^4)$、Fisher 信息 $6N/(a^4\sigma^2)$"口径表述并标注。若后续要做数值复现（例 6.9 无图），建议回英文原版 §6.6 取精确 $a$。
- 例 6.12 的 Fisher 信息行列式 $\det(\mathbf{I}_i)$ 的 $N$ 幂次（OCR 页 642 呈现 $N/(2\sigma^4)$、$N^2/(2\sigma^4)$、$N^3/(\cdots)$ 的部分）在扫描件中残缺，正文按"$\det(\mathbf{I}_0)\propto N$、$\det(\mathbf{I}_1)\propto N^2$、$\det(\mathbf{I}_2)\propto N^3$"的口径定性转述（与 MDL 罚项 $\frac{n_i}{2}\ln N$ 自洽），未引具体比例系数。
