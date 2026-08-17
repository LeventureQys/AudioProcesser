# Vol1 Ch14 素材核对清单

> 文档：`Documents/Vol1_Ch14_估计量总结.md`
> OCR 来源：`Temp/chapters_ocr/ch14/ocr_page_400~411.txt`（PDF 第 400~411 页，书内第 385~396 页，映射规则：书内页码 = PDF 页码 − 15）

## 1. 关键数值 / 公式编号 → OCR 出处

| 文中内容 | 原书编号 | OCR 出处（文件 + 附近内容） |
|---------|---------|---------------------------|
| 选估计量的总原则：模型复杂到足以描述特征、简单到足以允许最优且易实现；MVU 可能不存在、MMSE 可能不能实现 | — | ocr_page_400（"它的复杂性应该足以描述数据的基本特征…不能确定最佳估计量的存在性…MMSE估计量就是这样一个例子"） |
| 经典方法：数据信息总结在 PDF $p(\mathbf{x};\boldsymbol\theta)$；贝叶斯：联合 PDF $p(\mathbf{x},\boldsymbol\theta)$ 或 $p(\mathbf{x}|\boldsymbol\theta)p(\boldsymbol\theta)$ | — | ocr_page_400（"数据信息总结在概率密度函…联合PDF P(x,9)"） |
| CRLB 达界条件 $\partial\ln p/\partial\boldsymbol\theta=\mathbf{I}(\boldsymbol\theta)(g(\mathbf{x})-\boldsymbol\theta)$；$g(\mathbf{x})$ 达 CRLB=MVU=有效 | — | ocr_page_400~401（"如果CRLB等号条件 ln p(x;@) = I(0)(g(x)-0) 满足"） |
| RBLS：$p=g(T(\mathbf{x}),\boldsymbol\theta)h(\mathbf{x})$ 求充分统计量；$E[T]=\boldsymbol\theta$ 则 $\hat\theta=T$，否则求 $g$ 使 $E[g(T)]=\theta$ | — | ocr_page_401（"通过将PDF因式分解…求出一个充分统计量 T(x)…如果 E[T(x)} =0,那么=T(x)"） |
| BLUE：$E(\mathbf{x})=\mathbf{H}\boldsymbol\theta$、$\mathbf{C}$ 已知；$\hat\theta=(\mathbf{H}^T\mathbf{C}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{C}^{-1}\mathbf{x}$；若 $\mathbf{w}\sim\mathcal{N}(0,\mathbf{C})$ 也是 MVU | — | ocr_page_402（"E(x) = H8…x的协方差矩阵C是已知的…如果w是高斯随机矢量…那么é也是MVU估计量"） |
| MLE：渐近有效；渐近 $\hat\theta\sim\mathcal{N}(\boldsymbol\theta,\mathbf{I}^{-1}(\boldsymbol\theta))$；若有效估计量存在则 MLE 得之 | — | ocr_page_402~403（"在PDF一定的条件下，对于大数据记录…MLE是有效的…% N(0, 1-1(0)…如果有效估计量存在，那么最大似然方法将得到有效估计量"） |
| LSE：最小化 $J=(\mathbf{x}-\mathbf{s}(\boldsymbol\theta))^T(\mathbf{x}-\mathbf{s}(\boldsymbol\theta))$；若 $\mathbf{w}\sim\mathcal{N}(0,\sigma^2\mathbf{I})$ 则 LSE=MLE | — | ocr_page_403（"是使下式最小的0值 J(0)=(x -s(0))T(x - s(0))…如果是高斯随机矢量 W～N（,αI)那么LSE等价于 MLE"） |
| 矩方法：$\theta=\mathbf{h}(\boldsymbol\mu)$ 可逆，$\hat\theta=\mathbf{h}^{-1}(\hat{\boldsymbol\mu})$ | — | ocr_page_403~404（"如果h(),其中h是可逆的的p维函数…=h-1(i)"） |
| MMSE：$\hat\theta=E(\boldsymbol\theta|\mathbf{x})$；联合高斯时 $\hat\theta=E(\boldsymbol\theta)+\mathbf{C}_{\theta x}\mathbf{C}_{xx}^{-1}(\mathbf{x}-E(\mathbf{x}))$ | (14.1) | ocr_page_404（"= E(0|x)…如果和是联合高斯的,那么 = E(0) + CrC1(x - E(x)) (14.1)"） |
| 贝叶斯 MSE $\mathrm{Bmse}(\hat\theta_i)=E[(\theta_i-\hat\theta_i)^2]$；误差零均值、方差 | (14.2)(14.3) | ocr_page_404~405（"Bmse(0) -- E [(0; - 6.)? (14.2)…var(e:) = Bmse(6.) = / [Ce -luip(x)d x (14.3)"） |
| MAP：使后验最大 = 使 $p(\mathbf{x}|\boldsymbol\theta)p(\boldsymbol\theta)$ 最大；hit-or-miss 代价；均值=峰时 MMSE=MAP | — | ocr_page_405（"是使p(Ix)达到最大的值,或者等效地使p(xl)p(8)最大的值…使'成功-失败'代价函数最小…对于均值和模式相同的PDF，MMSE与MAP估计量是相同的"） |
| LMMSE：只需前二阶矩；$\hat\theta=E(\boldsymbol\theta)+\mathbf{C}_{\theta x}\mathbf{C}_{xx}^{-1}(\mathbf{x}-E(\mathbf{x}))$ | — | ocr_page_405（"联合PDF p（x，)的前二阶矩是已知的…= E(0) + CerC-l(x - E(x))"） |
| 省略卡尔曼：它是 MMSE 的特定实现 | — | ocr_page_406（"我们省略了卡尔曼滤波器的小结，这是因为它是MMSE估计量的特定实现"） |
| 经典一般线性模型 $\mathbf{x}=\mathbf{H}\boldsymbol\theta+\mathbf{w}$、$\mathbf{w}\sim\mathcal{N}(0,\mathbf{C})$、$\hat\theta=(\mathbf{H}^T\mathbf{C}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{C}^{-1}\mathbf{x}$ | (14.5) | ocr_page_406（"X = H8 + W…=(HTC-1H)-IHTC-}x (14.5)"） |
| 线性模型下 CRLB 达界、RBLS 充分统计量 $T=\hat\theta$、BLUE 同形、MLE 最小化 $(\mathbf{x}-\mathbf{H}\boldsymbol\theta)^T\mathbf{C}^{-1}(\mathbf{x}-\mathbf{H}\boldsymbol\theta)$、LSE=$(\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{x}$（$\mathbf{C}=\sigma^2\mathbf{I}$ 才 MVU，否则加权 $\mathbf{W}=\mathbf{C}^{-1}$） | — | ocr_page_406~407（"由于已经是x的线性函数…如果w不是高斯的，将仍然是BLUE，但不是MVU…LSE是=（H"H)-Hx,如果 C=I,它也是MVU"） |
| 表 14.1：高斯列勾有效/充分/MVU/BLUE/MLE/WLS 六项，非高斯列只勾 BLUE、WLS | 表14.1 | ocr_page_408（"神～高斯（线性模型） W~非高斯 有效估计量 充分统计望 MVL BLLE M.E WLS(W= C-I )"） |
| 贝叶斯线性模型后验均值/协方差 | (14.6)~(14.9) | ocr_page_408（"E(0|x) μg + C,HT(HCgHT + Cu.)-l(x - Hμg) (14.6)…(C, +HTCH)-1 (14.9)"） |
| 无先验 $\mathbf{C}_\theta^{-1}\to\mathbf{0}$ 退化到经典 MVU 形式；不能真正比较（习题 11.7） | — | ocr_page_409（"令C,1--0来表示…可以认为与经典一般线性模型的MVU粘计具有相同的形式…这个断言是不正确的"） |
| §14.4 决策：$x[n]=A[n]+w[n]$ 参数≈数据点、缺乏平均、性能差；有先验→贝叶斯；无先验→降维/更多数据→经典；图 14.1 三分支 | 图14.1 | ocr_page_409~411（"参数与数据点数一样多…由于缺乏平均…图14.1(a)…图14.1(b)描述的贝叶斯方法…图14.1(c)…经典方法"） |

## 2. 据英文原版校订之处及理由

1. **方法清单的矩阵公式**：OCR 把 $\mathbf{H}^T\mathbf{C}^{-1}\mathbf{H}$、$\mathbf{C}_{\theta x}\mathbf{C}_{xx}^{-1}$ 等识别成 `H"C-1H`、`CerC-l`，均按 Kay 英文原版 §14.2 校订。
2. **表 14.1 的勾选关系**：OCR 只给出表头和两列属性名，勾选符号 `*` 全部丢失；勾选关系按 Kay 英文原版 Table 14.1 重建（高斯列六项全勾，非高斯列勾 BLUE、WLS 两项），并经 §14.3 正文"w 非高斯时仍是 BLUE 但不是 MVU""加权 LSE 可能只是 BLUE"交叉印证。
3. **（14.6）~（14.9）后验均值/协方差**：OCR 残损（"μg + C,HT(HCgHT + Cu.)-l"），按英文原版补全两组等价形式。
4. **"苹果和桔子"无先验等价性**：OCR 末尾"断言没有先验信息的贝叶斯方法与经典方法的等效性…当用统计的观点进行考察时，这个断言是不正确的"与英文原版一致，本文按"伪等价"表述并引习题 11.7。

## 3. 图片清单

| 编号 | 文件 | 脚本 | 碰撞检测 | 状态 |
|------|------|------|---------|------|
| Fig019 | `Documents/figures/Fig019_估计量选型决策流程.png` | `Temp/scripts/make_fig019.py` | `check_figure` 通过 | ✅ 已生成 |

图注要点：Fig019 为三分支决策流程图（对应原书图 14.1），无仿真数据；(a) 根决策（有无先验）、(b) 贝叶斯分支（MMSE/MAP/LMMSE）、(c) 经典分支（CRLB→MVU→充分统计量→MLE→矩 / BLUE→LSE）。生成过程中曾因 Unicode 上标（θ̂、ᵀ、⁻¹）缺字形产生方框警告，已改用 mathtext 排版消除。

## 4. 未决疑问

- 无。本章无仿真数据，数字均来自 OCR 可辨内容与英文原版校订；表 14.1 勾选关系虽需英文原版补全，但与 §14.3 正文文字表述自洽。
