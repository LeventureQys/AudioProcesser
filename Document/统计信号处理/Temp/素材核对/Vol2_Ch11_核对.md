# Vol2_Ch11 素材核对清单

> 文档：`Document/统计信号处理/Documents/Vol2_Ch11_检测器总结.md`
> OCR 来源：`Temp/chapters_ocr/v2ch11/ocr_page_792~811.txt`（PDF 792~811，书内 777~796）
> 核对人：写作子代理；日期与生成脚本时间戳一致。

## 1. 关键数值/公式 → OCR 出处

| 文中引用 | OCR 出处（文件：行号附近） | 状态 |
|---------|--------------------------|------|
| §1.1 "既不能保证求出最佳检测器，也不能保证实现" | ocr_page_792.txt L5~11 | 一致 |
| 项目 1 NP：$L(\mathbf{x})=p(\mathbf{x};H_1)/p(\mathbf{x};H_0)>\gamma$，$P_{FA}=Pr[L>\gamma;H_0]=\alpha$ | ocr_page_792.txt L19~32 | 一致 |
| 项目 2 MAP（式 11.1）$L(\mathbf{x})>\frac{P(H_0)}{P(H_1)}$ 或 $P(H_1\mid\mathbf{x})>P(H_0\mid\mathbf{x})$ | ocr_page_793.txt L5~30 | 一致 |
| 项目 3 贝叶斯风险门限 $\frac{(C_{10}-C_{00})P(H_0)}{(C_{01}-C_{11})P(H_1)}$ | ocr_page_793.txt L32~43 | 校订（条件 $C_{10}>C_{00},C_{01}>C_{11}$ OCR 残缺，见 §2） |
| 项目 4 MAP 多元（式 11.2）$P(H_k\mid\mathbf{x})$ 最大；项目 5 贝叶斯风险 $C_k(\mathbf{x})$ 最小 | ocr_page_794.txt L18~39、ocr_page_795.txt L8~33 | 一致 |
| 项目 6 GLRT $L_G(\mathbf{x})=p(\mathbf{x};\hat{\boldsymbol{\theta}}_1,H_1)/p(\mathbf{x};\hat{\boldsymbol{\theta}}_0,H_0)$ | ocr_page_795.txt L36~40、ocr_page_796.txt L1~16 | 一致 |
| 项目 7 贝叶斯（先验积分） | ocr_page_796.txt L18~31 | 一致 |
| 项目 8 GLRT 无多余参数：$2\ln L_G\sim\chi_r^2$ / $\chi_r^{\prime2}(\lambda)$，$\lambda=(\boldsymbol{\theta}_1-\boldsymbol{\theta}_0)^T\mathbf{I}(\boldsymbol{\theta}_0)(\boldsymbol{\theta}_1-\boldsymbol{\theta}_0)$ | ocr_page_796.txt L33~39、ocr_page_797.txt L4~27 | 校订（$\lambda$ 下标 OCR 残缺，见 §2） |
| 项目 9 Wald、项目 10 Rao（无多余参数） | ocr_page_797.txt L28~39、ocr_page_798.txt L7~25 | 一致 |
| 项目 11 GLRT 有多余参数：$\lambda=(\boldsymbol{\theta}_{r_1}-\boldsymbol{\theta}_{r_0})^T[\mathbf{I}_{\theta_r\theta_r}-\mathbf{I}_{\theta_r\theta_s}\mathbf{I}_{\theta_s\theta_s}^{-1}\mathbf{I}_{\theta_s\theta_r}]^{-1}(\cdot)$ | ocr_page_798.txt L26~38、ocr_page_799.txt L1~26 | 校订（分块求逆 OCR 残缺，见 §2） |
| 项目 12 Wald、项目 13 Rao（有多余参数） | ocr_page_799.txt L28~39、ocr_page_800.txt L7~25 | 一致 |
| 项目 14 LMP：$T_{LMP}=\frac{\partial\ln p/\partial\theta\mid_{\theta_0}}{\sqrt{I(\theta_0)}}$，门限 $Q^{-1}(P_{FA})$，渐近 $\mathcal{N}(0,1)$ / $\mathcal{N}(\sqrt{I(\theta_0)}(\theta_1-\theta_0),1)$ | ocr_page_800.txt L27~42、ocr_page_801.txt L1~15 | 一致 |
| 项目 15 广义 ML：$\xi_i=\ln p(\mathbf{x};\hat{\boldsymbol{\theta}}_i;H_i)-\tfrac12\ln\det(\mathbf{I}(\hat{\boldsymbol{\theta}}_i))$ | ocr_page_801.txt L19~34 | 一致 |
| §11.3 经典线性模型 $\mathbf{x}=\mathbf{H}\boldsymbol{\theta}+\mathbf{w}$、PDF | ocr_page_801.txt L36~40、ocr_page_802.txt L4~13 | 一致 |
| 项目 16 已知确定信号 $T=\mathbf{x}^T\mathbf{C}^{-1}\mathbf{s}$，$P_D=Q(Q^{-1}(P_{FA})-\sqrt{\mathbf{s}^T\mathbf{C}^{-1}\mathbf{s}})$ | ocr_page_802.txt L14~32 | 一致 |
| 项目 17 未知参数确定信号 $T=\mathbf{x}^T\hat{\mathbf{s}}$，$\hat{\boldsymbol{\theta}}_1=(\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{x}$，$\lambda=\boldsymbol{\theta}_1^T\mathbf{H}^T\mathbf{H}\boldsymbol{\theta}_1/\sigma^2$ | ocr_page_802.txt L34~40、ocr_page_803.txt L8~27 | 一致 |
| 项目 18 未知参数+未知方差，门限 $F_{p,N-p}$，$\hat{\sigma}^2=(\mathbf{x}^T\mathbf{x}-\mathbf{x}^T\mathbf{H}\hat{\boldsymbol{\theta}}_1)/(N-p)$ | ocr_page_803.txt L30~40、ocr_page_804.txt L1~18 | 一致 |
| 项目 19 未知噪声参数 Rao $T_R=\mathbf{x}^T\mathbf{C}^{-1}(\hat{\boldsymbol{\theta}}_{w_0})\hat{\mathbf{s}}$ | ocr_page_804.txt L20~35、ocr_page_805.txt L1~20 | 一致 |
| 项目 20 贝叶斯线性模型 $\hat{\mathbf{s}}=\mathbf{H}\mathbf{C}_\theta\mathbf{H}^T(\mathbf{H}\mathbf{C}_\theta\mathbf{H}^T+\sigma^2\mathbf{I})^{-1}\mathbf{x}$（MMSE） | ocr_page_805.txt L21~33、ocr_page_806.txt L1~14 | 一致 |
| 项目 21 非高斯线性模型 Rao：$T_R=\mathbf{y}^T\mathbf{H}(\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{y}/i(A)$，$y[n]=g(x[n])$ | ocr_page_806.txt L15~40、ocr_page_807.txt L1~15 | 校订（$g(w)$、$i(A)$ OCR 残缺，见 §2） |
| §11.4 决策流程文字（图 11.1~11.4） | ocr_page_807.txt L16~35、ocr_page_808.txt、ocr_page_809.txt、ocr_page_810.txt L1~21 | 一致 |
| Wald 在检测中"很少采用"及理由 | ocr_page_809.txt L50~52 | 一致 |
| §11.5 UMPI/序贯/极小极大/非参数及出处 | ocr_page_810.txt L22~35 | 一致 |
| 参考文献清单 | ocr_page_811.txt L6~24 | 一致 |

## 2. 据 Kay 原书英文版校订处

1. **项目 3 贝叶斯风险的前提条件**（ocr_page_793.txt L43 显示"其中 >Cm, Co> Ca"）：据 Kay 原书英文版 §3.7 校订为"$C_{10}>C_{00}$、$C_{01}>C_{11}$"（判错比判对贵）。理由：贝叶斯风险门限的分母/分子要为正，且这是 Kay 对代价矩阵的标准约束；OCR 该句残缺严重。

2. **项目 8/11 的非中心参量 $\lambda$ 写法**（ocr_page_797.txt L23~26、ocr_page_799.txt L13~17）：OCR 分块下标（$\mathbf{I}_{\theta_r\theta_s}$ 等）残缺。据 Kay 原书英文版 §6.5 校订为无多余参数 $\lambda=(\boldsymbol{\theta}_1-\boldsymbol{\theta}_0)^T\mathbf{I}(\boldsymbol{\theta}_0)(\boldsymbol{\theta}_1-\boldsymbol{\theta}_0)$、有多余参数 $\lambda=(\boldsymbol{\theta}_{r_1}-\boldsymbol{\theta}_{r_0})^T[\mathbf{I}_{\theta_r\theta_r}-\mathbf{I}_{\theta_r\theta_s}\mathbf{I}_{\theta_s\theta_s}^{-1}\mathbf{I}_{\theta_s\theta_r}]^{-1}(\boldsymbol{\theta}_{r_1}-\boldsymbol{\theta}_{r_0})$。理由：与 Vol2_Ch06 已核对过的（6.24）（6.27）完全一致。

3. **项目 21 的 $g(w)$ 与 $i(A)$**（ocr_page_806.txt L35~40 显示"g(w) = d In p(w)/mp (dp(w)) (A) dw p(w)"残缺）：据 Kay 原书英文版 §10.5 校订为 $g(w)=-d\ln p(w)/dw$、$i(A)=E[(d\ln p(w)/dw)^2]$（位置参数 Fisher 信息）。理由：这是非高斯 Rao 检验的标准"得分函数"定义，且与偏移系数 $\lambda=i(A)\boldsymbol{\theta}_1^T\mathbf{H}^T\mathbf{H}\boldsymbol{\theta}_1$（ocr_page_807.txt L10 的"A = (A)0HTHO1"）自洽。

## 3. 图片清单

| 编号 | 文件 | 生成脚本 | 碰撞检测 |
|------|------|---------|---------|
| Fig032 | `Documents/figures/Fig032_检测器选型指南.png` | `Temp/scripts/make_fig032.py` | ✅ 通过（check_figure strict=True） |

Fig032 为自建选型决策流程图（纯结构图，无实测数值）：(a) 最佳路线（对应原书图 11.1/11.2）；(b) 复合假设准最佳路线（对应图 11.3）。括号内项目编号（1~21）为原书 §11.2/§11.3 项目序号的原样引用。

## 4. 未决疑问

- 项目 17/18/19 的"门限 $\gamma'$"在 OCR 中均以抽象记号给出（如 "= Q(Pm)" 等残缺），正文按第 7/9 章已核对的结论（$\chi^2_p$ 尾 / $F_{p,N-p}$ 尾）表述门限口径，未逐式转录原书可能的精确记号。若需原书 §11.3 的逐字门限定义，建议回英文原版核对。
- 项目 19 的"$\hat{\boldsymbol{\theta}}_{w_0}$ 是 $H_0$ 条件下 $\boldsymbol{\theta}_w$ 的 MLE"与项目 13 的"只要求 $H_0$ 下的 MLE"在 OCR 中均成立，正文据此表述；但项目 19 里信号估计 $\hat{\boldsymbol{\theta}}_1$ 用的是代入 $\mathbf{C}(\hat{\boldsymbol{\theta}}_{w_0})$ 后的加权最小二乘，而非联合 MLE——这一细节 OCR 页 805 L6~7 可辨，正文已按此表述。
