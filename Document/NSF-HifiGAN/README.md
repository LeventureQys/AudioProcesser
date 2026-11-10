# NSF-HiFiGAN 教学文章

当前正文是一篇独立教学稿，按 GAN → HiFi-GAN → NSF-HiFiGAN 递进，直接解释问题、数学关系和网络机制。

- [打开单文件阅读版](NSF-HiFiGAN详解.html)：图示和公式全部内嵌，离线阅读，不需要同目录中的其他文件。
- [Markdown 正文](NSF-HiFiGAN详解.md)：图片以内嵌数据保存，正文没有配套文件链接、具体代码或复现命令。

## 当前版本（2026-11-10）

全文 9 节、10 幅图，约 1.5 万字符。补充音频、基频、相位、声码器等概念的定义；保留必要的 GAN 数学推导、多感受野、周期重排、源信号与多级注入的解释。原来的实现版本对照、参数量统计、训练流水、硬件测速与复现附录已退出正文。

新增 8 幅教学图，并保留 2 幅不依赖实现上下文的数学示意。图面已逐张回读，新增图片保存前的自检问题数均为 0。单文件阅读版经过浏览器排版检查；所有图片和公式均在文件内部，不请求外部资源。核查结果记在 outputs/teaching_article_audit.json。

## 编写维护材料

这一节只服务于后续编写维护，不是读者阅读文章的前置条件。

- scripts/teaching_figures.py：教学插图的绘制逻辑，登记编号为 Fig018~Fig025。
- scripts/build_teaching_article.py：把正文中的图示与公式嵌入单文件阅读版，并检查外部依赖、代码残留与图片完整性。
- 图片编号登记.md：保留历史图片记录，末尾增加教学版图示的对应关系。

重新导出当前教学文章，可在本目录执行 python scripts/build_teaching_article.py。修改数学正文时，以 NSF-HiFiGAN详解.md 为准。

## 历史材料

旧工程稿及其审校记录保留于 Temp/NSF-HiFiGAN详解.工程审校版_20261110.md 与 Temp/README.工程审校版_20261110.md。原有训练结果、源代码快照、工程图表和历史审校 JSON 均属于编写过程的材料，不作为当前文章的前置上下文。

outputs/article_audit.json 与 outputs/figure_audit.json 记录的是旧工程稿的复核。旧的 audit_article.py 包含工程稿专用的表格与图片引用检查，不能直接用它判断当前教学稿是否合格；当前交付核查使用 teaching_article_audit.json。
