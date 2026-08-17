# -*- coding: utf-8 -*-
"""matplotlib 绘图公共工具：中文字体 + 程序化碰撞检测。

需求第 5 条要求作图严格防碰撞（防穿框、混乱、拧巴）。本会话模型不能读图，
因此所有图片脚本绘制完成后必须调用 check_figure()：
  1. 图中所有文字/标注/图例的包围盒两两相交检测；
  2. 文字/标注包围盒不得越出所在 Axes（防"穿框"）；
  3. 图例不得越出 Figure 边界。
任何一项不通过，check_figure 在 strict=True 时抛 RuntimeError（脚本报错、不产出图）。

用法（见 make_fig00X.py）：
    from plotutil import setup_cn, check_figure
    setup_cn()               # 配置中文字体
    fig, ax = plt.subplots(...)
    ...绘制...
    check_figure(fig)        # 通过才保存
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
]


def setup_cn():
    """配置中文字体（按可用性挑选），并关闭 unicode 负号问题。"""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in _CJK_CANDIDATES:
        if name in available:
            chosen = name
            break
    if chosen is None:
        raise RuntimeError(
            f"未找到可用中文字体，候选: {_CJK_CANDIDATES}; "
            f"请检查 C:/Windows/Fonts 下的 msyh/simhei 字体"
        )
    plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# 判定为"可能撞车"的文本类别：注释、标题、图例文字等（刻度标签由 matplotlib 自行排版，不参与两两检测）
_SKIP_ROLES = {"yticklabel", "xticklabel", "xlabel", "ylabel"}


def _texts(ax):
    """返回 [(text对象, role)]，role ∈ {annot, title, legend}。"""
    out = []
    for t in ax.texts:
        out.append((t, "annot"))
    if ax.get_title():
        out.append((ax.title, "title"))
    if ax.get_legend() is not None:
        for t in ax.get_legend().get_texts():
            out.append((t, "legend"))
    return out


def _boxes(fig, ax):
    """返回 [(label, bbox_display, role)]，其中 bbox 为显示坐标 Bbox。"""
    renderer = fig.canvas.get_renderer()
    items = []
    for t, role in _texts(ax):
        bb = t.get_window_extent(renderer)
        items.append((t.get_text()[:20], bb, role))
    if ax.get_legend() is not None:
        leg = ax.get_legend()
        bb = leg.get_window_extent(renderer)
        items.append(("LEGEND-BOX", bb, "legendbox"))
    # 注释箭头等 patch 暂不参与（matplotlib 注释文本已含在上方 texts）
    return items


def _overlap_area(a, b):
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def check_figure(fig, strict=True, min_overlap_ratio=0.02):
    """碰撞检测主入口。返回问题列表；strict=True 时发现问题直接抛错。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    fig_bbox = fig.bbox

    for ax in fig.axes:
        ax_bbox = ax.get_window_extent(renderer)
        items = _boxes(fig, ax)

        # 1) 越界检测（穿框）：注释/图例文字须位于所在 Axes 内（允许 1.5pt 容差）；
        #    标题、图例盒只要求不越出 Figure 边界
        for label, bb, role in items:
            if role in ("title", "legend", "legendbox"):
                if not fig_bbox.contains(bb.x0, bb.y0) or not fig_bbox.contains(bb.x1, bb.y1):
                    problems.append(f"[越界] {role} '{label}' 超出 Figure 边界: {bb}")
                continue
            tol = 1.5
            if (bb.x0 < ax_bbox.x0 - tol or bb.x1 > ax_bbox.x1 + tol
                    or bb.y0 < ax_bbox.y0 - tol or bb.y1 > ax_bbox.y1 + tol):
                problems.append(f"[穿框] '{label}' 越出 Axes 边界: text={bb} axes={ax_bbox}")

        # 2) 两两相交检测（排除刻度标签；只检测注释/标题/图例文字之间。
        #    图例文字位于自身图例框内是正常排版，跳过 legend 与自身 LEGEND-BOX 的组合）
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a_label, a_bb, a_role = items[i]
                b_label, b_bb, b_role = items[j]
                if (a_role == "legend" and b_role == "legendbox") or (
                    a_role == "legendbox" and b_role == "legend"
                ):
                    continue
                ov = _overlap_area(a_bb, b_bb)
                if ov <= 0:
                    continue
                small = min(a_bb.width * a_bb.height, b_bb.width * b_bb.height)
                if small > 0 and ov / small > min_overlap_ratio:
                    problems.append(
                        f"[重叠] '{a_label}' 与 '{b_label}' 包围盒相交 "
                        f"({ov:.0f}px² / {small:.0f}px²): a={a_bb} b={b_bb}"
                    )

    if problems and strict:
        raise RuntimeError("碰撞检测未通过，拒绝保存图片:\n  " + "\n  ".join(problems))
    return problems
