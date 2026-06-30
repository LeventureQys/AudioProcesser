"""
GTCRN 文章配图生成脚本

生成 9 张关键架构图，对应文章各章节：
  FIG-01  GTCRN 整体架构          (第二章)
  FIG-02  ERB 频带压缩示意        (第三章)
  FIG-03  GT-Conv 块结构          (第四章)
  FIG-04  Dilated Conv 感受野     (第四章)
  FIG-05  因果填充对比            (第四章)
  FIG-06  DPGRNN 双路径           (第五章)
  FIG-07  SFE 操作示意            (第六章)
  FIG-08  TRA 模块流程            (第六章)
  FIG-09  流式 Cache 更新机制     (第八章)

每张图右下角带 [FIG-XX] 标识，方便文章和外部引用追踪。
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Windows 下强制 UTF-8 输出，避免 GBK 编码错误
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# 统一配色
COLOR_INPUT  = '#E3F2FD'   # 浅蓝 - 输入/数据
COLOR_PROC   = '#FFF3E0'   # 浅橙 - 处理模块
COLOR_RNN    = '#F3E5F5'   # 浅紫 - RNN 模块
COLOR_OUT    = '#E8F5E9'   # 浅绿 - 输出
COLOR_ATTN   = '#FFEBEE'   # 浅红 - 注意力
COLOR_SKIP   = '#FFFDE7'   # 浅黄 - 跳连
EDGE         = '#37474F'   # 深灰边框
TEXT         = '#212121'   # 文字


def add_fig_id(ax, fid):
    """右下角加图编号"""
    ax.text(0.99, 0.01, f'[{fid}]', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#9E9E9E',
            family='monospace')


def rounded_box(ax, x, y, w, h, label, color=COLOR_PROC, fontsize=10, weight='normal'):
    """画一个圆角矩形模块"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.3, edgecolor=EDGE, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color=TEXT, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.4, style='-|>'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ─────────────────────────────────────────────────────────────
# FIG-01  GTCRN 整体架构
# ─────────────────────────────────────────────────────────────
def fig01_overall():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_title('GTCRN 整体架构：U-Net 编解码 + 双路径 RNN 瓶颈',
                 fontsize=14, weight='bold', pad=10)

    # 输入
    rounded_box(ax, 0.5, 9.7, 3, 0.7, '带噪 STFT  (B, 257, T, 2)', COLOR_INPUT, 10, 'bold')
    arrow(ax, 2, 9.7, 2, 9.2)

    # 特征工程
    rounded_box(ax, 0.5, 8.5, 3, 0.7, '拼接 [mag, real, imag]\n(B, 3, T, 257)', COLOR_PROC, 9)
    arrow(ax, 2, 8.5, 2, 8.0)

    # ERB BM
    rounded_box(ax, 0.5, 7.3, 3, 0.7, 'ERB BM\n(B, 3, T, 129)', COLOR_PROC, 9, 'bold')
    arrow(ax, 2, 7.3, 2, 6.8)

    # SFE
    rounded_box(ax, 0.5, 6.1, 3, 0.7, 'SFE  (B, 9, T, 129)', COLOR_PROC, 9, 'bold')
    arrow(ax, 2, 6.1, 2, 5.6)

    # Encoder
    enc_layers = [
        ('Conv (1×5, ↓2)', '(B, 16, T, 65)', 5.0),
        ('Conv (1×5, ↓2)', '(B, 16, T, 33)', 4.0),
        ('GT-Conv  d=1', '(B, 16, T, 33)', 3.0),
        ('GT-Conv  d=2', '(B, 16, T, 33)', 2.0),
        ('GT-Conv  d=5', '(B, 16, T, 33)', 1.0),
    ]
    for i, (name, shape, y) in enumerate(enc_layers):
        rounded_box(ax, 0.5, y, 3, 0.7, f'{name}    {shape}', COLOR_PROC, 9)
        if i < len(enc_layers) - 1:
            arrow(ax, 2, y, 2, y - 0.3)

    # Encoder 外框
    enc_box = FancyBboxPatch((0.3, 0.7), 3.4, 4.85,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=1.5, edgecolor='#1976D2',
                             facecolor='none', linestyle='--')
    ax.add_patch(enc_box)
    ax.text(0.3, 5.65, 'Encoder', fontsize=11, color='#1976D2', weight='bold')

    # 跳连箭头（多条）
    skip_ys = [5.35, 4.35, 3.35, 2.35, 1.35]
    for y in skip_ys:
        ax.annotate('', xy=(10.3, y), xytext=(3.6, y),
                    arrowprops=dict(arrowstyle='-|>', color='#F9A825',
                                    lw=1.0, linestyle='--', alpha=0.7))
    ax.text(6.5, 5.55, '5 条 skip connection (add)', fontsize=9,
            color='#F9A825', style='italic')

    # DPGRNN
    arrow(ax, 2, 0.7, 6.5, 0.7)
    arrow(ax, 6.5, 0.7, 6.5, 1.0)
    rounded_box(ax, 5, 1.0, 3, 0.7, 'DPGRNN ×1', COLOR_RNN, 10, 'bold')
    arrow(ax, 6.5, 1.7, 6.5, 2.0)
    rounded_box(ax, 5, 2.0, 3, 0.7, 'DPGRNN ×2', COLOR_RNN, 10, 'bold')
    ax.text(6.5, 2.95, 'Intra: 双向 GRU(沿F)\nInter: 单向 GRU(沿T)',
            ha='center', fontsize=8, color='#7B1FA2', style='italic')

    arrow(ax, 6.5, 2.7, 6.5, 3.6)

    # Decoder
    dec_layers = [
        ('GT-DeConv  d=5', '(B, 16, T, 33)', 3.6),
        ('GT-DeConv  d=2', '(B, 16, T, 33)', 4.6),
        ('GT-DeConv  d=1', '(B, 16, T, 33)', 5.6),
        ('DeConv (1×5, ↑2)', '(B, 16, T, 65)', 6.6),
        ('DeConv (1×5, ↑2) tanh', '(B, 2, T, 129)', 7.6),
    ]
    for i, (name, shape, y) in enumerate(dec_layers):
        rounded_box(ax, 10.3, y, 3.2, 0.7, f'{name}\n{shape}', COLOR_PROC, 8.5)
        if i < len(dec_layers) - 1:
            arrow(ax, 11.9, y + 0.7, 11.9, y + 1.0)

    # Decoder 外框
    dec_box = FancyBboxPatch((10.1, 3.4), 3.6, 4.95,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             linewidth=1.5, edgecolor='#1976D2',
                             facecolor='none', linestyle='--')
    ax.add_patch(dec_box)
    ax.text(10.1, 8.45, 'Decoder', fontsize=11, color='#1976D2', weight='bold')

    # ERB BS
    arrow(ax, 11.9, 8.3, 11.9, 8.5)
    rounded_box(ax, 10.5, 8.5, 3, 0.7, 'ERB BS\n(B, 2, T, 257)', COLOR_PROC, 9, 'bold')
    arrow(ax, 11.9, 9.2, 11.9, 9.4)

    # Complex Mask
    rounded_box(ax, 10.5, 9.4, 3, 0.7, 'Complex Mask × 原谱', COLOR_OUT, 9, 'bold')
    arrow(ax, 11.9, 10.1, 11.9, 10.4)
    ax.text(11.9, 10.6, '增强 STFT', ha='center', fontsize=10, weight='bold', color='#2E7D32')

    add_fig_id(ax, 'FIG-01')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig01_overall_architecture.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-01  整体架构')


# ─────────────────────────────────────────────────────────────
# FIG-02  ERB 频带压缩
# ─────────────────────────────────────────────────────────────
def fig02_erb():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle('ERB 频带压缩：低频不动 + 高频聚合（257 → 129）',
                 fontsize=14, weight='bold')

    # 上图：频带分配示意
    ax1.set_xlim(0, 8000); ax1.set_ylim(0, 1.6)
    ax1.set_xlabel('频率 (Hz)', fontsize=10)
    ax1.set_yticks([])

    # 低频块（线性，每个 31.25 Hz 一格）
    for i in range(65):
        f = i * 31.25
        rect = Rectangle((f, 0.2), 31.25, 0.6,
                         facecolor='#42A5F5', edgecolor='white', linewidth=0.3)
        ax1.add_patch(rect)
    # 高频块（ERB，宽度递增）
    erb_widths = np.linspace(40, 350, 64)
    f_start = 65 * 31.25
    for i, w in enumerate(erb_widths):
        rect = Rectangle((f_start, 0.2), w, 0.6,
                         facecolor='#EF5350', edgecolor='white', linewidth=0.5)
        ax1.add_patch(rect)
        f_start += w
        if f_start > 8000: break

    ax1.text(1000, 1.15, '前 65 个 STFT 频点（每 31.25 Hz 一格）\n保持线性，零损失',
             ha='center', fontsize=9, color='#1565C0',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='none'))
    ax1.text(5500, 1.15, '高频 192 点 → 64 个 ERB 频带\n（带宽随频率递增）',
             ha='center', fontsize=9, color='#C62828',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='none'))

    # 分界线
    ax1.axvline(x=65*31.25, color='#37474F', linestyle='--', lw=1.5)
    ax1.text(65*31.25 + 30, 0.05, '≈ 2 kHz', fontsize=8, color='#37474F')

    # 下图：ERB 滤波器组三角窗
    ax2.set_xlim(2000, 8000); ax2.set_ylim(0, 1.15)
    ax2.set_xlabel('频率 (Hz)', fontsize=10)
    ax2.set_ylabel('滤波器权重', fontsize=10)
    ax2.set_title('64 个 ERB 三角滤波器（高频段才使用）', fontsize=11, pad=5)

    # 在 ERB 尺度均匀分布
    def hz2erb(f): return 21.4 * np.log10(0.00437 * f + 1)
    def erb2hz(e): return (10**(e/21.4) - 1) / 0.00437
    erb_low, erb_high = hz2erb(2000), hz2erb(8000)
    centers = erb2hz(np.linspace(erb_low, erb_high, 16))  # 只画 16 个示意
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(centers)))

    for i in range(len(centers)):
        if i == 0 or i == len(centers) - 1: continue
        left  = centers[i-1]
        peak  = centers[i]
        right = centers[i+1]
        xs = [left, peak, right]
        ys = [0, 1, 0]
        ax2.fill_between(xs, ys, alpha=0.5, color=cmap[i])
        ax2.plot(xs, ys, color=cmap[i], lw=1.2)

    ax2.grid(alpha=0.3)
    add_fig_id(ax2, 'FIG-02')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig02_erb_compression.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-02  ERB 频带压缩')


# ─────────────────────────────────────────────────────────────
# FIG-03  GT-Conv 块结构
# ─────────────────────────────────────────────────────────────
def fig03_gtconv():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('GT-Conv 块：ShuffleNetV2 + Depthwise Separable + Dilated + TRA',
                 fontsize=13, weight='bold', pad=10)

    # 输入
    rounded_box(ax, 5.5, 9, 3, 0.6, '输入  x (B, C, T, F)', COLOR_INPUT, 10, 'bold')
    arrow(ax, 7, 9, 7, 8.6)

    # Channel Split
    rounded_box(ax, 5.5, 8, 3, 0.6, 'Channel Split  dim=1', COLOR_PROC, 9.5)
    # 两分支
    arrow(ax, 6, 8, 3, 7.5)
    arrow(ax, 8, 8, 11, 7.5)

    # 左路（处理路径）
    rounded_box(ax, 1.5, 6.9, 3, 0.6, 'x1  (B, C/2, T, F)', COLOR_INPUT, 9)
    arrow(ax, 3, 6.9, 3, 6.5)
    rounded_box(ax, 1.5, 5.9, 3, 0.6, 'SFE  k=3   (B, 3C/2, T, F)', '#FFCCBC', 9, 'bold')
    arrow(ax, 3, 5.9, 3, 5.5)
    rounded_box(ax, 1.5, 4.9, 3, 0.6, '1×1 Conv → BN → PReLU', COLOR_PROC, 9)
    arrow(ax, 3, 4.9, 3, 4.5)
    rounded_box(ax, 1.5, 3.9, 3, 0.6, '因果左 pad → 3×3 depthwise (dilated)', '#FFE0B2', 8.5)
    arrow(ax, 3, 3.9, 3, 3.5)
    rounded_box(ax, 1.5, 2.9, 3, 0.6, 'BN → PReLU', COLOR_PROC, 9)
    arrow(ax, 3, 2.9, 3, 2.5)
    rounded_box(ax, 1.5, 1.9, 3, 0.6, '1×1 Conv → BN', COLOR_PROC, 9)
    arrow(ax, 3, 1.9, 3, 1.5)
    rounded_box(ax, 1.5, 0.9, 3, 0.6, 'TRA  时序注意力', COLOR_ATTN, 9, 'bold')

    # 右路（identity）
    rounded_box(ax, 9.5, 6.9, 3, 0.6, 'x2  (B, C/2, T, F)', COLOR_INPUT, 9)
    # 长 identity 箭头
    arrow(ax, 11, 6.9, 11, 1.1, lw=1.2)
    ax.text(11.4, 4, 'identity\n（不计算）', fontsize=9, color='#558B2F', style='italic')

    # 合并
    arrow(ax, 3, 0.9, 5.5, 0.5)
    arrow(ax, 11, 0.9, 8.5, 0.5)
    rounded_box(ax, 5.5, 0.1, 3, 0.45, 'Concat + Channel Shuffle', '#C8E6C9', 9.5, 'bold')

    # 输出注解
    ax.text(7, -0.4, '↓ 输出  (B, C, T, F)', ha='center', fontsize=10, weight='bold', color='#2E7D32')

    # 分组说明
    ax.text(0.2, 9.6, '关键：通道分两半，只算一半，最后 shuffle 让通道交错',
            fontsize=9, color='#1565C0', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1976D2'))

    add_fig_id(ax, 'FIG-03')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig03_gtconv_block.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-03  GT-Conv 块结构')


# ─────────────────────────────────────────────────────────────
# FIG-04  Dilated Conv 感受野
# ─────────────────────────────────────────────────────────────
def fig04_dilated_rf():
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    fig.suptitle('Dilated Conv 感受野累积：3 层串联，dilation = 1 / 2 / 5',
                 fontsize=13, weight='bold')

    n_frames = 20
    dilations = [1, 2, 5]
    layer_names = ['Layer 1：dilation=1\n直接感受野 3 帧 (≈48ms)',
                   'Layer 2：dilation=2\n累积感受野 7 帧 (≈112ms)',
                   'Layer 3：dilation=5\n累积感受野 17 帧 (≈270ms)']

    cumulative_rf = [3, 7, 17]

    for idx, (ax, d, name) in enumerate(zip(axes, dilations, layer_names)):
        ax.set_xlim(-0.5, n_frames - 0.5); ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_title(name, fontsize=10, loc='left', pad=4)

        # 当前帧 (t)
        t_pos = n_frames - 1
        for i in range(n_frames):
            # 灰色背景方块（所有帧）
            ax.add_patch(Rectangle((i - 0.4, 0.3), 0.8, 0.6,
                                    facecolor='#ECEFF1', edgecolor='#B0BEC5', lw=0.8))

        # 高亮感受野范围
        rf = cumulative_rf[idx]
        for i in range(t_pos - rf + 1, t_pos + 1):
            if i >= 0:
                ax.add_patch(Rectangle((i - 0.4, 0.3), 0.8, 0.6,
                                        facecolor='#FFCCBC', edgecolor='#E64A19', lw=1.2))

        # 当前帧（最深色）
        ax.add_patch(Rectangle((t_pos - 0.4, 0.3), 0.8, 0.6,
                                facecolor='#E64A19', edgecolor='#BF360C', lw=1.5))
        ax.text(t_pos, 0.6, 't', ha='center', va='center', fontsize=10,
                weight='bold', color='white')

        # 帧序号
        if idx == len(axes) - 1:
            for i in range(0, n_frames, 2):
                ax.text(i, 0.15, f't{i-t_pos:+d}' if i != t_pos else 't',
                        ha='center', fontsize=7, color='#546E7A')

        ax.set_xlabel('时间帧（每帧 16ms）' if idx == len(axes) - 1 else '', fontsize=10)
        for spine in ax.spines.values(): spine.set_visible(False)

    axes[0].text(0.5, 1.4, '说明：橙色方块 = 当前层输出依赖的输入帧；深红 = 当前输出帧 t；',
                 fontsize=9, color='#37474F', style='italic')

    add_fig_id(axes[-1], 'FIG-04')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig04_dilated_receptive_field.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-04  Dilated 感受野')


# ─────────────────────────────────────────────────────────────
# FIG-05  因果填充对比
# ─────────────────────────────────────────────────────────────
def fig05_causal_padding():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('因果填充：只在时间前侧 pad，绝不"偷看"未来',
                 fontsize=13, weight='bold')

    def draw_pad_demo(ax, title, mode):
        ax.set_xlim(-0.5, 11); ax.set_ylim(0, 5)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis('off')

        # 真实帧
        real_frames = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
        if mode == 'sym':
            seq = ['pad'] + real_frames + ['pad']
            colors = ['#CFD8DC'] + ['#90CAF9'] * 6 + ['#FFAB91']
            xs = list(range(len(seq)))
        else:
            seq = ['pad', 'pad'] + real_frames
            colors = ['#CFD8DC', '#CFD8DC'] + ['#90CAF9'] * 6
            xs = list(range(len(seq)))

        for x, s, c in zip(xs, seq, colors):
            ax.add_patch(Rectangle((x - 0.4, 3.5), 0.8, 0.8,
                                   facecolor=c, edgecolor=EDGE, lw=1.0))
            ax.text(x, 3.9, s, ha='center', va='center', fontsize=9)

        # 卷积窗口（kernel=3 当前帧 = x2，所以窗口覆盖 ?）
        if mode == 'sym':
            # 中心 x2，窗覆盖 [x1, x2, x3]，包含未来 x3
            win_start, win_end = 2, 4
            danger = '注意：用到了 x3 (未来帧!)'
            danger_color = '#D32F2F'
        else:
            # 当前 x2，窗覆盖 [pad, x0, x1, x2] 改为 (kernel=3 at t=2 with left-pad =2 means... )
            # 让我重新表述：当前帧 = x0，窗覆盖 [pad, pad, x0]
            # 或者：当前帧 = x2，窗覆盖 [x0, x1, x2]
            win_start, win_end = 4, 6  # x0, x1, x2
            danger = '正确：只看 [x0, x1, x2] 历史 + 当前'
            danger_color = '#2E7D32'

        # 红色框框出卷积窗
        rect = Rectangle((win_start - 0.5, 3.3), win_end - win_start + 1.0, 1.2,
                         facecolor='none', edgecolor=danger_color, lw=2.5, linestyle='--')
        ax.add_patch(rect)
        ax.text((win_start + win_end) / 2, 4.7,
                f'kernel=3 卷积窗', ha='center', fontsize=9, color=danger_color, weight='bold')

        # 输出
        out_idx = (win_start + win_end) // 2
        arrow(ax, out_idx, 3.2, out_idx, 2.2, color=danger_color, lw=1.5)
        ax.add_patch(Rectangle((out_idx - 0.4, 1.4), 0.8, 0.8,
                               facecolor='#FFCCBC' if mode == 'sym' else '#C8E6C9',
                               edgecolor=EDGE, lw=1.0))
        target_label = 'y2' if mode == 'sym' else 'y2'
        ax.text(out_idx, 1.8, target_label, ha='center', va='center', fontsize=9)

        # 警告标签
        ax.text(5, 0.6, danger, ha='center', fontsize=10, color=danger_color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#FFEBEE' if mode == 'sym' else '#E8F5E9',
                          edgecolor=danger_color))

    draw_pad_demo(ax1, '[错] 对称 padding（默认）：破坏因果', 'sym')
    draw_pad_demo(ax2, '[对] 只填左侧（GTCRN 用法）：严格因果', 'left')

    add_fig_id(ax2, 'FIG-05')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig05_causal_padding.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-05  因果填充对比')


# ─────────────────────────────────────────────────────────────
# FIG-06  DPGRNN 双路径
# ─────────────────────────────────────────────────────────────
def fig06_dpgrnn():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('DPGRNN：Intra-frame（沿 F，双向）+ Inter-frame（沿 T，单向）',
                 fontsize=13, weight='bold')

    # 共用：时频网格 8×6 (F × T)
    F, T = 8, 6
    cell = 0.5

    def draw_grid(ax, title, mode):
        ax.set_xlim(-0.5, T + 4); ax.set_ylim(-0.5, F + 1)
        ax.set_title(title, fontsize=11, pad=5)
        ax.axis('off')

        # 画网格
        for t in range(T):
            for f in range(F):
                if mode == 'intra':
                    # 第 3 帧整列高亮
                    fc = '#FFCCBC' if t == 2 else '#ECEFF1'
                else:
                    # 第 4 个频点整行高亮
                    fc = '#C5CAE9' if f == 3 else '#ECEFF1'
                ax.add_patch(Rectangle((t, f), 0.9, 0.9,
                                        facecolor=fc, edgecolor='#90A4AE', lw=0.6))

        # 轴标签
        ax.text(T / 2 - 0.5, -0.4, 'T（时间帧）→', ha='center', fontsize=10, color='#37474F')
        ax.text(-0.3, F / 2, 'F\n(频点)', va='center', fontsize=10, color='#37474F', rotation=0)

        # RNN 流向
        if mode == 'intra':
            # 在 t=2 那一列上下双向箭头
            for f in range(F - 1):
                ax.annotate('', xy=(2.45, f + 1.4), xytext=(2.45, f + 0.4),
                            arrowprops=dict(arrowstyle='<|-|>', color='#D81B60', lw=1.5))
            ax.text(T + 0.5, F - 1, 'Intra RNN：\n双向 GRU\n沿 F 跑',
                    fontsize=10, color='#D81B60', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE4EC', edgecolor='#D81B60'))
            ax.text(T + 0.5, F - 3, '频域无因果，\n可以双向看', fontsize=9, color='#880E4F', style='italic')
        else:
            # 在 f=3 那一行从左到右单向箭头
            for t in range(T - 1):
                ax.annotate('', xy=(t + 1.4, 3.45), xytext=(t + 0.4, 3.45),
                            arrowprops=dict(arrowstyle='-|>', color='#1976D2', lw=1.5))
            ax.text(T + 0.5, F - 1, 'Inter RNN：\n单向 GRU\n沿 T 跑',
                    fontsize=10, color='#1976D2', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1976D2'))
            ax.text(T + 0.5, F - 3, '时间有因果，\n只能单向', fontsize=9, color='#0D47A1', style='italic')

    draw_grid(ax1, 'Intra-frame RNN：建模一帧的频谱形态', 'intra')
    draw_grid(ax2, 'Inter-frame RNN：建模跨帧的时间依赖', 'inter')

    add_fig_id(ax2, 'FIG-06')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig06_dpgrnn_dual_path.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-06  DPGRNN 双路径')


# ─────────────────────────────────────────────────────────────
# FIG-07  SFE 操作示意
# ─────────────────────────────────────────────────────────────
def fig07_sfe():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('SFE 子带特征提取：把相邻 3 个频带的特征堆到通道维（零参数）',
                 fontsize=13, weight='bold', pad=10)

    # 输入：3 通道 × 7 频带
    ax.text(2, 6.3, '输入  (B, 3, T, F)', fontsize=11, weight='bold', color='#1565C0')
    channels_in = ['mag', 'real', 'imag']
    colors_in = ['#42A5F5', '#66BB6A', '#FFA726']

    for f in range(7):
        for c in range(3):
            rect = Rectangle((1 + f * 0.7, 4.8 - c * 0.5), 0.6, 0.4,
                             facecolor=colors_in[c], edgecolor=EDGE, lw=0.6, alpha=0.85)
            ax.add_patch(rect)
            if f == 0:
                ax.text(0.7, 5.0 - c * 0.5, channels_in[c], ha='right', va='center',
                        fontsize=8.5, color=EDGE)
        ax.text(1.3 + f * 0.7, 3.55, f'f={f}', ha='center', fontsize=8, color='#546E7A')

    # 高亮 f=3 及其邻居
    highlight = Rectangle((1 + 2 * 0.7 - 0.05, 3.75), 0.7 * 3 + 0.1, 1.5,
                          facecolor='none', edgecolor='#E64A19', lw=2.0, linestyle='--')
    ax.add_patch(highlight)
    ax.text(1 + 3 * 0.7 + 0.3, 3.3, '取 f-1, f, f+1', ha='center',
            fontsize=9, color='#E64A19', weight='bold')

    # SFE 操作箭头
    arrow(ax, 6.5, 4.5, 8.5, 4.5, lw=2.0)
    ax.text(7.5, 4.8, 'nn.Unfold\n+ reshape', ha='center', fontsize=10,
            color='#7B1FA2', weight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E5F5', edgecolor='#7B1FA2'))

    # 输出：9 通道
    ax.text(11.5, 6.3, '输出  (B, 9, T, F)', fontsize=11, weight='bold', color='#2E7D32')

    output_labels = ['mag(f-1)', 'real(f-1)', 'imag(f-1)',
                     'mag(f)', 'real(f)', 'imag(f)',
                     'mag(f+1)', 'real(f+1)', 'imag(f+1)']
    output_colors = ['#90CAF9', '#A5D6A7', '#FFCC80',
                     '#42A5F5', '#66BB6A', '#FFA726',
                     '#90CAF9', '#A5D6A7', '#FFCC80']

    for f in range(5):  # 只画 5 个频带示意
        for c in range(9):
            rect = Rectangle((9 + f * 0.5, 5.6 - c * 0.3), 0.4, 0.25,
                             facecolor=output_colors[c], edgecolor=EDGE, lw=0.4)
            ax.add_patch(rect)
            if f == 0:
                ax.text(8.8, 5.7 - c * 0.3, output_labels[c], ha='right', va='center',
                        fontsize=7, color=EDGE)
        ax.text(9.2 + f * 0.5, 2.5, f'f={f+1}', ha='center', fontsize=7, color='#546E7A')

    # 总结说明
    ax.text(7, 1.5,
            '"通道翻 3 倍" 是错觉——SFE 本身没有任何可训练参数（纯 reshape），\n'
            '它只是把"频域邻居"重排到通道维，让后面的 1×1 卷积间接看到频域上下文。',
            ha='center', fontsize=10, color='#37474F', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7', edgecolor='#F9A825'))

    add_fig_id(ax, 'FIG-07')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig07_sfe_operation.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-07  SFE 操作示意')


# ─────────────────────────────────────────────────────────────
# FIG-08  TRA 模块流程
# ─────────────────────────────────────────────────────────────
def fig08_tra():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('TRA 时序注意力：让网络学会"哪一帧重要"',
                 fontsize=13, weight='bold', pad=10)

    boxes = [
        (0.3,  2.5, 1.8, 1.0, '输入\nx (B,C,T,F)', COLOR_INPUT, 9),
        (2.5,  2.5, 1.8, 1.0, '能量聚合\nz = mean(x²)\n沿 F 维', '#FFE0B2', 8.5),
        (4.7,  2.5, 1.8, 1.0, 'GRU\nC → 2C', COLOR_RNN, 9),
        (6.9,  2.5, 1.8, 1.0, 'FC\n2C → C', COLOR_PROC, 9),
        (9.1,  2.5, 1.8, 1.0, 'Sigmoid\n→ (0,1)', '#F8BBD0', 9),
        (11.3, 2.5, 1.8, 1.0, '逐元素相乘\n沿 F 广播', COLOR_ATTN, 9.5),
    ]
    for (x, y, w, h, label, color, fs) in boxes:
        rounded_box(ax, x, y, w, h, label, color, fs)

    # 箭头连接
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        arrow(ax, x1, y, x2, y, lw=1.6)

    # 形状标注
    shapes = ['(B,C,T,F)', '(B,C,T)', '(B,T,2C)', '(B,C,T)', '(B,C,T,1)', '(B,C,T,F)']
    for i, s in enumerate(shapes):
        x = boxes[i][0] + boxes[i][2] + 0.1
        ax.text(x, 4.0, s, fontsize=7.5, color='#7B1FA2', style='italic')

    # 旁路：x 也要进入相乘
    ax.annotate('', xy=(12.0, 3.0), xytext=(0.9, 2.4),
                arrowprops=dict(arrowstyle='-|>', color='#1976D2',
                                lw=1.3, linestyle=':',
                                connectionstyle='arc3,rad=-0.5'))
    ax.text(6, 1.3, 'x 旁路直达相乘节点', fontsize=9, color='#1976D2', style='italic')

    # 关键说明
    ax.text(7, 0.5,
            '本质：用一帧内的能量包络 → GRU 算出"每帧的重要性权重" → 乘回去做软选通\n'
            '参数仅约 1.3K，但补上了卷积视野的盲区（"哪一帧需要重点处理"）',
            ha='center', fontsize=9.5, color='#37474F',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#C62828'))

    add_fig_id(ax, 'FIG-08')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig08_tra_module.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-08  TRA 模块流程')


# ─────────────────────────────────────────────────────────────
# FIG-09  流式 Cache 更新机制
# ─────────────────────────────────────────────────────────────
def fig09_stream_cache():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.set_title('流式 Cache：dilated conv 的历史帧 FIFO 队列（以 dilation=2, kernel=3 为例）',
                 fontsize=13, weight='bold', pad=10)

    # 三个时间步骤，每步骤一个横条
    timesteps = [
        ('t = 0  (启动)', [('pad', '#CFD8DC'), ('pad', '#CFD8DC'),
                            ('pad', '#CFD8DC'), ('pad', '#CFD8DC')],
                          ('x0 当前输入', '#FFAB91')),
        ('t = 1',       [('pad', '#CFD8DC'), ('pad', '#CFD8DC'),
                            ('pad', '#CFD8DC'), ('x0', '#90CAF9')],
                          ('x1 当前输入', '#FFAB91')),
        ('t = 2',       [('pad', '#CFD8DC'), ('pad', '#CFD8DC'),
                            ('x0', '#90CAF9'), ('x1', '#90CAF9')],
                          ('x2 当前输入', '#FFAB91')),
        ('t = 3',       [('pad', '#CFD8DC'), ('x0', '#90CAF9'),
                            ('x1', '#90CAF9'), ('x2', '#90CAF9')],
                          ('x3 当前输入', '#FFAB91')),
        ('t = 4',       [('x0', '#90CAF9'), ('x1', '#90CAF9'),
                            ('x2', '#90CAF9'), ('x3', '#90CAF9')],
                          ('x4 当前输入', '#FFAB91')),
    ]

    for i, (label, cache, current) in enumerate(timesteps):
        y = 7.5 - i * 1.6
        ax.text(0.3, y + 0.3, label, fontsize=10, weight='bold', color='#37474F')

        # cache 队列
        ax.text(2.5, y + 0.8, 'cache (FIFO)', fontsize=8, color='#7B1FA2', ha='center')
        for j, (lbl, c) in enumerate(cache):
            rect = Rectangle((1.5 + j * 0.8, y), 0.7, 0.6,
                             facecolor=c, edgecolor=EDGE, lw=0.8)
            ax.add_patch(rect)
            ax.text(1.85 + j * 0.8, y + 0.3, lbl, ha='center', va='center', fontsize=9)

        # 加号
        ax.text(5.1, y + 0.3, '⊕', fontsize=16, color='#1976D2', ha='center', va='center')

        # 当前输入
        ax.text(5.8, y + 0.8, 'frame[t]', fontsize=8, color='#E64A19', ha='left')
        cur_lbl, cur_c = current
        rect = Rectangle((5.7, y), 0.9, 0.6, facecolor=cur_c, edgecolor='#BF360C', lw=1.4)
        ax.add_patch(rect)
        ax.text(6.15, y + 0.3, cur_lbl.split()[0], ha='center', va='center',
                fontsize=10, weight='bold')

        # 卷积
        ax.annotate('', xy=(7.6, y + 0.3), xytext=(6.7, y + 0.3),
                    arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.5))
        rounded_box(ax, 7.6, y, 1.5, 0.6, 'StreamConv\n(d=2)', COLOR_PROC, 8.5)

        ax.annotate('', xy=(10.0, y + 0.3), xytext=(9.1, y + 0.3),
                    arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.5))

        # 输出
        out_label = f'y{i}'
        rect = Rectangle((10.0, y), 0.9, 0.6, facecolor='#A5D6A7', edgecolor='#2E7D32', lw=1.2)
        ax.add_patch(rect)
        ax.text(10.45, y + 0.3, out_label, ha='center', va='center', fontsize=10, weight='bold')

        # cache 更新箭头
        ax.annotate('', xy=(3.5, y - 1.2), xytext=(6.15, y - 0.15),
                    arrowprops=dict(arrowstyle='-|>', color='#7B1FA2', lw=1.0,
                                    linestyle='--', alpha=0.7,
                                    connectionstyle='arc3,rad=-0.3'))

        if i == 0:
            ax.text(11.5, y + 0.3, 'cache 更新：\n弹出最早一帧\n推入当前 x', fontsize=8,
                    color='#7B1FA2', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#7B1FA2'))

    # 底部说明
    ax.text(7, -0.3,
            'cache 大小 = (kernel−1) × dilation 帧。GTCRN 三层 GT-Conv 加起来共需 2+4+10 = 16 帧缓存（约 70 KB FP32）',
            ha='center', fontsize=10, color='#37474F', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7', edgecolor='#F9A825'))

    add_fig_id(ax, 'FIG-09')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig09_stream_cache.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-09  流式 Cache 更新')


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('开始生成 GTCRN 文章配图...')
    print(f'输出目录：{OUTDIR}\n')

    fig01_overall()
    fig02_erb()
    fig03_gtconv()
    fig04_dilated_rf()
    fig05_causal_padding()
    fig06_dpgrnn()
    fig07_sfe()
    fig08_tra()
    fig09_stream_cache()

    print('\n全部完成。共 9 张图。')
