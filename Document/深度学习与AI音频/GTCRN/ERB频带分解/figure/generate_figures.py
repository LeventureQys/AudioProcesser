import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

OUTDIR = os.path.dirname(os.path.abspath(__file__))

COLOR_INPUT = '#E3F2FD'
COLOR_PROC = '#FFF3E0'
COLOR_ERB = '#F3E5F5'
COLOR_MEL = '#E8F5E9'
COLOR_OUT = '#FFEBEE'
EDGE = '#37474F'


def add_fig_id(ax, fid):
    ax.text(0.99, 0.01, f'[{fid}]', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#9E9E9E',
            family='monospace')


def rounded_box(ax, x, y, w, h, label, color, fontsize=10, weight='normal'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.02,rounding_size=0.08',
                         linewidth=1.3, edgecolor=EDGE, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fontsize, weight=weight, color='#212121')


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.5, style='-|>'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700.0)


def hz_to_erb(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def hz_to_bark(f):
    return 6 * np.arcsinh(f / 600.0)


def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('常见音频预处理流程：从波形到适合算法建模的表示', fontsize=13, weight='bold', pad=10)

    boxes = [
        (0.4, 1.7, 2.0, 1.0, '原始波形', COLOR_INPUT),
        (2.8, 1.7, 2.0, 1.0, '分帧 + 加窗', COLOR_PROC),
        (5.2, 1.7, 2.0, 1.0, 'STFT / 子带分解', COLOR_PROC),
        (7.6, 1.7, 2.0, 1.0, '感知映射\nMel / ERB / Bark', COLOR_ERB),
        (10.0, 1.7, 2.0, 1.0, '压缩 / 归一化\n包络 / 倒谱', COLOR_OUT),
    ]

    for x, y, w, h, label, color in boxes:
        rounded_box(ax, x, y, w, h, label, color, 10, 'bold')

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        arrow(ax, x1, 2.2, x2, 2.2)

    ax.text(6.5, 0.7,
            '目标不是“处理得更复杂”，而是把原始波形变成更适合当前任务的表示。',
            ha='center', fontsize=10, color='#37474F',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFDE7', edgecolor='#F9A825'))

    add_fig_id(ax, 'FIG-01')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig01_audio_preprocess_pipeline.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-01')


def fig02_scales():
    fig, ax = plt.subplots(figsize=(11, 6))
    freqs = np.linspace(0, 8000, 500)
    mel = hz_to_mel(freqs)
    erb = hz_to_erb(freqs)
    bark = hz_to_bark(freqs)

    ax.plot(freqs, freqs / 1000.0, label='Linear (scaled)', lw=2, color='#1976D2')
    ax.plot(freqs, mel / mel.max() * 8, label='Mel', lw=2, color='#43A047')
    ax.plot(freqs, bark / bark.max() * 8, label='Bark', lw=2, color='#FB8C00')
    ax.plot(freqs, erb / erb.max() * 8, label='ERB', lw=2, color='#8E24AA')

    ax.set_title('线性频率与常见感知频率刻度对比', fontsize=13, weight='bold')
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('相对刻度值')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.text(1800, 6.9, '共同趋势：低频展开得更细，高频压得更紧', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#F9A825'))

    add_fig_id(ax, 'FIG-02')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig02_frequency_scales.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-02')


def fig03_erb_filterbank():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_title('ERB 频带滤波器组示意：低频更密，高频更宽', fontsize=13, weight='bold')
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('权重')
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 1.15)

    centers = np.array([150, 300, 500, 800, 1200, 1800, 2600, 3800, 5200, 6800], dtype=float)
    widths = np.array([120, 140, 170, 220, 300, 420, 650, 950, 1300, 1700], dtype=float)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(centers)))

    for center, width, color in zip(centers, widths, colors):
        left = max(0, center - width / 2)
        right = min(8000, center + width / 2)
        xs = [left, center, right]
        ys = [0, 1, 0]
        ax.fill_between(xs, ys, alpha=0.35, color=color)
        ax.plot(xs, ys, lw=1.8, color=color)

    ax.grid(alpha=0.25)
    ax.text(3200, 1.03, '频率越高，子带越宽', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#8E24AA'))

    add_fig_id(ax, 'FIG-03')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig03_erb_filterbank.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-03')


def fig04_family_map():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('常见音频预处理方法分类：它们关注的是不同层面的信息', fontsize=13, weight='bold', pad=10)

    rounded_box(ax, 5.0, 5.8, 3.0, 0.8, '音频预处理', COLOR_INPUT, 11, 'bold')

    rounded_box(ax, 0.8, 3.8, 2.4, 1.0, '时间域\n包络 / 预加重 / VAD', COLOR_PROC, 10, 'bold')
    rounded_box(ax, 3.8, 3.8, 2.4, 1.0, '时频域\nSTFT / 频谱图', COLOR_PROC, 10, 'bold')
    rounded_box(ax, 6.8, 3.8, 2.4, 1.0, '感知频带\nMel / ERB / Bark', COLOR_ERB, 10, 'bold')
    rounded_box(ax, 9.8, 3.8, 2.4, 1.0, '低维压缩\nMFCC / 倒谱 / PLP', COLOR_OUT, 10, 'bold')

    for x in [2.0, 5.0, 8.0, 11.0]:
        arrow(ax, 6.5, 5.8, x, 4.8)

    rounded_box(ax, 0.8, 1.4, 2.4, 1.0, '关注能量轮廓', '#E8F5E9', 10)
    rounded_box(ax, 3.8, 1.4, 2.4, 1.0, '保留局部频谱结构', '#E8F5E9', 10)
    rounded_box(ax, 6.8, 1.4, 2.4, 1.0, '贴近听觉分辨率', '#E8F5E9', 10)
    rounded_box(ax, 9.8, 1.4, 2.4, 1.0, '强调低维稳健表示', '#E8F5E9', 10)

    for x in [2.0, 5.0, 8.0, 11.0]:
        arrow(ax, x, 3.8, x, 2.4)

    add_fig_id(ax, 'FIG-04')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig04_preprocess_family_map.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-04')


def fig05_erb_compute_process():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('ERB 是怎么从线性频谱算出来的', fontsize=13, weight='bold', pad=10)

    rounded_box(ax, 0.5, 4.9, 2.2, 1.0, '一帧线性频谱\n257 个频点', COLOR_INPUT, 10, 'bold')
    rounded_box(ax, 3.3, 4.9, 2.5, 1.0, 'ERB 滤波器组\n64 个子带模板', COLOR_ERB, 10, 'bold')
    rounded_box(ax, 6.5, 4.9, 2.6, 1.0, '逐子带加权汇聚\n不是简单平均', COLOR_PROC, 10, 'bold')
    rounded_box(ax, 10.0, 4.9, 2.0, 1.0, '一帧 ERB 向量\n64 个值', COLOR_OUT, 10, 'bold')

    arrow(ax, 2.7, 5.4, 3.3, 5.4)
    arrow(ax, 5.8, 5.4, 6.5, 5.4)
    arrow(ax, 9.1, 5.4, 10.0, 5.4)

    for i in range(10):
        rect = Rectangle((0.7 + i * 0.17, 3.2), 0.12, 0.8 + 0.2 * np.sin(i), facecolor='#90CAF9', edgecolor='white')
        ax.add_patch(rect)
    ax.text(1.6, 2.8, '线性频点 x0 ... x256', fontsize=9.5, ha='center', color='#1565C0')

    centers = [3.8, 4.45, 5.15]
    widths = [0.8, 1.1, 1.4]
    colors = ['#BA68C8', '#9575CD', '#7986CB']
    for c, w, color in zip(centers, widths, colors):
        xs = [c - w / 2, c, c + w / 2]
        ys = [0, 1, 0]
        ax.fill_between(xs, ys, [2.0, 2.0, 2.0], color=color, alpha=0.45)
        ax.plot(xs, np.array(ys) + 2.0, color=color, lw=1.8)
    ax.text(4.5, 1.6, '重叠的子带滤波器', fontsize=9.5, ha='center', color='#6A1B9A')

    ax.text(7.8, 2.2,
            '第 i 个 ERB 子带值\n= 一段频点 × 对应权重 后求和',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF8E1', edgecolor='#F9A825'))

    for i in range(6):
        rect = Rectangle((10.2 + i * 0.22, 3.15), 0.14, 0.35 + 0.45 * (i % 3 + 1) / 3, facecolor='#EF9A9A', edgecolor='white')
        ax.add_patch(rect)
    ax.text(11.0, 2.8, 'e0 ... e63', fontsize=9.5, ha='center', color='#C62828')

    ax.text(6.5, 0.6,
            '单帧输出是一组 1 维子带值；整段语音时，就是“时间帧 × ERB 子带”的二维特征图。',
            ha='center', fontsize=10, color='#37474F',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F5E9', edgecolor='#43A047'))

    add_fig_id(ax, 'FIG-05')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig05_erb_compute_process.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-05')


def fig06_gtcrn_erb_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    ax.set_title('GTCRN 中 ERB、复数谱与波形的关系', fontsize=13, weight='bold', pad=10)

    boxes = [
        (0.4, 3.8, 1.8, 0.95, '波形', COLOR_INPUT),
        (2.6, 3.8, 2.0, 0.95, 'STFT\n复数谱', COLOR_PROC),
        (5.0, 3.8, 2.2, 0.95, '三路特征\nmag / real / imag', COLOR_PROC),
        (7.6, 3.8, 1.8, 0.95, 'ERB BM', COLOR_ERB),
        (9.8, 3.8, 1.8, 0.95, 'GTCRN 主干\n编码/建模/解码', COLOR_OUT),
        (12.0, 3.8, 1.4, 0.95, 'ERB BS', COLOR_ERB),
    ]
    for x, y, w, h, label, color in boxes:
        rounded_box(ax, x, y, w, h, label, color, 9.5, 'bold')
    for i in range(len(boxes) - 1):
        arrow(ax, boxes[i][0] + boxes[i][2], 4.25, boxes[i + 1][0], 4.25)

    rounded_box(ax, 5.0, 1.5, 2.4, 0.95, '原始复数谱保留为参考', '#E8F5E9', 9.5, 'bold')
    rounded_box(ax, 8.5, 1.5, 2.4, 0.95, '输出复数掩码\n映回线性频点', '#FFECB3', 9.5, 'bold')
    rounded_box(ax, 11.5, 1.5, 2.0, 0.95, '复数乘法后\nISTFT 回波形', '#C8E6C9', 9.5, 'bold')

    arrow(ax, 6.2, 3.8, 6.2, 2.45)
    arrow(ax, 12.7, 3.8, 9.7, 2.45)
    arrow(ax, 7.4, 1.98, 8.5, 1.98)
    arrow(ax, 10.9, 1.98, 11.5, 1.98)

    ax.text(7.2, 5.3, 'ERB 是中间压缩表示', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#8E24AA'))
    ax.text(9.7, 0.55,
            '关键：GTCRN 不是只拿 ERB 能量回波形，\n而是始终围绕复数频谱做掩码修正。',
            ha='center', fontsize=10, color='#37474F',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF8E1', edgecolor='#F9A825'))

    add_fig_id(ax, 'FIG-06')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig06_gtcrn_erb_pipeline.png'), bbox_inches='tight', facecolor='white')
    plt.close()
    print('  ✓ FIG-06')


if __name__ == '__main__':
    print('开始生成 ERB 说明文配图...')
    print(f'输出目录：{OUTDIR}\n')
    fig01_pipeline()
    fig02_scales()
    fig03_erb_filterbank()
    fig04_family_map()
    fig05_erb_compute_process()
    fig06_gtcrn_erb_pipeline()
    print('\n全部完成。共 6 张图。')
