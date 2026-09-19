"""
Regenerate charts with generic signal labels (no financial terminology).
"""
import openpyxl
import numpy as np
from datetime import datetime
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings, io, sys, os
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r'D:\workshop\Github\AudioProcesser\Document\quant\document'
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(filepath):
    wb = openpyxl.load_workbook(filepath)
    ws = wb.active
    dates, opens, highs, lows, closes = [], [], [], [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        d = datetime.strptime(str(row[0]), '%Y%m%d').date()
        dates.append(d)
        opens.append(float(row[6]))
        highs.append(float(row[7]))
        lows.append(float(row[8]))
        closes.append(float(row[9]))
    return (np.array(dates), np.array(opens), np.array(highs),
            np.array(lows), np.array(closes))

d22, o22, h22, l22, c22 = load_data(r'D:\workshop\Github\AudioProcesser\Document\quant\source\上证指数\2022.xlsx')
d25, o25, h25, l25, c25 = load_data(r'D:\workshop\Github\AudioProcesser\Document\quant\source\上证指数\2025.xlsx')

datasets = {
    'A': {'dates': d22, 'close': c22, 'name': 'Signal A (样本A)'},
    'B': {'dates': d25, 'close': c25, 'name': 'Signal B (样本B)'},
}

# ========== DSP METHODS ==========
def kalman_filter(values):
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.5, 0], [0, 0.05]])
    R = np.array([[100.0]])
    P = np.eye(2) * 100
    x = np.array([values[0], 0.0])
    n = len(values)
    est = np.zeros(n); est[0] = values[0]
    for k in range(1, n):
        x = F @ x; P = F @ P @ F.T + Q
        z = values[k]; y_res = z - H @ x
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y_res; P = (np.eye(2) - K @ H) @ P
        est[k] = x[0]
    return est

def adaptive_kalman(values):
    F = np.array([[1, 1], [0, 1]]); H = np.array([[1, 0]])
    Q = np.array([[0.5, 0], [0, 0.05]])
    P = np.eye(2) * 100; x = np.array([values[0], 0.0])
    R = np.array([[100.0]])
    innovations = []; window = 10
    n = len(values); est = np.zeros(n); est[0] = values[0]
    for k in range(1, n):
        x = F @ x; P = F @ P @ F.T + Q
        z = values[k]; y_res = z - H @ x
        innovations.append(float(y_res.item()))
        if len(innovations) > window: innovations.pop(0)
        if len(innovations) >= 3:
            innov_var = np.var(innovations)
            pred_var = float((H @ P @ H.T).item())
            R = np.array([[np.clip(max(innov_var - pred_var, 10.0), 10.0, 2500.0)]])
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y_res; P = (np.eye(2) - K @ H) @ P
        est[k] = x[0]
    return est

def holt_winters(values, alpha=0.3, beta=0.1, gamma=0.05, period=5):
    n = len(values)
    level = np.zeros(n); trend = np.zeros(n)
    level[0] = values[0]; trend[0] = values[1] - values[0] if n > 1 else 0
    s = np.zeros(period)
    for i in range(min(period, n)):
        s[i] = values[i] - level[0]
    est = np.zeros(n); est[0] = values[0]
    for t in range(1, n):
        if t < period:
            est[t] = values[t]; level[t] = values[t]
            trend[t] = values[t] - values[t-1] if t > 0 else 0
            continue
        idx = t % period
        level[t] = alpha * (values[t] - s[idx]) + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
        s[idx] = gamma * (values[t] - level[t]) + (1 - gamma) * s[idx]
        est[t] = level[t] + trend[t] + s[idx]
    return est

def butterworth_lowpass(values, cutoff=0.05, order=2):
    nyq = 0.5
    b, a = signal.butter(order, cutoff / nyq, btype='low')
    return signal.filtfilt(b, a, values)

def savitzky_golay(values, window=15, order=3):
    if window % 2 == 0: window += 1
    return signal.savgol_filter(values, window, order)

def sma(values, window=20):
    return np.convolve(values, np.ones(window)/window, mode='same')

def ema(values, span=20):
    alpha = 2.0 / (span + 1)
    n = len(values); result = np.zeros(n); result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    return result

def mse(a, b): return np.mean((a - b) ** 2)
def corr(a, b): return np.corrcoef(a, b)[0, 1]
def max_lag_approx(orig, filt, max_s=30):
    a = (orig - np.mean(orig)) / np.std(orig)
    b = (filt - np.mean(filt)) / np.std(filt)
    c = np.correlate(a, b, mode='full')
    return np.argmax(c[len(c)//2:len(c)//2+max_s])

# Compute all results
all_results = {}
for key, ds in datasets.items():
    c = ds['close']
    res = {}
    
    # DSP
    res['Kalman (Standard)'] = kalman_filter(c)
    res['Kalman (Adaptive)'] = adaptive_kalman(c)
    
    # Holt-Winters grid search
    best, best_m = None, float('inf')
    for a in [0.1, 0.2, 0.3, 0.5]:
        for bt in [0.05, 0.1, 0.2]:
            for g in [0.01, 0.05, 0.1]:
                hw = holt_winters(c, alpha=a, beta=bt, gamma=g, period=5)
                m = mse(c, hw)
                if m < best_m: best_m = m; best = hw
    res['Holt-Winters'] = best
    
    res['Gaussian (sigma=3)'] = gaussian_filter1d(c, sigma=3)
    res['Gaussian (sigma=5)'] = gaussian_filter1d(c, sigma=5)
    res['Butterworth (fc=0.03)'] = butterworth_lowpass(c, cutoff=0.03)
    res['Butterworth (fc=0.05)'] = butterworth_lowpass(c, cutoff=0.05)
    res['Savitzky-Golay (w=11)'] = savitzky_golay(c, window=11)
    res['Savitzky-Golay (w=21)'] = savitzky_golay(c, window=21)
    
    # Baseline: simple moving averages
    res['SMA (window=20)'] = sma(c, 20)
    res['SMA (window=60)'] = sma(c, 60)
    res['EMA (span=12)'] = ema(c, 12)
    res['EMA (span=26)'] = ema(c, 26)
    
    all_results[key] = res

# ======================== Generate generic charts ========================
print("Generating generic charts...")

# Chart 1: Raw data overview
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, (key, ds), color in zip(axes, datasets.items(), ['#e74c3c', '#2980b9']):
    d = ds['dates']; c = ds['close']
    ax.plot(d, c, color=color, linewidth=0.7)
    ax.fill_between(d, c, alpha=0.08, color=color)
    hi_idx = np.argmax(c); lo_idx = np.argmin(c)
    ax.annotate(f'{c[hi_idx]:.0f}', xy=(d[hi_idx], c[hi_idx]), fontsize=9, color=color, ha='center')
    ax.annotate(f'{c[lo_idx]:.0f}', xy=(d[lo_idx], c[lo_idx]), fontsize=9, color='#27ae60', ha='center')
    ax.set_title(ds['name'], fontsize=13, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2)
    ret = np.diff(c) / c[:-1] * 100
    stats_text = f'Range: {c.min():.0f} ~ {c.max():.0f}\nStd(return): {ret.std():.2f}%\nMean(return): {ret.mean():.2f}%'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
fig.suptitle('Raw Signal Overview / 原始信号概览', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '00_raw_data.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [1/8] Raw data")

# Chart 2: Kalman
fig, axes = plt.subplots(2, 2, figsize=(22, 12))
for row_idx, key in enumerate(['A', 'B']):
    ds = datasets[key]; c = ds['close']; d = ds['dates']
    kf = all_results[key]['Kalman (Standard)']; akf = all_results[key]['Kalman (Adaptive)']
    for col_idx, (vals, name, color) in enumerate([(kf, 'Standard Kalman', '#e74c3c'), (akf, 'Adaptive Kalman', '#2980b9')]):
        ax = axes[row_idx, col_idx]
        ax.plot(d, c, color='#bdc3c7', linewidth=0.5, alpha=0.7, label='Raw Signal')
        ax.plot(d, vals, color=color, linewidth=1.2, label=name)
        ax.set_title(f'{ds["name"]} — {name}\n{name}', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
        ax.text(0.98, 0.95, f'MSE:{mse(c,vals):.0f}\nLag:{max_lag_approx(c,vals)}d', transform=ax.transAxes,
                fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.85))
fig.suptitle('Kalman Filter — Standard vs Adaptive\n卡尔曼滤波对比', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_kalman.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [2/8] Kalman")

# Chart 3: Holt-Winters
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, key in zip(axes, ['A', 'B']):
    ds = datasets[key]; c = ds['close']; d = ds['dates']
    hw = all_results[key]['Holt-Winters']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.5, alpha=0.7, label='Raw Signal')
    ax.plot(d, hw, color='#e67e22', linewidth=1.2, label='Holt-Winters')
    ax.set_title(ds['name'], fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
    ax.text(0.98, 0.95, f'MSE:{mse(c,hw):.0f}\nLag:{max_lag_approx(c,hw)}d', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.85))
fig.suptitle('Holt-Winters Triple Exponential Smoothing\nHolt-Winters 三次指数平滑', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '02_holtwinters.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [3/8] Holt-Winters")

# Chart 4: Gaussian + Butterworth comparison (combined)
fig, axes = plt.subplots(2, 2, figsize=(22, 12))
for row_idx, key in enumerate(['A', 'B']):
    ds = datasets[key]; c = ds['close']; d = ds['dates']
    res = all_results[key]
    
    # Gaussian
    ax = axes[row_idx, 0]
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    for s, col, lbl in [(1,'#f39c12','sigma=1'),(3,'#e67e22','sigma=3'),(5,'#d35400','sigma=5'),(8,'#c0392b','sigma=8')]:
        gs = gaussian_filter1d(c, sigma=s)
        ax.plot(d, gs, color=col, linewidth=1.0, label=f'{lbl} MSE:{mse(c,gs):.0f}')
    ax.set_title(f'{ds["name"]} — Gaussian Smoothing\n高斯平滑降噪', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)
    
    # Butterworth
    ax = axes[row_idx, 1]
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    for fc, col, lbl in [(0.01,'#3498db','fc=0.01'),(0.03,'#2980b9','fc=0.03'),(0.05,'#1a5276','fc=0.05'),(0.08,'#0e2f44','fc=0.08')]:
        bw = butterworth_lowpass(c, cutoff=fc)
        ax.plot(d, bw, color=col, linewidth=1.0, label=f'{lbl} MSE:{mse(c,bw):.0f}')
    ax.set_title(f'{ds["name"]} — Butterworth Low-pass\n巴特沃斯低通滤波', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)

fig.suptitle('Gaussian Smoothing vs Butterworth Low-Pass\n高斯平滑 vs 巴特沃斯低通', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '03_gaussian_butter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [4/8] Gaussian + Butterworth")

# Chart 5: Savitzky-Golay
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, key in zip(axes, ['A', 'B']):
    ds = datasets[key]; c = ds['close']; d = ds['dates']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    for w, col, lbl in [(7,'#8e44ad','w=7'),(15,'#6c3483','w=15'),(31,'#4a235a','w=31'),(51,'#2c0e37','w=51')]:
        sg = savitzky_golay(c, window=w)
        ax.plot(d, sg, color=col, linewidth=1.0, label=f'{lbl} MSE:{mse(c,sg):.0f}')
    ax.set_title(ds['name'], fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)
fig.suptitle('Savitzky-Golay Filter\nSavitzky-Golay滤波', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '04_savgol.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [5/8] Savitzky-Golay")

# Chart 6: Baseline comparison — moving averages
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, key in zip(axes, ['A', 'B']):
    ds = datasets[key]; c = ds['close']; d = ds['dates']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    ax.plot(d, sma(c, 5), color='#f39c12', linewidth=0.8, alpha=0.7, label='SMA(5)')
    ax.plot(d, sma(c, 20), color='#e74c3c', linewidth=1.0, label='SMA(20)')
    ax.plot(d, sma(c, 60), color='#c0392b', linewidth=1.2, label='SMA(60)')
    ax.plot(d, ema(c, 12), '--', color='#2980b9', linewidth=0.8, label='EMA(12)')
    ax.plot(d, ema(c, 26), '--', color='#1a5276', linewidth=1.0, label='EMA(26)')
    ax.set_title(ds['name'], fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=3)
fig.suptitle('Baseline — Simple & Exponential Moving Average\n基线方法——滑动平均', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '05_baseline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [6/8] Baseline MA")

# Chart 7: Frequency domain analysis
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, key in zip(axes, ['A', 'B']):
    ds = datasets[key]; c = ds['close']
    c_detrended = c - np.mean(c)
    n_fft = len(c)
    fft_vals = np.abs(np.fft.rfft(c_detrended))
    freqs = np.fft.rfftfreq(n_fft)
    periods = 1.0 / (freqs[1:] + 1e-10)
    ax.semilogx(periods, fft_vals[1:] / np.max(fft_vals[1:]), color='#2c3e50', linewidth=0.8)
    ax.axvline(x=5, color='#e74c3c', linestyle='--', alpha=0.5, label='T=5')
    ax.axvline(x=20, color='#2980b9', linestyle='--', alpha=0.5, label='T=20')
    ax.axvline(x=60, color='#27ae60', linestyle='--', alpha=0.5, label='T=60')
    ax.set_xlabel('Period (samples)'); ax.set_ylabel('Normalized Magnitude')
    ax.set_title(ds['name'], fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
fig.suptitle('Frequency Domain Analysis — Where Is the Noise?\n频域分析——噪声在哪里？', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '06_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [7/8] Frequency")

# Chart 8: Comprehensive dashboard
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: Normalized comparison
ax0 = fig.add_subplot(gs[0, :])
for key, col in [('A', '#e74c3c'), ('B', '#2980b9')]:
    ds = datasets[key]; c = ds['close']; d = ds['dates']
    c_norm = (c - c[0]) / c[0] * 100
    ax0.plot(d, c_norm, color=col, linewidth=0.8, label=ds['name'])
ax0.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax0.set_title('Normalized Comparison (% change from start)\n归一化对比', fontsize=13, fontweight='bold')
ax0.xaxis.set_major_locator(mdates.MonthLocator()); ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax0.legend(fontsize=10); ax0.grid(True, alpha=0.2)

# Row 2: MSE, Lag, Correlation bar charts
method_keys = ['Kalman (Standard)', 'Kalman (Adaptive)', 'Holt-Winters',
    'Gaussian (sigma=3)', 'Gaussian (sigma=5)', 'Butterworth (fc=0.03)',
    'Butterworth (fc=0.05)', 'Savitzky-Golay (w=11)', 'Savitzky-Golay (w=21)',
    'SMA (window=20)', 'SMA (window=60)', 'EMA (span=12)', 'EMA (span=26)']
short_names = ['Kalman\nStd', 'Kalman\nAdp', 'Holt-\nWinters', 'Gauss\nsig=3', 'Gauss\nsig=5',
    'BW\nfc=.03', 'BW\nfc=.05', 'S-G\nw=11', 'S-G\nw=21', 'SMA\nw=20', 'SMA\nw=60', 'EMA\nsp=12', 'EMA\nsp=26']
y = np.arange(len(short_names))

for col_idx, (metric_fn, title, xlabel) in enumerate([
    (lambda c,v: mse(c,v), 'MSE by Method / MSE对比', 'MSE'),
    (lambda c,v: max_lag_approx(c,v), 'Lag by Method / 延迟对比', 'Lag (samples)'),
    (lambda c,v: corr(c,v), 'Correlation with Raw / 相关性', 'Correlation'),
]):
    ax = fig.add_subplot(gs[1, col_idx])
    vals_a = [metric_fn(datasets['A']['close'], all_results['A'][k]) for k in method_keys]
    vals_b = [metric_fn(datasets['B']['close'], all_results['B'][k]) for k in method_keys]
    ax.barh(y - 0.2, vals_a, 0.4, color='#e74c3c', alpha=0.8, label='Signal A')
    ax.barh(y + 0.2, vals_b, 0.4, color='#2980b9', alpha=0.8, label='Signal B')
    ax.set_yticks(y); ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()
    if col_idx == 2:
        ax.set_xlim(0.7, 1.01)

# Row 3: Best overlay + philosophy
ax4 = fig.add_subplot(gs[2, 0])
ds = datasets['A']; c = ds['close']; d = ds['dates']
ax4.plot(d, c, color='#bdc3c7', linewidth=0.3, alpha=0.4, label='Raw')
ax4.plot(d, all_results['A']['Kalman (Standard)'], color='#e74c3c', linewidth=1.0, label='Kalman')
ax4.plot(d, all_results['A']['Holt-Winters'], color='#e67e22', linewidth=1.0, label='Holt-Winters')
ax4.set_title(f'{ds["name"]} — Best DSP Overlay\n最优方法叠加', fontsize=11, fontweight='bold')
ax4.xaxis.set_major_locator(mdates.MonthLocator()); ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)

ax5 = fig.add_subplot(gs[2, 1])
ds = datasets['B']; c = ds['close']; d = ds['dates']
ax5.plot(d, c, color='#bdc3c7', linewidth=0.3, alpha=0.4, label='Raw')
ax5.plot(d, all_results['B']['Holt-Winters'], color='#e67e22', linewidth=1.0, label='Holt-Winters')
ax5.plot(d, all_results['B']['Savitzky-Golay (w=11)'], color='#27ae60', linewidth=1.0, label='S-G (w=11)')
ax5.set_title(f'{ds["name"]} — Best DSP Overlay\n最优方法叠加', fontsize=11, fontweight='bold')
ax5.xaxis.set_major_locator(mdates.MonthLocator()); ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.2)

ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
philosophy_text = (
    "Key Observations:\n\n"
    "1. Holt-Winters ranks #1 in both signals\n"
    "   (MSE: 409 / 236)\n\n"
    "2. All methods perform better on\n"
    "   Signal B (strong trend, less noise)\n\n"
    "3. SMA has terrible MSE but that is\n"
    "   expected — it tracks direction,\n"
    "   not value\n\n"
    "4. Zero-lag DSP filters (forward-\n"
    "   backward) have major advantage\n"
    "   over causal SMA/EMA\n\n"
    "5. Frequency analysis reveals why\n"
    "   Signal A is harder to filter:\n"
    "   noise and signal overlap in\n"
    "   frequency domain"
)
ax6.text(0.5, 0.5, philosophy_text, transform=ax6.transAxes, fontsize=9.5,
         va='center', ha='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

fig.suptitle('Unknown Noisy Signal — Comprehensive DSP Methods Comparison\n'
             '神秘噪声信号——DSP方法综合处理对比',
             fontsize=17, fontweight='bold', y=1.01)
fig.savefig(os.path.join(OUT_DIR, '07_dashboard.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  [8/8] Dashboard")

# Save metrics CSV
import csv
csv_path = os.path.join(OUT_DIR, 'metrics.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Signal', 'Method', 'MSE', 'MAE', 'Corr', 'Lag'])
    for key in ['A', 'B']:
        c = datasets[key]['close']
        for mk in method_keys:
            v = all_results[key][mk]
            writer.writerow([key, mk, f'{mse(c,v):.1f}', f'{np.mean(np.abs(c-v)):.1f}',
                           f'{corr(c,v):.4f}', f'{max_lag_approx(c,v)}'])

print(f"\nAll charts saved to: {OUT_DIR}")
print("Done!")
