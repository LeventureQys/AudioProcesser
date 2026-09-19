"""
Comprehensive chart generation for DSP vs Financial methods comparison on SSE Composite Index.
Generates all charts needed for the article.
"""
import openpyxl
import numpy as np
from datetime import datetime
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from collections import defaultdict
import warnings, io, sys, os
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ======================== CONFIG ========================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r'D:\workshop\Github\AudioProcesser\Document\quant\document'
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(filepath):
    wb = openpyxl.load_workbook(filepath)
    ws = wb.active
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        d = datetime.strptime(str(row[0]), '%Y%m%d').date()
        dates.append(d)
        opens.append(float(row[6]))
        highs.append(float(row[7]))
        lows.append(float(row[8]))
        closes.append(float(row[9]))
        volumes.append(float(row[12]))
    return (np.array(dates), np.array(opens), np.array(highs),
            np.array(lows), np.array(closes), np.array(volumes))

print("Loading data...")
d22, o22, h22, l22, c22, v22 = load_data(r'D:\workshop\Github\AudioProcesser\Document\quant\source\上证指数\2022.xlsx')
d25, o25, h25, l25, c25, v25 = load_data(r'D:\workshop\Github\AudioProcesser\Document\quant\source\上证指数\2025.xlsx')

datasets = {
    '2022': {'dates': d22, 'close': c22, 'open': o22, 'high': h22, 'low': l22},
    '2025': {'dates': d25, 'close': c25, 'open': o25, 'high': h25, 'low': l25},
}

# ======================== DSP METHODS ========================
def kalman_filter(values, Q_scale=1.0, R_val=100.0):
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.5 * Q_scale, 0], [0, 0.05 * Q_scale]])
    R = np.array([[R_val]])
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
    R = np.array([[100.0]]); R_min, R_max = np.array([[10.0]]), np.array([[2500.0]])
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
            est_R = max(innov_var - pred_var, R_min[0, 0])
            R = np.array([[np.clip(est_R, R_min[0, 0], R_max[0, 0])]])
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y_res; P = (np.eye(2) - K @ H) @ P
        est[k] = x[0]
    return est

def holt_winters(values, alpha=0.3, beta=0.1, gamma=0.05, period=5, seasonal='add'):
    """Triple exponential smoothing (Holt-Winters)"""
    n = len(values)
    level = np.zeros(n); trend = np.zeros(n); seasonal_comp = np.zeros(n)
    # Initialize
    level[0] = values[0]; trend[0] = values[1] - values[0] if n > 1 else 0
    seasonal_init = np.zeros(period)
    for i in range(min(period, n)):
        seasonal_init[i] = values[i] - level[0] if seasonal == 'add' else values[i] / max(level[0], 1)
    s = seasonal_init.copy()
    est = np.zeros(n); est[0] = values[0]
    for t in range(1, n):
        if t < period:
            est[t] = values[t]
            level[t] = values[t]
            trend[t] = values[t] - values[t-1] if t > 0 else 0
            continue
        s_idx = t % period
        if seasonal == 'add':
            level[t] = alpha * (values[t] - s[s_idx]) + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
            s[s_idx] = gamma * (values[t] - level[t]) + (1 - gamma) * s[s_idx]
            est[t] = level[t] + trend[t] + s[s_idx]
        else:
            level[t] = alpha * (values[t] / max(s[s_idx], 1e-10)) + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
            s[s_idx] = gamma * (values[t] / max(level[t], 1e-10)) + (1 - gamma) * s[s_idx]
            est[t] = (level[t] + trend[t]) * s[s_idx]
    return est

def butterworth_lowpass(values, cutoff=0.05, order=2):
    """Apply Butterworth low-pass filter (forward-backward for zero phase)"""
    nyq = 0.5
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype='low')
    return signal.filtfilt(b, a, values)

def savitzky_golay(values, window=15, order=3):
    """Savitzky-Golay filter"""
    if window % 2 == 0:
        window += 1
    return signal.savgol_filter(values, window, order)

def sma(values, window=20):
    """Simple Moving Average"""
    return np.convolve(values, np.ones(window)/window, mode='same')

def ema(values, span=20):
    """Exponential Moving Average"""
    alpha = 2.0 / (span + 1)
    n = len(values); result = np.zeros(n); result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    return result

def macd(values, fast=12, slow=26, signal_span=9):
    """MACD indicator"""
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(values, window=20, num_std=2):
    """Bollinger Bands"""
    mid = sma(values, window)
    rolling_std = np.array([np.std(values[max(0,i-window+1):i+1]) for i in range(len(values))])
    upper = mid + num_std * rolling_std
    lower = mid - num_std * rolling_std
    return upper, mid, lower

# ======================== METRICS ========================
def mse(a, b): return np.mean((a - b) ** 2)
def mae(a, b): return np.mean(np.abs(a - b))
def correlation(a, b): return np.corrcoef(a, b)[0, 1]
def max_lag_approx(original, filtered, max_search=30):
    """Estimate lag by finding max cross-correlation offset"""
    a = (original - np.mean(original)) / np.std(original)
    b = (filtered - np.mean(filtered)) / np.std(filtered)
    corr = np.correlate(a, b, mode='full')
    mid = len(corr) // 2
    search_range = corr[mid:mid+max_search]
    lag = np.argmax(search_range)
    return lag

# ======================== APPLY ALL METHODS ========================
print("Applying methods...")
all_results = {}

for year, ds in datasets.items():
    c = ds['close']
    results = {}
    
    # DSP methods
    results['Kalman (Standard)'] = kalman_filter(c)
    results['Kalman (Adaptive)'] = adaptive_kalman(c)
    
    # Holt-Winters (best alpha found by grid search)
    best_hw = None; best_mse = float('inf')
    for a in [0.1, 0.2, 0.3, 0.5]:
        for b_val in [0.05, 0.1, 0.2]:
            for g in [0.01, 0.05, 0.1]:
                hw = holt_winters(c, alpha=a, beta=b_val, gamma=g, period=5)
                m = mse(c, hw)
                if m < best_mse:
                    best_mse = m; best_hw = hw
    results['Holt-Winters'] = best_hw
    
    results['Butterworth (fc=0.03)'] = butterworth_lowpass(c, cutoff=0.03, order=2)
    results['Butterworth (fc=0.05)'] = butterworth_lowpass(c, cutoff=0.05, order=2)
    results['Gaussian (σ=3)'] = gaussian_filter1d(c, sigma=3)
    results['Gaussian (σ=5)'] = gaussian_filter1d(c, sigma=5)
    results['Savitzky-Golay (w=11)'] = savitzky_golay(c, window=11, order=3)
    results['Savitzky-Golay (w=21)'] = savitzky_golay(c, window=21, order=3)
    
    # Financial methods
    results['SMA (20)'] = sma(c, 20)
    results['SMA (60)'] = sma(c, 60)
    results['EMA (12)'] = ema(c, 12)
    results['EMA (26)'] = ema(c, 26)
    
    macd_line, sig_line, hist = macd(c)
    results['MACD_line'] = macd_line
    results['MACD_signal'] = sig_line
    results['MACD_hist'] = hist
    
    bb_upper, bb_mid, bb_lower = bollinger_bands(c, 20, 2)
    results['BB_upper'] = bb_upper
    results['BB_mid'] = bb_mid
    results['BB_lower'] = bb_lower
    
    all_results[year] = results

# ======================== Compute all metrics ========================
print("Computing metrics...")
metrics_table = []
for year in ['2022', '2025']:
    c = datasets[year]['close']
    res = all_results[year]
    for name, vals in res.items():
        if name.startswith('MACD') or name.startswith('BB'):
            continue
        metrics_table.append({
            'Year': year, 'Method': name,
            'MSE': mse(c, vals), 'MAE': mae(c, vals),
            'Corr': correlation(c, vals),
            'Lag': max_lag_approx(c, vals),
        })

# ======================== CHART 1: Raw data overview ========================
print("Generating charts...")
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for ax, (year, ds), color in zip(axes, datasets.items(), ['#e74c3c', '#2980b9']):
    d = ds['dates']; c = ds['close']
    ax.plot(d, c, color=color, linewidth=0.7)
    ax.fill_between(d, c, alpha=0.1, color=color)
    # Mark high and low
    hi_idx = np.argmax(c); lo_idx = np.argmin(c)
    ax.annotate(f'{c[hi_idx]:.0f}', xy=(d[hi_idx], c[hi_idx]), fontsize=9, color=color, ha='center')
    ax.annotate(f'{c[lo_idx]:.0f}', xy=(d[lo_idx], c[lo_idx]), fontsize=9, color='green', ha='center')
    ax.set_title(f'SSE Composite Index {year}\n上证综指 {year}年 日收盘价', fontsize=13, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.25)
    # Stats box
    ret = np.diff(c) / c[:-1] * 100
    stats_text = f'Range: {c.min():.0f}~{c.max():.0f}\nVolatility: {ret.std():.2f}%\nMean return: {ret.mean():.2f}%'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
fig.suptitle('Raw Data Overview / 原始数据概览', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '00_raw_data.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [1/12] Raw data overview")

# ======================== CHART 2: Kalman Filter detailed ========================
fig, axes = plt.subplots(2, 2, figsize=(22, 12))
for row_idx, year in enumerate(['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    kf = all_results[year]['Kalman (Standard)']; akf = all_results[year]['Kalman (Adaptive)']
    
    ax = axes[row_idx, 0]
    ax.plot(d, c, color='#bdc3c7', linewidth=0.5, alpha=0.7, label='Raw')
    ax.plot(d, kf, color='#e74c3c', linewidth=1.2, label='Kalman')
    ax.set_title(f'{year} — Standard Kalman Filter\n标准卡尔曼滤波', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
    ax.text(0.98, 0.95, f'MSE:{mse(c,kf):.0f}\nLag:{max_lag_approx(c,kf)}d', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.85))
    
    ax = axes[row_idx, 1]
    ax.plot(d, c, color='#bdc3c7', linewidth=0.5, alpha=0.7, label='Raw')
    ax.plot(d, akf, color='#2980b9', linewidth=1.2, label='Adaptive Kalman')
    ax.set_title(f'{year} — Adaptive Kalman Filter\n自适应卡尔曼滤波', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
    ax.text(0.98, 0.95, f'MSE:{mse(c,akf):.0f}\nLag:{max_lag_approx(c,akf)}d', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#dfe6e9', alpha=0.85))

fig.suptitle('Kalman Filter — Standard vs Adaptive\n卡尔曼滤波对比', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_kalman.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [2/12] Kalman filter")

# ======================== CHART 3: Holt-Winters ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    hw = all_results[year]['Holt-Winters']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.5, alpha=0.7, label='Raw')
    ax.plot(d, hw, color='#e67e22', linewidth=1.2, label='Holt-Winters')
    ax.set_title(f'{year} — Holt-Winters Triple ES\nHolt-Winters 三次指数平滑', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
    ax.text(0.98, 0.95, f'MSE:{mse(c,hw):.0f}\nLag:{max_lag_approx(c,hw)}d', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.85))
fig.suptitle('Holt-Winters Exponential Smoothing\nHolt-Winters指数平滑', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '02_holtwinters.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [3/12] Holt-Winters")

# ======================== CHART 4: Gaussian Smoothing ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
sigmas = [1, 3, 5, 8]
colors_sigma = ['#f39c12', '#e67e22', '#d35400', '#c0392b']
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    for s, col in zip(sigmas, colors_sigma):
        gs = gaussian_filter1d(c, sigma=s)
        ax.plot(d, gs, color=col, linewidth=1.0, label=f'σ={s} (MSE:{mse(c,gs):.0f})')
    ax.set_title(f'{year} — Gaussian Smoothing\n高斯平滑降噪', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)
fig.suptitle('Gaussian Smoothing with Different σ\n不同σ值的高斯平滑对比', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '03_gaussian.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [4/12] Gaussian smoothing")

# ======================== CHART 5: Butterworth ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
cutoffs = [0.01, 0.03, 0.05, 0.08]
colors_bw = ['#3498db', '#2980b9', '#1a5276', '#0e2f44']
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    for fc, col in zip(cutoffs, colors_bw):
        bw = butterworth_lowpass(c, cutoff=fc, order=2)
        ax.plot(d, bw, color=col, linewidth=1.0, label=f'fc={fc} (MSE:{mse(c,bw):.0f})')
    ax.set_title(f'{year} — Butterworth Low-pass\n巴特沃斯低通滤波', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)
fig.suptitle('Butterworth Low-Pass Filter with Different Cutoff Frequencies\n不同截止频率的巴特沃斯滤波', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '04_butterworth.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [5/12] Butterworth")

# ======================== CHART 6: Savitzky-Golay ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
windows = [7, 15, 31, 51]
colors_sg = ['#8e44ad', '#6c3483', '#4a235a', '#2c0e37']
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    for w, col in zip(windows, colors_sg):
        sg = savitzky_golay(c, window=w, order=3)
        ax.plot(d, sg, color=col, linewidth=1.0, label=f'w={w} (MSE:{mse(c,sg):.0f})')
    ax.set_title(f'{year} — Savitzky-Golay Filter\nSavitzky-Golay滤波', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)
fig.suptitle('Savitzky-Golay Filter with Different Window Sizes\n不同窗口的SG滤波', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '05_savgol.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [6/12] Savitzky-Golay")

# ======================== CHART 7: SMA + EMA comparison ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    ax.plot(d, c, color='#bdc3c7', linewidth=0.4, alpha=0.5, label='Raw')
    sma5 = sma(c, 5); sma20 = sma(c, 20); sma60 = sma(c, 60)
    ema12 = ema(c, 12); ema26 = ema(c, 26)
    ax.plot(d, sma5, color='#f39c12', linewidth=0.8, alpha=0.7, label='SMA(5)')
    ax.plot(d, sma20, color='#e74c3c', linewidth=1.0, label='SMA(20)')
    ax.plot(d, sma60, color='#c0392b', linewidth=1.2, label='SMA(60)')
    ax.plot(d, ema12, '--', color='#2980b9', linewidth=0.8, label='EMA(12)')
    ax.plot(d, ema26, '--', color='#1a5276', linewidth=1.0, label='EMA(26)')
    ax.set_title(f'{year} — Moving Averages\n移动平均线', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=3)
fig.suptitle('SMA & EMA — Financial Domain Standard Tools\n金融领域：移动平均线', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '06_moving_avg.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [7/12] Moving averages")

# ======================== CHART 8: MACD ========================
fig, axes = plt.subplots(2, 2, figsize=(22, 12))
for row_idx, year in enumerate(['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    res = all_results[year]
    
    # Top: price + EMAs
    ax = axes[row_idx, 0]
    ax.plot(d, c, color='#2c3e50', linewidth=0.6, label='Close')
    ax.plot(d, ema(c,12), color='#e74c3c', linewidth=0.8, label='EMA12')
    ax.plot(d, ema(c,26), color='#2980b9', linewidth=0.8, label='EMA26')
    ax.set_title(f'{year} — Price & EMAs\n价格与指数均线', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7)
    
    # Bottom: MACD
    ax = axes[row_idx, 1]
    ax.bar(d, res['MACD_hist'], color=['#27ae60' if h >= 0 else '#e74c3c' for h in res['MACD_hist']],
           width=1, alpha=0.6, label='Histogram')
    ax.plot(d, res['MACD_line'], color='#e74c3c', linewidth=0.8, label='MACD')
    ax.plot(d, res['MACD_signal'], color='#2980b9', linewidth=0.8, label='Signal')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(f'{year} — MACD(12,26,9)\nMACD指标', fontsize=11, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7)

fig.suptitle('MACD Indicator — Trend & Momentum\nMACD指标——趋势与动量', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '07_macd.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [8/12] MACD")

# ======================== CHART 9: Bollinger Bands ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    res = all_results[year]
    ax.fill_between(d, res['BB_upper'], res['BB_lower'], alpha=0.15, color='#3498db')
    ax.plot(d, res['BB_upper'], color='#3498db', linewidth=0.6, alpha=0.6, label='Upper (±2σ)')
    ax.plot(d, res['BB_mid'], color='#e74c3c', linewidth=0.8, label='SMA(20)')
    ax.plot(d, res['BB_lower'], color='#3498db', linewidth=0.6, alpha=0.6, label='Lower (±2σ)')
    ax.plot(d, c, color='#2c3e50', linewidth=0.5, label='Close')
    ax.set_title(f'{year} — Bollinger Bands (20,2)\n布林带', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
fig.suptitle('Bollinger Bands — Volatility Envelope\n布林带——波动率包络', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '08_bollinger.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [9/12] Bollinger Bands")

# ======================== CHART 10: DSP Method Showdown ========================
fig, axes = plt.subplots(2, 2, figsize=(22, 12))

# Panel 1: 2022 DSP methods
ax = axes[0, 0]
c22_data = datasets['2022']['close']; d22_data = datasets['2022']['dates']
ax.plot(d22_data, c22_data, color='#bdc3c7', linewidth=0.3, alpha=0.4, label='Raw')
methods_show = [
    ('Kalman', all_results['2022']['Kalman (Standard)'], '#e74c3c'),
    ('Holt-Winters', all_results['2022']['Holt-Winters'], '#e67e22'),
    ('Gaussian σ=5', all_results['2022']['Gaussian (σ=5)'], '#8e44ad'),
    ('Butterworth fc=0.03', all_results['2022']['Butterworth (fc=0.03)'], '#2980b9'),
    ('Savitzky-Golay w=21', all_results['2022']['Savitzky-Golay (w=21)'], '#27ae60'),
]
for name, vals, col in methods_show:
    ax.plot(d22_data, vals, color=col, linewidth=1.0, alpha=0.8, label=f'{name} (MSE:{mse(c22_data,vals):.0f})')
ax.set_title('2022 — DSP Methods Comparison\nDSP方法对比 (高波动年)', fontsize=12, fontweight='bold')
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)

# Panel 2: 2025 DSP methods
ax = axes[0, 1]
c25_data = datasets['2025']['close']; d25_data = datasets['2025']['dates']
ax.plot(d25_data, c25_data, color='#bdc3c7', linewidth=0.3, alpha=0.4, label='Raw')
methods_show_25 = [
    ('Kalman', all_results['2025']['Kalman (Standard)'], '#e74c3c'),
    ('Holt-Winters', all_results['2025']['Holt-Winters'], '#e67e22'),
    ('Gaussian σ=5', all_results['2025']['Gaussian (σ=5)'], '#8e44ad'),
    ('Butterworth fc=0.03', all_results['2025']['Butterworth (fc=0.03)'], '#2980b9'),
    ('Savitzky-Golay w=21', all_results['2025']['Savitzky-Golay (w=21)'], '#27ae60'),
]
for name, vals, col in methods_show_25:
    ax.plot(d25_data, vals, color=col, linewidth=1.0, alpha=0.8, label=f'{name} (MSE:{mse(c25_data,vals):.0f})')
ax.set_title('2025 — DSP Methods Comparison\nDSP方法对比 (趋势年)', fontsize=12, fontweight='bold')
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.grid(True, alpha=0.2); ax.legend(fontsize=7, ncol=2)

# Panel 3: MSE Bar chart
ax = axes[1, 0]
dsp_names = ['Kalman', 'Holt-Winters', 'Gaussian\nσ=5', 'Butterworth\nfc=0.03', 'S-G\nw=21']
mse_2022 = [mse(c22_data, all_results['2022']['Kalman (Standard)']),
            mse(c22_data, all_results['2022']['Holt-Winters']),
            mse(c22_data, all_results['2022']['Gaussian (σ=5)']),
            mse(c22_data, all_results['2022']['Butterworth (fc=0.03)']),
            mse(c22_data, all_results['2022']['Savitzky-Golay (w=21)'])]
mse_2025 = [mse(c25_data, all_results['2025']['Kalman (Standard)']),
            mse(c25_data, all_results['2025']['Holt-Winters']),
            mse(c25_data, all_results['2025']['Gaussian (σ=5)']),
            mse(c25_data, all_results['2025']['Butterworth (fc=0.03)']),
            mse(c25_data, all_results['2025']['Savitzky-Golay (w=21)'])]
x_pos = np.arange(len(dsp_names))
w = 0.35
ax.bar(x_pos - w/2, mse_2022, w, color='#e74c3c', alpha=0.8, label='2022')
ax.bar(x_pos + w/2, mse_2025, w, color='#2980b9', alpha=0.8, label='2025')
ax.set_xticks(x_pos); ax.set_xticklabels(dsp_names, fontsize=9)
ax.set_ylabel('MSE'); ax.set_title('DSP Methods — MSE Comparison\nDSP方法MSE对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')

# Panel 4: Lag comparison
ax = axes[1, 1]
lag_2022 = [max_lag_approx(c22_data, all_results['2022']['Kalman (Standard)']),
            max_lag_approx(c22_data, all_results['2022']['Holt-Winters']),
            max_lag_approx(c22_data, all_results['2022']['Gaussian (σ=5)']),
            max_lag_approx(c22_data, all_results['2022']['Butterworth (fc=0.03)']),
            max_lag_approx(c22_data, all_results['2022']['Savitzky-Golay (w=21)'])]
lag_2025 = [max_lag_approx(c25_data, all_results['2025']['Kalman (Standard)']),
            max_lag_approx(c25_data, all_results['2025']['Holt-Winters']),
            max_lag_approx(c25_data, all_results['2025']['Gaussian (σ=5)']),
            max_lag_approx(c25_data, all_results['2025']['Butterworth (fc=0.03)']),
            max_lag_approx(c25_data, all_results['2025']['Savitzky-Golay (w=21)'])]
ax.bar(x_pos - w/2, lag_2022, w, color='#e74c3c', alpha=0.8, label='2022')
ax.bar(x_pos + w/2, lag_2025, w, color='#2980b9', alpha=0.8, label='2025')
ax.set_xticks(x_pos); ax.set_xticklabels(dsp_names, fontsize=9)
ax.set_ylabel('Lag (days)'); ax.set_title('DSP Methods — Lag Comparison (lower is better)\nDSP方法延迟对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('DSP Methods Comprehensive Comparison\nDSP方法综合对比', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '09_dsp_showdown.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [10/12] DSP showdown")

# ======================== CHART 11: Frequency domain analysis ========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, year in zip(axes, ['2022', '2025']):
    ds = datasets[year]; c = ds['close']
    # Remove DC component
    c_detrended = c - np.mean(c)
    n_fft = len(c)
    fft_vals = np.abs(np.fft.rfft(c_detrended))
    freqs = np.fft.rfftfreq(n_fft)
    # Convert to period in trading days
    periods = 1.0 / (freqs[1:] + 1e-10)
    ax.semilogx(periods, fft_vals[1:] / np.max(fft_vals[1:]), color='#2c3e50', linewidth=0.8)
    ax.axvline(x=5, color='#e74c3c', linestyle='--', alpha=0.5, label='5-day (week)')
    ax.axvline(x=20, color='#2980b9', linestyle='--', alpha=0.5, label='20-day (month)')
    ax.axvline(x=60, color='#27ae60', linestyle='--', alpha=0.5, label='60-day (quarter)')
    ax.set_xlabel('Period (trading days)'); ax.set_ylabel('Normalized Magnitude')
    ax.set_title(f'{year} — Frequency Spectrum\n频谱分析', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2); ax.legend(fontsize=8)
fig.suptitle('Frequency Domain Analysis — Where is the "Noise"?\n频域分析——"噪声"在哪里？', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '10_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [11/12] Frequency analysis")

# ======================== CHART 12: Comprehensive comparison dashboard ========================
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: Raw data comparison
ax0 = fig.add_subplot(gs[0, :])
for year, col in [('2022', '#e74c3c'), ('2025', '#2980b9')]:
    ds = datasets[year]; c = ds['close']; d = ds['dates']
    c_norm = (c - c[0]) / c[0] * 100  # Normalize to % change from start
    ax0.plot(d, c_norm, color=col, linewidth=0.8, label=f'{year} (from {c[0]:.0f})')
ax0.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax0.set_title('Normalized Comparison (% from start)\n归一化对比（距起始点涨跌幅%）', fontsize=13, fontweight='bold')
ax0.xaxis.set_major_locator(mdates.MonthLocator()); ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax0.legend(fontsize=10); ax0.grid(True, alpha=0.2)
# Annotate
ax0.text(0.01, 0.95, f'2022: {c22[-1]-c22[0]:+.0f} ({((c22[-1]/c22[0])-1)*100:+.1f}%)\n2025: {c25[-1]-c25[0]:+.0f} ({((c25[-1]/c25[0])-1)*100:+.1f}%)',
         transform=ax0.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Row 2: MSE heatmap-like comparison
all_method_names = [
    'Kalman\n(Standard)', 'Kalman\n(Adaptive)', 'Holt-\nWinters',
    'Gaussian\nσ=3', 'Gaussian\nσ=5', 'Butterworth\nfc=0.03',
    'Butterworth\nfc=0.05', 'S-G\nw=11', 'S-G\nw=21',
    'SMA\n(20)', 'SMA\n(60)', 'EMA\n(12)', 'EMA\n(26)'
]
method_keys_2022 = [
    'Kalman (Standard)', 'Kalman (Adaptive)', 'Holt-Winters',
    'Gaussian (σ=3)', 'Gaussian (σ=5)', 'Butterworth (fc=0.03)',
    'Butterworth (fc=0.05)', 'Savitzky-Golay (w=11)', 'Savitzky-Golay (w=21)',
    'SMA (20)', 'SMA (60)', 'EMA (12)', 'EMA (26)'
]

# MSE comparison
ax1 = fig.add_subplot(gs[1, 0])
mse_vals_22 = [mse(c22_data, all_results['2022'][k]) for k in method_keys_2022]
mse_vals_25 = [mse(c25_data, all_results['2025'][k]) for k in method_keys_2022]
y = np.arange(len(all_method_names))
ax1.barh(y - 0.2, mse_vals_22, 0.4, color='#e74c3c', alpha=0.8, label='2022', zorder=3)
ax1.barh(y + 0.2, mse_vals_25, 0.4, color='#2980b9', alpha=0.8, label='2025', zorder=3)
ax1.set_yticks(y); ax1.set_yticklabels(all_method_names, fontsize=8)
ax1.set_xlabel('MSE'); ax1.set_title('MSE by Method\n各方法MSE', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right'); ax1.grid(True, alpha=0.2, axis='x')
ax1.invert_yaxis()

# Lag comparison
ax2 = fig.add_subplot(gs[1, 1])
lag_22 = [max_lag_approx(c22_data, all_results['2022'][k]) for k in method_keys_2022]
lag_25 = [max_lag_approx(c25_data, all_results['2025'][k]) for k in method_keys_2022]
ax2.barh(y - 0.2, lag_22, 0.4, color='#e74c3c', alpha=0.8, label='2022', zorder=3)
ax2.barh(y + 0.2, lag_25, 0.4, color='#2980b9', alpha=0.8, label='2025', zorder=3)
ax2.set_yticks(y); ax2.set_yticklabels(all_method_names, fontsize=8)
ax2.set_xlabel('Lag (days)'); ax2.set_title('Lag by Method (lower=better)\n各方法延迟', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2, axis='x')
ax2.invert_yaxis()

# Correlation comparison
ax3 = fig.add_subplot(gs[1, 2])
corr_22 = [correlation(c22_data, all_results['2022'][k]) for k in method_keys_2022]
corr_25 = [correlation(c25_data, all_results['2025'][k]) for k in method_keys_2022]
ax3.barh(y - 0.2, corr_22, 0.4, color='#e74c3c', alpha=0.8, label='2022', zorder=3)
ax3.barh(y + 0.2, corr_25, 0.4, color='#2980b9', alpha=0.8, label='2025', zorder=3)
ax3.set_yticks(y); ax3.set_yticklabels(all_method_names, fontsize=8)
ax3.set_xlabel('Correlation'); ax3.set_title('Correlation with Raw\n与原始数据相关性', fontsize=12, fontweight='bold')
ax3.set_xlim(0.85, 1.01)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.2, axis='x')
ax3.invert_yaxis()

# Row 3: Side-by-side filtered view
ax4 = fig.add_subplot(gs[2, 0])
# 2022 with best DSP + best Finance method
c22_s = c22_data; d22_s = d22_data
ax4.plot(d22_s, c22_s, color='#bdc3c7', linewidth=0.3, alpha=0.4, label='Raw')
ax4.plot(d22_s, all_results['2022']['Kalman (Standard)'], color='#e74c3c', linewidth=1.0, label='Best DSP: Kalman')
ax4.plot(d22_s, all_results['2022']['EMA (12)'], '--', color='#2980b9', linewidth=1.0, label='Best Finance: EMA(12)')
ax4.set_title('2022 — Best Methods Overlay\n2022年最优方法叠加', fontsize=11, fontweight='bold')
ax4.xaxis.set_major_locator(mdates.MonthLocator()); ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2)

ax5 = fig.add_subplot(gs[2, 1])
c25_s = c25_data; d25_s = d25_data
ax5.plot(d25_s, c25_s, color='#bdc3c7', linewidth=0.3, alpha=0.4, label='Raw')
ax5.plot(d25_s, all_results['2025']['Holt-Winters'], color='#e67e22', linewidth=1.0, label='Best DSP: Holt-Winters')
ax5.plot(d25_s, all_results['2025']['EMA (12)'], '--', color='#2980b9', linewidth=1.0, label='Best Finance: EMA(12)')
ax5.set_title('2025 — Best Methods Overlay\n2025年最优方法叠加', fontsize=11, fontweight='bold')
ax5.xaxis.set_major_locator(mdates.MonthLocator()); ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.2)

ax6 = fig.add_subplot(gs[2, 2])
# Philosophy comparison text
ax6.axis('off')
philosophy_text = (
    "DSP vs Finance: Philosophy\n\n"
    "DSP Approach:\n"
    "- Treats price as signal + noise\n"
    "- Goal: extract underlying signal\n"
    "- Metric: MSE, lag, smoothness\n"
    "- Frequency-domain thinking\n"
    "- Forward-backward filtering (0 lag)\n\n"
    "Finance Approach:\n"
    "- Price IS the signal\n"
    "- Goal: generate trading signals\n"
    "- Metric: P&L, hit rate, Sharpe\n"
    "- Causal only (no future leak)\n"
    "- Accepts lag as feature (trend ID)\n\n"
    "Key Insight:\n"
    "The 'noise' DSP removes is the\n"
    "'opportunity' traders exploit."
)
ax6.text(0.5, 0.5, philosophy_text, transform=ax6.transAxes, fontsize=10,
         va='center', ha='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

fig.suptitle('SSE Composite Index — DSP vs Financial Methods Comprehensive Comparison\n'
             '上证综合指数 — DSP方法与金融方法综合对比',
             fontsize=17, fontweight='bold', y=1.01)
fig.savefig(os.path.join(OUT_DIR, '11_dashboard.png'), dpi=180, bbox_inches='tight')
plt.close()
print("  [12/12] Dashboard")

# ======================== Export metrics CSV ========================
import csv
csv_path = os.path.join(OUT_DIR, 'metrics.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['Year', 'Method', 'MSE', 'MAE', 'Corr', 'Lag'])
    writer.writeheader()
    for row in metrics_table:
        writer.writerow(row)

print(f"\nAll charts saved to: {OUT_DIR}")
print(f"Metrics CSV saved to: {csv_path}")
print("Done!")
