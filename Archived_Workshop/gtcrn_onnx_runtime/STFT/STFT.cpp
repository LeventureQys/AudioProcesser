

#include "STFT.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm> // 用于 std::clamp

// -------------------------- 构造函数实现 --------------------------
STFT::STFT(int frame_size, int hop_size, const std::string& window_type, bool apply_window_sqrt)
    : frameSize(frame_size), hopSize(hop_size), fftSize(frame_size), applyWindowSqrt(apply_window_sqrt) {

    // 1. 校验参数合法性
    if (frame_size <= 0 || hop_size <= 0) {
        throw std::invalid_argument("Frame size and hop size must be positive");
    }
    if (window_type != "hanning" && window_type != "hamming") {
        throw std::invalid_argument("Unsupported window type (only 'hanning' or 'hamming' allowed)");
    }

    // 2. 生成指定类型的窗口函数
    if (window_type == "hanning") {
        window = generateHanningWindow(frame_size);
    }
    else { // "hamming"
        window = generateHammingWindow(frame_size);
    }

    // 3. 若需要，对窗口函数应用平方根（匹配 PyTorch 的 .pow(0.5)）
    if (applyWindowSqrt) {
        for (auto& val : window) {
            val = std::sqrt(val);
        }
    }
}

// -------------------------- 窗口函数实现 --------------------------
// 生成汉宁窗（Hanning Window）
std::vector<double> STFT::generateHanningWindow(int size) {
    std::vector<double> win(size);
    for (int i = 0; i < size; ++i) {
        win[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
    }
    return win;
}

// 生成汉明窗（Hamming Window）
std::vector<double> STFT::generateHammingWindow(int size) {
    std::vector<double> win(size);
    for (int i = 0; i < size; ++i) {
        win[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
    return win;
}

// -------------------------- FFT 辅助函数（位反转） --------------------------
void STFT::bitReverse(std::vector<Complex>& x) {
    int n = x.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1; // 等价于 n/2
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
}

// -------------------------- FFT 核心实现 --------------------------
// 快速傅里叶变换（Cooley-Tukey 算法，仅支持 2 的幂次长度）
void STFT::fft(std::vector<Complex>& x) {
    int n = x.size();
    bitReverse(x); // 第一步：位反转

    // 迭代实现 FFT（按子问题长度划分）
    for (int len = 2; len <= n; len <<= 1) { // len: 子问题长度（2,4,8,...n）
        double ang = 2 * M_PI / len; // 旋转因子的角度
        Complex wlen(std::cos(ang), std::sin(ang)); // 初始旋转因子

        for (int i = 0; i < n; i += len) { // 遍历每个子问题
            Complex w(1); // 当前旋转因子
            for (int j = 0; j < len / 2; ++j) { // 蝴蝶操作
                Complex u = x[i + j];
                Complex v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen; // 更新旋转因子
            }
        }
    }
}

// 逆快速傅里叶变换（基于 FFT 实现）
void STFT::ifft(std::vector<Complex>& x) {
    int n = x.size();
    // 第一步：对所有元素取共轭
    for (auto& val : x) {
        val = std::conj(val);
    }

    // 第二步：执行 FFT
    fft(x);

    // 第三步：再次取共轭，并除以长度 n（归一化）
    for (auto& val : x) {
        val = std::conj(val) / static_cast<double>(n);
    }
}

// -------------------------- STFT 正变换（double 输入） --------------------------
std::vector<std::vector<STFT::Complex>> STFT::compute(const std::vector<double>& input) {
    // 校验输入长度（至少要能容纳一帧）
    if (input.size() < static_cast<size_t>(frameSize)) {
        throw std::invalid_argument("Input signal length is shorter than frame size");
    }

    // 计算总帧数：(信号长度 - 帧大小) / 跳跃大小 + 1
    int num_frames = 1 + (static_cast<int>(input.size()) - frameSize) / hopSize;
    if (num_frames <= 0) {
        throw std::invalid_argument("No valid frames can be extracted from input");
    }

    // 存储 STFT 结果：外层是帧数，内层是每帧的频率 bin（复数）
    std::vector<std::vector<Complex>> result;
    result.reserve(num_frames); // 预分配空间，提升效率

    // 逐帧处理
    for (int i = 0; i < num_frames; ++i) {
        int frame_start = i * hopSize; // 当前帧的起始索引
        std::vector<Complex> current_frame(frameSize); // 存储当前帧（复数形式）

        // 1. 提取当前帧并应用窗口函数
        for (int j = 0; j < frameSize; ++j) {
            current_frame[j] = input[frame_start + j] * window[j];
        }

        // 2. 对当前帧执行 FFT
        fft(current_frame);

        // 3. 只保留前半段频率（实信号的 FFT 具有共轭对称性，后半段冗余）
        std::vector<Complex> half_frame(
            current_frame.begin(),
            current_frame.begin() + frameSize / 2 + 1
        );
        result.push_back(half_frame);
    }

    return result;
}

// -------------------------- 新增：STFT 正变换（int16_t 输入） --------------------------
std::vector<std::vector<STFT::Complex>> STFT::compute(const std::vector<int16_t>& input) {
    // 1. 将 int16_t 转换为 double（标准化到 [-1.0, 1.0]，适配音频 PCM 格式）
    std::vector<double> input_double;
    input_double.reserve(input.size()); // 预分配空间
    const double int16_max = 32767.0;   // int16_t 的最大值（范围：-32768 ~ 32767）

    for (int16_t sample : input) {
        // 转换公式：int16样本值 / 32767.0 → 标准化到 [-1.0, 1.0]
        input_double.push_back(static_cast<double>(sample) / int16_max);
    }

    // 2. 调用已实现的 double 版本 compute（避免代码重复）
    return compute(input_double);
}

// -------------------------- 逆 STFT 实现（输出 double） --------------------------
std::vector<double> STFT::computeInverse(const std::vector<std::vector<Complex>>& stft_result) {
    // 校验输入合法性
    if (stft_result.empty()) {
        throw std::invalid_argument("STFT result is empty");
    }
    int num_frames = static_cast<int>(stft_result.size());
    int freq_bins = static_cast<int>(stft_result[0].size());
    if (freq_bins != frameSize / 2 + 1) {
        throw std::invalid_argument("STFT result frequency bins do not match frame size");
    }

    // 计算输出信号长度：(帧数 - 1) * 跳跃大小 + 帧大小
    int output_size = (num_frames - 1) * hopSize + frameSize;
    std::vector<double> output(output_size, 0.0);       // 存储重建信号
    std::vector<double> window_sum(output_size, 0.0);   // 存储窗口重叠求和（用于后续归一化）

    // 逐帧逆变换并重叠相加
    for (int i = 0; i < num_frames; ++i) {
        int frame_start = i * hopSize; // 当前帧的起始索引
        std::vector<Complex> full_frame(fftSize, 0.0);   // 恢复完整的 FFT 帧（包含对称部分）

        // 1. 恢复完整频谱（利用实信号 FFT 的共轭对称性）
        for (int j = 0; j < freq_bins; ++j) {
            full_frame[j] = stft_result[i][j]; // 前半段直接复制
        }
        for (int j = 1; j < freq_bins - 1; ++j) {
            // 后半段为前半段的共轭（除了直流分量和 Nyquist 频率）
            full_frame[fftSize - j] = std::conj(stft_result[i][j]);
        }

        // 2. 对完整频谱执行逆 FFT
        ifft(full_frame);

        // 3. 重叠相加（应用窗口并累加到输出）
        for (int j = 0; j < frameSize; ++j) {
            if (frame_start + j >= output_size) break; // 避免越界
            // 累加信号（乘以窗口，抵消正变换时的窗口影响）
            output[frame_start + j] += full_frame[j].real() * window[j];
            // 累加窗口能量（用于后续归一化，解决重叠区域能量叠加问题）
            window_sum[frame_start + j] += window[j] * window[j];
        }
    }

    // 4. 窗口能量归一化（避免重叠区域信号幅度失真）
    for (int i = 0; i < output_size; ++i) {
        if (window_sum[i] > 1e-9) { // 避免除以 0（浮点精度保护）
            output[i] /= window_sum[i];
        }
    }

    return output;
}

// -------------------------- 新增：逆 STFT 实现（输出 int16_t） --------------------------
std::vector<int16_t> STFT::computeInverseInt16(const std::vector<std::vector<Complex>>& stft_result) {
    // 1. 先调用 double 版本逆变换，得到标准化的 [-1.0, 1.0] 信号
    std::vector<double> double_signal = computeInverse(stft_result);

    // 2. 将 double 信号反标准化为 int16_t（适配 PCM 音频格式）
    std::vector<int16_t> int16_signal;
    int16_signal.reserve(double_signal.size()); // 预分配空间，提升效率
    const double int16_max = 32767.0;          // int16_t 最大值（范围：-32768 ~ 32767）

    for (double sample : double_signal) {
        // 步骤1：反标准化（乘以 32767.0，恢复到 int16_t 数值范围）
        double scaled_sample = sample * int16_max;

        // 步骤2：四舍五入到最近整数（减少量化误差）
        int rounded_sample = static_cast<int>(std::round(scaled_sample));

        // 步骤3：钳位到 int16_t 合法范围（避免溢出，关键！）
        int16_t clamped_sample = static_cast<int16_t>(
            std::clamp(rounded_sample, static_cast<int>(INT16_MIN), static_cast<int>(INT16_MAX))
            );

        int16_signal.push_back(clamped_sample);
    }

    return int16_signal;
}