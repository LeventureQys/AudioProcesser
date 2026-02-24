#ifndef STFT_H
#define STFT_H

#include <vector>
#include <complex>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class STFT {
public:
    // 复数类型定义（简化代码）
    using Complex = std::complex<double>;

    // FFT 核心操作函数（私有实现）
    void fft(std::vector<Complex>& x);       // 快速傅里叶变换
    void ifft(std::vector<Complex>& x);      // 逆快速傅里叶变换
    void bitReverse(std::vector<Complex>& x); // FFT 所需的位反转操作

    // 窗口函数生成（私有实现）
    std::vector<double> generateHanningWindow(int size); // 汉宁窗
    std::vector<double> generateHammingWindow(int size); // 汉明窗

    // 私有成员变量
    int frameSize;         // 帧大小（每帧样本数）
    int hopSize;           // 跳跃大小（帧间重叠步长）
    int fftSize;           // FFT 变换长度（通常与帧大小一致）
    std::vector<double> window; // 窗口函数数据
    bool applyWindowSqrt;  // 是否对窗口函数应用平方根（匹配 PyTorch .pow(0.5)）

public:
    // 构造函数
    // 参数：frame_size-帧大小, hop_size-跳跃大小, window_type-窗口类型("hanning"/"hamming"), apply_window_sqrt-是否对窗口开平方
    STFT(int frame_size, int hop_size, const std::string& window_type = "hanning", bool apply_window_sqrt = false);

    // 析构函数（默认空实现）
    ~STFT() = default;

    // 核心接口：执行 STFT 变换（支持 double 输入）
    std::vector<std::vector<Complex>> compute(const std::vector<double>& input);

    // 新增接口：执行 STFT 变换（支持 int16_t 输入，如 PCM 音频）
    std::vector<std::vector<Complex>> compute(const std::vector<int16_t>& input);

    // 核心接口：执行逆 STFT 变换（输出 double 信号）
    std::vector<double> computeInverse(const std::vector<std::vector<Complex>>& stft_result);

    // 新增接口：执行逆 STFT 变换（输出 int16_t 信号，适配 PCM 音频）
    std::vector<int16_t> computeInverseInt16(const std::vector<std::vector<Complex>>& stft_result);

    // 辅助接口：获取参数
    int getFrameSize() const { return frameSize; }
    int getHopSize() const { return hopSize; }
    int getFFTSize() const { return fftSize; }
};
#endif