#include "../api/api.h"
#include "../wav_reader/wav_reader.h"
#include "windows.h"
// 保存处理后的音频为WAV文件
bool saveWavFile(const std::string& output_path, const int16_t* data, size_t data_size,
    uint32_t sample_rate, uint16_t num_channels) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开输出文件: " << output_path << std::endl;
        return false;
    }

    // WAV文件头结构
    struct WavHeader {
        char riff[4] = { 'R', 'I', 'F', 'F' };
        uint32_t file_size;
        char wave[4] = { 'W', 'A', 'V', 'E' };

        char fmt[4] = { 'f', 'm', 't', ' ' };
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1;  // PCM
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 16;

        char data[4] = { 'd', 'a', 't', 'a' };
        uint32_t data_size;
    };

    WavHeader header;
    header.num_channels = num_channels;
    header.sample_rate = sample_rate;
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = static_cast<uint32_t>(data_size * sizeof(int16_t));
    header.file_size = 36 + header.data_size;

    // 写入文件头
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // 写入音频数据
    file.write(reinterpret_cast<const char*>(data), header.data_size);

    if (!file) {
        std::cerr << "WAV文件写入失败" << std::endl;
        return false;
    }

    std::cout << "处理后的音频已保存至: " << output_path << std::endl;
    return true;
}

int main() {
    // 设置控制台编码为UTF-8，方便中文显示
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    // 音频文件路径
    const std::string input_audio_path = "D:\\AudioSample\\16khz\\white\\D4_750.wav";
    const std::string output_audio_path = "D:\\AudioSample\\16khz\\white\\D4_750_processed.wav";
    const std::string model_path = "D:\\WorkShop\\Github\\gtcrn_onnx_runtime\\model\\model_trained_on_dns3.onnx";

    // 音频处理参数
    const int sample_rate = 16000;  // 16kHz采样率
    const int frame_size = 512;     // 帧大小
    const int hop_size = 256;       // 跳跃大小
    const std::string window_type = "hanning";  // 窗口类型

    try {
        // 1. 读取输入音频文件
        WavReader wav_reader(input_audio_path);
        if (!wav_reader.isOpen()) {
            std::cerr << "无法打开音频文件: " << input_audio_path << std::endl;
            return 1;
        }

        // 打印音频信息
        std::cout << "输入音频信息:" << std::endl;
        wav_reader.printInfo();

        // 检查采样率是否为16000Hz
        if (wav_reader.getSampleRate() != sample_rate) {
            std::cerr << "错误: 音频采样率必须为16000Hz" << std::endl;
            return 1;
        }

        // 获取音频数据
        const std::vector<int16_t>& audio_data = wav_reader.getAudioData();
        std::cout << "音频样本数: " << audio_data.size() << std::endl;

        // 2. 初始化音频处理器
        AudioProcessorContext* processor = audio_processor_init(
            model_path.c_str(),
            sample_rate,
            frame_size,
            hop_size       
        );

        if (!processor) {
            std::cerr << "音频处理器初始化失败" << std::endl;
            return 1;
        }

        // 3. 准备处理音频
        std::vector<int16_t> output_data(audio_data.size() * 2);  // 预留足够大的缓冲区
        size_t output_size = output_data.size();

        // 4. 处理音频
        ProcessStatus status = audio_processor_process(
            processor,
            audio_data.data(),
            audio_data.size(),
            output_data.data(),
            &output_size
        );

        if (status != PROCESS_SUCCESS) {
            std::cerr << "音频处理失败，错误代码: " << status << std::endl;
            audio_processor_destroy(processor);
            return 1;
        }

        // 调整输出向量大小以匹配实际处理结果
        output_data.resize(output_size);

        // 5. 保存处理后的音频
        bool save_success = saveWavFile(
            output_audio_path,
            output_data.data(),
            output_data.size(),
            sample_rate,
            wav_reader.getNumChannels()  // 保持与输入相同的声道数
        );

        if (!save_success) {
            std::cerr << "保存处理后的音频失败" << std::endl;
            audio_processor_destroy(processor);
            return 1;
        }

        // 6. 清理资源
        audio_processor_destroy(processor);

        std::cout << "音频处理完成！" << std::endl;
        std::cout << "输入样本数: " << audio_data.size() << std::endl;
        std::cout << "输出样本数: " << output_data.size() << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
        return 1;
    }
}