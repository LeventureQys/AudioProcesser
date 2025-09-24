//Author:Leventure
//Info : 有关
//



#include "STFT.h"
#include <iostream>
#include <cmath>
#include <vector>
#include "windows.h"
#include "wav_reader/wav_reader.h"
#include "onnx/include/onnxruntime_cxx_api.h"
// WAV文件头结构
struct WavHeader {
    // RIFF chunk
    char riff[4] = { 'R', 'I', 'F', 'F' };
    uint32_t file_size;
    char wave[4] = { 'W', 'A', 'V', 'E' };

    // fmt subchunk
    char fmt[4] = { 'f', 'm', 't', ' ' };
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;

    // data subchunk
    char data[4] = { 'd', 'a', 't', 'a' };
    uint32_t data_size;
};

// 保存int16_t格式的音频数据为WAV文件
bool writeWavFile(const std::string& filename, const std::vector<int16_t>& audio_data,
    uint32_t sample_rate, uint16_t num_channels) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件进行写入: " << filename << std::endl;
        return false;
    }

    WavHeader header;
    header.num_channels = num_channels;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 16;  // 我们使用int16_t格式
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = audio_data.size() * sizeof(int16_t);
    header.file_size = 36 + header.data_size;  // 36是固定部分的大小

    // 写入文件头
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // 写入音频数据
    file.write(reinterpret_cast<const char*>(audio_data.data()), header.data_size);

    if (!file) {
        std::cerr << "WAV写入失败" << std::endl;
        return false;
    }

    std::cout << "WAV写入成功g: " << filename << std::endl;
    return true;
}
// 生成16kHz采样率的测试信号
std::vector<double> generateTestSignal(int duration_ms, int sample_rate) {
    int num_samples = (duration_ms * sample_rate) / 1000;
    std::vector<double> signal(num_samples);

    double t = 0.0;
    double dt = 1.0 / sample_rate;

    for (int i = 0; i < num_samples; ++i) {
        signal[i] = 0.5 * sin(2 * M_PI * 440 * t) + 0.3 * sin(2 * M_PI * 880 * t);
        t += dt;
    }
    return signal;
}

// 转换C++ STFT结果为PyTorch格式的维度 (频率bins, 时间帧, 2)
std::vector<std::vector<std::vector<double>>> convertToPyTorchFormat(
    const std::vector<std::vector<std::complex<double>>>& stft_result) {

    if (stft_result.empty()) return {};

    int num_time_frames = stft_result.size();
    int num_freq_bins = stft_result[0].size();

    // 创建PyTorch格式的三维数组: [freq_bins][time_frames][2]
    std::vector<std::vector<std::vector<double>>> pytorch_format(
        num_freq_bins,
        std::vector<std::vector<double>>(
            num_time_frames,
            std::vector<double>(2)
        )
    );

    // 填充数据（转换维度顺序并分离实部虚部）
    for (int t = 0; t < num_time_frames; ++t) {
        for (int f = 0; f < num_freq_bins; ++f) {
            pytorch_format[f][t][0] = stft_result[t][f].real();  // 实部
            pytorch_format[f][t][1] = stft_result[t][f].imag();  // 虚部
        }
    }

    return pytorch_format;
}

// 定义模型输入输出的相关信息
struct ModelInfo {
    std::string input_name;    // 输入节点名称
    std::vector<int64_t> input_shape;  // 输入形状
    ONNXTensorElementDataType input_type;  // 输入数据类型

    std::string output_name;   // 输出节点名称
    std::vector<int64_t> output_shape; // 输出形状
    ONNXTensorElementDataType output_type; // 输出数据类型
};

// 将 std::string 转换为 wchar_t*，主要是因为Ort的Session需要传参wchar_t*，不然会报错
wchar_t* stringToWchar(const std::string& str) {
    // 计算需要的宽字符缓冲区大小
    int bufferSize = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (bufferSize == 0) {
        return nullptr; // 转换失败
    }

    // 分配缓冲区（注意：需要手动释放）
    wchar_t* wideStr = new wchar_t[bufferSize];

    // 执行转换
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, wideStr, bufferSize);
    return wideStr;
}

int main() {
    try {
        //main代码中，我将对一些主要流程进行切片，打几个断点，用于专门的性能测试
            // 设置控制台输出编码为 UTF-8
        SetConsoleOutputCP(65001);
        // 设置控制台输入编码为 UTF-8（如果需要输入中文）
        SetConsoleCP(65001);
        ///
        ///声明基本参数
        /// 
        std::string str_filepath = "D:\\AudioSample\\Audio\\AudioSample-16000hz\\sing_02.wav";
        std::string str_onnx_filepath = "D:\\WorkShop\\Github\\gtcrn_onnx_runtime\\model\\model_trained_on_dns3.onnx";
        wchar_t* wstr_onnx_filepath = stringToWchar(str_onnx_filepath);

        const int frame_size = 512;     // n_fft=512
        const int hop_size = 256;       // hop_length=256
        const int sample_rate = 16000;  // 16kHz采样率
        const int duration_ms = 1850;   // 调整时长以匹配PyTorch的1165帧
        const std::string window_type = "hanning";


        ///
        ///声明一个wav reader，读取一个wav文件
        /// 此处会占用内存，但是此处内存不做考虑
        /// 
        WavReader reader(str_filepath);

        if (!reader.isOpen()) {                         // 检查文件是否成功打开
            std::cerr << "无法打开WAV文件" << std::endl;
            return 1;
        }
        reader.printInfo();                             // 打印WAV文件信息
        const auto& original_audio = reader.getAudioData();         // 获取并处理音频数据
        std::cout << "音频数据样本数: " << original_audio.size() << std::endl;
        std::vector<int16_t> mono_audio;                                                //需要提取单通道数据，如果是立体声，出来的效果比较差（反正立体声也就是两个单通道）
        int num_channels = reader.getNumChannels();                                     // 假设WavReader有获取声道数的方法
        if (num_channels > 1) {
            std::cout << " channel : " << num_channels << "found,make it mono";

            for (size_t i = 0; i < original_audio.size(); i += num_channels) {          // 每隔num_channels个样本取一个（左声道）
                mono_audio.push_back(original_audio[i]);
            }
        }
        else {
            mono_audio = original_audio;
        } 
        SetConsoleOutputCP(CP_UTF8);            // 设置控制台输出为UTF-8编码
        SetConsoleCP(CP_UTF8);                  // 设置控制台输入为UTF-8编码（如果需要中文输入）

        ///
        /// 初始化onnx_runtime的环境，
        /// 此处为系统强制申请，不得不申请
        /// 
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-environment");                // 对于ONNX Runtime 1.17.x，直接使用环境构造函数
        Ort::SessionOptions session_options;        // 验证会话选项基本功能 
        session_options.SetIntraOpNumThreads(1);    // 设置一些兼容1.17版本的选项
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);    
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();// 检查可用的执行 providers
        std::cout << "ONNX Runtime 版本: 1.17.1" << std::endl;
        std::cout << "可用的执行 providers:" << std::endl;        

        ///
        /// 傅里叶变化，此处还可以优化使用rfft来提升速度和优化内存，我这里用的AI帮我写的stft库，后续可以再调研
        /// 
        STFT stft(frame_size, hop_size, window_type, true);// 创建STFT实例
        auto stft_result = stft.compute(mono_audio);    
        int time_frames = stft_result.size();// 输出结果形状（匹配 PyTorch 格式：[freq_bins, time_frames]）
        int freq_bins = stft_result[0].size();
        std::cout << "STFT 结果形状: (" << freq_bins << ", " << time_frames << ")" << std::endl;
        std::cout << "对应 PyTorch 输出: [freq_bins, time_frames, 2] = [" << freq_bins << ", " << time_frames << ", 2]" << std::endl;
        auto pytorch_format = convertToPyTorchFormat(stft_result);// 转换为PyTorch格式
        std::cout << "\nC++原生STFT维度: "
            << "(" << stft_result.size() << ", "
            << (stft_result.empty() ? 0 : stft_result[0].size()) << ")" << std::endl;
        std::cout << "PyTorch格式STFT维度: "
            << "(" << pytorch_format.size() << ", "
            << (pytorch_format.empty() ? 0 : pytorch_format[0].size()) << ", "
            << (pytorch_format.empty() || pytorch_format[0].empty() ? 0 : pytorch_format[0][0].size()) << ")" << std::endl;

        ModelInfo model_info;
        model_info.input_name = "input";  // 替换为你的模型实际输入名称
        model_info.input_shape = { 1, (int64_t)pytorch_format.size(),
                                 (int64_t)pytorch_format[0].size(), 2 };  // [batch, freq, time, 2]
        model_info.input_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        model_info.output_name = "output";  // 替换为你的模型实际输出名称


        ///
        /// onnx_runtime session 申请内存
        /// 此为系统申请，不可以不给 
        /// 
        Ort::Session session(env, wstr_onnx_filepath, session_options);
        std::vector<float> input_data;
        input_data.reserve(model_info.input_shape[1] * model_info.input_shape[2] * model_info.input_shape[3]);
        for (const auto& freq_bin : pytorch_format) {
            for (const auto& time_frame : freq_bin) {
                input_data.push_back(static_cast<float>(time_frame[0]));  // 实部
                input_data.push_back(static_cast<float>(time_frame[1]));  // 虚部
            }
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            model_info.input_shape.data(),
            model_info.input_shape.size()
        );

        const char* input_names[] = { model_info.input_name.c_str() };
        const char* output_names[] = { model_info.output_name.c_str() };

        ///
        /// 这里是onnx_runtime 直接运行在向量上
        /// 模型的主要发力点，这里cpu和内存消耗是敏感的
        /// 

        Ort::RunOptions run_options;  // 创建运行选项
        std::vector<Ort::Value> output_tensors = session.Run(
            run_options,
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        if (output_tensors.empty()) {
            throw std::runtime_error(" 无输出 ");
        }
      
        ///
        ///     输出数据，并转换成mono数据
        /// 
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::cout <<"模型输出形状";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
        }
        std::cout << std::endl;

  
        if (output_shape.size() != 4 || output_shape[3] != 2) {
            throw std::runtime_error("输出形状不符合预期 [batch, freq, time, 2]");
        }

        int batch_size = output_shape[0];
        freq_bins = output_shape[1];
        time_frames = output_shape[2];

        std::vector<std::vector<STFT::Complex>> stft_result_(time_frames, std::vector<STFT::Complex>(freq_bins));        // 注意：模型输出格式是 [batch, freq, time, 2]
                                                                                                                        // 而STFT::computeInverseInt16需要的是 [time, freq] 的复数数组

        for (int t = 0; t < time_frames; ++t) {
            for (int f = 0; f < freq_bins; ++f) {   // 计算当前位置的索引
                size_t index = 0 * freq_bins * time_frames * 2 +  // batch索引
                    f * time_frames * 2 +              // 频率索引
                    t * 2;                             // 时间索引
                float real = output_data[index];// 实部和虚部
                float imag = output_data[index + 1];           
                stft_result_[t][f] = STFT::Complex(real, imag);// 存储为复数形式
            }
        }

        ///
        /// 逆傅里叶 
        /// 
 
        STFT istft(frame_size, hop_size, window_type, true); // 执行逆STFT转换得到音频数据 使用相同的参数
        std::vector<int16_t> mono_output = istft.computeInverseInt16(stft_result_);
 
        

       ///
        /// 输出数据的保存
        /// 
        std::vector<int16_t> stereo_output; // 将单声道扩展为双声道（左右声道相同）
        stereo_output.reserve(mono_output.size() * 2);
        for (int16_t sample : mono_output) {
            stereo_output.push_back(sample);  // 左声道
            stereo_output.push_back(sample);  // 右声道（与左声道相同）
        }

        std::cout << "单声道处理后样本数: " << mono_output.size() << std::endl;
        std::cout << "扩展为双声道后样本数: " << stereo_output.size() << std::endl;    
        std::cout << "成功将模型输出转换为音频数据，样本数: " << stereo_output.size() << std::endl;// 12. 现在可以使用audio_output了，例如保存为WAV文件
        
        std::string output_file = "D:\\AudioSample\\output.wav";   // 保存音频数据为WAV文件 指定输出文件路径
       
        bool save_success = writeWavFile(output_file, stereo_output, 16000, 2); // 假设输入是单声道，采样率16000Hz
        if (!save_success) {
            std::cerr << "保存WAV文件失败" << std::endl;
            return 1;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
