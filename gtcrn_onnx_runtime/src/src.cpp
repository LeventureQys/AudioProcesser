#include "../api/api.h"
#include "STFT.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "windows.h"
#include "../wav_reader/wav_reader.h"
#include "../onnx/include/onnxruntime_cxx_api.h"

// ����ģ����������������Ϣ
struct ModelInfo {
    std::string input_name;    // ����ڵ�����
    std::vector<int64_t> input_shape;  // ������״
    ONNXTensorElementDataType input_type;  // ������������

    std::string output_name;   // ����ڵ�����
    std::vector<int64_t> output_shape; // �����״
    ONNXTensorElementDataType output_type; // �����������
};

// ��Ƶ�����������Ľṹ
struct AudioProcessorContext {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    STFT stft;
    STFT istft;
    ModelInfo model_info;
    int sample_rate;
    int frame_size;
    int hop_size;
    std::string window_type;

    // ���캯��
    AudioProcessorContext(const char* model_path, int sr, int fs, int hs, const char* wt)
        : env(ORT_LOGGING_LEVEL_WARNING, "audio-processor"),
        session_options(),
        session(nullptr),
        stft(fs, hs, wt, true),
        istft(fs, hs, wt, true),
        sample_rate(sr),
        frame_size(fs),
        hop_size(hs),
        window_type(wt) {
        // ���ûỰѡ��
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // ת��ģ��·��Ϊ���ַ�
        wchar_t* w_model_path = stringToWchar(model_path);
        if (!w_model_path) {
            throw std::runtime_error("�޷�ת��ģ��·��Ϊ���ַ�");
        }

        // �����Ự
        session = Ort::Session(env, w_model_path, session_options);
        delete[] w_model_path;

        // ��ʼ��ģ����Ϣ
        model_info.input_name = "input";  // �滻Ϊʵ����������
        model_info.input_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        model_info.output_name = "output";  // �滻Ϊʵ���������
    }

    // �� std::string ת��Ϊ wchar_t*
    wchar_t* stringToWchar(const std::string& str) {
        int bufferSize = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
        if (bufferSize == 0) {
            return nullptr;
        }

        wchar_t* wideStr = new wchar_t[bufferSize];
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, wideStr, bufferSize);
        return wideStr;
    }

    // ת��C++ STFT���ΪPyTorch��ʽ
    std::vector<std::vector<std::vector<double>>> convertToPyTorchFormat(
        const std::vector<std::vector<STFT::Complex>>& stft_result) {

        if (stft_result.empty()) return {};

        int num_time_frames = stft_result.size();
        int num_freq_bins = stft_result[0].size();

        // ����PyTorch��ʽ����ά����: [freq_bins][time_frames][2]
        std::vector<std::vector<std::vector<double>>> pytorch_format(
            num_freq_bins,
            std::vector<std::vector<double>>(
                num_time_frames,
                std::vector<double>(2)
            )
        );

        // �������
        for (int t = 0; t < num_time_frames; ++t) {
            for (int f = 0; f < num_freq_bins; ++f) {
                pytorch_format[f][t][0] = stft_result[t][f].real();  // ʵ��
                pytorch_format[f][t][1] = stft_result[t][f].imag();  // �鲿
            }
        }

        return pytorch_format;
    }
};

// ��ʼ����Ƶ������
AudioProcessorContext* audio_processor_init(
    const char* model_path,
    int sample_rate,
    int frame_size,
    int hop_size) {
    try {
        if (!model_path || sample_rate <= 0 || frame_size <= 0 || hop_size <= 0) {
            return nullptr;
        }

        // ���ÿ���̨����
        SetConsoleOutputCP(65001);
        SetConsoleCP(65001);

        return new AudioProcessorContext(model_path, sample_rate, frame_size, hop_size, "hanning");
    }
    catch (const std::exception& e) {
        std::cerr << "��ʼ��ʧ��: " << e.what() << std::endl;
        return nullptr;
    }
}

// ����16k��������Ƶ����
ProcessStatus audio_processor_process(
    AudioProcessorContext* context,
    const int16_t* input,
    size_t input_size,
    int16_t* output,
    size_t* output_size) {

    if (!context || !input || input_size == 0 || !output || !output_size) {
        return PROCESS_INVALID_INPUT;
    }

    try {
        // ת��Ϊ��������������������Ƕ�������
        std::vector<int16_t> mono_audio;
        // ����򻯴������������Ѿ��ǵ�����
        mono_audio.assign(input, input + input_size);

        // ִ��STFT
        auto stft_result = context->stft.compute(mono_audio);
        int time_frames = stft_result.size();
        if (time_frames == 0) {
            return PROCESS_INTERNAL_ERROR;
        }

        // ת��ΪPyTorch��ʽ
        auto pytorch_format = context->convertToPyTorchFormat(stft_result);
        int freq_bins = pytorch_format.size();
        if (freq_bins == 0) {
            return PROCESS_INTERNAL_ERROR;
        }

        // ����������״
        context->model_info.input_shape = {
            1,
            (int64_t)freq_bins,
            (int64_t)pytorch_format[0].size(),
            2
        };

        // ׼����������
        std::vector<float> input_data;
        input_data.reserve(
            context->model_info.input_shape[1] *
            context->model_info.input_shape[2] *
            context->model_info.input_shape[3]
        );

        for (const auto& freq_bin : pytorch_format) {
            for (const auto& time_frame : freq_bin) {
                input_data.push_back(static_cast<float>(time_frame[0]));  // ʵ��
                input_data.push_back(static_cast<float>(time_frame[1]));  // �鲿
            }
        }

        // ������������
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            context->model_info.input_shape.data(),
            context->model_info.input_shape.size()
        );

        const char* input_names[] = { context->model_info.input_name.c_str() };
        const char* output_names[] = { context->model_info.output_name.c_str() };

        // ����ģ��
        Ort::RunOptions run_options;
        std::vector<Ort::Value> output_tensors = context->session.Run(
            run_options,
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        if (output_tensors.empty()) {
            return PROCESS_INTERNAL_ERROR;
        }

        // �������
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        if (output_shape.size() != 4 || output_shape[3] != 2) {
            return PROCESS_INTERNAL_ERROR;
        }

        // ת�����ΪSTFT��ʽ
        int batch_size = output_shape[0];
        freq_bins = output_shape[1];
        time_frames = output_shape[2];

        std::vector<std::vector<STFT::Complex>> stft_result_(time_frames, std::vector<STFT::Complex>(freq_bins));

        for (int t = 0; t < time_frames; ++t) {
            for (int f = 0; f < freq_bins; ++f) {
                size_t index = 0 * freq_bins * time_frames * 2 +  // batch����
                    f * time_frames * 2 +              // Ƶ������
                    t * 2;                             // ʱ������
                float real = output_data[index];
                float imag = output_data[index + 1];
                stft_result_[t][f] = STFT::Complex(real, imag);
            }
        }

        // ִ����STFT
        std::vector<int16_t> mono_output = context->istft.computeInverseInt16(stft_result_);

        // ��������������С
        if (*output_size < mono_output.size()) {
            *output_size = mono_output.size();
            return PROCESS_MEMORY_ERROR; // ���������̫С
        }

        // ���ƽ�������������
        std::copy(mono_output.begin(), mono_output.end(), output);
        *output_size = mono_output.size();

        return PROCESS_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "�������: " << e.what() << std::endl;
        return PROCESS_INTERNAL_ERROR;
    }
}

// �ͷ���Ƶ��������Դ
void audio_processor_destroy(AudioProcessorContext* context) {
    if (context) {
        delete context;
    }
}
