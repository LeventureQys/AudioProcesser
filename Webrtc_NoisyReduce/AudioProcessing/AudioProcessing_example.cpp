#include "./agc2/rnn_vad/common.h"
#include <windows.h>
#include <mmreg.h> // 推荐使用这个

#include "NoiseSuppressor.h"
//#include "VoiceActivityDetector2.h"
//#include "VoiceActivityDetector.h"
#include "audio_util.h"
#include "audio_buffer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem> // C++17 or later for filesystem operations
#define LDTTL_WAVE_PACKET_LENGTH (1472)
namespace fs = std::filesystem;
#if __has_include(<chrono>)
#    include <chrono>
static uint64_t std_get_time_now_by_ms()
{
	auto timeNow = std::chrono::system_clock::now().time_since_epoch();
	return std::chrono::duration_cast<std::chrono::milliseconds>(timeNow).count();
}
#endif

static inline uint64_t GetTimeInterval(const uint64_t a, const uint64_t b)
{
#ifdef __cplusplus
	return (std::min)(a - b, ((uint64_t)-1) - a + b + 1);
#else
	return min(a - b, ((uint64_t)-1) - a + b + 1);
#endif
};


struct WavHeader {
    char chunkId[4];         // "RIFF"
    uint32_t chunkSize;      // 文件大小 - 8
    char format[4];          // "WAVE"
    char subchunk1Id[4];     // "fmt "
    uint32_t subchunk1Size;  // 16 for PCM
    uint16_t audioFormat;    // 1 for PCM
    uint16_t numChannels;    // 声道数
    uint32_t sampleRate;     // 采样率
    uint32_t byteRate;       // 每秒字节数
    uint16_t blockAlign;     // 每个采样点的字节数
    uint16_t bitsPerSample;  // 每个采样的比特数
    char subchunk2Id[4];     // "data"
    uint32_t subchunk2Size;  // 数据大小
};

bool process_wav_audio(const std::string& input_filename, const std::string& output_filename) {
    std::ifstream input_file(input_filename, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error opening input file: " << input_filename << std::endl;
        return false;
    }
    auto config = webrtc::NsConfig();
    config.target_level = webrtc::NsConfig::SuppressionLevel::k21dB;
    //初始化指针
    std::unique_ptr<webrtc::NoiseSuppressor> m_NoiseSuppressor
        = std::make_unique<webrtc::NoiseSuppressor>(config, 48000, 2);

    WavHeader header;
    input_file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    if (std::strncmp(header.chunkId, "RIFF", 4) != 0 ||
        std::strncmp(header.format, "WAVE", 4) != 0 ||
        std::strncmp(header.subchunk1Id, "fmt ", 4) != 0 ||
        std::strncmp(header.subchunk2Id, "data", 4) != 0 ||
        header.audioFormat != 1 || // 必须是 PCM
        header.numChannels != 2 ||
        header.sampleRate != 48000 ||
        header.bitsPerSample != 16) {
        std::cerr << "Invalid WAV file format." << std::endl;
        return false;
    }

    size_t num_samples = header.subchunk2Size / (header.bitsPerSample / 8) / header.numChannels;

    // 计算每次处理的帧长（例如，10ms）
    const int frame_size = header.sampleRate / 100; // 10ms的帧大小
    const int num_channels = header.numChannels;

    std::vector<int16_t> pcm_data(frame_size * num_channels);

    std::ofstream output_file(output_filename, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error opening output file: " << output_filename << std::endl;
        return false;
    }
    // 先写入头部信息
    output_file.write(reinterpret_cast<char*>(&header), sizeof(WavHeader));

    // 循环处理音频数据
    size_t total_processed_samples = 0;
    while (total_processed_samples < num_samples) {
        // 读取一帧数据
        size_t samples_to_read = min(frame_size, (int)(num_samples - total_processed_samples));
        input_file.read(reinterpret_cast<char*>(pcm_data.data()), samples_to_read * num_channels * sizeof(int16_t));

        if (input_file.gcount() != samples_to_read * num_channels * sizeof(int16_t)) {
            // 可能已经到达文件末尾，进行补零操作
            std::fill(pcm_data.begin() + input_file.gcount() / sizeof(int16_t), pcm_data.end(), 0);
        }

        webrtc::AudioBuffer audio_buffer(header.sampleRate, num_channels, header.sampleRate, num_channels, header.sampleRate, num_channels);
       // const float* float_flow = (float*)pcm_data.data();
        audio_buffer.CopyFrom(pcm_data.data(), StreamConfig(header.sampleRate, header.numChannels));
        
        // 噪声抑制处理
        m_NoiseSuppressor->Analyze(audio_buffer);
        m_NoiseSuppressor->Process(&audio_buffer);

        std::vector<int16_t> output_pcm_data(frame_size * num_channels);
        audio_buffer.CopyTo(StreamConfig(header.sampleRate, header.numChannels), output_pcm_data.data());

        // 写入输出文件
        output_file.write(reinterpret_cast<char*>(output_pcm_data.data()), samples_to_read * num_channels * sizeof(int16_t));

        total_processed_samples += samples_to_read;
    }

    input_file.close();
    output_file.close();

    // 更新输出文件的头部信息中的数据大小
    std::fstream update_output_file(output_filename, std::ios::in | std::ios::out | std::ios::binary);
    if (update_output_file.is_open()) {
        header.subchunk2Size = total_processed_samples * num_channels * sizeof(int16_t);
        header.chunkSize = 36 + header.subchunk2Size;
        update_output_file.seekp(0, std::ios::beg);
        update_output_file.write(reinterpret_cast<char*>(&header), sizeof(WavHeader));
        update_output_file.close();
    }
    else {
        std::cerr << "Error updating output file header." << std::endl;
        return false;
    }

    return true;
}
int main(int argc, char* argv[])
{

    std::string input_folder;
    input_folder = "D:/temp/48k";
    fs::path output_dir = fs::path(input_folder) / "output";
    if (!fs::exists(output_dir)) {
        if (!fs::create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << std::endl;
            return 1;
        }
    }

    // Iterate through all files in the input folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wav") {
            // Construct input and output file paths
            std::string input_filename = entry.path().string();
            std::string output_filename = (output_dir / entry.path().filename()).string();

            // Process the WAV file
            if (!process_wav_audio(input_filename, output_filename)) {
                std::cerr << "Failed to process: " << input_filename << std::endl;
            }
        }
    }

    std::cout << "Processing complete." << std::endl;

    return 0;
}
