#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include "include/df/deep_filter.h"

namespace fs = std::filesystem;

#define FRAME_SIZE 480
#define SAMPLE_RATE 16000

// WAV文件头结构
#pragma pack(push, 1)
struct WavHeader {
    char     riff[4];        // "RIFF"
    uint32_t fileSize;       // 文件总大小-8
    char     wave[4];        // "WAVE"
    char     fmt[4];         // "fmt "
    uint32_t fmtSize;        // fmt chunk大小(16)
    uint16_t audioFormat;    // 音频格式(1=PCM)
    uint16_t numChannels;    // 声道数
    uint32_t sampleRate;     // 采样率
    uint32_t byteRate;       // 每秒字节数
    uint16_t blockAlign;     // 每个样本的字节数
    uint16_t bitsPerSample;  // 每样本位数
    char     data[4];        // "data"
    uint32_t dataSize;       // 数据大小
};
#pragma pack(pop)

// 获取当前时间的毫秒时间戳
long long currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

// 获取文件夹下所有wav文件
std::vector<fs::path> getWavFiles(const fs::path& folderPath) {
    std::vector<fs::path> wavFiles;

    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "Invalid directory: " << folderPath << std::endl;
        return wavFiles;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wav") {
            wavFiles.push_back(entry.path());
        }
    }

    return wavFiles;
}

// 读取WAV文件并提取PCM数据
bool readWavFile(const fs::path& wavPath, std::vector<int16_t>& pcmData, uint16_t& numChannels, uint32_t& sampleRate) {
    std::ifstream file(wavPath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << wavPath << std::endl;
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // 验证WAV文件头
    if (std::string(header.riff, 4) != "RIFF" ||
        std::string(header.wave, 4) != "WAVE" ||
        std::string(header.fmt, 4) != "fmt " ||
        header.audioFormat != 1) {
        std::cerr << "Invalid WAV file format: " << wavPath << std::endl;
        return false;
    }

    // 检查采样率
    if (header.sampleRate != SAMPLE_RATE) {
        std::cerr << "Unsupported sample rate (expected 16kHz): " << wavPath << std::endl;
        return false;
    }

    // 检查数据块
    if (std::string(header.data, 4) != "data") {
        std::cerr << "Data chunk not found: " << wavPath << std::endl;
        return false;
    }

    numChannels = header.numChannels;
    sampleRate = header.sampleRate;

    // 读取PCM数据
    pcmData.resize(header.dataSize / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(pcmData.data()), header.dataSize);

    return true;
}

// 写入WAV文件
bool writeWavFile(const fs::path& wavPath, const std::vector<int16_t>& pcmData, uint16_t numChannels, uint32_t sampleRate) {
    std::ofstream file(wavPath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create file: " << wavPath << std::endl;
        return false;
    }

    WavHeader header;
    std::copy_n("RIFF", 4, header.riff);
    std::copy_n("WAVE", 4, header.wave);
    std::copy_n("fmt ", 4, header.fmt);
    std::copy_n("data", 4, header.data);

    header.fmtSize = 16;
    header.audioFormat = 1; // PCM
    header.numChannels = numChannels;
    header.sampleRate = sampleRate;
    header.bitsPerSample = 16;
    header.blockAlign = numChannels * header.bitsPerSample / 8;
    header.byteRate = sampleRate * header.blockAlign;
    header.dataSize = static_cast<uint32_t>(pcmData.size() * sizeof(int16_t));
    header.fileSize = sizeof(header) - 8 + header.dataSize;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(pcmData.data()), header.dataSize);

    return true;
}

// 处理PCM数据
void processPcmData(std::vector<int16_t>& pcmData, DFState* st) {
    size_t totalFrames = (pcmData.size() + FRAME_SIZE - 1) / FRAME_SIZE;

    for (size_t i = 0; i < totalFrames; ++i) {
        size_t start = i * FRAME_SIZE;
        size_t end = std::min(start + FRAME_SIZE, pcmData.size());
        size_t frameSize = end - start;

        if (frameSize < FRAME_SIZE) {
            // 最后一帧可能不足FRAME_SIZE，填充零
            std::vector<int16_t> paddedFrame(FRAME_SIZE, 0);
            std::copy(pcmData.begin() + start, pcmData.begin() + end, paddedFrame.begin());

            long long firstTime = currentTimestamp();
            df_process_frame_i16(st, paddedFrame.data(), paddedFrame.data());
            long long lastTime = currentTimestamp();

            std::copy(paddedFrame.begin(), paddedFrame.begin() + frameSize, pcmData.begin() + start);
            std::cout << "Processed padded frame, time cost: " << (lastTime - firstTime) << " ms\n";
        }
        else {
            long long firstTime = currentTimestamp();
            df_process_frame_i16(st, pcmData.data() + start, pcmData.data() + start);
            long long lastTime = currentTimestamp();
            std::cout << "Processed frame, time cost: " << (lastTime - firstTime) << " ms\n";
        }
    }
}

int main(int argc, char** argv) {

    fs::path inputFolder = "D:/AudioSample/16khz_crownd/";
    fs::path outputFolder ="D:/AudioSample/16khz_crownd/deepfilter";
    std::string modelPath = (argc > 3) ? argv[3] : "D:/WorkShop/Github/AudioProcesser/DeepFilterDemo/Demo/model/DeepFilterNet3_onnx.tar.gz";

    // 创建输出文件夹
    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    // 初始化DeepFilter
    DFState* st = df_create(modelPath.c_str(), 25.);
    if (!st) {
        std::cerr << "Failed to initialize DeepFilter\n";
        return 1;
    }

    // 获取所有wav文件
    auto wavFiles = getWavFiles(inputFolder);
    if (wavFiles.empty()) {
        std::cerr << "No WAV files found in input folder\n";
        df_free(st);
        return 1;
    }

    // 处理每个wav文件
    for (const auto& wavFile : wavFiles) {
        std::cout << "Processing file: " << wavFile << std::endl;

        // 读取WAV文件
        std::vector<int16_t> pcmData;
        uint16_t numChannels;
        uint32_t sampleRate;

        if (!readWavFile(wavFile, pcmData, numChannels, sampleRate)) {
            std::cerr << "Failed to read WAV file: " << wavFile << std::endl;
            continue;
        }

        // 处理PCM数据
        processPcmData(pcmData, st);

        // 写入输出WAV文件
        fs::path outputPath = outputFolder / wavFile.filename().replace_filename(
            wavFile.stem().string() + "_processed.wav");

        if (!writeWavFile(outputPath, pcmData, numChannels, sampleRate)) {
            std::cerr << "Failed to write output WAV file: " << outputPath << std::endl;
            continue;
        }

        std::cout << "Successfully processed: " << wavFile << "\nOutput: " << outputPath << "\n";
    }

    df_free(st);
    return 0;
}