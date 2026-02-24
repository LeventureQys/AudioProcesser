#include "wav_reader.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

WavReader::WavReader() : is_open(false) {}

WavReader::WavReader(const std::string& file_path) : is_open(false) {
    open(file_path);
}

WavReader::~WavReader() {
    close();
}

bool WavReader::read_header(std::ifstream& file) {
    // 读取RIFF头部
    file.read(header.riff_header, 4);
    if (std::strncmp(header.riff_header, "RIFF", 4) != 0) {
        std::cerr << "不是有效的WAV文件: 缺少RIFF标识" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&header.file_size), 4);
    file.read(header.wave_header, 4);
    if (std::strncmp(header.wave_header, "WAVE", 4) != 0) {
        std::cerr << "不是有效的WAV文件: 缺少WAVE标识" << std::endl;
        return false;
    }

    // 寻找fmt子块
    bool found_fmt = false;
    while (!found_fmt && file.good()) {
        char subchunk_header[4];
        file.read(subchunk_header, 4);
        
        if (std::strncmp(subchunk_header, "fmt ", 4) == 0) {
            found_fmt = true;
            std::memcpy(header.fmt_header, subchunk_header, 4);
            
            // 读取fmt子块内容
            file.read(reinterpret_cast<char*>(&header.fmt_chunk_size), 4);
            file.read(reinterpret_cast<char*>(&header.audio_format), 2);
            file.read(reinterpret_cast<char*>(&header.num_channels), 2);
            file.read(reinterpret_cast<char*>(&header.sample_rate), 4);
            file.read(reinterpret_cast<char*>(&header.byte_rate), 4);
            file.read(reinterpret_cast<char*>(&header.block_align), 2);
            file.read(reinterpret_cast<char*>(&header.bits_per_sample), 2);
            
            // 如果fmt块大小大于16，说明有额外参数，我们跳过
            if (header.fmt_chunk_size > 16) {
                uint16_t extra_params_size;
                file.read(reinterpret_cast<char*>(&extra_params_size), 2);
                file.ignore(extra_params_size);
            }
        } else {
            // 不是fmt子块，跳过
            uint32_t subchunk_size;
            file.read(reinterpret_cast<char*>(&subchunk_size), 4);
            file.ignore(subchunk_size);
        }
    }

    if (!found_fmt) {
        std::cerr << "WAV文件中未找到fmt子块" << std::endl;
        return false;
    }

    // 寻找data子块
    bool found_data = false;
    while (!found_data && file.good()) {
        char subchunk_header[4];
        file.read(subchunk_header, 4);
        
        if (std::strncmp(subchunk_header, "data", 4) == 0) {
            found_data = true;
            std::memcpy(header.data_header, subchunk_header, 4);
            file.read(reinterpret_cast<char*>(&header.data_size), 4);
        } else {
            // 不是data子块，跳过
            uint32_t subchunk_size;
            file.read(reinterpret_cast<char*>(&subchunk_size), 4);
            file.ignore(subchunk_size);
        }
    }

    if (!found_data) {
        std::cerr << L"WAV文件中未找到data子块" << std::endl;
        return false;
    }

    // 检查是否为PCM格式
    if (header.audio_format != 1) {
        std::cerr << L"不支持的音频格式，仅支持PCM格式" << std::endl;
        return false;
    }

    // 检查采样位数是否为16位（这个实现仅支持16位）
    if (header.bits_per_sample != 16) {

        return false;
    }

    return true;
}

bool WavReader::read_data(std::ifstream& file) {
    // 计算样本数量
    uint32_t num_samples = header.data_size / (header.bits_per_sample / 8);
    audio_data.resize(num_samples);
    
    // 读取音频数据
    file.read(reinterpret_cast<char*>(audio_data.data()), header.data_size);
    
    if (!file) {
        std::cerr << "读取音频数据失败" << std::endl;
        return false;
    }
    
    return true;
}

bool WavReader::open(const std::string& file_path) {
    // 关闭已打开的文件
    close();
    
    // 尝试打开文件
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return false;
    }
    
    // 读取并解析头部
    if (!read_header(file)) {
        file.close();
        return false;
    }
    
    // 读取音频数据
    if (!read_data(file)) {
        file.close();
        return false;
    }
    
    // 成功打开和读取
    filename = file_path;
    is_open = true;
    file.close();
    return true;
}

void WavReader::close() {
    if (is_open) {
        audio_data.clear();
        filename.clear();
        is_open = false;
        std::memset(&header, 0, sizeof(WavHeader));
    }
}

bool WavReader::isOpen() const {
    return is_open;
}

uint16_t WavReader::getNumChannels() const {
    if (!is_open) {
        throw std::runtime_error("WAV文件未打开");
    }
    return header.num_channels;
}

uint32_t WavReader::getSampleRate() const {
    if (!is_open) {
        throw std::runtime_error("WAV文件未打开");
    }
    return header.sample_rate;
}

uint16_t WavReader::getBitsPerSample() const {
    if (!is_open) {
        throw std::runtime_error("WAV文件未打开");
    }
    return header.bits_per_sample;
}

uint32_t WavReader::getNumSamples() const {
    if (!is_open) {
        throw std::runtime_error("WAV文件未打开");
    }
    return header.data_size / (header.bits_per_sample / 8);
}

uint32_t WavReader::getDurationMs() const {
    if (!is_open) {
        throw std::runtime_error("WAV文件未打开");
    }
    // 计算音频时长（毫秒）
    return (getNumSamples() / header.num_channels) * 1000 / header.sample_rate;
}

const std::vector<int16_t>& WavReader::getAudioData() const {
    if (!is_open) {
        throw std::runtime_error("WAV文件未打开");
    }
    return audio_data;
}

void WavReader::printInfo() const {
    if (!is_open) {
        std::cerr << L"WAV文件未打开" << std::endl;
        return;
    }
    std::cout<< "WAV文件信息:" << std::endl;
    std::cout << "文件名 : " << filename << std::endl;
    std::cout << "声道数 : " << static_cast<int>(header.num_channels) << std::endl;
    std::cout << "采样率 : " << header.sample_rate << "Hz" << std::endl;
    std::cout << "字节率 : " << header.byte_rate << std::endl;
    std::cout << "音频数据大小 : " << header.data_size << "字节" << std::endl;
    std::cout << "总样本数 : " << getNumSamples() << std::endl;
    std::cout << "时长 : " << getDurationMs() << "毫秒" << std::endl;

}
