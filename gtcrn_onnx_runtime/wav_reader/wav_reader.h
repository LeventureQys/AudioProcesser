#ifndef WAV_READER_H
#define WAV_READER_H

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <iostream>
class WavReader {
private:
    // WAV文件头结构
    struct WavHeader {
        // RIFF Chunk
        char riff_header[4];   // "RIFF"
        uint32_t file_size;    // 文件大小 - 8
        char wave_header[4];   // "WAVE"
        
        // fmt Subchunk
        char fmt_header[4];    // "fmt "
        uint32_t fmt_chunk_size; // 子块大小
        uint16_t audio_format;  // 音频格式，1表示PCM
        uint16_t num_channels;  // 声道数
        uint32_t sample_rate;   // 采样率
        uint32_t byte_rate;     // 字节率
        uint16_t block_align;   // 块对齐
        uint16_t bits_per_sample; // 每个样本的位数
        
        // data Subchunk
        char data_header[4];    // "data"
        uint32_t data_size;     // 数据大小
    };

    WavHeader header;
    std::vector<int16_t> audio_data;  // 存储音频数据
    std::string filename;
    bool is_open;

    // 从文件读取头部信息
    bool read_header(std::ifstream& file);
    
    // 从文件读取音频数据
    bool read_data(std::ifstream& file);

public:
    // 构造函数和析构函数
    WavReader();
    WavReader(const std::string& file_path);
    ~WavReader();

    // 打开WAV文件
    bool open(const std::string& file_path);
    
    // 关闭WAV文件
    void close();
    
    // 检查文件是否已打开
    bool isOpen() const;
    
    // 获取音频信息
    uint16_t getNumChannels() const;
    uint32_t getSampleRate() const;
    uint16_t getBitsPerSample() const;
    uint32_t getNumSamples() const;
    uint32_t getDurationMs() const;
    
    // 获取音频数据
    const std::vector<int16_t>& getAudioData() const;
    
    // 打印WAV文件信息
    void printInfo() const;
};




#endif // WAV_READER_H
