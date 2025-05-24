
#pragma once
#include <windows.h>
#include <mmsystem.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <shlobj.h> // 用于获取特殊文件夹路径
#include <functional>
#include <mmreg.h>    // 基础音频格式定义
#include <ks.h>       // Kernel Streaming 定义
#include <ksmedia.h>  // 媒体相关定义
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "shell32.lib") // 用于SHGetSpecialFolderPath
#include "QObject.h"

#include "qmap"
namespace AR {

    using namespace std;
    // WAV 文件头结构
    struct WAVHeader {
        char chunkId[4] = { 'R', 'I', 'F', 'F' };
        uint32_t chunkSize;
        char format[4] = { 'W', 'A', 'V', 'E' };
        char subchunk1Id[4] = { 'f', 'm', 't', ' ' };
        uint32_t subchunk1Size = 16;
        uint16_t audioFormat = 1; // PCM
        uint16_t numChannels;
        uint32_t sampleRate;
        uint32_t byteRate;
        uint16_t blockAlign;
        uint16_t bitsPerSample;
        char subchunk2Id[4] = { 'd', 'a', 't', 'a' };
        uint32_t subchunk2Size;
    };
    struct AudioTimeInfo {
        size_t bytes_recorded;    // 累计字节数
        size_t samples_recorded;  // 累计样本数
        uint64_t usecs_elapsed;   // 累计时间（微秒，μs）
    };
    class Recorder_Core_Windows : public QObject {
        Q_OBJECT
    public:
        static const size_t BUFFER_SAMPLES = 4800; // 统一使用1024个样本作为缓冲区大小
        Recorder_Core_Windows(QObject* parent);
        ~Recorder_Core_Windows();
        bool InitRecording(const wstring& target_device_name);
        void StartRecording();
        void StopRecording();
        void PauseRecording();
        void ResumeRecording();

        void SetAudioFormat(WAVEFORMATEX format);

        bool SaveAsWav(const wstring& filename);
        void Cleanup();
        void SetTargetDevice(const wstring& device_name);
        void PlayRecordedAudio();
        void StopRecordedAudio();  // 新增：停止播放

        void HandleWaveInMessage(UINT uMsg, DWORD_PTR dwParam1, DWORD_PTR dwParam2, AudioTimeInfo currentTimeSec);
        void HandleWaveOutMessage(UINT uMsg, DWORD_PTR dwParam1, DWORD_PTR dwParam2);
        const WAVEFORMATEX& GetWaveFormat() const;
        size_t GetRecordedBytesCount() const;

        // 存储录制音频数据的动态数组（BYTE = unsigned char）
// 录音时，音频数据会被实时追加到这个容器中
        vector<BYTE> recordedAudio;
    private:
        /// <summary>
        /// key : time value : volume
        /// </summary>
        QMap<size_t, QPair<size_t, double>> map_time_volume;
        // 音频格式结构体，用于定义录制的音频参数（采样率、位深、声道数等）
        WAVEFORMATEX waveFormat;
        // 录音状态标志：
        //   - true: 正在录音
        //   - false: 未在录音
        bool isRecording = false;

        // 选择的音频设备ID：
        //   - WAVE_MAPPER (= -1) 表示由系统自动选择默认录音设备
        //   - 也可以指定具体设备ID（通过waveInGetDevCaps获取设备列表）
        UINT selectedDeviceId = WAVE_MAPPER;

        // 音频缓冲区头结构体（Wave Header）
        // 用于管理音频数据块，包括缓冲区地址、长度、状态标志等
        // 需要先填充此结构体，再传递给waveInAddBuffer函数
        WAVEHDR waveHdr;
        wstring str_target_device_name;
        // 添加在全局变量部分
        HWAVEOUT hWaveOut = NULL;
        WAVEHDR waveOutHdr;
        bool isPaused = false; // 暂停状态标志

        // 音频输入设备的句柄（Handle to Waveform Audio Input）
        // 用于操作麦克风等音频输入设备，初始为NULL表示未打开设备
        HWAVEIN hWaveIn = NULL;

        /// <summary>
        /// 这个接口初始化设备，从这个接口进来之后，就会尝试fetch特定的设备
        /// </summary>
        /// <returns></returns>
        UINT FindTargetDevice();
      void RecordVolume(double volume);
    signals:
        //level值从0-100,level_time代表电平对应的usec
        void Sig_UpdateLevel(double level);
        void Sig_UpdatePlayLevel(double playLevel); // 新增播放电平信号
        void Sig_PlaybackFinished();



        //这个录音模块需要移动到线程中去处理，否则会卡死UI线程



    };


};