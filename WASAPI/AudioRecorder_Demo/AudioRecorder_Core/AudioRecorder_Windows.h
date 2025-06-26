
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
#include "mutex"
#include "qmap"
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

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

        ///windows的接口下暂时没有支持这两个接口
        void PauseRecording();
        void ResumeRecording();

        void SetAudioFormat(WAVEFORMATEX format);

        bool SaveAsWav(const wstring& filename);
        void SetTargetDevice(const wstring& device_name);
        void PlayRecordedAudio();
        void StopRecordedAudio(); 


    signals:
        //level值从0-100,level_time代表电平对应的usec
        void Sig_UpdateLevel(double level);
        void Sig_UpdatePlayLevel(double playLevel); // 新增播放电平信号
        void Sig_PlaybackFinished();



        //这个录音模块需要移动到线程中去处理，否则会卡死UI线程

    private:
        DWORD duration_time = 100 * 10000;
        IMMDevice* GetTargetDevice();
        IMMDevice* pDevice;
        std::wstring str_target_device;
        std::vector<BYTE> audioBuffer;
        std::mutex bufferMutex;
        std::condition_variable bufferCV;
        bool isRecording = false;
        bool isPlaying = false;
        bool shouldStop = false;
        std::thread recordThread;       //录制线程
        std::thread playThread;         //播放线程
        std::map<size_t, short> levelMap;
        size_t currentPos = 0;
        std::mutex mapMutex;

        void StartRecordThreadFunction();           //录制音频数据线程
        void PlayDataThreadFunction();              //播放以录制音频数据线程
    };


};
