#include "AudioRecorder_Windows.h"

bool AR::Recorder_Core_Windows::SaveAsWav(const wstring& filename)
{
    if (audioBuffer.empty())
        return false;

    // 获取音频设备格式
    IAudioClient* pAudioClient = nullptr;
    WAVEFORMATEX* pWaveFormat = nullptr;
    HRESULT hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
    if (FAILED(hr)) return false;

    hr = pAudioClient->GetMixFormat(&pWaveFormat);
    if (FAILED(hr))
    {
        pAudioClient->Release();
        return false;
    }

    // 打开文件进行写入
    ofstream file(filename, ios::binary);
    if (!file.is_open())
    {
        CoTaskMemFree(pWaveFormat);
        pAudioClient->Release();
        return false;
    }
    pWaveFormat->wFormatTag = WAVE_FORMAT_PCM;
    pWaveFormat->nChannels = 2;
    pWaveFormat->nSamplesPerSec = 48000;
    pWaveFormat->wBitsPerSample = 16;
    pWaveFormat->nBlockAlign = pWaveFormat->nChannels * pWaveFormat->wBitsPerSample / 8;
    pWaveFormat->nAvgBytesPerSec = pWaveFormat->nSamplesPerSec * pWaveFormat->nBlockAlign;
    pWaveFormat->cbSize = 0;
    // 准备WAV文件头
    WAVHeader header;
    //header.numChannels = pWaveFormat->nChannels;
    header.numChannels = pWaveFormat->nChannels;
    header.sampleRate = pWaveFormat->nSamplesPerSec;
    header.bitsPerSample = pWaveFormat->wBitsPerSample;
    header.byteRate = pWaveFormat->nAvgBytesPerSec;
    header.blockAlign = pWaveFormat->nBlockAlign;

    // 计算大小
    header.subchunk2Size = static_cast<uint32_t>(audioBuffer.size());
    header.chunkSize = 36 + header.subchunk2Size; // 36 = 总头部大小 - 8 (RIFF和大小字段)

    // 写入WAV头
    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    // 写入音频数据
    file.write(reinterpret_cast<const char*>(audioBuffer.data()), audioBuffer.size());

    // 清理资源
    file.close();
    CoTaskMemFree(pWaveFormat);
    pAudioClient->Release();

    return true;
}

void AR::Recorder_Core_Windows::SetTargetDevice(const wstring& device_name)
{
	if (!device_name.empty()) {
		this->str_target_device = device_name;
	}
}

void AR::Recorder_Core_Windows::PlayRecordedAudio()
{
    if (!isPlaying)
    {
        isPlaying = true;
        shouldStop = false;
        if (playThread.joinable()) {
            playThread.join();
        }
        playThread = std::thread([=]() {this->PlayDataThreadFunction();});
    }
    else
    {
        std::cout << "Already playing." << std::endl;
    }
}

void AR::Recorder_Core_Windows::StopRecordedAudio()
{//关闭播放线程
    if (isPlaying) {
        isPlaying = false;
        if (playThread.joinable()) {
            playThread.join();
        }
    }
}

IMMDevice* AR::Recorder_Core_Windows::GetTargetDevice()
{
    IMMDeviceEnumerator* pEnumerator = NULL;
    IMMDeviceCollection* pCollection = NULL;
    IMMDevice* pDevice = NULL;
    IPropertyStore* pProps = NULL;
    LPWSTR pwszID = NULL;
    HRESULT hr;

    CoInitialize(NULL);

    // 创建设备枚举器
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        NULL,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&pEnumerator);
    if (FAILED(hr)) goto Exit;

    // 获取所有音频输入设备
    hr = pEnumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &pCollection);
    if (FAILED(hr)) goto Exit;

    UINT count;
    hr = pCollection->GetCount(&count);
    if (FAILED(hr)) goto Exit;

    // 遍历设备查找包含str_target_device的设备
    for (UINT i = 0; i < count; i++)
    {
        hr = pCollection->Item(i, &pDevice);
        if (FAILED(hr)) continue;

        hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
        if (FAILED(hr)) { pDevice->Release(); continue; }

        PROPVARIANT varName;
        PropVariantInit(&varName);

        hr = pProps->GetValue(PKEY_Device_FriendlyName, &varName);
        if (SUCCEEDED(hr))
        {
            std::wstring deviceName(varName.pwszVal);
            if (deviceName.find(this->str_target_device) != std::wstring::npos)
            {
                PropVariantClear(&varName);
                pProps->Release();
                pCollection->Release();
                pEnumerator->Release();
                return pDevice;
            }
        }
        PropVariantClear(&varName);
        pProps->Release();
        pDevice->Release();
    }

Exit:
    if (pCollection) pCollection->Release();
    if (pEnumerator) pEnumerator->Release();
    return NULL;
}
const float kMinDB = -60.0f;
const float kMaxDB = 0.0f;
// 将分贝值线性映射到0-100范围
short MapDBToRange(float db)
{
    float normalized = (db - kMinDB) / (kMaxDB - kMinDB);
    return static_cast<short>(normalized * 100.0f);
}
// 将RMS值转换为分贝值
float RMSToDB(float rms)
{
    if (rms <= 0.0f) return kMinDB;
    float db = 20.0f * log10f(rms);
    return std::clamp(db, kMinDB, kMaxDB);
}
float CalculateRMS(const BYTE* pData, UINT32 numFrames, UINT32 bytesPerFrame)
{
    if (numFrames == 0) return 0.0f;

    float sum = 0.0f;
    const float* samples = reinterpret_cast<const float*>(pData);
    UINT32 numSamples = numFrames * bytesPerFrame / sizeof(float);

    for (UINT32 i = 0; i < numSamples; ++i)
    {
        float sample = samples[i];
        sum += sample * sample;
    }

    return sqrtf(sum / numSamples);
}

void AR::Recorder_Core_Windows::StartRecordThreadFunction()
{
    long sampe_count = 0;
    IAudioClient* pAudioClient = NULL;
    IAudioCaptureClient* pCaptureClient = NULL;
    WAVEFORMATEX* pWaveFormat = NULL;
    UINT32 numFramesAvailable;
    UINT32 packetLength = 0;
    BYTE* pData;
    DWORD flags;
    HRESULT hr;
    UINT32 bufferFrameCount = 0;
    IPropertyStore* pProps = nullptr;
    PROPVARIANT varName;
    const REFERENCE_TIME hnsRequestedDuration = duration_time; // 100 * 10000 (100ms)
    const REFERENCE_TIME bufferDuration = 100 * 10000; // 100ms
    // 使用PCM格式
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    //给当前的pDevice找一下当前设备
    this->pDevice = this->GetTargetDevice();

    // 获取设备ID（唯一标识符）
    LPWSTR pwszDeviceId = nullptr;
    hr = pDevice->GetId(&pwszDeviceId);
    if (SUCCEEDED(hr)) {
        std::wcout << L"Device ID: " << pwszDeviceId << std::endl;
        CoTaskMemFree(pwszDeviceId); // 必须释放
    }

    // 2. 获取设备友好名称（Friendly Name）
    hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
    if (SUCCEEDED(hr)) {
        PROPVARIANT varName;
        PropVariantInit(&varName);

        // 获取 PKEY_Device_FriendlyName
        hr = pProps->GetValue(PKEY_Device_FriendlyName, &varName);
        if (SUCCEEDED(hr)) {
            std::wcout << L"Device Name: " <<std::wstring(varName.pwszVal) << std::endl;
        }
        PropVariantClear(&varName); // 清理 PROPVARIANT
        pProps->Release(); // 释放属性存储
    }




    hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
    if (FAILED(hr)) goto Exit;

    // 替换你的初始化代码：
    hr = pAudioClient->GetMixFormat(&pWaveFormat);  // 先获取设备支持的格式
    if (FAILED(hr)) {
        std::cerr << "GetMixFormat failed: 0x" << std::hex << hr << std::endl;
        goto Exit;
    }

    //pWaveFormat->wFormatTag = WAVE_FORMAT_PCM;
    //pWaveFormat->nChannels = 2;
    //pWaveFormat->nSamplesPerSec = 48000;
    //pWaveFormat->wBitsPerSample = 16;
    //pWaveFormat->nBlockAlign = pWaveFormat->nChannels * pWaveFormat->wBitsPerSample / 8;
    //pWaveFormat->nAvgBytesPerSec = pWaveFormat->nSamplesPerSec * pWaveFormat->nBlockAlign;
    //pWaveFormat->cbSize = 0;
    pWaveFormat->wFormatTag = WAVE_FORMAT_PCM;
    //pWaveFormat->nChannels = 1;  // 保持与设备一致
    //pWaveFormat->wBitsPerSample = 32;
    //pWaveFormat->nBlockAlign = 4;  // 1声道 * 32bit / 8 = 4字节/帧
    //pWaveFormat->nAvgBytesPerSec = 192000; // 48000 * 4
    pWaveFormat->cbSize = 0;
    WAVEFORMATEX* pWaveFormat2;

    hr = pAudioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, pWaveFormat, &pWaveFormat2);
    
    hr = pAudioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
        hnsRequestedDuration, 0, pWaveFormat, NULL);
    if (FAILED(hr))
    {
        std::cerr << "Initialize failed: 0x" << std::hex << hr << std::dec << std::endl;
        if (hr == AUDCLNT_E_UNSUPPORTED_FORMAT)
            std::cerr << "Unsupported format" << std::endl;
        else if (hr == AUDCLNT_E_DEVICE_IN_USE)
            std::cerr << "Device already in use" << std::endl;
        else if (hr == E_INVALIDARG)
            std::cerr << "Invalid arguments" << std::endl;
        goto Exit;
    }

    hr = pAudioClient->SetEventHandle(hEvent);

    hr = pAudioClient->GetBufferSize(&bufferFrameCount);
    if (FAILED(hr)) goto Exit;

    hr = pAudioClient->GetService(
        __uuidof(IAudioCaptureClient),
        (void**)&pCaptureClient);
    if (FAILED(hr)) goto Exit;

    hr = pAudioClient->Start();
    if (FAILED(hr)) goto Exit;

    std::cout << "Recording started..." << std::endl;

    while (isRecording) {
        WaitForSingleObject(hEvent, INFINITE);

        hr = pCaptureClient->GetBuffer(&pData, &numFramesAvailable, &flags, NULL, NULL);
        if (FAILED(hr)) break;

        if (pData != NULL) {
            // 电平计算（支持 32-bit）
            const float kMinDB = -60.0f;
            const float kMaxDB = 0.0f;
            float rmsLevel = 0.0f;

            if (pWaveFormat->wBitsPerSample == 32) {
                const int32_t* samples32 = reinterpret_cast<const int32_t*>(pData);
                size_t sampleCount = numFramesAvailable * pWaveFormat->nChannels;

                for (size_t i = 0; i < sampleCount; i++) {
                    float sample = samples32[i] / 2147483648.0f; // 32-bit → [-1.0, 1.0]
                    rmsLevel += sample * sample;
                }
            }
            else { // 16-bit
                const int16_t* samples16 = reinterpret_cast<const int16_t*>(pData);
                size_t sampleCount = numFramesAvailable * pWaveFormat->nChannels;

                for (size_t i = 0; i < sampleCount; i++) {
                    float sample = samples16[i] / 32768.0f; // 16-bit → [-1.0, 1.0]
                    rmsLevel += sample * sample;
                }
            }

            rmsLevel = sqrt(rmsLevel / (numFramesAvailable * pWaveFormat->nChannels));
            float rmsDb = 20.0f * log10(rmsLevel + 0.000001f);
            rmsDb = std::max(kMinDB, std::min(kMaxDB, rmsDb));
            short rmsMapped = static_cast<short>(((rmsDb - kMinDB) / (kMaxDB - kMinDB)) * 100.0f);

            // 存储电平数据
            {
                std::lock_guard<std::mutex> lock(mapMutex);
                levelMap[currentPos] = rmsMapped;
            }
            if (sampe_count++ > 2) {
                sampe_count = 0;
                emit this->Sig_UpdateLevel(rmsMapped);
            }

            // 更新音频缓冲区
            std::lock_guard<std::mutex> lock(bufferMutex);
            size_t dataSize = numFramesAvailable * pWaveFormat->nBlockAlign;

            if (pWaveFormat->nChannels == 1 && pWaveFormat->wBitsPerSample == 32) {
                // 单声道 32-bit → 双声道 16-bit
                const int32_t* pMono32Data = reinterpret_cast<const int32_t*>(pData);
                std::vector<int16_t> stereoData;
                stereoData.reserve(numFramesAvailable * 2);

                for (size_t i = 0; i < numFramesAvailable; ++i) {
                    int16_t sample16 = static_cast<int16_t>(pMono32Data[i] >> 16);
                    stereoData.push_back(sample16); // 左
                    stereoData.push_back(sample16); // 右
                }

                audioBuffer.insert(
                    audioBuffer.end(),
                    reinterpret_cast<const char*>(stereoData.data()),
                    reinterpret_cast<const char*>(stereoData.data() + stereoData.size())
                );
                currentPos += numFramesAvailable * 4; // 双声道 16-bit 大小
            }
            else if (pWaveFormat->nChannels == 2 && pWaveFormat->wBitsPerSample == 32) {
                // 双声道 32-bit → 双声道 16-bit
                const int32_t* pStereo32Data = reinterpret_cast<const int32_t*>(pData);
                std::vector<int16_t> stereo16Data;
                stereo16Data.reserve(numFramesAvailable * 2);

                for (size_t i = 0; i < numFramesAvailable; ++i) {
                    // 左声道 32→16
                    int16_t left16 = static_cast<int16_t>(pStereo32Data[i * 2] >> 16);
                    // 右声道 32→16
                    int16_t right16 = static_cast<int16_t>(pStereo32Data[i * 2 + 1] >> 16);

                    stereo16Data.push_back(left16);
                    stereo16Data.push_back(right16);
                }

                audioBuffer.insert(
                    audioBuffer.end(),
                    reinterpret_cast<const char*>(stereo16Data.data()),
                    reinterpret_cast<const char*>(stereo16Data.data() + stereo16Data.size())
                );
                currentPos += numFramesAvailable * 4; // 双声道 16-bit 大小
            }
            else {
                // 其他格式（如 16-bit），直接写入
                audioBuffer.insert(audioBuffer.end(), pData, pData + dataSize);
                currentPos += dataSize;
            }

            bufferCV.notify_one();
        }

        hr = pCaptureClient->ReleaseBuffer(numFramesAvailable);
        if (FAILED(hr)) break;
    }

    pAudioClient->Stop();
    if (FAILED(hr)) goto Exit;

    std::cout << "Recording stopped." << std::endl;
    this->StopRecording();
Exit:
    if (pWaveFormat) CoTaskMemFree(pWaveFormat);
    if (pCaptureClient) pCaptureClient->Release();
    if (pAudioClient) pAudioClient->Release();
    
}

void AR::Recorder_Core_Windows::PlayDataThreadFunction()
{
    IMMDeviceEnumerator* pEnumerator = NULL;
    IMMDevice* pDevice = NULL;
    IAudioClient* pAudioClient = NULL;
    IAudioRenderClient* pRenderClient = NULL;
    WAVEFORMATEX* pWaveFormat = NULL;
    UINT32 bufferFrameCount;
    UINT32 numFramesPadding;
    long sampe_count = 0;
    BYTE* pData;
    HRESULT hr;
    size_t pos = 0;
    CoInitialize(NULL);

    // 获取默认音频输出设备
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        NULL,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&pEnumerator);
    if (FAILED(hr)) goto Exit;

    hr = pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice);
    if (FAILED(hr)) goto Exit;

    hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
    if (FAILED(hr)) goto Exit;

    // 使用PCM格式
    pWaveFormat = (WAVEFORMATEX*)CoTaskMemAlloc(sizeof(WAVEFORMATEX));
    if (!pWaveFormat) goto Exit;

    pWaveFormat->wFormatTag = WAVE_FORMAT_PCM;
    pWaveFormat->nChannels = 2;
    pWaveFormat->nSamplesPerSec = 48000;
    pWaveFormat->wBitsPerSample = 16;
    pWaveFormat->nBlockAlign = pWaveFormat->nChannels * pWaveFormat->wBitsPerSample / 8;
    pWaveFormat->nAvgBytesPerSec = pWaveFormat->nSamplesPerSec * pWaveFormat->nBlockAlign;
    pWaveFormat->cbSize = 0;

    hr = pAudioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        0,
        duration_time,
        0,
        pWaveFormat,
        NULL);
    if (FAILED(hr))
    {
        std::cerr << "Initialize failed: 0x" << std::hex << hr << std::dec << std::endl;
        if (hr == AUDCLNT_E_UNSUPPORTED_FORMAT)
            std::cerr << "Unsupported format" << std::endl;
        else if (hr == AUDCLNT_E_DEVICE_IN_USE)
            std::cerr << "Device already in use" << std::endl;
        else if (hr == E_INVALIDARG)
            std::cerr << "Invalid arguments" << std::endl;
        goto Exit;
    }

    hr = pAudioClient->GetBufferSize(&bufferFrameCount);
    if (FAILED(hr)) goto Exit;

    hr = pAudioClient->GetService(
        __uuidof(IAudioRenderClient),
        (void**)&pRenderClient);
    if (FAILED(hr)) goto Exit;

    hr = pAudioClient->Start();
    if (FAILED(hr)) goto Exit;

    std::cout << "Playing started..." << std::endl;


    while (isPlaying && !shouldStop)
    {
        hr = pAudioClient->GetCurrentPadding(&numFramesPadding);
        if (FAILED(hr))
            break;

        UINT32 numFramesAvailable = bufferFrameCount - numFramesPadding;

        // 当没有数据可写时，检查是否所有数据都已提交
        if (numFramesAvailable == 0)
        {
            // 检查是否已经提交了所有音频数据
            std::unique_lock<std::mutex> lock(bufferMutex);
            bool allDataSubmitted = (pos >= audioBuffer.size());
            lock.unlock();

            if (allDataSubmitted)
            {
                // 直接退出循环，无需等待 numFramesPadding == 0
                 // （因为数据已全部提交，后续由外部循环等待播放完毕）
                break; // 退出主循环
            }

            Sleep(10);
            continue;
        }

        hr = pRenderClient->GetBuffer(numFramesAvailable, &pData);
        if (FAILED(hr))
            break;

        std::unique_lock<std::mutex> lock(bufferMutex);
        size_t bytesAvailable = audioBuffer.size() - pos;
        size_t bytesToCopy = numFramesAvailable * pWaveFormat->nBlockAlign;
        bool isLastBlock = false;

        if (bytesAvailable < bytesToCopy)
        {
            bytesToCopy = bytesAvailable;
            isLastBlock = true; // 标记这是最后一块数据
        }
       

        if (bytesToCopy > 0) {
            memcpy(pData, &audioBuffer[pos], bytesToCopy);

            // 从map中获取电平数据
            short rmsMapped = 0;
            {
                std::lock_guard<std::mutex> lock(mapMutex);
                auto it = levelMap.lower_bound(pos);
                if (it != levelMap.end()) {
                    rmsMapped = it->second;
                }
            }
            if (sampe_count++ > 2) {
                sampe_count = 0;
                emit this->Sig_UpdatePlayLevel(rmsMapped);
            }
           // emit this->Sig_UpdatePlayLevel(rmsMapped);
            pos += bytesToCopy;
        }
        else
        {
            memset(pData, 0, bytesToCopy);
        }

        lock.unlock();

        UINT32 framesToWrite = (UINT32)(bytesToCopy / pWaveFormat->nBlockAlign);  // 计算实际写入的帧数
        hr = pRenderClient->ReleaseBuffer(framesToWrite, isLastBlock ? AUDCLNT_BUFFERFLAGS_SILENT : 0);
        if (FAILED(hr)) break;
    }
    // 确保所有数据都已播放完毕
    if (isPlaying)
    {
        // 等待缓冲区完全播放完毕
        UINT32 paddingFrames = 0;
        do {
            hr = pAudioClient->GetCurrentPadding(&paddingFrames);
            if (FAILED(hr)) break;
            if (paddingFrames > 0) Sleep(10);
        } while (paddingFrames > 0);
    }
    hr = pAudioClient->Stop();
    if (FAILED(hr)) goto Exit;

    isPlaying = false;

    std::cout << "Playing stopped." << std::endl;

    
    emit this->Sig_PlaybackFinished();
    this->StopRecordedAudio();
Exit:
    if (pWaveFormat) CoTaskMemFree(pWaveFormat);
    if (pRenderClient) pRenderClient->Release();
    if (pAudioClient) pAudioClient->Release();
    if (pDevice) pDevice->Release();
    if (pEnumerator) pEnumerator->Release();
    CoUninitialize();
    emit this->Sig_PlaybackFinished();
}

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <mutex>
#include <vector>
#include <cmath>


AR::Recorder_Core_Windows::Recorder_Core_Windows(QObject* parent)
{
}

AR::Recorder_Core_Windows::~Recorder_Core_Windows()
{
    if (this->playThread.joinable()) {
        this->playThread.join();
    }

    if (this->recordThread.joinable()) {
        this->recordThread.join();
    }
}

bool AR::Recorder_Core_Windows::InitRecording(const wstring& target_device_name)
{
    this->str_target_device = target_device_name;
    this->pDevice = GetTargetDevice();
    return true;
}

void AR::Recorder_Core_Windows::StartRecording()
{
    if (!isRecording) {
        isRecording = true;
        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            this->audioBuffer.clear();
        }
        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            this->levelMap.clear();
        }
        //线程中运行此函数
        if (recordThread.joinable()) {
            recordThread.join();
        }
        recordThread = std::thread([=]() {this->StartRecordThreadFunction();});
    }
    else {
        std::cout << " Recording is already running!!";
    }
}

void AR::Recorder_Core_Windows::StopRecording()
{
    if (isRecording) {
        isRecording = false;
        if (recordThread.joinable())
        {
            recordThread.join();
        }
    }
}

void AR::Recorder_Core_Windows::PauseRecording()
{
    std::cout << "暂时没有支持这个接口!" << std::endl;
}

void AR::Recorder_Core_Windows::ResumeRecording()
{
    std::cout << "暂时没有支持这个接口!" << std::endl;
}

void AR::Recorder_Core_Windows::SetAudioFormat(WAVEFORMATEX format)
{
    //暂时似乎不需要这个接口，麦克风支持什么就录什么
    std::cout << "暂时不支持这个接口，麦克风支持什么格式就录什么格式";
}
