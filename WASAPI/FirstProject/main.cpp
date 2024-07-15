#include <iostream>
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

#pragma comment(lib, "Ole32.lib")

int main() {
    HRESULT hr;

    // 初始化COM库
    hr = CoInitialize(nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM library: " << hr << std::endl;
        return -1;
    }

    // 获取设备枚举器
    IMMDeviceEnumerator* pEnumerator = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&pEnumerator));
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator: " << hr << std::endl;
        CoUninitialize();
        return -1;
    }

    // 获取默认音频捕获设备
    IMMDevice* pDevice = nullptr;
    hr = pEnumerator->GetDefaultAudioEndpoint(eCapture, eCommunications, &pDevice);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint: " << hr << std::endl;
        pEnumerator->Release();
        CoUninitialize();
        return -1;
    }

    // 激活音频客户端
    IAudioClient* pAudioClient = nullptr;
    hr = pDevice->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, (void**)&pAudioClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client: " << hr << std::endl;
        pDevice->Release();
        pEnumerator->Release();
        CoUninitialize();
        return -1;
    }

    // 获取音频捕获格式
    WAVEFORMATEX* pwfx = nullptr;
    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        std::cerr << "Failed to get mix format: " << hr << std::endl;
        pAudioClient->Release();
        pDevice->Release();
        pEnumerator->Release();
        CoUninitialize();
        return -1;
    }

    // 初始化音频客户端
    hr = pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, 0, 0, pwfx, nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client: " << hr << std::endl;
        CoTaskMemFree(pwfx);
        pAudioClient->Release();
        pDevice->Release();
        pEnumerator->Release();
        CoUninitialize();
        return -1;
    }

    // 获取音频捕获服务
    IAudioCaptureClient* pCaptureClient = nullptr;
    hr = pAudioClient->GetService(IID_IAudioCaptureClient, (void**)&pCaptureClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to get capture client: " << hr << std::endl;
        pAudioClient->Release();
        pDevice->Release();
        pEnumerator->Release();
        CoUninitialize();
        return -1;
    }

    // 开始捕获
    hr = pAudioClient->Start();
    if (FAILED(hr)) {
        std::cerr << "Failed to start audio client: " << hr << std::endl;
        pCaptureClient->Release();
        pAudioClient->Release();
        pDevice->Release();
        pEnumerator->Release();
        CoUninitialize();
        return -1;
    }

    // 捕获音频数据
    while (true) {
        UINT32 packetLength = 0;
        hr = pCaptureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            std::cerr << "Failed to get next packet size: " << hr << std::endl;
            break;
        }

        while (packetLength != 0) {
            BYTE* pData;
            UINT32 numFramesAvailable;
            DWORD flags;

            hr = pCaptureClient->GetBuffer(&pData, &numFramesAvailable, &flags, nullptr, nullptr);
            if (FAILED(hr)) {
                std::cerr << "Failed to get buffer: " << hr << std::endl;
                break;
            }

            // 处理音频数据
            // 这里可以将pData中的音频数据保存到文件或进行其他处理

            hr = pCaptureClient->ReleaseBuffer(numFramesAvailable);
            if (FAILED(hr)) {
                std::cerr << "Failed to release buffer: " << hr << std::endl;
                break;
            }

            hr = pCaptureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                std::cerr << "Failed to get next packet size: " << hr << std::endl;
                break;
            }
        }
    }

    // 停止捕获
    pAudioClient->Stop();

    // 释放资源
    pCaptureClient->Release();
    pAudioClient->Release();
    pDevice->Release();
    pEnumerator->Release();
    CoUninitialize();

    return 0;
}
