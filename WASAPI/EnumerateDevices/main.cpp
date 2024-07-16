#include <Windows.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <audioclient.h>
#include <Functiondiscoverykeys_devpkey.h>
#include <iostream>
#include <atlbase.h>
#include <fcntl.h>
#include <io.h>



int main() {
    // 设置控制台输出模式为Unicode
    _setmode(_fileno(stdout), _O_U16TEXT);

    HRESULT hr = CoInitialize(nullptr);
    if (FAILED(hr)) {
        std::wcerr << L"Failed to initialize COM library: " << hr << std::endl;
        return -1;
    }

    CComPtr<IMMDeviceEnumerator> pEnumerator;
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator), nullptr,
        CLSCTX_INPROC_SERVER, __uuidof(IMMDeviceEnumerator),
        (void**)&pEnumerator);
    if (FAILED(hr)) {
        std::wcerr << L"Failed to create device enumerator: " << hr << std::endl;
        CoUninitialize();
        return -1;
    }

    CComPtr<IMMDeviceCollection> pCollection;
    hr = pEnumerator->EnumAudioEndpoints(eAll, DEVICE_STATE_ACTIVE, &pCollection);
    if (FAILED(hr)) {
        std::wcerr << L"Failed to enumerate audio endpoints: " << hr << std::endl;
        CoUninitialize();
        return -1;
    }

    UINT count;
    hr = pCollection->GetCount(&count);
    if (FAILED(hr)) {
        std::wcerr << L"Failed to get device count: " << hr << std::endl;
        CoUninitialize();
        return -1;
    }

    for (UINT i = 0; i < count; i++) {
        CComPtr<IMMDevice> pDevice;
        hr = pCollection->Item(i, &pDevice);
        if (FAILED(hr)) {
            std::wcerr << L"Failed to get device: " << hr << std::endl;
            continue;
        }

        LPWSTR pwszID = nullptr;
        hr = pDevice->GetId(&pwszID);
        if (FAILED(hr)) {
            std::wcerr << L"Failed to get device ID: " << hr << std::endl;
            continue;
        }

        CComPtr<IPropertyStore> pProps;
        hr = pDevice->OpenPropertyStore(STGM_READ, &pProps);
        if (FAILED(hr)) {
            std::wcerr << L"Failed to open property store: " << hr << std::endl;
            CoTaskMemFree(pwszID);
            continue;
        }

        PROPVARIANT varName;
        PropVariantInit(&varName);

        hr = pProps->GetValue(PKEY_Device_FriendlyName, &varName);
        if (FAILED(hr)) {
            std::wcerr << L"Failed to get device friendly name: " << hr << std::endl;
            CoTaskMemFree(pwszID);
            continue;
        }

        // 获取IAudioClient接口
        CComPtr<IAudioClient> pAudioClient;
        hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
        if (FAILED(hr)) {
            std::wcerr << L"Failed to activate audio client: " << hr << std::endl;
            CoTaskMemFree(pwszID);
            continue;
        }

        // 获取混音格式
        WAVEFORMATEX* pWaveFormat = nullptr;
        hr = pAudioClient->GetMixFormat(&pWaveFormat);
        if (FAILED(hr)) {
            std::wcerr << L"Failed to get mix format: " << hr << std::endl;
            CoTaskMemFree(pwszID);
            continue;
        }

        // 判断是否为立体声
        bool isStereo = (pWaveFormat->nChannels == 2);

        std::wcout << L"Device " << i + 1 << L": " << varName.pwszVal
            << L" (Stereo: " << (isStereo ? L"Yes" : L"No") << L")" << std::endl;

        PropVariantClear(&varName);
        CoTaskMemFree(pWaveFormat);
        CoTaskMemFree(pwszID);
    }

    CoUninitialize();
    return 0;
}
