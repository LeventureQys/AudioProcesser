#include "AudioRecorder_Windows.h"
void CALLBACK waveInProc(HWAVEIN hwi, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    if (uMsg == WIM_DATA) {
        AR::Recorder_Core_Windows* pThis = (AR::Recorder_Core_Windows*)dwInstance;
        WAVEHDR* pHdr = (WAVEHDR*)dwParam1;

        if (pHdr->dwBytesRecorded > 0) {
            // 关键修改：保存音频数据到容器
            BYTE* pData = (BYTE*)pHdr->lpData;
            pThis->recordedAudio.insert(
                pThis->recordedAudio.end(),
                pData,
                pData + pHdr->dwBytesRecorded
            );

            // 计算时间信息（保持原有逻辑）
            const WAVEFORMATEX& fmt = pThis->GetWaveFormat();
            AR::AudioTimeInfo timeInfo;
            timeInfo.bytes_recorded = pThis->GetRecordedBytesCount();
            timeInfo.samples_recorded = timeInfo.bytes_recorded / (fmt.wBitsPerSample / 8);
            timeInfo.usecs_elapsed =
                (static_cast<uint64_t>(timeInfo.samples_recorded) * 1'000'000ULL) /
                fmt.nSamplesPerSec;

            // 传递数据和时间信息
            pThis->HandleWaveInMessage(uMsg, dwParam1, dwParam2, timeInfo);
        }
    }
}

// 实现播放回调函数
void CALLBACK waveOutProc(HWAVEOUT hwo, UINT uMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    AR::Recorder_Core_Windows* pThis = reinterpret_cast<AR::Recorder_Core_Windows*>(dwInstance);
    if (pThis) {
        pThis->HandleWaveOutMessage(uMsg, dwParam1, dwParam2);
    }
}
AR::Recorder_Core_Windows::Recorder_Core_Windows(QObject* parent):QObject(parent)
{

}

AR::Recorder_Core_Windows::~Recorder_Core_Windows()
{
}
// 获取桌面路径（宽字符版本）
std::wstring GetDesktopPath()
{
    wchar_t desktopPath[MAX_PATH] = { 0 };

    // 使用 Unicode 版本的 API (SHGetSpecialFolderPathW)
    if (SHGetSpecialFolderPathW(NULL, desktopPath, CSIDL_DESKTOP, FALSE))
    {
        // 确保路径以反斜杠结尾
        size_t len = wcslen(desktopPath);
        if (len > 0 && desktopPath[len - 1] != L'\\')
        {
            wcscat_s(desktopPath, L"\\");
        }
        return desktopPath;
    }

    return L""; // 如果获取失败，返回空路径
}
bool AR::Recorder_Core_Windows::InitRecording(const wstring& target_device_name)
{
    this->SetTargetDevice(target_device_name);
    selectedDeviceId = FindTargetDevice();

   

    
    return true;
}

void AR::Recorder_Core_Windows::StartRecording()
{
    if (isRecording) {
        cout << "已经在录音中" << endl;
        return;
    }

    //开始之前必须重新尝试找到当前设备
    this->FindTargetDevice();

    recordedAudio.clear();
    MMRESULT result = waveInStart(hWaveIn);
    if (result != MMSYSERR_NOERROR) {
        cerr << "无法开始录音，错误代码: " << result << endl;
        return;
    }

    isRecording = true;
    //cout << "开始录音... 按任意键停止" << endl;
    //system("pause"); // 等待用户按键
    //StopRecording();
}

void AR::Recorder_Core_Windows::StopRecording()
{
    if (!isRecording) return;

    if (isPaused) {
        // 如果处于暂停状态，需要先恢复再停止
        waveInStart(hWaveIn);
    }

    waveInStop(hWaveIn);
    isRecording = false;
    isPaused = false;
    cout << "录音已停止，共录制 " << recordedAudio.size() << " 字节" << endl;
}

bool AR::Recorder_Core_Windows::SaveAsWav(const wstring& filename)
{
    if (recordedAudio.empty()) {
        cerr << "错误：没有录音数据可保存" << endl;
        return false;
    }

    // 修改后的代码段
    std::wstring fullPath;
    if (filename.find(L':') != std::wstring::npos ||   // 检查Windows驱动器（C:）
        filename.find(L'/') != std::wstring::npos ||   // 检查Unix风格路径
        filename.find(L'\\') != std::wstring::npos)    // 检查Windows风格路径
    {
        fullPath = filename; // 已经是完整路径
    }
    else {
        std::wstring desktopPath = GetDesktopPath();
        // 确保桌面路径以反斜杠结尾
        if (!desktopPath.empty() && desktopPath.back() != L'\\') {
            desktopPath += L'\\';
        }
        fullPath = desktopPath + filename; // 保存到桌面
    }

    // 使用_wfopen以支持宽字符路径（兼容中文）
    FILE* outFile = _wfopen(fullPath.c_str(), L"wb");
    if (!outFile) {
        std::wcerr << L"无法创建文件: " << fullPath << std::endl;
        return false;
    }

    // 准备WAV文件头
    WAVHeader header;
    header.numChannels = waveFormat.nChannels;
    header.sampleRate = waveFormat.nSamplesPerSec;
    header.bitsPerSample = waveFormat.wBitsPerSample;
    header.byteRate = waveFormat.nAvgBytesPerSec;
    header.blockAlign = waveFormat.nBlockAlign;
    header.subchunk2Size = static_cast<uint32_t>(recordedAudio.size());
    header.chunkSize = 36 + header.subchunk2Size;

    // 使用C风格文件操作写入数据
    fwrite(&header, sizeof(header), 1, outFile);
    fwrite(recordedAudio.data(), 1, recordedAudio.size(), outFile);

    fclose(outFile);

    // 输出保存信息（需要将wstring转换为当前控制台编码）
    std::wcout << L"已保存到: " << fullPath << std::endl;

    // 在资源管理器中显示文件

    return true;
}

void AR::Recorder_Core_Windows::Cleanup()
{
    if (hWaveIn) {
        waveInReset(hWaveIn);
        if (waveHdr.lpData) {
            waveInUnprepareHeader(hWaveIn, &waveHdr, sizeof(WAVEHDR));
            delete[] waveHdr.lpData;
            waveHdr.lpData = NULL;
        }
        waveInClose(hWaveIn);
        hWaveIn = NULL;
    }

    // 添加输出设备清理
    if (hWaveOut) {
        waveOutReset(hWaveOut);
        waveOutClose(hWaveOut);
        hWaveOut = NULL;
    }
}

void AR::Recorder_Core_Windows::SetTargetDevice(const wstring& device_name)
{
    this->str_target_device_name = device_name;
}

void AR::Recorder_Core_Windows::PlayRecordedAudio()
{
    if (recordedAudio.empty()) {
        cerr << "错误：没有录音数据可播放" << endl;
        return;
    }
    // 添加播放进度跟踪
    static uint64_t playbackPosition = 0; // 使用静态变量跟踪播放进度
    this->StopRecordedAudio();

    MMRESULT result = waveOutOpen(&hWaveOut, WAVE_MAPPER, &waveFormat,
        (DWORD_PTR)waveOutProc, (DWORD_PTR)this, CALLBACK_FUNCTION);
    if (result != MMSYSERR_NOERROR) {
        cerr << "无法打开音频输出设备，错误代码: " << result << endl;
        return;
    }

    // 分块处理音频数据（每块1024个样本）
    const size_t chunkSamples = 1024;
    const size_t chunkSize = BUFFER_SAMPLES * waveFormat.nBlockAlign;
    size_t remaining = recordedAudio.size();
    size_t offset = 0;

    while (remaining > 0) {
        size_t thisChunk = min(chunkSize, remaining);

        WAVEHDR* pHdr = new WAVEHDR;
        ZeroMemory(pHdr, sizeof(WAVEHDR));
        pHdr->dwBufferLength = thisChunk;
        pHdr->lpData = new char[thisChunk];
        memcpy(pHdr->lpData, recordedAudio.data() + offset, thisChunk);

        waveOutPrepareHeader(hWaveOut, pHdr, sizeof(WAVEHDR));
        waveOutWrite(hWaveOut, pHdr, sizeof(WAVEHDR));

        offset += thisChunk;
        remaining -= thisChunk;
        // 记录块的时间信息
        const size_t chunkDuration = (chunkSamples * 1'000'000ULL) / waveFormat.nSamplesPerSec;
        playbackPosition += chunkDuration;
    }


    cout << "正在播放录音..." << endl;
}

void AR::Recorder_Core_Windows::StopRecordedAudio()
{
    if (!hWaveOut) return;

    // 重置设备会自动取消所有未完成的缓冲区
    waveOutReset(hWaveOut);
    waveOutClose(hWaveOut);
    hWaveOut = NULL;
    cout << "播放已停止" << endl;
}

void AR::Recorder_Core_Windows::PauseRecording()
{
    if (!isRecording || isPaused) return;

    MMRESULT result = waveInStop(hWaveIn);
    if (result != MMSYSERR_NOERROR) {
        cerr << "暂停录音失败，错误代码: " << result << endl;
        return;
    }

    isPaused = true;
    cout << "录音已暂停" << endl;
}

void AR::Recorder_Core_Windows::ResumeRecording()
{
    if (!isRecording || !isPaused) return;

    MMRESULT result = waveInStart(hWaveIn);
    if (result != MMSYSERR_NOERROR) {
        cerr << "恢复录音失败，错误代码: " << result << endl;
        return;
    }

    isPaused = false;
    cout << "录音已恢复" << endl;
}

void AR::Recorder_Core_Windows::SetAudioFormat(WAVEFORMATEX format)
{

}

void AR::Recorder_Core_Windows::HandleWaveInMessage(UINT uMsg, DWORD_PTR dwParam1, DWORD_PTR dwParam2, AudioTimeInfo currentTimeSec)
{
    if (uMsg == WIM_DATA) // 当有新音频数据到达时
    {
        WAVEHDR* pWaveHdr = reinterpret_cast<WAVEHDR*>(dwParam1);
        if (pWaveHdr && pWaveHdr->lpData && pWaveHdr->dwBytesRecorded > 0)
        {
            // 假设音频是 16-bit 单声道 PCM
            const short* pSamples = reinterpret_cast<const short*>(pWaveHdr->lpData);
            int numSamples = pWaveHdr->dwBytesRecorded / sizeof(short);

            // 1. 计算 RMS（均方根）
            float rms = 0.0f;
            for (int i = 0; i < numSamples; i++)
            {
                float sample = pSamples[i] / 32768.0f; // 归一化到 [-1.0, 1.0]
                rms += sample * sample;
            }
            rms = sqrt(rms / numSamples);

            // 2. 防止对数计算出现负无穷（避免 log10(0)）
            const float kMinLevel = 0.0001f; // -80dB（接近静音）
            rms = std::max(rms, kMinLevel);

            // 3. 将 RMS 值转换为分贝（dB）
            float rmsDB = 20.0f * log10(rms);

            // 4. 定义最小和最大分贝值（用于归一化）
            const float kMinDB = -60.0f; // 最小显示分贝（例如 -60dB）
            const float kMaxDB = 0.0f;   // 最大分贝（0dB 是最大值）

            // 5. 将分贝值归一化到 0 - 1 范围
            float normalized = (rmsDB - kMinDB) / (kMaxDB - kMinDB);
            normalized = std::clamp(normalized, 0.0f, 1.0f); // 限制在 [0, 1]

            // 6. 转换为 0 - 100 范围（用于 UI 显示）
            int level = static_cast<int>(normalized * 100.0f);

            // 更新电平条（示例：存储或触发 UI 更新）
            double m_currentLevel = level;
            // 新增：存储时间戳和电平值（按微秒存储）
            map_time_volume.insert(
                currentTimeSec.usecs_elapsed,
                QPair<size_t, double>(currentTimeSec.usecs_elapsed, level)
            );
            emit this->Sig_UpdateLevel(level); // 假设有一个更新 UI 的方法
        }

        // 重新提交缓冲区（继续录音）
        waveInAddBuffer(hWaveIn, pWaveHdr, sizeof(WAVEHDR));
    }
}

void AR::Recorder_Core_Windows::HandleWaveOutMessage(UINT uMsg, DWORD_PTR dwParam1, DWORD_PTR dwParam2)
{
    if (uMsg == WOM_DONE) {
        WAVEHDR* pHdr = reinterpret_cast<WAVEHDR*>(dwParam1);
        if (pHdr && pHdr->lpData && pHdr->dwBufferLength > 0) {
            // 计算当前播放位置对应的微秒时间
            static uint64_t accumulatedUsecs = 0;
            const size_t samplesPerChunk = pHdr->dwBufferLength / (waveFormat.wBitsPerSample / 8);
            accumulatedUsecs += (samplesPerChunk * 1'000'000ULL) / waveFormat.nSamplesPerSec;

            // 查找最近的时间戳电平数据
            auto it = map_time_volume.lowerBound(accumulatedUsecs);
            if (it != map_time_volume.end()) {
                emit Sig_UpdatePlayLevel(it.value().second); // 使用录制时存储的电平值
            }

            // 清理资源
            waveOutUnprepareHeader(hWaveOut, pHdr, sizeof(WAVEHDR));
            delete[] pHdr->lpData;
            delete pHdr;
        }
    }
}

const WAVEFORMATEX& AR::Recorder_Core_Windows::GetWaveFormat() const
{
    return waveFormat;
}

size_t AR::Recorder_Core_Windows::GetRecordedBytesCount() const
{
    return recordedAudio.size();
}



UINT AR::Recorder_Core_Windows::FindTargetDevice()
{
    waveFormat.wFormatTag = WAVE_FORMAT_PCM;
    waveFormat.nChannels = 1;
    waveFormat.nSamplesPerSec = 48000;
    waveFormat.wBitsPerSample = 16;
    waveFormat.nBlockAlign = waveFormat.nChannels * waveFormat.wBitsPerSample / 8;
    waveFormat.nAvgBytesPerSec = waveFormat.nSamplesPerSec * waveFormat.nBlockAlign;
    waveFormat.cbSize = 0;

    UINT deviceCount = waveInGetNumDevs();
    UINT int_target_device = WAVE_MAPPER; // 默认使用映射器
    if (deviceCount == 0) {
        cout << "未找到音频输入设备" << endl;
        return WAVE_MAPPER;
    }

    cout << "\n可用音频输入设备:" << endl;

    bool blnFindAvaliableDevice = false;
    for (UINT i = 0; i < deviceCount; i++) {
        WAVEINCAPS caps;
        if (waveInGetDevCaps(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR) {
            wcout << i << ": " << caps.szPname << endl;

            wstring deviceName(caps.szPname);
            transform(deviceName.begin(), deviceName.end(), deviceName.begin(), ::towlower);
            // 将目标设备名称也转换为小写
            wstring targetLower = this->str_target_device_name;
            transform(targetLower.begin(), targetLower.end(), targetLower.begin(), ::towlower);

            if (deviceName.find(targetLower) != wstring::npos) {
                cout << "\n找到目标设备: ID " << i << endl;
                int_target_device = i;
                blnFindAvaliableDevice = true;
                break;
            }
        }
    }

    // 如果找到合适的设备，关闭现有句柄并重新创建
    if (blnFindAvaliableDevice) {
        // 关闭现有音频输入设备
        if (hWaveIn != NULL) {
            waveInReset(hWaveIn);  // 停止任何正在进行的录音
            waveInClose(hWaveIn);  // 关闭句柄
            hWaveIn = NULL;

            // 清理之前的缓冲区
            if (waveHdr.lpData != NULL) {
                delete[] waveHdr.lpData;
                waveHdr.lpData = NULL;
            }
        }


        // 使用找到的设备ID打开新设备
        MMRESULT result = waveInOpen(&hWaveIn, int_target_device, &waveFormat,
            (DWORD_PTR)waveInProc, (DWORD_PTR)this, CALLBACK_FUNCTION);
        if (result != MMSYSERR_NOERROR) {
            cerr << "无法打开音频输入设备，错误代码: " << result << endl;
            return WAVE_MAPPER;
        }
        else {
            cout << "设备注册成功 " << endl;
        }

        // 准备新的缓冲区
        ZeroMemory(&waveHdr, sizeof(WAVEHDR));
        waveHdr.dwBufferLength = BUFFER_SAMPLES * waveFormat.nBlockAlign; // 统一大小
        waveHdr.lpData = new char[waveHdr.dwBufferLength];

        result = waveInPrepareHeader(hWaveIn, &waveHdr, sizeof(WAVEHDR));
        if (result != MMSYSERR_NOERROR) {
            cerr << "无法准备音频头，错误代码: " << result << endl;
            return WAVE_MAPPER;
        }

        result = waveInAddBuffer(hWaveIn, &waveHdr, sizeof(WAVEHDR));
        if (result != MMSYSERR_NOERROR) {
            cerr << "无法添加音频缓冲区，错误代码: " << result << endl;
            return WAVE_MAPPER;
        }

        return int_target_device;
    }

    // 如果没有找到目标设备，尝试使用默认设备
    MMRESULT result = waveInOpen(&hWaveIn, WAVE_MAPPER, &waveFormat,
        (DWORD_PTR)waveInProc, (DWORD_PTR)this, CALLBACK_FUNCTION);
    if (result != MMSYSERR_NOERROR) {
        cerr << "无法打开音频输入设备，错误代码: " << result << endl;
        return WAVE_MAPPER;
    }

    ZeroMemory(&waveHdr, sizeof(WAVEHDR));
    waveHdr.dwBufferLength = 44100 * 2; // 1秒的缓冲区
    waveHdr.lpData = new char[waveHdr.dwBufferLength];

    result = waveInPrepareHeader(hWaveIn, &waveHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR) {
        cerr << "无法准备音频头，错误代码: " << result << endl;
        return WAVE_MAPPER;
    }

    result = waveInAddBuffer(hWaveIn, &waveHdr, sizeof(WAVEHDR));
    if (result != MMSYSERR_NOERROR) {
        cerr << "无法添加音频缓冲区，错误代码: " << result << endl;
        return WAVE_MAPPER;
    }

    return WAVE_MAPPER;
}

void AR::Recorder_Core_Windows::RecordVolume(double volume)
{
    ////记录一个当前umsec的时间和volume
    //size_t process_time = volume_time;

    //this->map_time_volume.insert(this->map_time_volume.size(), QPair<size_t, double>(process_time, volume));
}
