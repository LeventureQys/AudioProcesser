#include "audiorecorder_core.h"
#ifdef _WIN32
#endif 
AR::AudioRecorder::AudioRecorder(QObject* parent)
{
    connect(this, &AudioRecorder::SThread_Initialize, this, &AudioRecorder::Initialize);
    connect(this, &AudioRecorder::SThread_SetAudioFormat, this, &AudioRecorder::SetAudioFormat);
    connect(this, &AudioRecorder::SThread_SetAudioDeviceInfo, this, &AudioRecorder::SetAudioDeviceInfo);
    connect(this, &AudioRecorder::SThread_StartRecord, this, &AudioRecorder::StartRecord);

    connect(this, &AudioRecorder::SThread_StopRecord, this, &AudioRecorder::StopRecord);
    connect(this, &AudioRecorder::SThread_PauseRecord, this, &AudioRecorder::PauseRecord);
    connect(this, &AudioRecorder::SThread_ResumeRecord, this, &AudioRecorder::ResumeRecord);
    connect(this, &AudioRecorder::SThread_PlayRecordedData, this, &AudioRecorder::PlayRecordedData);
    connect(this, &AudioRecorder::SThread_StopPlayBack, this, &AudioRecorder::StopPlayBack);
    connect(this, &AudioRecorder::SThread_SaveRecordAsWavFile, this, &AudioRecorder::SaveRecordAsWavFile);
    connect(this, &AudioRecorder::SThread_SetTargetDeviceName, this, &AudioRecorder::SetTargetDeviceName);
}

AR::AudioRecorder::~AudioRecorder()
{
#ifdef _WIN32
    this->recorder->deleteLater();
#endif 
}

void AR::AudioRecorder::IInitialize(const QString& str_target_device)
{
    this->SThread_Initialize(str_target_device);
}

void AR::AudioRecorder::ISetAudioFormat(QAudioFormat format)
{
    this->SThread_SetAudioFormat(format);
}

void AR::AudioRecorder::ISetAudioDeviceInfo(QAudioDevice info)
{
    this->SThread_SetAudioDeviceInfo(info);
}

void AR::AudioRecorder::IStartRecord()
{
    this->SThread_StartRecord();
}

void AR::AudioRecorder::IStopRecord()
{
    this->SThread_StopRecord();
}

void AR::AudioRecorder::IPauseRecord()
{
    this->SThread_PauseRecord();
}

void AR::AudioRecorder::IResumeRecord()
{
    this->SThread_ResumeRecord();
}

void AR::AudioRecorder::IPlayRecordedData()
{
    this->SThread_PlayRecordedData();
}

void AR::AudioRecorder::IStopPlayBack()
{
    this->SThread_StopPlayBack();
}

void AR::AudioRecorder::ISaveRecordAsWavFile(const QString& filePath)
{
    this->SThread_SaveRecordAsWavFile(filePath);
}

void AR::AudioRecorder::ISetTargetDeviceName(const QString& str_target_device)
{
    this->SThread_SetTargetDeviceName(str_target_device);
}



void AR::AudioRecorder::Initialize(const QString& str_target_device)
{
    



#ifdef _WIN32

    //设置控制台获得打印正确
    // 设置控制台输出为UTF-8编码
    SetConsoleOutputCP(CP_UTF8);

    // 设置控制台字体（可选）
    CONSOLE_FONT_INFOEX font = { sizeof(font) };
    GetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &font);
    wcscpy_s(font.FaceName, L"Consolas"); // 或使用支持中文的字体如"SimSun"
    SetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &font);

    std::cout << "中文测试" << std::endl;

	if (this->recorder == nullptr) {
		this->recorder = new Recorder_Core_Windows(this);
		connect(this->recorder, &Recorder_Core_Windows::Sig_UpdateLevel, this, &AudioRecorder::Sig_Volumechanged);
        connect(this->recorder, &Recorder_Core_Windows::Sig_UpdatePlayLevel, this, &AudioRecorder::Sig_VolumePlayed);
        connect(this->recorder, &Recorder_Core_Windows::Sig_PlaybackFinished, this, &AudioRecorder::Sig_PlaybackFinished);
    }
	//设置目标设备
	//this->recorder->SetTargetDevice(str_target_device.toStdWString());
    this->recorder->InitRecording(str_target_device.toStdWString());


#endif
}

bool AR::AudioRecorder::SetAudioFormat(QAudioFormat format)
{
#ifdef _WIN32
    WAVEFORMATEX wfx;
    ZeroMemory(&wfx, sizeof(WAVEFORMATEX));

    // Set basic parameters
    wfx.nChannels = format.channelCount();
    wfx.nSamplesPerSec = format.sampleRate();

    // Qt6 uses SampleFormat enum
    switch (format.sampleFormat()) {
    case QAudioFormat::Int16:
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.wBitsPerSample = 16;
        break;
    case QAudioFormat::Int32:
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.wBitsPerSample = 32;
        break;
    case QAudioFormat::Float:
        wfx.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
        wfx.wBitsPerSample = 32;
        break;
    case QAudioFormat::UInt8:
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.wBitsPerSample = 8;
        break;
    default:
        throw std::runtime_error("Unsupported Qt6 sample format");
        return false;
    }

    // Calculate derived parameters
    wfx.nBlockAlign = wfx.nChannels * (wfx.wBitsPerSample / 8);
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
    wfx.cbSize = 0; // Must be 0 for PCM/IEEE_FLOAT

    // Validate
    if (wfx.nChannels == 0 || wfx.nSamplesPerSec == 0) {
        throw std::runtime_error("Invalid channel count or sample rate");
        return false;
    }

    this->recorder->SetAudioFormat(wfx);
    return true;
#endif
 

}

bool AR::AudioRecorder::SetAudioDeviceInfo(QAudioDevice info)
{
#ifdef _WIN32
    QString str_target_device_name = info.description();
    this->recorder->SetTargetDevice(str_target_device_name.toStdWString());
    return true;
#endif
}

bool AR::AudioRecorder::StartRecord()
{
#ifdef _WIN32
    this->recorder->StartRecording();
    return true;
#endif
}

bool AR::AudioRecorder::StopRecord()
{
#ifdef _WIN32
    this->recorder->StopRecording();
    return true;
#endif
}

bool AR::AudioRecorder::PauseRecord()
{
#ifdef _WIN32
    this->recorder->PauseRecording();
    return true;
#endif
}

bool AR::AudioRecorder::ResumeRecord()
{
#ifdef _WIN32
    this->recorder->ResumeRecording();
    return true;
#endif
}

bool AR::AudioRecorder::PlayRecordedData()
{
#ifdef _WIN32
    this->recorder->PlayRecordedAudio();
    return true;
#endif
}

bool AR::AudioRecorder::StopPlayBack()
{
#ifdef _WIN32
    this->recorder->StopRecordedAudio();
    return true;
#endif
}

bool AR::AudioRecorder::SaveRecordAsWavFile(const QString& filePath)
{
#ifdef _WIN32
    this->recorder->SaveAsWav(filePath.toStdWString());
    return true;
#endif
}

void AR::AudioRecorder::SetTargetDeviceName(const QString& str_target_device)
{
#ifdef _WIN32
    this->recorder->SetTargetDevice(str_target_device.toStdWString());
    
#endif
}

//void AR::AudioRecorder::RecordVolume(double volume, size_t volume_time)
//{
//	//记录一个当前umsec的时间和volume
//	size_t process_time = volume_time;
//
//	this->map_time_volume.insert(this->map_time_volume.size(), QPair<size_t, double>(process_time, volume));
//	//qDebug() << "time : " << this->audioInput->processedUSecs() << " volume : " << volume;
//	emit this->Sig_Volumechanged(volume);
//}
