#ifndef AUDIORECORDER_WINDOWS_H
#define AUDIORECORDER_WINDOWS_H
#include <QAudioDevice>
#include <QAudioInput>
#include "AudioRecorder_Core_global.h"
#include "QObject"
#include "qmap"
#ifdef _WIN32
// Windows 系统专用的代码
#include "AudioRecorder_Windows.h"
#else
// 非 Windows 系统（如 Linux、macOS）
printf("This is not Windows.\n");
#endif

/// <summary>
/// AudioRecorder_Windows
/// </summary>
namespace AR {
    using namespace std;
   
    class AUDIORECORDER_WINDOWS_EXPORT AudioRecorder : public QObject
    {
        Q_OBJECT
    public:
        AudioRecorder(QObject* parent);
        ~AudioRecorder();
        void IInitialize(const QString& str_target_device);
        void ISetAudioFormat(QAudioFormat format);
        void ISetAudioDeviceInfo(QAudioDevice info);

        void IStartRecord();
        void IStopRecord();
        void IPauseRecord();
        void IResumeRecord();
        void IPlayRecordedData();
        void IStopPlayBack();
        void ISaveRecordAsWavFile(const QString& filePath);
        void ISetTargetDeviceName(const QString& str_target_device);


    signals:
        void Sig_Volumechanged(double volume);
        void Sig_VolumePlayed(double volume);
        void Sig_PlaybackFinished();


    private:
        bool blnRecordTempLevelVolume = false;
        /// <summary>
    /// key : time value : volume
    /// </summary>
        //QMap<size_t, QPair<size_t, double>> map_time_volume;

        /// <summary>
        /// 记录音频的电平信号和时间的联系，来自Core的信号需要连接到这个类
        /// </summary>
        /// <param name="volume"></param>
        /// <param name="volume_time"></param>
        //void RecordVolume(double volume, size_t volume_time);


        void Initialize(const QString& str_target_device);
        bool SetAudioFormat(QAudioFormat format);
        bool SetAudioDeviceInfo(QAudioDevice info);

        bool StartRecord();
        bool StopRecord();
        bool PauseRecord();
        bool ResumeRecord();
        bool PlayRecordedData();
        bool StopPlayBack();
        bool SaveRecordAsWavFile(const QString& filePath);
        void SetTargetDeviceName(const QString& str_target_device);

    signals:
        void SThread_Initialize(const QString& str_target_device);
        void SThread_SetAudioFormat(const QAudioFormat& format);
        void SThread_SetAudioDeviceInfo(const QAudioDevice& info);
        
        void SThread_StartRecord();
        void SThread_StopRecord();
        void SThread_PauseRecord();
        void SThread_ResumeRecord();

        void SThread_PlayRecordedData();
        void SThread_StopPlayBack();
        void SThread_SaveRecordAsWavFile(const QString& file_path);
        void SThread_SetTargetDeviceName(const QString& str_target_device);
#ifdef _WIN32
        Recorder_Core_Windows* recorder = nullptr;



#endif
    };
}


#endif // AUDIORECORDER_WINDOWS_H
