#ifndef AUDIODEVICE_H
#define AUDIODEVICE_H

#include <QAudioBuffer>
#include <QApplication>
#include <QMediaPlayer>
#include <QAudioOutput>
#include <QIODevice>
#include <QAudioFormat>
#include <QTimer>
#include <QAudioInput>
#include <QAudioDeviceInfo>
#include <QQueue>
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>

class AudioOutputThread : public QThread {
    Q_OBJECT

public:
    AudioOutputThread(QMutex* mutex, QWaitCondition* condition, QObject* parent = nullptr)
        : QThread(parent), mutex(mutex), condition(condition) {
        // Initialize audio format
        QAudioFormat format;
        format.setSampleRate(48000);
        format.setChannelCount(1);  // Mono
        format.setSampleSize(16);
        format.setCodec("audio/pcm");
        format.setByteOrder(QAudioFormat::LittleEndian);
        format.setSampleType(QAudioFormat::SignedInt);

        // Initialize audio output device
        QAudioDeviceInfo outputDeviceInfo = QAudioDeviceInfo::defaultOutputDevice();
        audioOutput = new QAudioOutput(outputDeviceInfo, format);
        outputIODevice = audioOutput->start();
    }

    void run() override {
        while (true) {
            mutex->lock();
            while (buffer.size() < 4800) {
                condition->wait(mutex);
            }

            short output[4800];
            for (int i = 0; i < 4800; ++i) {
                output[i] = buffer.dequeue();
            }
            mutex->unlock();

            outputIODevice->write(reinterpret_cast<const char*>(output), sizeof(output));
        }
    }

    void enqueueData(const short* data, int size) {
        mutex->lock();
        for (int i = 0; i < size; ++i) {
            buffer.enqueue(data[i]);
        }
        condition->wakeAll();
        mutex->unlock();
    }

private:
    QQueue<short> buffer;
    QMutex* mutex;
    QWaitCondition* condition;
    QAudioOutput* audioOutput;
    QIODevice* outputIODevice;
};

#include "qdir.h"
class AudioDevice_ : public QIODevice {
    Q_OBJECT

public:
    AudioDevice_(QObject* parent = nullptr) : QIODevice(parent) {
        // Get the directory of the current executable
        QString executableDir = QCoreApplication::applicationDirPath();

        // Use QDir to construct the path
        QDir dir(executableDir);
        QString modelPath = dir.filePath("model/DeepFilterNet3_onnx.tar.gz");

        // Convert to const char* type
        QByteArray modelPathBytes = modelPath.toLocal8Bit();
        const char* model_path = modelPathBytes.constData();

        // Initialize audio output thread
        outputThread = new AudioOutputThread(&mutex, &condition);
        outputThread->start();
    }

    ~AudioDevice_() {
        outputThread->quit();
        outputThread->wait();
        delete outputThread;
    }

    bool open(OpenMode mode) override {
        return QIODevice::open(mode | QIODevice::ReadWrite); // Open in ReadWrite mode
    }

    qint64 readData(char* data, qint64 maxlen) override {
        // No need to implement because we only use writeData
        Q_UNUSED(data);
        Q_UNUSED(maxlen);
        return 0;
    }

    qint64 writeData(const char* data, qint64 len) override {
        // Store the received data into the buffer
        const short* samples = reinterpret_cast<const short*>(data);
        qint64 sampleCount = len / sizeof(short);
        for (qint64 i = 0; i < sampleCount; ++i) {
            buffer.enqueue(samples[i]);
            if (buffer.size() >= 480) {
                // Buffer is full, process the data
                short input[480];
                short output[480];
                for (int j = 0; j < 480; ++j) {
                    input[j] = buffer.dequeue();
                }
                qint64 first_time = QDateTime::currentDateTime().toMSecsSinceEpoch();
                if (blnDF) {
                    // Here is where you put your algorithm
                    // float snr = df_process_frame_i16(df_state, input, output);
                    qint64 lastTime = QDateTime::currentDateTime().toMSecsSinceEpoch();

                    // Processed data can be output or passed to other places here
                    // qDebug() << "Processed frame with Time " << lastTime - first_time << " ms, SNR:" << snr;
                }
                else {
                    // Use memcpy for memory copy
                    memcpy(output, input, 480 * sizeof(short));
                }

                // Put the processed data into the output buffer
                if (this->blnReturn)
                    outputThread->enqueueData(output, 480);
            }
        }
        return len;  // Return the length of the written data
    }

    /// <summary>
    /// Set monitoring
    /// </summary>
    /// <param name="blnReturn"></param>
    void setReturn(bool blnReturn) {
        this->blnReturn = blnReturn;
    }

    void setDF(bool blnDF) {
        this->blnDF = blnDF;
    }

private:
    QQueue<short> buffer;
    AudioOutputThread* outputThread;
    QMutex mutex;
    QWaitCondition condition;
    bool blnReturn;
    bool blnDF;
};

#endif // AUDIODEVICE_H
