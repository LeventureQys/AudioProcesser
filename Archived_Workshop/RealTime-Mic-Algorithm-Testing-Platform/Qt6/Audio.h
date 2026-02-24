#ifndef AUDIODEVICE_H
#define AUDIODEVICE_H

#include <QAudioFormat>
#include <QAudioSink>
#include <QIODevice>
#include <QTimer>
#include <QAudioSource>
#include <QMediaDevices>
#include <QAudioDevice>
#include <QQueue>
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QCoreApplication>
#include <QDir>

class AudioOutputThread : public QThread {
    Q_OBJECT

public:
    AudioOutputThread(QMutex* mutex, QWaitCondition* condition, QObject* parent = nullptr)
        : QThread(parent), mutex(mutex), condition(condition), audioSink(nullptr), outputIODevice(nullptr) {
        // 初始化音频格式
        QAudioFormat format;
        format.setSampleRate(48000);
        format.setChannelConfig(QAudioFormat::ChannelConfigMono);  // 单声道
        format.setSampleFormat(QAudioFormat::Int16);  // 16位有符号整数

        // 初始化音频输出设备
        QAudioDevice outputDeviceInfo = QMediaDevices::defaultAudioOutput();

        // 检查格式是否支持
        if (!outputDeviceInfo.isFormatSupported(format)) {
            qDebug() << " 输出格式有误，不合适 ";
        }

        audioSink = new QAudioSink(outputDeviceInfo, format);
        outputIODevice = audioSink->start();
    }

    ~AudioOutputThread() {
        if (audioSink) {
            audioSink->stop();
            delete audioSink;
        }
    }

    void run() override {
        while (!isInterruptionRequested()) {
            mutex->lock();
            while (buffer.size() < 4800 && !isInterruptionRequested()) {
                condition->wait(mutex, 100);  // 添加超时以允许线程中断
            }

            if (isInterruptionRequested()) {
                mutex->unlock();
                break;
            }

            short output[4800];
            for (int i = 0; i < 4800; ++i) {
                output[i] = buffer.dequeue();
            }
            mutex->unlock();

            if (outputIODevice && outputIODevice->isWritable()) {
                outputIODevice->write(reinterpret_cast<const char*>(output), sizeof(output));
            }
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
    QAudioSink* audioSink;
    QIODevice* outputIODevice;
};

class AudioDevice_ : public QIODevice {
    Q_OBJECT

public:
    AudioDevice_(QObject* parent = nullptr) : QIODevice(parent), blnReturn(false), blnDF(false) {
        // 获取当前可执行文件的目录
        QString executableDir = QCoreApplication::applicationDirPath();

        // 使用 QDir 构造路径
        QDir dir(executableDir);
        QString modelPath = dir.filePath("model/DeepFilterNet3_onnx.tar.gz");

        // 转换为 const char* 类型
        QByteArray modelPathBytes = modelPath.toLocal8Bit();
        const char* model_path = modelPathBytes.constData();

        // 初始化音频输出线程
        outputThread = new AudioOutputThread(&mutex, &condition);
        outputThread->start();
    }

    ~AudioDevice_() {
        outputThread->requestInterruption();
        outputThread->quit();
        outputThread->wait();
        delete outputThread;
    }

    bool open(OpenMode mode) override {
        return QIODevice::open(mode | QIODevice::ReadWrite); // 以读写模式打开
    }

    qint64 readData(char* data, qint64 maxlen) override {
        // 不需要实现，因为我们只使用 writeData
        Q_UNUSED(data);
        Q_UNUSED(maxlen);
        return 0;
    }

    qint64 writeData(const char* data, qint64 len) override {
        // 将接收到的数据存储到缓冲区
        const short* samples = reinterpret_cast<const short*>(data);
        qint64 sampleCount = len / sizeof(short);

        for (qint64 i = 0; i < sampleCount; ++i) {
            buffer.enqueue(samples[i]);

            if (buffer.size() >= 480) {
                // 缓冲区已满，处理数据
                short input[480];
                short output[480];

                for (int j = 0; j < 480; ++j) {
                    input[j] = buffer.dequeue();
                }

                qint64 first_time = QDateTime::currentMSecsSinceEpoch();

                if (blnDF) {
                    // 这里是放置算法的地方
                    // float snr = df_process_frame_i16(df_state, input, output);
                    qint64 lastTime = QDateTime::currentMSecsSinceEpoch();

                    // 处理后的数据可以在这里输出或传递到其他地方
                    // qDebug() << "处理帧，耗时 " << lastTime - first_time << " 毫秒，信噪比:" << snr;
                }
                else {
                    // 使用 memcpy 进行内存复制
                    memcpy(output, input, 480 * sizeof(short));
                }

                // 将处理后的数据放入输出缓冲区
                if (this->blnReturn)
                    outputThread->enqueueData(output, 480);
            }
        }
        return len;  // 返回写入数据的长度
    }

    /// <summary>
    /// 设置监听
    /// </summary>
    /// <param name="blnReturn">是否返回处理后的音频</param>
    void setReturn(bool blnReturn) {
        this->blnReturn = blnReturn;
    }

    /// <summary>
    /// 设置是否使用降噪算法
    /// </summary>
    /// <param name="blnDF">是否启用 DeepFilter</param>
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