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
#include "./include/df/deep_filter.h"

class AudioOutputThread : public QThread {
	Q_OBJECT

public:
	AudioOutputThread(QMutex* mutex, QWaitCondition* condition, QObject* parent = nullptr)
		: QThread(parent), mutex(mutex), condition(condition) {
		// 初始化音频格式
		QAudioFormat format;
		format.setSampleRate(48000);
		format.setChannelCount(1);  // 单声道
		format.setSampleSize(16);
		format.setCodec("audio/pcm");
		format.setByteOrder(QAudioFormat::LittleEndian);
		format.setSampleType(QAudioFormat::SignedInt);

		// 初始化音频输出设备
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
		// 获取当前可执行文件的目录
		QString executableDir = QCoreApplication::applicationDirPath();

		// 使用QDir拼接路径
		QDir dir(executableDir);
		QString modelPath = dir.filePath("model/DeepFilterNet3_onnx.tar.gz");

		// 转换为const char*类型
		QByteArray modelPathBytes = modelPath.toLocal8Bit();
		const char* model_path = modelPathBytes.constData();
		this->df_state = df_create(model_path, 100.);

		// 初始化音频输出线程
		outputThread = new AudioOutputThread(&mutex, &condition);
		outputThread->start();
	}

	~AudioDevice_() {
		outputThread->quit();
		outputThread->wait();
		delete outputThread;
	}

	bool open(OpenMode mode) override {
		return QIODevice::open(mode | QIODevice::ReadWrite); // 使用ReadWrite模式打开
	}

	qint64 readData(char* data, qint64 maxlen) override {
		// 不需要实现，因为我们只使用writeData
		Q_UNUSED(data);
		Q_UNUSED(maxlen);
		return 0;
	}

	qint64 writeData(const char* data, qint64 len) override {
		// 将接收到的数据存入缓冲区
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
				qint64 first_time = QDateTime::currentDateTime().toMSecsSinceEpoch();
				if (blnDF) {
					float snr = df_process_frame_i16(df_state, input, output);
					qint64 lastTime = QDateTime::currentDateTime().toMSecsSinceEpoch();

					// 处理后的数据可以在这里输出或传递到其他地方
					qDebug() << "Processed frame with Time " << lastTime - first_time << " ms, SNR:" << snr;
				}
				else {
					// 使用 memcpy 进行内存拷贝
					memcpy(output, input, 480 * sizeof(short));
				}
				

				// 将处理后的数据放入输出缓冲区
				if (this->blnReturn)
					outputThread->enqueueData(output, 480);
			}
		}
		return len;  // 返回写入的数据长度
	}
	/// <summary>
	/// 设置耳返
	/// </summary>
	/// <param name="blnReturn"></param>
	void setReturn(bool blnReturn) {
		this->blnReturn = blnReturn;
	}
	void setDF(bool blnDF) {
		this->blnDF = blnDF;
	}
private:
	DFState* df_state;
	QQueue<short> buffer;
	AudioOutputThread* outputThread;
	QMutex mutex;
	QWaitCondition condition;
	bool blnReturn;
	bool blnDF;
};

#endif // AUDIODEVICE_H
