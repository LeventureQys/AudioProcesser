# 前言

最近在做一些有关DeepFilter的开发，写了份代码，这里简单说说代码，怎么使用Qt + DeepFilter进行实时的AI音频降噪，并获得耳返

# 环境
Windows11 + VisualStudio + Qt 5.14.2

# 流程

我们大致需要两个组件，一个是AudioDevice_，用于充当QIODevice来从QAudioInpu中源源不断获取当前默认写入设备的数据流，一个是AudioOutputThread，则在不影响AudioDevice的情况下将读取到的音频数据以耳返的方式播放出来。

代码如下：

Audio.h

```cpp
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

```


调用示例：mainwindow.cpp

```cpp
#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // 获取默认音频输入设备
    QAudioDeviceInfo info = QAudioDeviceInfo::defaultInputDevice();

    // 设置音频格式
    QAudioFormat format;
    format.setSampleRate(48000);
    format.setChannelCount(1);  // 使用单声道，方便处理
    format.setSampleSize(16);
    format.setCodec("audio/pcm");
    format.setByteOrder(QAudioFormat::LittleEndian);
    format.setSampleType(QAudioFormat::SignedInt);

    // 检查设备是否支持所设置的格式
    if (!info.isFormatSupported(format)) {
        qWarning() << "Default format not supported, trying to use the nearest.";
        format = info.nearestFormat(format);
    }

    // 创建音频输入对象
    audioInput = new QAudioInput(info, format, this);

    // 创建自定义的QIODevice来处理音频数据
    audioDevice = new AudioDevice_(this);
    audioDevice->open(QIODevice::ReadWrite);

    // 初始化 DeepFilterNet
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_2_clicked()
{
    // 停止录音
    audioInput->stop();
}

void MainWindow::on_cbx_return_clicked(bool blnchecked)
{
    audioDevice->setReturn(blnchecked);
}

void MainWindow::on_cbx_df_clicked(bool blnchecked)
{
    audioDevice->setDF(blnchecked);
}

void MainWindow::on_pushButton_clicked()
{
    // 开始录音
    audioInput->start(audioDevice);
}

```
