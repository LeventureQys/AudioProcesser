#include "AudioCondition.h"
#include <QMediaDevices>
#include <QAudioDevice>

AudioCondition::AudioCondition(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::AudioConditionClass())
    , audioSource(nullptr)
    , audioDevice(nullptr)
{
    ui->setupUi(this);

    // 获取默认音频输入设备
    QAudioDevice inputDeviceInfo = QMediaDevices::defaultAudioInput();

    // 设置音频格式
    QAudioFormat format;
    format.setSampleRate(48000);
    format.setChannelConfig(QAudioFormat::ChannelConfigMono);  // 使用单声道以便于处理
    format.setSampleFormat(QAudioFormat::Int16);  // 16位有符号整数

    // 检查设备是否支持设置的格式
    if (!inputDeviceInfo.isFormatSupported(format)) {
        qWarning() << "默认格式不支持，尝试使用默认格式。";
        // Qt6 没有 nearestFormat，系统会尽力适配
    }

    // 创建音频源对象（Qt6: QAudioInput -> QAudioSource）
    audioSource = new QAudioSource(inputDeviceInfo, format, this);

    // 创建自定义 QIODevice 来处理音频数据
    audioDevice = new AudioDevice_(this);
    audioDevice->open(QIODevice::ReadWrite);

    // 初始化 DeepFilterNet
}

AudioCondition::~AudioCondition()
{
    if (audioSource) {
        audioSource->stop();
    }
    delete ui;
}

void AudioCondition::on_btn_stop_clicked()
{
    // 停止录音
    if (audioSource) {
        audioSource->stop();
    }
}

void AudioCondition::on_btn_start_clicked()
{
    // 开始录音
    if (audioSource && audioDevice) {
        audioSource->start(audioDevice);
    }
}

void AudioCondition::on_cbx_algorithm_clicked(bool blnchecked)
{
    if (audioDevice) {
        audioDevice->setDF(blnchecked);
    }
}

void AudioCondition::on_cbx_monitoring_clicked(bool blnchecked)
{
    if (audioDevice) {
        audioDevice->setReturn(blnchecked);
    }
}