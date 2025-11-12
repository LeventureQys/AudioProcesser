#include "AudioCondition.h"

AudioCondition::AudioCondition(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::AudioConditionClass())
{
    ui->setupUi(this);
    // Get the default audio input device
    QAudioDeviceInfo info = QAudioDeviceInfo::defaultInputDevice();

    // Set the audio format
    QAudioFormat format;
    format.setSampleRate(48000);
    format.setChannelCount(1);  // Use mono for easier processing
    format.setSampleSize(16);
    format.setCodec("audio/pcm");
    format.setByteOrder(QAudioFormat::LittleEndian);
    format.setSampleType(QAudioFormat::SignedInt);

    // Check if the device supports the set format
    if (!info.isFormatSupported(format)) {
        qWarning() << "Default format not supported, trying to use the nearest.";
        format = info.nearestFormat(format);
    }

    // Create audio input object
    audioInput = new QAudioInput(info, format, this);

    // Create a custom QIODevice to handle audio data
    audioDevice = new AudioDevice_(this);
    audioDevice->open(QIODevice::ReadWrite);

    // Initialize DeepFilterNet
}

AudioCondition::~AudioCondition()
{
    delete ui;
}

void AudioCondition::on_btn_stop_clicked()
{
    // Stop recording
    audioInput->stop();
}

void AudioCondition::on_btn_start_clicked()
{
    // Start recording
    audioInput->start(audioDevice);
}

void AudioCondition::on_cbx_algorithm_clicked(bool blnchecked)
{
    audioDevice->setDF(blnchecked);
}

void AudioCondition::on_cbx_monitoring_clicked(bool blnchecked)
{
    audioDevice->setReturn(blnchecked);
}
