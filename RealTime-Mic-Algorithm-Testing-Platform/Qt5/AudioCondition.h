#pragma once

#include <QMainWindow>
#include "ui_AudioCondition.h"
#include "Audio.h"
QT_BEGIN_NAMESPACE
namespace Ui { class AudioConditionClass; };
QT_END_NAMESPACE

class AudioCondition : public QMainWindow
{
	Q_OBJECT

public:
	AudioCondition(QWidget *parent = nullptr);
	~AudioCondition();
private slots:
	
	void on_btn_stop_clicked();
	void on_btn_start_clicked();
	void on_cbx_algorithm_clicked(bool blnchecked);
	void on_cbx_monitoring_clicked(bool blnchecked);
private:
	Ui::AudioConditionClass *ui;
	QAudioInput* audioInput;
	AudioDevice_* audioDevice;
};
