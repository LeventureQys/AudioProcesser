#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->recorder = new AR::AudioRecorder(this);
    this->thread = new QThread(this);
    this->thread->start();
    this->recorder->moveToThread(this->thread);
    connect(this->recorder, &AR::AudioRecorder::Sig_Volumechanged, this, [=](double volume) {
        QString string;
        string.append(QString::number(volume) + '\n');
        this->ui->plainTextEdit->appendPlainText(string);
        });
    connect(this->recorder, &AR::AudioRecorder::Sig_VolumePlayed, this, [=](double volume) {
        QString string;
        string.append(QString::number(volume) + '\n');
        this->ui->plainTextEdit_2->appendPlainText(string);
        });

    connect(this->recorder, &AR::AudioRecorder::Sig_PlaybackFinished, this, [=]() {
        QString string;
        string.append("播放完毕了！！！！" + '\n');
        this->ui->plainTextEdit_2->appendPlainText(string);
        });
}

MainWindow::~MainWindow()
{
    
    if (thread) {
        thread->quit();  // 请求线程退出事件循环
        thread->wait();  // 等待线程完全结束
    }


    this->recorder->deleteLater();
    delete ui;
}

void MainWindow::on_btn_init_clicked()
{
    this->recorder->IInitialize("AI");
}

void MainWindow::on_btn_set_format_clicked()
{

}

void MainWindow::on_btn_set_device_clicked()
{

}

void MainWindow::on_btn_start_record_clicked()
{
    this->recorder->IStartRecord();
}

void MainWindow::on_btn_stop_record_clicked()
{
    this->recorder->IStopRecord();
}

void MainWindow::on_btn_play_data_clicked()
{
    this->recorder->IPlayRecordedData();
}

void MainWindow::on_btn_stop_data_clicked()
{
    this->recorder->IStopPlayBack();
}

void MainWindow::on_btn_save_file_clicked()
{
    // 检查录音器是否有效
    if (!this->recorder) {
        QMessageBox::warning(this, tr("Error"), tr("Recorder is not initialized!"));
        return;
    }

    // 弹出文件保存对话框
    QString filePath = QFileDialog::getSaveFileName(
        this,                                   // 父窗口
        tr("Save Recording"),                   // 对话框标题
        QStandardPaths::writableLocation(QStandardPaths::MusicLocation), // 默认音乐目录
        tr("WAV Files (*.wav)")                // 文件过滤器
    );

    // 检查用户是否选择了文件路径
    if (filePath.isEmpty()) {
        return;  // 用户取消了操作
    }

    // 确保文件以.wav结尾（Qt不会自动添加扩展名）
    if (!filePath.endsWith(".wav", Qt::CaseInsensitive)) {
        filePath += ".wav";
    }

    // 调用录音器的保存接口
    bool success = true;
        this->recorder->ISaveRecordAsWavFile(filePath);

    // 显示操作结果
    if (success) {
        QMessageBox::information(this, tr("Success"), tr("Recording saved successfully!"));
    }
    else {
        QMessageBox::warning(this, tr("Error"), tr("Failed to save recording!"));
    }
}

void MainWindow::on_btn_set_device_name_clicked()
{
    QString str_ret = this->ui->lineEdit->text();
    if (str_ret.isEmpty()) {
        this->ui->plainTextEdit->appendPlainText("设备名称为空，设置失败");
        return;
    }

    this->recorder->ISetTargetDeviceName(str_ret);

}
