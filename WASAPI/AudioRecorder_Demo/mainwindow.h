#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "AudioRecorder_Core/audiorecorder_core.h"
#include "QMessageBox.h"
#include "qfiledialog.h"
#include "QStandardPaths"
QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
public slots:

    void on_btn_init_clicked();
    void on_btn_set_format_clicked();
    void on_btn_set_device_clicked();
    void on_btn_start_record_clicked();
    void on_btn_stop_record_clicked();
    void on_btn_play_data_clicked();
    void on_btn_stop_data_clicked();
    void on_btn_save_file_clicked();
    void on_btn_set_device_name_clicked();
private:
    Ui::MainWindow *ui;
    AR::AudioRecorder* recorder = nullptr;
};
#endif // MAINWINDOW_H
