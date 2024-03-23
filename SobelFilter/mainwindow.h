#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include<opencv2/opencv.hpp>
#include <QMainWindow>
#include<QString>
#include<vector>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

extern QString sayHello();
extern int getnum();
extern void cudapro(int *h_a,int *h_b,int *h_c);
extern void sobelFilter(int* img_pt);
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_sobel_bt_clicked();

private:
    Ui::MainWindow *ui;
private:
    cv::Mat image;
    cv::cuda::GpuMat gpuImage;
};

#endif // MAINWINDOW_H
