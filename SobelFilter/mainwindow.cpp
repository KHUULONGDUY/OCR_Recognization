#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<QImage>
#include<time.h>
#include <chrono>
#define N 1048576*100
#define Threadperblock 64

using namespace std;
using namespace cv;
using namespace cv::cuda;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    image = cv::imread("/home/thinkalpha/data/Image/ThinkAlphaCam__21968493__20240206_102845605_0001.bmp");
    //cv::imshow("Image",image);
    //cv::waitKey(0);
    //apply gpu calculation library
/*
    auto start = std::chrono::high_resolution_clock::now();
    gpuImage.upload(image);

    // Apply Sobel filter (using appropriate data type for output)
    GpuMat gx, gy;
    Sobel(gpuImage, gx, CV_32F, 1, 0, 3, 1, 0, BORDER_REPLICATE);
    Sobel(gpuImage, gy, CV_32F, 0, 1, 3, 1, 0, BORDER_REPLICATE);
    Mat result;
    convertScaleAbs(gx, result);
    Mat result_g, result_y;
    gx.copyTo(result_g);
    gy.copyTo(result_y);
    addWeighted(result_g, 0.5, result_y, 0.5, 0, result);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the result and elapsed time
    cout << "Calculation result: " <<endl;
    cout << "Elapsed time: " << elapsed.count() << " milliseconds" << endl;
    ui->label->setText("Cuda running time: " + QString::number(elapsed.count()) + " milliseconds");
    //QPixmap pix
    QImage qimage((uchar*)result.data, result.cols, result.rows, result.step, QImage::Format_RGB888);
    ui->label_2->setPixmap(QPixmap::fromImage(qimage));
*/
    // Apply normal Opencv
    auto start = std::chrono::high_resolution_clock::now();
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;
    int ddepth = CV_16S;
    int ksize=3;
    Sobel(image, grad_x, ddepth, 1, 0, ksize, 1, 0, cv::BORDER_DEFAULT);
    Sobel(image, grad_y, ddepth, 0, 1, ksize, 1, 0, cv::BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the result and elapsed time
    cout << "Calculation result: " <<endl;
    cout << "Elapsed time: " << elapsed.count() << " milliseconds" << endl;
    ui->label->setText("Running time: " + QString::number(elapsed.count()) + " milliseconds");

    QImage qimage((uchar*)grad.data, grad.cols, grad.rows, grad.step,QImage::Format_RGB888);
    ui->label_2->setPixmap(QPixmap::fromImage(qimage));

    //test call cuda
/*
    //call cuda code
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(N*sizeof(int));
    h_b = (int*)malloc(N*sizeof(int));
    h_c = (int*)malloc(N*sizeof(int));
    //random value for host memory
    for(int i=0; i<N; i++){
        h_a[i] = rand()%100;
        h_b[i] = rand()%100;
    }
    cudapro(h_a,h_b,h_c);
    //cout<<h_c[1]<<endl;
    ui->label_3->setText(QString::number(h_a[10]));
    ui->label_4->setText(QString::number(h_b[10]));
    ui->label_2->setText(QString::number(h_c[10]));
    */
}
vector<vector<int>> imageVector(cv::Mat image)
{
    vector<vector<int>> imgVector;
    for(int i=0;i<image.rows;i++)
    {
        imgVector.push_back(vector<int>());
        for(int j=0;j<image.cols;j++)
        {
            imgVector[i].push_back(image.at<uchar>(i,j));
        }
    }

    return imgVector;
}
//print vector
/*
    for(int i=0; i<200;i++)
    {
        for(int j=0; j<imgVector[i].size();j++){
            cout<<"Value: "<<imgVector[i][j]<<" ; "<<(int)image.at<uchar>(i, j)<<endl;
        }
    }
*/
//convert 2D vector into an image
/*
 * // Check dimensions and data type (adjust as needed)
    int rows = imageIntensityVector.size();
    int cols = imageIntensityVector[0].size();

    // Create cv::Mat (assuming grayscale image)
    cv::Mat image(rows, cols, CV_8UC1); // 8UC1 for single-channel (grayscale) image

    // Convert 2D vector to cv::Mat
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Clamp intensity values to valid range (0-255) for uchar
            image.at<uchar>(y, x) = static_cast<uchar>(std::max(0.0f, std::min(255.0f, imageIntensityVector[y][x])));
        }
    }

    // Display image using imshow
    cv::imshow("Image from 2D Vector", image);
    cv::waitKey(0);
 */
void MainWindow::on_sobel_bt_clicked()
{
    image = imread("/home/thinkalpha/data/Image/ThinkAlphaCam__21968493__20240206_102845605_0001.bmp");
    //cv::imshow("Image",image);
    //cv::waitKey(0);
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int block_dimension = 16;
    unsigned int gridDimX = 1280 / block_dimension;
    unsigned int gridDimY = 960 / block_dimension;
    dim3 grid_Dim(gridDimX, gridDimY);
    dim3 block_Dim(block_dimension, block_dimension);
    //***Apply sobel filter by cuda programming***
    int padding =1;
    //convert Mat data type to pointer
    int row = image.rows;
    int col = image.cols;
    int *ori_img = (int*)malloc(row*col*sizeof(int));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
          // Assuming grayscale image, access pixel using at<uchar>
          int index = y*col+x;
          ori_img[index] = image.at<uchar>(y, x);
        }
      }
    cout<<"Before cuda: "<<endl;
    for(int i=0; i<10;i++){
        cout<<"Value: "<<ori_img[i]<<endl;
    }
    cout<<"After cuda: "<<endl;
    //call cuda program
    sobelFilter(ori_img);
    for(int i=0; i<10;i++){
        cout<<"Value: "<<ori_img[i]<<endl;
    }


    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the result and elapsed time
    cout << "Calculation result: " <<endl;
    cout << "Elapsed time: " << elapsed.count() << " milliseconds" << endl;
    ui->label->setText("Cuda running time: " + QString::number(elapsed.count()) + " milliseconds");
    int width = 1280;
    int height = 960;
    QImage newimage(width, height, QImage::Format_Grayscale8);
    uchar* imageData = newimage.bits();
    for (int i = 0; i < width*height; ++i) {
      imageData[i] = ori_img[i];
    }
    //QPixmap pix
    ui->label_2->setPixmap(QPixmap::fromImage(newimage));
    ui->label_2->setScaledContents(true);
    //cout<<"CUDA status: "<< getCudaEnabledDeviceCount()<<endl;

}
