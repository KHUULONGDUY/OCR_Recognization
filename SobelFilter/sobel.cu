//#include"mainwindow.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include<time.h>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>
#define WIDTH_IMAGE 1282
#define HEIGHT_IMAGE 962
//define for cuda programming
#define N 1048576*100
#define Threadperblock 32



__global__ void sobelKernel(int* padding_img, int* sobelX, int* sobelY)
{

    int col_g = blockIdx.x * blockDim.x + threadIdx.x;
    int row_g = blockIdx.y * blockDim.y + threadIdx.y;
    int index_sobel = row_g * (WIDTH_IMAGE - 2) + col_g;
    int index_padding = row_g * WIDTH_IMAGE + col_g;
    sobelX[index_sobel] = padding_img[index_padding + 2] + 2*padding_img[index_padding + WIDTH_IMAGE + 2] + padding_img[index_padding + 2 * WIDTH_IMAGE + 2] - (padding_img[index_padding] + 2 * padding_img[index_padding + WIDTH_IMAGE] + padding_img[index_padding + 2 * WIDTH_IMAGE]);
    sobelY[index_sobel] = padding_img[index_padding + 2*WIDTH_IMAGE] + 2* padding_img[index_padding + 2 * WIDTH_IMAGE + 1] + padding_img[index_padding + 2 * WIDTH_IMAGE + 2] - (padding_img[index_padding] + 2*padding_img[index_padding + 1] + padding_img[index_padding + 2]);
}

__global__ void add_sobel_kernel(int *a, int *b, int *c)
{
    int scale = 2;
    int col_g = blockIdx.x * blockDim.x + threadIdx.x;
    int row_g = blockIdx.y * blockDim.y + threadIdx.y;
    int index_sobel = row_g * (WIDTH_IMAGE - 2) + col_g;
    //c[index_sobel] = abs(a[index_sobel])+abs(b[index_sobel]);
    c[index_sobel] = scale*sqrtf(a[index_sobel] * a[index_sobel] + b[index_sobel] * b[index_sobel]);
}

__global__ void add_matrix(int *a, int *b, int *c){
    int tid=threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<N)
      c[tid] = a[tid] + b[tid];
}
//apply sobel filter
extern void sobelFilter(int* img_pt){
    //host memory
    int *padding_img, *result;
    padding_img = (int*)malloc(WIDTH_IMAGE * HEIGHT_IMAGE * sizeof(int));
    result = (int*)malloc((WIDTH_IMAGE - 2) * (HEIGHT_IMAGE - 2) * sizeof(int));
    //device memory
    int *dev_pad_img,*dev_sobelX, *dev_sobelY, *dev_result;
    cudaMalloc((void**)&dev_pad_img, WIDTH_IMAGE * HEIGHT_IMAGE * sizeof(int));
    cudaMalloc((void**)&dev_sobelX, (WIDTH_IMAGE - 2) * (HEIGHT_IMAGE - 2) * sizeof(int));
    cudaMalloc((void**)&dev_sobelY, (WIDTH_IMAGE - 2) * (HEIGHT_IMAGE - 2) * sizeof(int));
    cudaMalloc((void**)&dev_result, (WIDTH_IMAGE - 2) * (HEIGHT_IMAGE - 2) * sizeof(int));
    //devide blocks and threads
    unsigned int gridDimX = 1280/Threadperblock;
    unsigned int gridDimY = 960/Threadperblock;
    dim3 grid_Dim(gridDimX,gridDimY);
    dim3 block_Dim(Threadperblock,Threadperblock);
    // For padding

    for(int i=-1; i<((WIDTH_IMAGE-2)*( HEIGHT_IMAGE-2));i++)
    {
        if(img_pt[i] != NULL){
            padding_img[i]=img_pt[i];
        }else
        {
            padding_img[i]=0;
        }
    }

    //copy value from host to device
    cudaMemcpy(dev_pad_img, padding_img, WIDTH_IMAGE * HEIGHT_IMAGE  * sizeof(int), cudaMemcpyHostToDevice);
    //call cuda kernel
    sobelKernel << <grid_Dim, block_Dim >> > (dev_pad_img, dev_sobelX, dev_sobelY);
    add_sobel_kernel << <grid_Dim, block_Dim >> > (dev_sobelX, dev_sobelY,dev_result);
    cudaMemcpy(img_pt, dev_result, 1280*960 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_pad_img);
    cudaFree(dev_sobelX);
    cudaFree(dev_sobelY);
    cudaFree(dev_result);

}

extern void cudapro(int *h_a,int *h_b, int *h_c){
    //device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N*sizeof(int));
    cudaMalloc((void **)&d_b, N*sizeof(int));
    cudaMalloc((void **)&d_c, N*sizeof(int));

    //copy value from host to device
    cudaMemcpy(d_a,h_a,N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*sizeof(int), cudaMemcpyHostToDevice);
    //call kernel
    add_matrix<<<N/Threadperblock,Threadperblock>>>(d_a,d_b,d_c);
    //return value to host memory
    cudaMemcpy(h_c,d_c,N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory (add these lines)
}
