import cv2
import numpy as np 
from numba import cuda
import timeit
from timeit import default_timer as timer

#print(cv2.cuda.getCudaEnabledDeviceCount())
def imgprocessing2():
    s = timer()
    img = 255-img_src
    #Find contour area
    img2 = np.uint8(np.power((img/255),10)*255)

    img2 = cv2.GaussianBlur(img2,(3,3),3)
    edges = cv2.Canny(img2, threshold1=50, threshold2=60)
    #cv2_imshow(img2)
    #cv2_imshow(edges)
    #Apply dilation
    kernel = np.ones((3,3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel)
    e = timer()
    print("Execution time :  %s miliseconds" %((e-s)*1000))
    cv2.imshow("img",src)
    cv2.waitKey(0)
def sobelCuda():
    img_src = cv2.imread('/home/thinkalpha/data/Image/ThinkAlphaCam__21968493__20240206_102845605_0001.bmp',cv2.COLOR_BGR2GRAY)
    #apply filter
            
    # Convert image to appropriate format for CUDA
    image_gpu = cv2.cuda.GpuMat()
    image_gpu.upload(img_src)

    # Define Sobel filter kernels
    sobel_x = cv2.cuda.createSobelFilter(cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.cuda.createSobelFilter(cv2.CV_32F, 0, 1, ksize=3)

    # Allocate GPU memory for intermediate and output results
    d_gx = cv2.cuda.GpuMat(image.shape[0], image.shape[1], cv2.CV_32F)
    d_gy = cv2.cuda.GpuMat(image.shape[0], image.shape[1], cv2.CV_32F)
    d_mag = cv2.cuda.GpuMat(image.shape[0], image.shape[1], cv2.CV_32F)
    d_atan = cv2.cuda.GpuMat(image.shape[0], image.shape[1], cv2.CV_32F)

    # Apply Sobel filters in horizontal and vertical directions
    sobel_x.apply(image_gpu, d_gx)
    sobel_y.apply(image_gpu, d_gy)

    # Calculate magnitude (using optimized approach)
    cv2.cuda.absdiff(d_gx, cv2.cuda.Scalar(0), d_gx)  # Avoid negative square root
    cv2.cuda.absdiff(d_gy, cv2.cuda.Scalar(0), d_gy)
    cv2.cuda.pow(d_gx, 2.0, d_gx)
    cv2.cuda.pow(d_gy, 2.0, d_gy)
    cv2.cuda.addWeighted(d_gx, 0.5, d_gy, 0.5, 0.0, d_mag)
    cv2.cuda.sqrt(d_mag, d_mag)

    # Calculate gradient direction (atan2 for better handling)
    cv2.cuda.cartToPolar(d_gx, d_gy, d_mag, d_atan, angleInDegrees=False)

    # Download results back to CPU for further processing (if needed)
    mag = d_mag.download()
    atan = d_atan.download()
    cv2.imshow("Sobel Filtered Image (Magnitude)", mag)
    cv2.waitKey(0)
#Load images
def gaussianCuda():
    #load img
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA is not available. Using CPU instead.")

    img_src = cv2.imread('/home/thinkalpha/data/Image/ThinkAlphaCam__21968493__20240206_102845605_0001.bmp',cv2.COLOR_BGR2GRAY)
    #apply filter
    blurred_image = cv2.GaussianBlur(img_src, (5,5), 0.5)
    cv2.imshow('Filtered Image', blurred_image)
    cv2.waitKey(0)
    # Create a CUDA-enabled image
    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(img_src)
    # Create a Gaussian filter
    s = timer()
    gaussian_filter = cv2.cuda.createGaussianFilter(img_gpu.type(), img_gpu.type(), (5, 5) , 0.5)    
    filtered_img = gaussian_filter.apply(img_gpu)
    
    # Download the filtered image from GPU to CPU
    filtered_img_cpu = filtered_img.download()
    e = timer()
    print("Execution time :  %s miliseconds" %((e-s)*1000))
    # Display the original and filtered images
    #cv2.imshow('Original Image', img_src)
    cv2.imshow('Filtered Image with cuda', filtered_img_cpu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Hoandoi(X,Y):
    tem=0
    tem=X
    X=Y
    Y=tem
    return X,Y
def SortArr(X,Y,w,h):
    for i in range(len(X)-1):
        for j in range(i+1,len(X)):
            if X[j]<X[i]:
                X[i],X[j]=Hoandoi(X[i],X[j])
                Y[i],Y[j]=Hoandoi(Y[i],Y[j])
                w[i],w[j]=Hoandoi(w[i],w[j])
                h[i],h[j]=Hoandoi(h[i],h[j])
def mainfunction():
    #load gray img 
    img0 = cv2.imread('/home/thinkalpha/data/Image/ThinkAlphaCam__21968493__20240206_103932423_0010.bmp',cv2.COLOR_BGR2GRAY)
    s = timer()
    img1 = 255-img0    
    #Find contour area
    img2 = np.uint8(np.power((img1/255),10)*255)
    img3 = cv2.GaussianBlur(img2,(3,3),3)
    edges = cv2.Canny(img3, threshold1=50, threshold2=60)
    
    #Apply dilation
    kernel = np.ones((3,3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel)

    #cv2.imshow("Edge",edges)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    #crop image by box:
    cropped_img = img2[box[1][1]:box[0][1],box[1][0]:box[2][0]]

    #Convert to binary --> Chuyển sang "sauvola" thresholding
    roi = np.uint8((cropped_img-np.min(cropped_img))/(np.max(cropped_img)-np.min(cropped_img))*255)
    ret,roi_bin = cv2.threshold(roi,190,255,cv2.THRESH_BINARY)
    
    kernel = np.uint8(np.ones([3,3]))
    
    roi_bin = cv2.dilate(roi_bin, kernel,iterations=2)
    
    roi_bin = cv2.erode(roi_bin,kernel,iterations=3)
    
    roi_bin = cv2.dilate(roi_bin, kernel, iterations=7)
    roi_bin = cv2.erode(roi_bin,kernel,iterations=2)
    roi_bin = cv2.dilate(roi_bin, kernel, iterations=3)
    #cv2.imshow("Result",roi_bin)
    #cv2.waitKey(0)

    #detect contours around characters
    contours, hierarchy = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    X = []
    Y = []
    W = []
    H = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (w>60) and (w<110):
            X.append(x)
            Y.append(y)
            W.append(w)
            H.append(h)
    SortArr(X,Y,W,H)
    #if(len(X)==len(name)):

    cropped_img = 255 - cropped_img
    #cv2.imshow("cropped image",cropped_img)
    characters = []
    for i in range(len(X)):
        #x,y,w,h = cv2.boundingRect(c)
        #cropped_img=cv2.rectangle(cropped_img,(x,y),(x+w,y+h),255,2)
        img=cropped_img[Y[i]:(Y[i]+H[i]),X[i]:(X[i]+W[i])]
        characters.append(img)
    e = timer()
    print("Execution time :  %s miliseconds" %((e-s)*1000))
    cv2.imshow("cropped image",cropped_img)
    for i in range(len(characters)):
        cv2.imshow("result",characters[i])
    cv2.waitKey(0)
def training():
    pass


     
mainfunction()