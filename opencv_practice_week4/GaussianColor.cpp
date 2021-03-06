//
//  GaussianColor.cpp
//  opencv_practice_week4
//
//  Created by 류예나 on 2018. 3. 26..
//  Copyright © 2018년 류예나. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE    CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {
    
    Mat input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week4/opencv_practice_week4/opencv_practice_week4/lena.jpg", CV_LOAD_IMAGE_COLOR);

    Mat output;
    
    if (!input.data)
    {
        std::cout << "Could not open" << std::endl;
        return -1;
    }
    
    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input);
    output = gaussianfilter(input, 1,1,1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel
    
    namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);
    imshow("Gaussian Filter", output);
    
    
    waitKey(0);
    
    return 0;
}


Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {
    
    Mat kernel;
    
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempa;
    int tempb;
    float denom;
    float kernelvalue;
    
    // Initialiazing Kernel Matrix
    kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
    
    denom = 0.0;
    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
            kernel.at<float>(a+n, b+n) = value1;
            denom += value1;
        }
    }
    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            kernel.at<float>(a+n, b+n) /= denom;
        }
    }
    
    Mat output = Mat::zeros(row, col, input.type());
    
    //input 크기만큼 반복
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            
            if (!strcmp(opt, "zero-paddle")) {
                float sumR = 0.0;
                float sumG = 0.0;
                float sumB = 0.0;
                //n = mask size
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        
                        /* Gaussian filter with Zero-paddle boundary process:
                         
                         Fill the code:
                         */
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)){
                            sumR += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[2]);
                            sumG += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[1]);
                            sumB += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[0]);
                        }
                    }
                }
                output.at<Vec3b>(i,j)[0]=(G)sumB;
                output.at<Vec3b>(i,j)[1]=(G)sumG;
                output.at<Vec3b>(i,j)[2]=(G)sumR;
            }
            
            else if (!strcmp(opt, "mirroring")) {
                float sumR = 0.0;
                float sumG = 0.0;
                float sumB = 0.0;
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        
                        /* Gaussian filter with Zero-paddle boundary process:
                         
                         Fill the code:
                         */
                        if (i + a > row - 1) {  //mirroring for the border pixels
                            tempa = i - a;
                        }
                        else if (i + a < 0) {
                            tempa = -(i + a);
                        }
                        else {
                            tempa = i + a;
                        }
                        if (j + b > col - 1) {
                            tempb = j - b;
                        }
                        else if (j + b < 0) {
                            tempb = -(j + b);
                        }
                        else {
                            tempb = j + b;
                        }
                        sumR += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[2]);
                        sumG += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[1]);
                        sumB += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[0]);
                    }
                }
                output.at<Vec3b>(i,j)[0]=(G)sumB;
                output.at<Vec3b>(i,j)[1]=(G)sumG;
                output.at<Vec3b>(i,j)[2]=(G)sumR;
            }
            
            
            else if (!strcmp(opt, "adjustkernel")) {
                float sumR = 0.0;
                float sumG = 0.0;
                float sumB = 0.0;
                float sumR2 = 0.0;
                float sumG2 = 0.0;
                float sumB2 = 0.0;
                for (int a = -n; a <= n; a++) { // for each kernel window
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            sumR += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[2]);
                            sumG += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[1]);
                            sumB += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[0]);
                            sumR2 += kernel.at<float>(a+n,b+n);
                            sumG2 += kernel.at<float>(a+n,b+n);
                            sumB2 += kernel.at<float>(a+n,b+n);
                        }
                    }
                }
                output.at<Vec3b>(i,j)[0]=(G)(sumB/sumB2);
                output.at<Vec3b>(i,j)[1]=(G)(sumG/sumG2);
                output.at<Vec3b>(i,j)[2]=(G)(sumR/sumR2);
            }
        }
    }
    return output;
}

