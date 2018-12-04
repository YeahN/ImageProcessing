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
    
    Mat kernel_s;
    Mat kernel_t;
    
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempa;
    int tempb;
    float denom_s;
    float denom_t;
    float kernelvalue;
    
    // Initialiazing Kernel Matrix
    kernel_s = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
    kernel_t = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
    
    denom_s = 0.0;
    denom_t = 0.0;

    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
            float value2 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
            
            kernel_s.at<float>(a+n, b+n) = value1;
            kernel_t.at<float>(a+n, b+n) = value2;
            
            denom_s += value1;
            denom_t += value2;
        }
    }
    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            kernel_s.at<float>(a+n, b+n) /= denom_s;
            kernel_t.at<float>(a+n, b+n) /= denom_t;
        }
    }
    
    Mat output = Mat::zeros(row, col, input.type());
    
    //input 크기만큼 반복
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            
            if (!strcmp(opt, "zero-paddle")) {
                float sumR_x = 0.0;
                float sumR_y = 0.0;
                float sumG_x = 0.0;
                float sumG_y = 0.0;
                float sumB_x = 0.0;
                float sumB_y = 0.0;
                //n = mask size
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        
                        /* Gaussian filter with Zero-paddle boundary process:
                         
                         Fill the code:
                         */
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)){
//                            sumR += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[2]);
//                            sumG += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[1]);
//                            sumB += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[0]);
//                            sum1 += kernel_s.at<float>(a+n, b+n)* (float)(input.at<G>(i+a,j+b));
//                            sum2 += kernel_t.at<float>(a+n, b+n);
                            sumR_x += kernel_s.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[2]);
                            sumR_y += kernel_t.at<float>(a+n,b+n);
                            sumG_x += kernel_s.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[1]);
                            sumG_y += kernel_t.at<float>(a+n,b+n);
                            sumB_x += kernel_s.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[0]);
                            sumB_y += kernel_t.at<float>(a+n,b+n);
                        }
                    }
                }
                output.at<Vec3b>(i,j)[0]=(G)sumB_x*sumB_y;
                output.at<Vec3b>(i,j)[1]=(G)sumG_x*sumG_y;
                output.at<Vec3b>(i,j)[2]=(G)sumR_x*sumB_y;
            }
            
            else if (!strcmp(opt, "mirroring")) {
                float sumR_x = 0.0;
                float sumR_y = 0.0;
                float sumG_x = 0.0;
                float sumG_y = 0.0;
                float sumB_x = 0.0;
                float sumB_y = 0.0;
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
//                        sumR += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[2]);
//                        sumG += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[1]);
//                        sumB += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[0]);
                        sumR_x += kernel_s.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[2]);
                        sumR_y += kernel_t.at<float>(a+n,b+n);
                        sumG_x += kernel_s.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[1]);
                        sumG_y += kernel_t.at<float>(a+n,b+n);
                        sumB_x += kernel_s.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(tempa,tempb)[0]);
                        sumB_y += kernel_t.at<float>(a+n,b+n);
                    }
                }
                output.at<Vec3b>(i,j)[0]=(G)sumB_x*sumB_y;
                output.at<Vec3b>(i,j)[1]=(G)sumG_x*sumG_y;
                output.at<Vec3b>(i,j)[2]=(G)sumR_x*sumR_y;
            }
            
            
            else if (!strcmp(opt, "adjustkernel")) {
                float sumR_s1 = 0.0;
                float sumR_s2 = 0.0;
                float sumR_t1 = 0.0;
                float sumR_t2 = 0.0;
                float sumG_s1 = 0.0;
                float sumG_s2 = 0.0;
                float sumG_t1 = 0.0;
                float sumG_t2 = 0.0;
                float sumB_s1 = 0.0;
                float sumB_s2 = 0.0;
                float sumB_t1 = 0.0;
                float sumB_t2 = 0.0;
                for (int a = -n; a <= n; a++) { // for each kernel window
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
//                            sumR += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[2]);
//                            sumG += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[1]);
//                            sumB += kernel.at<float>(a+n,b+n) * (float)(input.at<Vec3b>(i+a,j+b)[0]);
//                            sumR2 += kernel.at<float>(a+n,b+n);
//                            sumG2 += kernel.at<float>(a+n,b+n);
//                            sumB2 += kernel.at<float>(a+n,b+n);
                            sumR_s1 += kernel_s.at<float>(a+n, b+n) * (float) (input.at<Vec3b>(i+a,j+b)[2]);
                            sumR_s2 += kernel_s.at<float>(a+n,b+n);
                            sumR_t1 += kernel_t.at<float>(a+n, b+n);
                            sumR_t2 += kernel_t.at<float>(a+n,b+n);
                            sumG_s1 += kernel_s.at<float>(a+n, b+n) * (float) (input.at<Vec3b>(i+a,j+b)[1]);
                            sumG_s2 += kernel_s.at<float>(a+n,b+n);
                            sumG_t1 += kernel_t.at<float>(a+n, b+n);
                            sumG_t2 += kernel_t.at<float>(a+n,b+n);
                            sumB_s1 += kernel_s.at<float>(a+n, b+n) * (float) (input.at<Vec3b>(i+a,j+b)[0]);
                            sumB_s2 += kernel_s.at<float>(a+n,b+n);
                            sumB_t1 += kernel_t.at<float>(a+n, b+n);
                            sumB_t2 += kernel_t.at<float>(a+n,b+n);
                        }
                    }
                }
                output.at<Vec3b>(i,j)[0]=(G)(sumB_s1*sumB_t1)/(sumB_s2*sumB_t2);
                output.at<Vec3b>(i,j)[1]=(G)(sumG_s1*sumG_t1)/(sumG_s2*sumG_t2);
                output.at<Vec3b>(i,j)[2]=(G)(sumR_s1*sumR_t1)/(sumR_s2*sumR_t2);
            }
        }
    }
    return output;
}

