////
////  main.cpp
////  opencv_practice_week2
////
////  Created by 류예나 on 2018. 3. 12..
////  Copyright © 2018년 류예나. All rights reserved.
////
//
//#include <opencv2\opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//
//template <typename T>
//Mat myrotate(const Mat input, float angle, const char* opt);
//
//int main()
//{
//    Mat input, rotated;
//
//    // Read each image
//    input = imread("lena.jpg");
//
//    // Check for invalid input
//    if (!input.data) {
//        std::cout << "Could not open or find the image" << std::endl;
//        return -1;
//    }
//
//    // original imagea
//    namedWindow("image");
//    imshow("image", input);
//
//    rotated = myrotate<Vec3b>(input, 45, "bilinear");
//
//    // rotated image
//    namedWindow("rotated");
//    imshow("rotated", rotated);
//
//    waitKey(0);
//
//    return 0;
//}
//
//template <typename T>
//Mat myrotate(const Mat input, float angle, const char* opt) {
//    int row = input.rows;
//    int col = input.cols;
//
//    float radian = angle * CV_PI / 180;
//
//    //size of output image
//    float sq_row = ceil(row * sin(radian) + col * cos(radian));
//    float sq_col = ceil(col * sin(radian) + row * cos(radian));
//
//    Mat output = Mat::zeros(sq_row, sq_col, input.type());
//
//    for (int i = 0; i < sq_row; i++) {
//        for (int j = 0; j < sq_col; j++) {
//            //회전 후 크기가 변경된 이미지의 각각의 좌표를 다시 돌려서 float 좌표로 나타냄
//            float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
//            float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;
//
//            //각각의 interpolation을 사용해서 float를 int값으로 바꾸기
//            if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
//                if (!strcmp(opt, "nearest")) {
//
//                    output.at<Vec3b>(j, i)[0] = input.at<Vec3b>(round(x), round(y))[0];
//                    output.at<Vec3b>(j, i)[1] = input.at<Vec3b>(round(x), round(y))[1];
//                    output.at<Vec3b>(j, i)[2] = input.at<Vec3b>(round(x), round(y))[2];
//
//                }
//                else if (!strcmp(opt, "bilinear")) {
//                    int linear_x1 = floor(x);
//                    int linear_x2 = ceil(x);
//                    int linear_y1 = floor(y);
//                    int linear_y2 = ceil(y);
//
//                    int new_x;
//                    int new_y;
//
//                    if((linear_x2==linear_x1) && (linear_y1==linear_y2)){
//                        new_x = linear_x2;
//                        new_y = linear_y2;
//                    }
//                    else if(linear_x1 == linear_x2){
//                        new_x = linear_x1;
//                        new_y = (linear_y2+linear_y1)/2;
//                    }else if(linear_y1 == linear_y2){
//                        new_y = linear_y2;
//                        new_x = (linear_x2+linear_x1)/2;
//                    }else{
//                        new_x = ((linear_x2 - linear_x1) / (linear_y2 - linear_y1) * (y - linear_y1)) + linear_x1;
//                        new_y = ((linear_y2 - linear_y1) / (linear_x2 - linear_x1) * (x - linear_x1)) + linear_y1;
//                    }
//
//                    output.at<Vec3b>(j, i)[0] = input.at<Vec3b>(new_x, new_y)[0];
//                    output.at<Vec3b>(j, i)[1] = input.at<Vec3b>(new_x, new_y)[1];
//                    output.at<Vec3b>(j, i)[2] = input.at<Vec3b>(new_x, new_y)[2];
//
//                }
//            }
//        }
//    }
//
//    return output;
//}
//
//
//


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
    Mat input, rotated;
    
    // Read each image
    input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week2/opencv_practice_week2/opencv_practice_week2/lena.jpg");
    
    // Check for invalid input
    if (!input.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    // original image
    namedWindow("image");
    imshow("image", input);
    
    rotated = myrotate<Vec3b>(input, 45, "bilinear");
    
    // rotated image
    namedWindow("rotated");
    imshow("rotated", rotated);
    
    waitKey(0);
    
    return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
    int row = input.rows;
    int col = input.cols;
    
    float radian = angle * CV_PI / 180;
    
    float sq_row = ceil(row * sin(radian) + col * cos(radian));
    float sq_col = ceil(col * sin(radian) + row * cos(radian));
    
    Mat output = Mat::zeros(sq_row, sq_col, input.type());
    
    for (int i = 0; i < sq_row; i++) {
        for (int j = 0; j < sq_col; j++) {
            float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
            float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;
            
            if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
                if (!strcmp(opt, "nearest")) {
                    int newx = floor(x);
                    int newy = floor(y);
                    
                    if (x - newx < 0.5) x = newx;
                    else if (x - newx >= 0.5) x = ceil(x);
                    if (y - newy < 0.5) y = newy;
                    else if (y - newy >= 0.5) y = ceil(y);
                    
                    output.at<Vec3b>(i, j)[0] = input.at<Vec3b>(y, x)[0];
                    output.at<Vec3b>(i, j)[1] = input.at<Vec3b>(y, x)[1];
                    output.at<Vec3b>(i, j)[2] = input.at<Vec3b>(y, x)[2];
                    
                }
                else if (!strcmp(opt, "bilinear")) {
                    int fx= floor(x);
                    int cx = ceil(x);
                    int fy = floor(y);
                    int cy = ceil(y);
                    
                    int new_x;
                    int new_y;
                    
                    if ((fx == cx) && (fy == cy)) {
                        new_x = cx;
                        new_y = cy;
                    }
                    else if (fx == cx) {
                        new_x = fx;
                        new_y = (fy + cy) / 2;
                    }
                    else if (fy == cy) {
                        new_y = cy;
                        new_x = (cx + fx) / 2;
                    }
                    else {
                        new_x = ((cx - fx) / (cy - fy) * (y - fy)) + fx;
                        new_y = ((cy - fy) / (cx - fx) * (x - fx)) + fy;
                    }
                    
                    output.at<Vec3b>(i, j)[0] = input.at<Vec3b>(new_y, new_x)[0];
                    output.at<Vec3b>(i, j)[1] = input.at<Vec3b>(new_y, new_x)[1];
                    output.at<Vec3b>(i, j)[2] = input.at<Vec3b>(new_y, new_x)[2];
                    
                }
            }
        }
    }
    
    return output;
}
