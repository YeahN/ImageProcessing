//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

void type2str(int type);
int morphology_test();
int hit_miss();

int main() {
    
    morphology_test();
    
    hit_miss();
    
    return 0;
}

int morphology_test()
{
    Mat src = imread("nicework.tif", 0);
    Mat element = getStructuringElement(0, Size(7, 7));
    Mat dst;
    
    if (!src.data)
    {
        std::cout << "Could not open" << std::endl;
        return -1;
    }
    
    morphologyEx(src, dst, MORPH_DILATE, element);
    namedWindow("Dilation");
    imshow("Dilation", dst);
    imwrite("Dilation.png", dst);
    
    morphologyEx(src, dst, MORPH_ERODE, element);
    namedWindow("Erosion");
    imshow("Erosion", dst);
    imwrite("Erosion.png", dst);
    
    morphologyEx(src, dst, MORPH_OPEN, element);
    namedWindow("Opening");
    imshow("Opening", dst);
    imwrite("Opening.png", dst);
    
    morphologyEx(src, dst, MORPH_CLOSE, element);
    namedWindow("Closing");
    imshow("Closing", dst);
    imwrite("Closing.png", dst);
    
    return 1;
}

int hit_miss()
{
    Mat input_gray = imread("license_clean.png", 0);
    Mat plate_gray = imread("character_template\\E.png", 0);
    
    if (!input_gray.data || !plate_gray.data)
    {
        std::cout << "Could not open" << std::endl;
        return -1;
    }
    
    Mat input_binary, input_binary_inv, input_binary1, input_binary2;
    Mat plate_binary, plate1, plate2, _plate_a, _plate_b;
    
    // Generate a binary image (input_gray) with Otsu thresholding
    // Fill here using 'threshold'
    threshold(input_gray, input_binary_inv, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    input_binary = Mat::zeros(input_binary_inv.size(), input_binary_inv.type());
    for (int i = 0; i<input_binary.size().height; i++)
        for (int j = 0; j < input_binary.size().width; j++)
            input_binary.at<uchar>(i, j) = 255 - input_binary_inv.at<uchar>(i, j);
    
    namedWindow("input_binary");      imshow("input_binary", input_binary);
    namedWindow("input_binary_inv");   imshow("input_binary_inv", input_binary_inv);
    
    
    imwrite("input_binary.png", input_binary);
    imwrite("input_binary_inv.png", input_binary);
    
    // Generate a binary image (plate_gray) with Otsu thresholding
    // Fill here using 'threshold'
    threshold(plate_gray, plate_binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    for (int i = 0; i<plate_binary.size().height; i++)
        for (int j = 0; j < plate_binary.size().width; j++)
            plate_binary.at<uchar>(i, j) = 255 - plate_binary.at<uchar>(i, j);
    
    imwrite("plate_binary.png", plate_binary);
    
    
    //// Hit-or-miss method
    // Step 1: Generate structure element (B1) by applying the erosion to the binary template image (SE: 3x3 square) to mitigate the effect of slightly mismatched regions.
    Mat element_3_3 = getStructuringElement(0, Size(3, 3));
    // Fill here using 'morphologyEx'
    morphologyEx(plate_binary, plate1, MORPH_ERODE, element_3_3);
    
    
    // Step 2: Generate structure element (B2) by using 'the dilated template image using 5x5 square - the dilated template image using 3x3 square.'
    Mat element_5_5 = getStructuringElement(0, Size(5, 5));
    // Fill here using 'morphologyEx'
    morphologyEx(plate_binary, _plate_a, MORPH_DILATE, element_5_5); morphologyEx(plate_binary, _plate_b, MORPH_DILATE, element_3_3);
    
    plate2 = Mat::zeros(plate1.size(), plate1.type());
    for (int i = 0; i<_plate_a.size().height; i++)
        for (int j = 0; j < _plate_a.size().width; j++)
            plate2.at<uchar>(i, j) = _plate_a.at<uchar>(i, j) - _plate_b.at<uchar>(i, j);
    
    imwrite("plate1.png", plate1);
    imwrite("plate2.png", plate2);
    
    // Step 3: Perform hit-or-miss method
    Mat output_location, output;
    // Fill here using 'morphologyEx' and 'bitwise_and' plate2 and plate1
    morphologyEx(input_binary, input_binary1, MORPH_ERODE, plate1);
    morphologyEx(input_binary_inv, input_binary2, MORPH_ERODE, plate2);
    bitwise_and(input_binary1, input_binary2, output_location);
    
    imwrite("output_location.png", output_location);
    
    
    // Step 4: Overlay the detected result on the grayscale image for visualization
    //Due to bug in dilation operation of 'morphologyEx', kernel should be mirrored.
    
    Mat plate1_mirror = Mat::zeros(plate1.size(), plate1.type());
    int h = plate1_mirror.size().height;
    int w = plate1_mirror.size().width;
    for (int i = 0; i<h; i++)   for (int j = 0; j < w; j++)
        plate1_mirror.at<uchar>(i, j) = plate1.at<uchar>(h - 1 - i, w - 1 - j);
    
    morphologyEx(output_location, output, MORPH_DILATE, plate1_mirror);
    
    for (int i = 0; i<output.size().height; i++)
        for (int j = 0; j < output.size().width; j++)
        {
            if (output.at<uchar>(i, j) == 0)
                output.at<uchar>(i, j) = 0.5*input_gray.at<uchar>(i, j);
        }
    
    namedWindow("output");   imshow("output", output);
    imwrite("output.png", output);
    
    waitKey(0);
    return 1;
}

void type2str(int type) {
    string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans + '0');
    
    printf("Matrix: %s \n", r.c_str());
}
