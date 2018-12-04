#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#include<iostream>
#include<tuple> // for tuple
#define IM_TYPE	CV_8UC3
#define L 256		// # of intensity levels

using namespace cv;
using namespace std;

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

tuple<float, Mat> otsu_gray_seg(const Mat input);

int main() {

	Mat input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week6/opencv_practice_week6/opencv_practice_week6/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;
	float t;

	cvtColor(input, input_gray, CV_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	tie(t, output) = otsu_gray_seg(input_gray);

	namedWindow("Otsu", WINDOW_AUTOSIZE);
	imshow("Otsu", output);
	std::cout << t << std::endl;

	waitKey(0);

	return 0;
}


tuple<float, Mat> otsu_gray_seg(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	Mat output = Mat::zeros(row, col, input.type());
	int n = row*col;
    float T = 0, var = 0, var_max = 0, sum = 0, sumB = 0, q1 = 0, q2 = 0, sigma1 = 0, sigma2 = 0, sigma=0;
	int histogram[L] = { 0 };  // initializing histogram values
    float m1 = 0.0, m2 =0.0;

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {   // finding histogram of the image
			histogram[input.at<G>(i, j)]++;

		}
	}

	for (int i = 0; i < L; i++) {     //auxiliary value for computing mean value
		//Fill code
        //thresold가 0일때, 뒷 부분 값
        sumB += (histogram[i] * i);
        sigma = histogram[i] * pow(i,2);
    }
    

	for (int t = 0; t < L; t++) {  //update q
		//Fill code
        q1 += histogram[t];
        q2 = n-q1;
        
        sum += t*histogram[t];
        
        m1 = sum/q1;
        m2 = (sumB - sum)/q2;
        
        //Between
        var = q1 * q2 * pow((m1-m2),2);

        if (var > var_max) {
            T = t; //threshold
            var_max = var;
        }
	}

    printf("%f", T);
    ///*
    //Fill code that makes output image's pixel intensity to 255 if the intensity of the input image is bigger
    //than the threshold value else 0.
    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++){
            if( (float)input.at<G>(i,j) > T)
                output.at<G>(i,j) = 255;
            else
                output.at<G>(i,j) = 0;
        }

	return make_tuple(T, output);
}
