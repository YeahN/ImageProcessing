// opencv_test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Inversefilter(const Mat input, int n, double sigma_t, double sigma_s, double d);
Mat FourierTransform(const Mat input, int m, int n, bool inverse);

int main() {

	Mat input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week5/opencv_practice_week5/opencv_practice_week5/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 7;
	double sigma_t = 5.0;
	double sigma_s = 5.0;

	//AWGN noise variance
	double noise_var = 0.03;

	//Deconvolution threshold
	double decon_thres = 0.1;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point


	Mat h_f = Gaussianfilter(input_gray, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)
	Mat g = Add_Gaussian_noise(h_f, 0, noise_var);		//					+ n(x, y)

	Mat F = Inversefilter(g, window_radius, sigma_t, sigma_s, decon_thres);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Gaussian Noise", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise", g);

	namedWindow("Deconvolution result", WINDOW_AUTOSIZE);
	imshow("Deconvolution result", F);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);

	Mat output = Mat::zeros(row, col, input.type());

	// convolution with zero padding
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            float sum = 0.0;
			for (int x = -n; x <= n; x++) { // for each kernel window
				for (int y = -n; y <= n; y++) {

					/* Gaussian filter with "zero-padding" boundary process:

					Fill the code:
					*/
                    
                    if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0))
                        sum += kernel.at<float>(0,0) * (float)(input.at<float>(i+x, j+y));
                        //sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));
				}
			}
            output.at<G>(i,j) = (G)sum;
		}
	}

	return output;
}

Mat Inversefilter(const Mat input, int n, double sigma_t, double sigma_s, double d) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);

	// Fill the code to
	// Perform Fourier Transform on Noise Image(G) and Gaussian Kernel(H)
    Mat H = Mat::zeros(row, col, input.type());
    Mat G = Mat::zeros(row, col, input.type());
    Mat F = Mat::zeros(row, col, input.type());
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			/* Element-wise division for compute F (F = G / H)

			Fill the code:
			*/
            double h1 = H.at<Vec2d>(i,j)[0];
            double h2 = H.at<Vec2d>(i,j)[1];
            double g1 = H.at<Vec2d>(i,j)[0];
            double g2 = H.at<Vec2d>(i,j)[1];
            
            double magH = sqrt(h1*h1 + h2*h2);
            
            if(magH >= d){
                F.at<Vec2d>(i,j)[0] =  (h1 * g1 + h2 * g2)/(magH * magH);
            }
		}
	}

	// Fill the code to perform Inverse Fourier Transform

	return F;
}

Mat FourierTransform(const Mat input, int m, int n, bool inverse) {

	//expand input image to optimal size
	Mat padded;
	copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat Transformed;

	// Applying DFT
	if (!inverse) {
		dft(padded, Transformed, DFT_COMPLEX_OUTPUT);
	}
	// Reconstructing original image from the DFT coefficients
	else {
		idft(padded, Transformed, DFT_SCALE | DFT_REAL_OUTPUT);
//		normalize(Transformed, Transformed, 0, 1, CV_MINMAX);
	}

	return Transformed;
}

Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {

	int kernel_size = (2 * n + 1);

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	// if "normalize" is true
	// return normalized Guassian Kernel
	// else, return unnormalized one
	if (normalize) {
		// Fill code to normalize kernel
	}

	return kernel;
}
