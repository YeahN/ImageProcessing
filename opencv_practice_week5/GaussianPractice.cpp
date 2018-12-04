#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);

int main() {

	Mat input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week5/opencv_practice_week5/opencv_practice_week5/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	
	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Gaussianfilter_Gray(noise_Gray, 3, 10, 10, "adjustkernel");
    Mat Denoised_RGB = Gaussianfilter_RGB(noise_RGB, 3, 10, 10, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

    namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
    imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

    namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
    imshow("Denoised (RGB)", Denoised_RGB);

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

Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
    float denom;

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
    
    denom = 0.0;
    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
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

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

            if (!strcmp(opt, "zero-padding")) {
                float sum = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
                        
						/* Gaussian filter with "zero-padding" boundary process:		

                         Fill the code:
						*/

                        if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)){
                            sum += kernel.at<float>(x+n,y+n) * (float)input.at<G>(i + x, j + y);
                        }
					}
				}
                output.at<G>(i,j) = (G)sum;
			}

			else if (!strcmp(opt, "mirroring")) {
                float tempa = 0.0;
                float tempb = 0.0;
                float sum = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "mirroring" boundary process:		

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
                        sum += kernel.at<float>(a+n,b+n) * (float)(input.at<G>(tempa,tempb));
                    }
                }
                output.at<G>(i,j)=(G)sum;
			}

			else if (!strcmp(opt, "adjustkernel")) {
                float sum1 = 0.0;
                float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" boundary process:		

						Fill the code:
						*/
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            sum1 += kernel.at<float>(a+n,b+n) * (float)(input.at<G>(i+a,j+b));
                            sum2 += kernel.at<float>(a+n,b+n);
                        }
                    }
                }
                output.at<G>(i,j)=(G)(sum1/sum2);
                
            }
		}
	}
	return output;
}

Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {

    Mat kernel;
    
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    double denom;
    
    // Initialiazing Gaussian Kernel Matrix
    // Fill code to initialize Gaussian filter kernel matrix

    kernel = Mat::ones(kernel_size, kernel_size, CV_64F) / (double)(kernel_size * kernel_size);
    
    denom = 0.0;
    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            double value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
            kernel.at<double>(a+n, b+n) = value1;
            denom += value1;
        }
    }
    
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            kernel.at<double>(a+n, b+n) /= denom;
        }
    }
    
    Mat output = Mat::zeros(row, col, input.type());

    // convolution
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {

            if (!strcmp(opt, "zero-padding")) {
                double sumR = 0.0;
                double sumG = 0.0;
                double sumB = 0.0;
                for (int a = -n; a <= n; a++) { // for each kernel window
                    for (int b = -n; b <= n; b++) {

                        /* Gaussian filter with "zero-padding" boundary process (3-channel input):

                        Fill the code:
                        */
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)){
                            sumR += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(i+a,j+b)[2]);
                            sumG += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(i+a,j+b)[1]);
                            sumB += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(i+a,j+b)[0]);
                        }
                    }
                }
                output.at<Vec3d>(i,j)[0]=(G)sumB;
                output.at<Vec3d>(i,j)[1]=(G)sumG;
                output.at<Vec3d>(i,j)[2]=(G)sumR;
            }

            else if (!strcmp(opt, "mirroring")) {
                double sumR = 0.0;
                double sumG = 0.0;
                double sumB = 0.0;
                int tempa;
                int tempb;
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
                        sumR += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(tempa,tempb)[2]);
                        sumG += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(tempa,tempb)[1]);
                        sumB += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(tempa,tempb)[0]);
                    }
                }
                output.at<Vec3d>(i,j)[0]=(G)sumB;
                output.at<Vec3d>(i,j)[1]=(G)sumG;
                output.at<Vec3d>(i,j)[2]=(G)sumR;
            }
            
            
            else if (!strcmp(opt, "adjustkernel")) {
                double sumR = 0.0;
                double sumG = 0.0;
                double sumB = 0.0;
                double sumR2 = 0.0;
                double sumG2 = 0.0;
                double sumB2 = 0.0;
                for (int a = -n; a <= n; a++) { // for each kernel window
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            sumR += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(i+a,j+b)[2]);
                            sumG += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(i+a,j+b)[1]);
                            sumB += kernel.at<double>(a+n,b+n) * (double)(input.at<Vec3d>(i+a,j+b)[0]);
                            sumR2 += kernel.at<double>(a+n,b+n);
                            sumG2 += kernel.at<double>(a+n,b+n);
                            sumB2 += kernel.at<double>(a+n,b+n);
                        }
                    }
                }
                output.at<Vec3d>(i,j)[0]=(G)(sumB/sumB2);
                output.at<Vec3d>(i,j)[1]=(G)(sumG/sumG2);
                output.at<Vec3d>(i,j)[2]=(G)(sumR/sumR2);
            }
        }
    }
    return output;
}






