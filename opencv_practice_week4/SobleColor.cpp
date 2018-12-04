#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

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

Mat sobelfilter(const Mat input);

int main() {

	Mat input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week4/opencv_practice_week4/opencv_practice_week4/lena.jpg", CV_LOAD_IMAGE_COLOR);

	Mat output;



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Colorscale", WINDOW_AUTOSIZE);
	imshow("Colorscale", input);
	output = sobelfilter(input);

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
    int suma;
    int sumb;
    float outputR;
    float outputG;
    float outputb;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
    int Sx[3][3] = { -1,0,1,-2,0,2,-1,0,1 };
    int Sy[3][3] = {-1,-2,-1,0,0,0,1,2,1 };

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            float xR = 0;
            float yR = 0;
            float xG = 0;
            float yG = 0;
            float xB = 0;
            float yB = 0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy )
                    if (i + a > row - 1) {  //mirroring for the border pixels
                        suma = i - a;
                    }
                    else if (i + a < 0) {
                        suma = -(i + a);
                    }
                    else {
                        suma = i + a;
                    }
                    if (j + b > col - 1) {
                        sumb = j - b;
                    }
                    else if (j + b < 0) {
                        sumb = -(j + b);
                    }
                    else {
                        sumb = j + b;
                    }
                    xR += (float)(input.at<Vec3b>(suma, sumb)[2]) * Sx[a+n][b+n];
                    yR += (float)(input.at<Vec3b>(suma, sumb)[2]) * Sy[a+n][b+n];
                    xG += (float)(input.at<Vec3b>(suma, sumb)[1]) * Sx[a+n][b+n];
                    yG += (float)(input.at<Vec3b>(suma, sumb)[1]) * Sy[a+n][b+n];
                    xB += (float)(input.at<Vec3b>(suma, sumb)[0]) * Sx[a+n][b+n];
                    yB += (float)(input.at<Vec3b>(suma, sumb)[0]) * Sy[a+n][b+n];
				}
			}
            output.at<Vec3b>(i,j)[2] = sqrt(pow(xR,2) + pow(yR,2));
            output.at<Vec3b>(i,j)[1] = sqrt(pow(xG,2) + pow(yG,2));
            output.at<Vec3b>(i,j)[0] = sqrt(pow(xB,2) + pow(yB,2));
		}
	}
	return output;
}
