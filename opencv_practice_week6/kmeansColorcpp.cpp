#include <iostream>
#include <opencv2/opencv.hpp>

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

// Note that this code is for the case when an input data is a color value.
int main() {

	Mat input = imread("/Users/yeahna/Documents/EWHA/18-1/SmartSoftwareProject/Week6/opencv_practice_week6/opencv_practice_week6/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);


	Mat samples(input.rows * input.cols, 3, CV_32F);
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
            for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y, x)[z];


	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


	Mat new_image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
            //레이블은 이미지 사이즈랑 크기 같음
			int cluster_idx = labels.at<int>(y + x*input.rows, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
            for (int z = 0; z < 3; z++)
                new_image.at<Vec3b>(y,x)[z] = centers.at<float>(cluster_idx,z);
		}
	imshow("clustered image", new_image);

	waitKey(0);

	return 0;
}

