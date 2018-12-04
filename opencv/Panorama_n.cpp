#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <stdlib.h>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;


void type2str(int type);
double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int flag);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

Point2d applyHomography(const Point2d& _point, const Mat& _H)
{
	Point2d ret = Point2d(-1, -1);

	const double u = _H.at<double>(0, 0) * _point.x + _H.at<double>(0, 1) * _point.y + _H.at<double>(0, 2);
	const double v = _H.at<double>(1, 0) * _point.x + _H.at<double>(1, 1) * _point.y + _H.at<double>(1, 2);
	const double s = _H.at<double>(2, 0) * _point.x + _H.at<double>(2, 1) * _point.y + _H.at<double>(2, 2);
	if (s != 0)
	{
		ret.x = (u / s);
		ret.y = (v / s);
	}
	return ret;
};
void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha) {

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;

	for (int i = 0; i < I1.rows; i++) {
		for (int j = 0; j < I1.cols; j++) {
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;
			if (cond1 && cond2) {
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i, j);
			}
			else
				I_f.at<Vec3f>(i, j) = I1.at<Vec3f>(i, j);

		}
	}
}

Mat stitching(Mat input1, Mat input2) {

	Mat input1_gray, input2_gray;

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1);//point(x,y)
	extractor->compute(input1_gray, keypoints1, descriptors1);//description SIFT

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	//Generating keypoint
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);

	Mat h = findHomography(dstPoints, srcPoints, RANSAC);
	return h;
}
int main() {

	Mat input1 = imread("i2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("i1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input3 = imread("i3.jpg", CV_LOAD_IMAGE_COLOR);

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));
	resize(input3, input3, Size(input3.cols / 2, input3.rows / 2));

	Mat H = stitching(input1, input2);
	Mat H_ = stitching(input3, input1);
	H_ = H*H_;
	Mat result, result1;

	std::vector<Point2f> obj_corners(4);
	std::vector<Point2f> scene_corners(4);
	std::vector<Point2f> scene_corners1(4);

	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(input1.cols, 0);
	obj_corners[2] = cvPoint(input1.cols, input1.rows); obj_corners[3] = cvPoint(0, input1.rows);
	perspectiveTransform(obj_corners, scene_corners, H);

	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(input3.cols, 0);
	obj_corners[2] = cvPoint(input3.cols, input3.rows); obj_corners[3] = cvPoint(0, input3.rows);
	perspectiveTransform(obj_corners, scene_corners1, H_);


	warpPerspective(input1, result, H, cv::Size(scene_corners[2].x, scene_corners[2].y));
	warpPerspective(input3, result1, H_, cv::Size(scene_corners1[2].x, scene_corners1[2].y));

	result.convertTo(result, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);
	result1.convertTo(result1, CV_32FC3, 1.0 / 255);


	blend_stitching(input2, result, result, scene_corners[0].x, scene_corners[0].y, 0.5);
	blend_stitching(result, result1, result1, scene_corners1[0].x, scene_corners1[0].y, 0.5);

	// Display mathing image
	imshow("Warped", result1);
	waitKey(0);

	return 0;
}

/**
* Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors, int flag) {
	int neighbor = -1;
	double dist2 = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		double dist = euclidDistance(v, vec);
		if (dist<minDist) {
			dist2 = minDist;
			minDist = dist;
			neighbor = i;
		}

	}
	if (flag == 2) {
		double compare = minDist / dist2;
		if (compare >= RATIO_THR) {
			return 1;
		}
		else
			return 0;
	}
	else if (flag == 1)
		return neighbor;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);//specific descriptor

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2, 1);//index of keypoint

																	 // Refine matching points using ratio_based thresholding
		if (ratio_threshold) {
			if (nearestNeighbor(desc1, keypoints2, descriptors2, 2)) {
				continue;
			}
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			Mat desc2 = descriptors2.row(nn);
			int rnn = nearestNeighbor(desc2, keypoints1, descriptors1, 1);
			if (rnn != i) {
				continue;
			}
		}

		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
}

//If you want to know the type of 'Mat', use the following function
//For instance, for 'Mat input'
//type2str(input.type());

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
