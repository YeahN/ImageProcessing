#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define RATIO_THR 0.4
#define K 5 //6   // the needed points amount of calculating homography matrix
#define S 500 //293  // the needed try times
#define THETA 3


using namespace std;
using namespace cv;

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


void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha) {

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// for check validation of I1 & I2f
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;

			// I2 is already in I_f by inverse warping
			// So, It is not necessary to check that only I2 is valid
			// if both are valid
			if (cond1 && cond2) {
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3f>(i, j);
			}
			// only I1 is valid
			else if (cond1) {
				I_f.at<Vec3f>(i, j) = I1.at<Vec3f>(i - diff_x, j - diff_y);
			}
		}
	}
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
template <typename T>
Mat homography(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points) {
	// initialize matrix
	Mat M(2 * number_of_points, 9, CV_32F, Scalar(0));
	// initialize every value in M
	for (int i = 0; i < number_of_points; i++) {
		M.at<T>(2 * i, 0) = ptl_x[i]; M.at<T>(2 * i, 1) = ptl_y[i]; M.at<T>(2 * i, 2) = 1;
		M.at<T>(2 * i, 3) = 0;		M.at<T>(2 * i, 4) = 0;		M.at<T>(2 * i, 5) = 0;
		M.at<T>(2 * i, 6) = -ptr_x[i] * ptl_x[i];
		M.at<T>(2 * i, 7) = -ptr_x[i] * ptl_y[i];		M.at<T>(2 * i, 8) = -ptr_x[i];
		M.at<T>(2 * i + 1, 0) = 0;	M.at<T>(2 * i + 1, 1) = 0;	M.at<T>(2 * i + 1, 2) = 0;
		M.at<T>(2 * i + 1, 3) = ptl_x[i];	M.at<T>(2 * i + 1, 4) = ptl_y[i];				M.at<T>(2 * i + 1, 5) = 1;
		M.at<T>(2 * i + 1, 6) = -ptr_y[i] * ptl_x[i];
		M.at<T>(2 * i + 1, 7) = -ptr_y[i] * ptl_y[i]; M.at<T>(2 * i + 1, 8) = -ptr_y[i];
	}
	Mat U, s, Vt, V;
	// use SVD discomposition
	SVD::compute(M, U, s, Vt);
	// the matrix gotten in SVD is the transposition of V
	transpose(Vt, V);
	return V.col(8);
}

int* Ransac(vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
	int maxInliers;
	int best[K];
	int ptl_x[28];
	int ptl_y[28];
	int ptr_x[28];
	int ptr_y[28];

	Mat A12, A21;
	for (int trytime = 0; trytime<S; trytime++)
	{
		maxInliers = 0;
		// randomly choose k pairs of corresponding points
		srand((unsigned)time(NULL));
		int index[K];
		for (int i = 0; i<K; i++)
		{
			// randomly pick out points in srcPoints
			int temp = rand() % srcPoints.size();
			// store these index in a temperate array
			index[i] = temp;
			ptl_x[i] = dstPoints[temp].x;
			ptl_y[i] = dstPoints[temp].y;
			ptr_x[i] = srcPoints[temp].x;
			ptr_y[i] = srcPoints[temp].y;
		}
		// estimate Affine matrix
		A12 = homography<float>(ptl_x, ptl_y, ptr_x, ptr_y, K);
		A21 = homography<float>(ptr_x, ptr_y, ptl_x, ptl_y, K);
		// count inlier points
		int count = 0;
		for (int i = 0; i<dstPoints.size(); i++)
		{
			// threshold is defined as 3
			// if |p' - p''| < threshold
			float d = A21.at<float>(6) * srcPoints[i].x + A21.at<float>(7) * srcPoints[i].y + A21.at<float>(8);
			float x = (A21.at<float>(0) * srcPoints[i].x + A21.at<float>(1) * srcPoints[i].y + A21.at<float>(2)) / d - dstPoints[i].x;
			float y = (A21.at<float>(3) * srcPoints[i].x + A21.at<float>(4) * srcPoints[i].y + A21.at<float>(5)) / d - dstPoints[i].y;
			// using formula |Mx - b|^2 < θ^2 
			if (x*x + y*y < THETA * THETA)
				count += 1;
		}
		// find maximum amount of inlier points
		if (count>maxInliers)
		{
			// update the max inlier amount
			for (int i = 0; i<K; i++)
			{
				best[i] = index[i];
			}
		}
	}
	return best;
}
int main() {
	Mat input1 = imread("i1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("i2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

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

	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);



	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);

	Mat I1, I2;
	// Read each image
	I1 = imread("i1.jpg");
	I2 = imread("i2.jpg");
	//I1 = imread("Img01.jpg");
	//I2 = imread("Img02.jpg");
	// Check for invalid input
	if (!I1.data || !I2.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	I1.convertTo(I1, CV_32FC3, 1.0 / 255);
	I2.convertTo(I2, CV_32FC3, 1.0 / 255);

	int ptl_x[28];
	int ptl_y[28];
	int ptr_x[28];
	int ptr_y[28];

	Mat A12, A21;
	int* best = Ransac(srcPoints, dstPoints);
	for (int i = 0; i<K; i++)
	{
		ptl_y[i] = dstPoints[best[i]].x;
		ptl_x[i] = dstPoints[best[i]].y;
		ptr_y[i] = srcPoints[best[i]].x;
		ptr_x[i] = srcPoints[best[i]].y;
	}

	// estimate homography matrix
	A12 = homography<float>(ptl_x, ptl_y, ptr_x, ptr_y, K);
	A21 = homography<float>(ptr_x, ptr_y, ptl_x, ptl_y, K);

	// height(row), width(col) of each image
	const float I1_row = I1.rows;
	const float I1_col = I1.cols;
	const float I2_row = I2.rows;
	const float I2_col = I2.cols;

	// compute corners (p1, p2, p3, p4)
	float tmp = A21.at<float>(6) * 0 + A21.at<float>(7) * 0 + A21.at<float>(8);
	Point2f p1((A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2)) / tmp,
		(A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5)) / tmp);

	tmp = A21.at<float>(6) * 0 + A21.at<float>(7) * (I2_col - 1) + A21.at<float>(8);
	Point2f p2((A21.at<float>(0) * 0 + A21.at<float>(1) * (I2_col - 1) + A21.at<float>(2)) / tmp,
		(A21.at<float>(3) * 0 + A21.at<float>(4) * (I2_col - 1) + A21.at<float>(5)) / tmp);

	tmp = A21.at<float>(6) * (I2_row - 1) + A21.at<float>(7) * 0 + A21.at<float>(8);
	Point2f p3((A21.at<float>(0) * (I2_row - 1) + A21.at<float>(1) * 0 + A21.at<float>(2)) / tmp,
		(A21.at<float>(3) * (I2_row - 1) + A21.at<float>(4) * 0 + A21.at<float>(5)) / tmp);

	tmp = A21.at<float>(6) * (I2_row - 1) + A21.at<float>(7) * (I2_col - 1) + A21.at<float>(8);
	Point2f p4((A21.at<float>(0) * (I2_row - 1) + A21.at<float>(1) * (I2_col - 1) + A21.at<float>(2)) / tmp,
		(A21.at<float>(3) * (I2_row - 1) + A21.at<float>(4) * (I2_col - 1) + A21.at<float>(5)) / tmp);

	// for inverse warping
	tmp = A12.at<float>(6) * 0 + A12.at<float>(7) * 0 + A12.at<float>(8);
	Point2f p1_((A12.at<float>(0) * 0 + A12.at<float>(1) * 0 + A12.at<float>(2)) / tmp,
		(A12.at<float>(3) * 0 + A12.at<float>(4) * 0 + A12.at<float>(5)) / tmp);

	tmp = A12.at<float>(6) * 0 + A12.at<float>(7) * (I1_col - 1) + A12.at<float>(8);
	Point2f p2_((A12.at<float>(0) * 0 + A12.at<float>(1) * (I1_col - 1) + A12.at<float>(2)) / tmp,
		(A12.at<float>(3) * 0 + A12.at<float>(4) * (I1_col - 1) + A12.at<float>(5)) / tmp);

	tmp = A12.at<float>(6) * (I1_row - 1) + A12.at<float>(7) * 0 + A12.at<float>(8);
	Point2f p3_((A12.at<float>(0) * (I1_row - 1) + A12.at<float>(1) * 0 + A12.at<float>(2)) / tmp,
		(A12.at<float>(3) * (I1_row - 1) + A12.at<float>(4) * 0 + A12.at<float>(5)) / tmp);

	tmp = A12.at<float>(6) * (I1_row - 1) + A12.at<float>(7) * (I1_col - 1) + A12.at<float>(8);
	Point2f p4_((A12.at<float>(0) * (I1_row - 1) + A12.at<float>(1) * (I1_col - 1) + A12.at<float>(2)) / tmp,
		(A12.at<float>(3) * (I1_row - 1) + A12.at<float>(4) * (I1_col - 1) + A12.at<float>(5)) / tmp);

	// compute boundary for merged image(I_f)
	int bound_u = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_b = (int)round(std::max(I1_row, std::max(p3.x, p4.x)));
	int bound_l = (int)round(min(0.0f, min(p1.y, p3.y)));
	int bound_r = (int)round(std::max(I1_col, std::max(p2.y, p4.y)));

	// compute boundary for inverse warping
	int bound_u_ = (int)round(min(0.0f, min(p1_.x, p2_.x)));
	int bound_b_ = (int)round(std::max(I2_row, std::max(p3_.x, p4_.x)));
	int bound_l_ = (int)round(min(0.0f, min(p1_.y, p3_.y)));
	int bound_r_ = (int)round(std::max(I2_col, std::max(p2_.y, p4_.y)));

	int diff_x = abs(bound_u);
	int diff_y = abs(bound_l);

	int diff_x_ = abs(bound_u_);
	int diff_y_ = abs(bound_l_);

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));

	// inverse warping with bilinear interplolation
	for (int i = -diff_x_; i < I_f.rows - diff_x_; i++) {
		for (int j = -diff_y_; j < I_f.cols - diff_y_; j++) {
			tmp = A12.at<float>(6)*i + A12.at<float>(7)*j + A12.at<float>(8);
			float x = (A12.at<float>(0) * i + A12.at<float>(1) * j + A12.at<float>(2)) / tmp + diff_x_;
			float y = (A12.at<float>(3) * i + A12.at<float>(4) * j + A12.at<float>(5)) / tmp + diff_y_;
			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);
			float mu = y - y1;
			float lambda = x - x1;
			if (x1 >= 0 && x2 < I2_row && y1 >= 0 && y2 < I2_col)
				I_f.at<Vec3f>(i + diff_x_, j + diff_y_) = lambda * mu * I2.at<Vec3f>(x2, y2) + lambda * (1 - mu) * I2.at<Vec3f>(x2, y1) +
				(1 - lambda) * mu * I2.at<Vec3f>(x1, y2) + (1 - lambda) * (1 - mu) * I2.at<Vec3f>(x1, y1);
		}
	}
	// image stitching with blend
	blend_stitching(I1, I2, I_f, diff_x, diff_y, 0.5);

	namedWindow("Left Image");
	imshow("Left Image", I1);

	namedWindow("Right Image");
	imshow("Right Image", I2);

	namedWindow("result");
	imshow("result", I_f);

	waitKey(0);

	return 0;
}
