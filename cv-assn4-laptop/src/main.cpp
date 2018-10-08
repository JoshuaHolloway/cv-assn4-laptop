#include "Matlab_Class.h"
//#include "part_4.h"
//#include "helper.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
using std::vector;
using namespace cv;
int main()
{
	// Link to MATLAB environment
	matlabClass matlab;

	// Read images
	vector<Mat> imgs;
	imgs.push_back(imread("keble_a.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imgs.push_back(imread("keble_b.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imgs.push_back(imread("keble_c.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	// ==== Image 2 and 3 =============================================
	// ==== Image 2 and 3 =============================================
	// ==== Image 2 and 3 =============================================

	// Pass data back into C++ from Matlab:
	//matlab.command("matlab_script");
	matlab.command("load('sift_features.mat');");
	matlab.command("load('corners.mat');");

	// Step 1: Read the features and descriptors from MATLAB:

	// Read in features/descriptors:
	Mat sift_desc_2 = matlab.return_matrix_as_cvMat_from_matlab("sift_center"); 	// Each row-vector is one feature
	Mat sift_desc_3 = matlab.return_matrix_as_cvMat_from_matlab("sift_right");

	// Raw 2d-coordinates of features
	Mat feature_coords_2 = matlab.return_matrix_as_cvMat_from_matlab("corners_center");
	Mat feature_coords_3 = matlab.return_matrix_as_cvMat_from_matlab("corners_right");

	assert(sift_desc_2.rows == sift_desc_3.rows);
	assert(sift_desc_2.cols == sift_desc_3.cols);
	size_t num_features = sift_desc_2.rows;
	size_t desc_dim = sift_desc_2.cols;
	
	imshow("test", imgs[0]);
	waitKey(0);

	return 0;
}