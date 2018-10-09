#include "Matlab_Class.h"
//#include "part_4.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <random>
//#include <opencv2/opencv.hpp>
//#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/core/utility.hpp>
//#include <opencv2/core/ocl.hpp>
using std::vector;
//using namespace cv;
using cv::Mat;
using cv::Rect;
using cv::Point;
using cv::Point2f;
using cv::Point2i;
using cv::Scalar;
using cv::KeyPoint;
using cv::DMatch;
using cv::imread;
using cv::imshow;
using cv::waitKey;
//--------
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

	// From matrix storing L2-norm: 
	//  -image-2's feature number in rows
	//  -image-3's feature number in cols
	Mat l2_mat(num_features, num_features, CV_64FC1);
	for (cv::MatIterator_<double> it = l2_mat.begin<double>(), end = l2_mat.end<double>(); it != end; ++it)
		*it = 1e6; // Initialize with large number

	for (int i = 0; i != num_features; ++i)   // Feature from image-2
	{
		// Only compare feature i to feature j once  =>  j=i:num_features
		for (int j = i; j != num_features; ++j) // Feature from image-3
		{
			double l2 = 0.0;
			for (int k = 0; k < desc_dim; ++k) // Itterate over elements of each descriptor
			{
				auto e = sift_desc_2.at<double>(i, k) - sift_desc_3.at<double>(j, k); // Each row is one descriptor
				l2 += e * e;
			}
			l2_mat.at<double>(i, j) = sqrt(l2) / static_cast<double>(desc_dim);
		}
		//cout << "\nl2_mat.row(i):\n\n" << l2_mat.row(i) << "\n";
		//getchar();
	}

	// Perform matching with Lowe-Ratio-Test Thresholding
	Mat index_pairs;
	vector<DMatch> index_pairs_matches;
	vector<KeyPoint> X, x;
	size_t num_matches{0};
	for (int i = 0; i != num_features; ++i) // iterate down the rows
	{
		// Extract row of the l2-matrix
		Mat row = l2_mat.row(i);

		double min_1{ 0 }, min_2{ 0 }, max{ 0 };
		Point min_idx_1, min_idx_2, max_idx;
		minMaxLoc(row, &min_1, &max, &min_idx_1, &max_idx);
		auto min_idx_1_ = min_idx_1.x;

		// Find 2nd smallest value
		row.at<double>(0, min_idx_1_) = 1e6; // Remove the minimum value
		minMaxLoc(row, &min_2, &max, &min_idx_2, &max_idx);
		auto min_idx_2_ = min_idx_2.x;

		// Compute ratio of 1st to 2nd min-val
		auto lowe_ratio = min_1 / min_2; // Threshold ratio

		// Ratio-test:
		double lowe_thresh = 0.6;
		if (lowe_ratio < lowe_thresh)
		{ // If ratio-test is met then we have a match
			
			// Store match
			auto index_1 = i;
			auto index_2 = min_idx_1_; // Col-index of l2-mat corresponds to index of feature from image-2 corresponds to current (ith) row
			Point2i match_indices(index_1, index_2);
			index_pairs.push_back(match_indices);

			// min_1 is the L2-norm for this match
			index_pairs_matches.push_back(DMatch(index_1, index_2, min_1));
			
			// Set column to large value to ensure unique matches
			l2_mat.col(min_idx_1_).setTo(Scalar(1e6));

			auto X1 = feature_coords_2.at<double>(index_1, 0),	X2 = feature_coords_2.at<double>(index_1, 1);
			auto x1 = feature_coords_3.at<double>(index_2, 0),	x2 = feature_coords_3.at<double>(index_2, 1);

			X.push_back(KeyPoint(X1, X2, 1.f));
			x.push_back(KeyPoint(x1, x2, 1.f));

			num_matches++;
		}
	}

	// Draw the matches
	Mat outImg;
	Scalar matchColor, singlePointColor;
	vector<DMatch> matches;
	for (int i = 0; i < num_matches; ++i)
	{
		// The matches have already been made => Just need to index into them like (0)<->(0), (1)<->(1), etc.
		int queryIdx = i;	
		int trainIdx = i;
		float distance = 1.0f;
		matches.push_back(DMatch(i, i, distance));
		drawMatches(imgs[1], X, imgs[2], x, matches, outImg,
			matchColor = Scalar::all(-1),
			singlePointColor = Scalar::all(-1));
		imshow("matched features", outImg);

		// The other way to do this is to use index_pairs_matches 
		// and the original unordered set of points

	}
	//waitKey(0);

	// -----------------------------------------------------------------
	/// Randomly select four samples
	// -----------------------------------------------------------------



	// Instantiate object of templated normal_distribution class
	std::random_device seed;
	std::mt19937 generator(seed());
	//const float mean = 0.0f;
	//const float variance = 5.0f;
	//std::normal_distribution<float> distribution(mean, variance);
	std::uniform_int_distribution<> distribution(0, num_matches);
	
	vector<int>rand(num_matches);
	cout << "num_matches = " << num_matches << "\n";

	// Use logic like below to ensure each value chosen is unique
	for (int i = 0; i != 4; ++i)
	{ 
		// Sample value from uniform distribution:
		rand[i] = distribution(generator);

		// Ensure uique rand value  chosen:
		for (int j = 0; j != i; ++j)
		{
			// Does newly chosen value match any of the pevious values j = 0:i
			if (rand[j] == rand[i])
			{
				--i;
				break;
			}		
		}
	}

	for (int i = 0; i != 4; ++i)
		cout << "rand[i=" << i << "] = " << rand[i] << "\n";

	// -Use this random number to index into the row of the cv::Mat of index_pairs which you use to index into feature_coords
	// -Use this to compute the

	float thresh = 0.5f;
	float reproj_error = 1.0f;

	while (reproj_error > thresh)
	{
		// Grab the four coordinates:
		Mat X_mat; //(4, 2, CV_64FC1);
		Mat x_mat; //(4, 2, CV_64FC1);
		for (int i = 0; i != 4; ++i)
		{
			auto feature_coord_2 = feature_coords_2.row(index_pairs.at<int>(rand[i], 0));
			auto feature_coord_3 = feature_coords_3.row(index_pairs.at<int>(rand[i], 1));

			//cout << "\nrandom index into index_pair = " << rand[i];

			//cout << "\nindex_pairs at the randon index for image 2 is: " << index_pairs.at<int>(rand[i], 0);
			//cout << "\nindex_pairs at the randon index for image 3 is: " << index_pairs.at<int>(rand[i], 1);

			cout << "\nfeature_coord_2 = " << feature_coord_2 << "\n";
			//cout << "\nfeature_coord_3 = " << feature_coord_3 << "\n";

			// Push the coodinates onto the matrices
			X_mat.push_back(feature_coord_2);
			x_mat.push_back(feature_coord_3);

			cout << "\nX_mat:\n" << X_mat << "\n";
		}

		// Compute the homography estimation


		// Project the points through the homography estimate


	}



	// DUDE - you have to do the FFT filtering today!!!!!!!!!
	// DUDE - you have to do the FFT filtering today!!!!!!!!!
	// DUDE - you have to do the FFT filtering today!!!!!!!!!
	// DUDE - you have to do the FFT filtering today!!!!!!!!!


	// BELOW IS THE PORT OF THE HOMOGRAPHY CODE
	// BELOW IS THE PORT OF THE HOMOGRAPHY CODE
	// BELOW IS THE PORT OF THE HOMOGRAPHY CODE
	// BELOW IS THE PORT OF THE HOMOGRAPHY CODE
	//	-Make the mods to move it in here
	

	// Send the matches to MATLAB and run the script to compute the homography
	
	////// Grab Homography matrix
	////Mat H2 = matlab.return_matrix_as_cvMat_from_matlab("H_to_cpp");

	////// Correpsondences for images 2 <-> 3
	////auto vects_2 = mat_2_points(X, x);

	/////// Compute re-projection error
	////// Step 1: Grab the full set of point correspondances
	////auto N = matlab.return_scalar_from_matlab("num_matches__to_cpp");
	////Mat X2_full = matlab.return_matrix_as_cvMat_from_matlab("X2_full__to_cpp", (int)N, 2);
	////Mat x2_full = matlab.return_matrix_as_cvMat_from_matlab("x2_full__to_cpp", (int)N, 2);

	////// Step 2: Pass into re-projection function
	////float threshold = 4.0f;
	////auto l2_error = reproj_error(H2, X2_full, x2_full);
	////vector<Mat> inliers = ransac(H2, X2_full, x2_full, threshold);
	////cout << "\n\ninliers[0]:\n" << inliers[0];
	////cout << "\n\ninliers[1]:\n" << inliers[1];

	
	return 0;
}