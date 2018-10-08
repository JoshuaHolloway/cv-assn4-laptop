#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
using std::vector;
using namespace cv;
//===================================================================
//Mat est_homog(const Mat& X, const Mat& x)
Mat est_homog(vector<Point2f> X, vector<Point2f> x)
{
	// -X stores the feature coordinates for image-1
	// -x stores the feature coordinates for image-2
	// -Both are 2D-arrays stored like this:
	// CHANGED ----- [0][j] = x-coordinate of feature-j
	// CHANGED ----- [1][j] = y-coordinate of feature-j
	// TO:     vector<Point2f>  i.e., x[0].x, x[0].y, etc...

	//auto X1 = X.at<double>(0, 0), Y1 = X.at<double>(1, 0);
	//auto X2 = X.at<double>(0, 1), Y2 = X.at<double>(1, 1);
	//auto X3 = X.at<double>(0, 2), Y3 = X.at<double>(1, 2);
	//auto X4 = X.at<double>(0, 3), Y4 = X.at<double>(1, 3);

	//auto x1 = x.at<double>(0, 0), y1 = x.at<double>(1, 0);
	//auto x2 = x.at<double>(0, 1), y2 = x.at<double>(1, 1);
	//auto x3 = x.at<double>(0, 2), y3 = x.at<double>(1, 2);
	//auto x4 = x.at<double>(0, 3), y4 = x.at<double>(1, 3);

	// NOTE: Going from X -> x
	auto x1 = X[0].x;    auto y1 = X[0].y;
	auto x2 = X[1].x;    auto y2 = X[1].y;
	auto x3 = X[2].x;    auto y3 = X[2].y;
	auto x4 = X[3].x;    auto y4 = X[3].y;

	auto X1 = x[0].x;    auto Y1 = x[0].y;
	auto X2 = x[1].x;    auto Y2 = x[1].y;
	auto X3 = x[2].x;    auto Y3 = x[2].y;
	auto X4 = x[3].x;    auto Y4 = x[3].y;

	Mat A(8, 9, CV_64FC1);
	A.at<double>(0, 0) = -X1; A.at<double>(0, 1) = -Y1; A.at<double>(0, 2) = -1; A.at<double>(0, 3) = 0;   A.at<double>(0, 4) = 0;   A.at<double>(0, 5) = 0;  A.at<double>(0, 6) = X1 * x1; A.at<double>(0, 7) = Y1 * x1; A.at<double>(0, 8) = x1;
	A.at<double>(1, 0) = 0;   A.at<double>(1, 1) = 0;   A.at<double>(1, 2) = 0;  A.at<double>(1, 3) = -X1; A.at<double>(1, 4) = -Y1; A.at<double>(1, 5) = -1; A.at<double>(1, 6) = X1 * y1; A.at<double>(1, 7) = Y1 * y1; A.at<double>(1, 8) = y1;

	A.at<double>(2, 0) = -X2; A.at<double>(2, 1) = -Y2; A.at<double>(2, 2) = -1; A.at<double>(2, 3) = 0;   A.at<double>(2, 4) = 0;   A.at<double>(2, 5) = 0;  A.at<double>(2, 6) = X2 * x2; A.at<double>(2, 7) = Y2 * x2; A.at<double>(2, 8) = x2;
	A.at<double>(3, 0) = 0;   A.at<double>(3, 1) = 0;   A.at<double>(3, 2) = 0;  A.at<double>(3, 3) = -X2; A.at<double>(3, 4) = -Y2; A.at<double>(3, 5) = -1; A.at<double>(3, 6) = X2 * y2; A.at<double>(3, 7) = Y2 * y2; A.at<double>(3, 8) = y2;

	A.at<double>(4, 0) = -X3; A.at<double>(4, 1) = -Y3; A.at<double>(4, 2) = -1; A.at<double>(4, 3) = 0;   A.at<double>(4, 4) = 0;   A.at<double>(4, 5) = 0;  A.at<double>(4, 6) = X3 * x3; A.at<double>(4, 7) = Y3 * x3; A.at<double>(4, 8) = x3;
	A.at<double>(5, 0) = 0;   A.at<double>(5, 1) = 0;   A.at<double>(5, 2) = 0;  A.at<double>(5, 3) = -X3; A.at<double>(5, 4) = -Y3; A.at<double>(5, 5) = -1; A.at<double>(5, 6) = X3 * y3; A.at<double>(5, 7) = Y3 * y3; A.at<double>(5, 8) = y3;

	A.at<double>(6, 0) = -X4; A.at<double>(6, 1) = -Y4; A.at<double>(6, 2) = -1; A.at<double>(6, 3) = 0;   A.at<double>(6, 4) = 0;   A.at<double>(6, 5) = 0;  A.at<double>(6, 6) = X4 * x4; A.at<double>(6, 7) = Y4 * x4; A.at<double>(6, 8) = x4;
	A.at<double>(7, 0) = 0;   A.at<double>(7, 1) = 0;   A.at<double>(7, 2) = 0;  A.at<double>(7, 3) = -X4; A.at<double>(7, 4) = -Y4; A.at<double>(7, 5) = -1; A.at<double>(7, 6) = X4 * y4; A.at<double>(7, 7) = Y4 * y4; A.at<double>(7, 8) = y4;

	// Drop an SVD on A
	Mat Sigma, U, Vt;
	int flag = 4; // Full
	cv::SVD::compute(A, Sigma, U, Vt, flag = 4);

	//matlabClass matlab;
	//matlab.passImageIntoMatlab(Vt, "Vt_cpp");

	// Modify SVD
	Mat V = Vt.t(); // SVD in OpenCV is different than MATLAB and Numpy

	// Extract right most column of V
	const size_t I = V.rows;
	const size_t J = V.cols;
	Mat v(I, 1, CV_64FC1); // Col-vector to store right most col of V
	for (int i = 0; i < I; ++i)
		v.at<double>(i, 0) = V.at<double>(i, J - 1); // Itterate down right-most col of V

													 // Re-shape into 3x3 homography matrix
	int cn = 0;
	int rows = 3;
	Mat H = v.reshape(cn, rows);
	//cout << "H:\n" << H;

	//matlab.command("cd C:/Dropbox/fall2018_cv/ass4/cv-ass4-3/cv-ass4-matlab");
	//matlab.command("[ H, A, U, S, V ] = est_homography([488,124; 523,26; 266,254; 711,322], [908,124; 946,29; 880,255; 1116,327])");
	//matlab.passImageIntoMatlab(A, "A_cpp");
	//matlab.passImageIntoMatlab(U, "U_cpp");
	//matlab.passImageIntoMatlab(Sigma, "S_cpp");
	//matlab.passImageIntoMatlab(V, "V_cpp");
	//matlab.passImageIntoMatlab(H, "H_cpp");

	return H;
}
//===================================================================
#include <iostream>
using std::cout;
vector<vector<Point2f>> mat_2_points(Mat X, Mat x)
{
	// Put the point correspondences into a set of vector<Point2f>'s
	vector<Point2f> x_points;
	vector<Point2f> X_points;
	cout << "rows x cols = " << X.rows << " x " << X.cols << "\n";
	for (int i = 0; i < X.rows; ++i)
	{

		Point2f x_point;
		Point2f X_point;

		x_point.x = x.at<double>(i, 0); x_point.y = x.at<double>(i, 1);
		X_point.x = X.at<double>(i, 0); X_point.y = X.at<double>(i, 1);

		x_points.push_back(x_point);
		X_points.push_back(X_point);

		cout << "X_points" << X_points << "\n";
	}
	cout << "X_points" << X_points << "\n";
	cout << "x_points" << x_points << "\n\n\n\n";

	//vector<vector<Point2f>> return_vect;
	//return_vect.push_back(X_points);
	//return_vect.push_back(x_points);
	return vector<vector<Point2f>>({ X_points,  x_points });
}
//===================================================================
vector<Point2f> mat_2_points(Mat X)
{
	// Same as previous function except this one works with just one input Mat
	// Put the point correspondences into a set of vector<Point2f>'s
	vector<Point2f> X_points;
	for (int i = 0; i < X.rows; ++i)
	{
		Point2f X_point;
		X_point.x = X.at<double>(i, 0); X_point.y = X.at<double>(i, 1);
		X_points.push_back(X_point);
		//cout << "X_points" << X_points << "\n";
	}
	//cout << "X_points" << X_points << "\n";
	//vector<vector<Point2f>> return_vect;
	//return_vect.push_back(X_points);
	//return_vect.push_back(x_points);
	return vector<Point2f>({ X_points });
}
//===================================================================
Mat to_homo(Mat x);
Mat from_homo(Mat x_);
//void reproj_error(Mat H, vector<Point2f> X, vector<Point2f> x)
double reproj_error(Mat H, Mat Xt, Mat xt)
{	
	assert(Xt.size() == xt.size());

	// Comes in as Nx2 cv::Mat's, fix this to 2xN cv::Mat's
	Mat X = Xt.t();
	Mat x = xt.t();

	/// Compute re-projection error
	// Input (1 and 2): Vector of point2f correspondances X <-> x
	//	-Each point corresondance is a point2f
	// Input (3): Homography matrix
	//	-3x3 cv::Mat<double>

	/// Description of internal workings of the function
	// Input X is mapped into x_hat through estimated homography H
	//	-i.e.	x_hat = H * X
	// x_hat is the compared against x via L2-norm

	// Number of point correspondances
	const auto N = x.cols; 

	// Map points into homogeneous coordinates

	Mat X_ = to_homo(X);

	cout << "\nX size = " << X.rows << " x " << X.cols << "\n";
	cout << "\nX_ size = " << X_.rows << " x " << X_.cols << "\n";
	cout << "\nH size = " << H.rows << " x " << H.cols << "\n";

	// Project points through H
	Mat x_hat_ = H * X_;  // (3x3) * (3xN) => 3xN matrix where each col is homo-coordinates of one point
	Mat x_hat = from_homo(x_hat_);

	cout << "\nNumber of matches = " << N << "\n";
	cout << "\nx_hat_ size = " << x_hat_.rows << " x " << x_hat_.cols << "\n";
	cout << "\nx_hat size = " << x_hat.rows << " x " << x_hat.cols << "\n";
	cout << "\n\nx_hat:\n" << x_hat;
	
	// Get into form that matches notation from LM-implementation notes:
	vector<double> x_u, x_v;					// row-vec
	vector<double> x_u_hat, x_v_hat;  // row-vec
	for (int j = 0; j < N; ++j) // itterate over matches
	{
		x_u.push_back(x.at<double>(0, j));
		x_v.push_back(x.at<double>(1, j));

		x_u_hat.push_back(x_hat.at<double>(0, j));
		x_v_hat.push_back(x_hat.at<double>(1, j));
	}

	// Form d-vector
	vector<double> d; // To be used in Levenberg-Marquardt
	double l2 = 0; // l2-norm
	for (int j = 0; j < N; ++j)
	{
		auto d_jx = x_u[j] - x_u_hat[j];
		auto d_jy = x_v[j] - x_v_hat[j];

		d.push_back(d_jx);
		d.push_back(d_jy);

		l2 += d_jx * d_jx + d_jy * d_jy;
	}
	l2 = sqrt(l2) / static_cast<double>(N);

	// Perform inner product with self
	cout << "l2 error manual = " << l2 << "\n";
	return l2;
}
//===================================================================
vector<Mat> ransac(Mat H, Mat Xt, Mat xt, float threshold)
{
	assert(Xt.size() == xt.size());

	// Comes in as Nx2 cv::Mat's, fix this to 2xN cv::Mat's
	Mat X = Xt.t();
	Mat x = xt.t();

	/// Compute re-projection error and count number of inliers
	// Input (1 and 2): Vector of point2f correspondances X <-> x
	//	-Each point corresondance is a point2f
	// Input (3): Homography matrix
	//	-3x3 cv::Mat<double>

	/// Description of internal workings of the function
	// Input X is mapped into x_hat through estimated homography H
	//	-i.e.	x_hat = H * X
	// x_hat is the compared against x via L2-norm

	// Number of point correspondances
	const auto N = x.cols;

	// Map points into homogeneous coordinates

	Mat X_ = to_homo(X);

	cout << "\nX size = " << X.rows << " x " << X.cols << "\n";
	cout << "\nX_ size = " << X_.rows << " x " << X_.cols << "\n";
	cout << "\nH size = " << H.rows << " x " << H.cols << "\n";

	// Project points through H
	Mat x_hat_ = H * X_;  // (3x3) * (3xN) => 3xN matrix where each col is homo-coordinates of one point
	Mat x_hat = from_homo(x_hat_);

	cout << "\nNumber of matches = " << N << "\n";
	cout << "\nx_hat_ size = " << x_hat_.rows << " x " << x_hat_.cols << "\n";
	cout << "\nx_hat size = " << x_hat.rows << " x " << x_hat.cols << "\n";
	cout << "\n\nx_hat:\n" << x_hat;

	// Get into form that matches notation from LM-implementation notes:
	vector<double> x_u, x_v;					// row-vec
	vector<double> x_u_hat, x_v_hat;  // row-vec
	for (int j = 0; j < N; ++j) // itterate over matches
	{
		x_u.push_back(x.at<double>(0, j));
		x_v.push_back(x.at<double>(1, j));

		x_u_hat.push_back(x_hat.at<double>(0, j));
		x_v_hat.push_back(x_hat.at<double>(1, j));
	}

	// Form d-vector
	vector<double> d; // To be used in Levenberg-Marquardt
	double l2 = 0; // l2-norm
	size_t num_inliers(0);
	double error = 1e6;
	vector<Mat> inlier_output;
	while (num_inliers < 4)
	{
		Mat x_inliers;//(N, 2, CV_64FC1); // N x 2 x JOSH matrix storing inlier coordinates in rows
		Mat X_inliers;//(N, 2, CV_64FC1); // N x 2 x JOSH matrix storing inlier coordinates in rows
		for (int j = 0; j < N; ++j)
		{
			auto d_jx = x_u[j] - x_u_hat[j];
			auto d_jy = x_v[j] - x_v_hat[j];

			d.push_back(d_jx);
			d.push_back(d_jy);

			l2 += d_jx * d_jx + d_jy * d_jy;

			error = sqrt(d_jx * d_jx + d_jy * d_jy);
			if (error < threshold)
			{
				// Store inliers
				//num_inliers++;
				Mat x_inliers_row(1, 2, CV_64FC1);
				Mat X_inliers_row(1, 2, CV_64FC1);
				x_inliers_row.at<double>(0, 0) = x.at<double>(0, j);
				x_inliers_row.at<double>(0, 1) = x.at<double>(1, j);

				X_inliers_row.at<double>(0, 0) = X.at<double>(0, j);
				X_inliers_row.at<double>(0, 1) = X.at<double>(1, j);

				// Push rows onto matrices:
				x_inliers.push_back(x_inliers_row);
				X_inliers.push_back(X_inliers_row);

				cout << "\n\nx_inliers\n" << x_inliers;

				// Increment the inlier index;
				num_inliers++;

				//cout << "\n\nX_inliers = " << X_inliers << "\n";
				//getchar();
			}				
		}
		l2 = sqrt(l2) / static_cast<double>(N);
		cout << "l2 error manual = " << l2 << "\n";
		cout << "number inliers = " << num_inliers << "\n";

		if (num_inliers >= 4)
		{
			// If sufficient number of inliers then store them in output
			inlier_output.push_back(X_inliers);
			inlier_output.push_back(x_inliers);
		}
		else 
		{
			// reset number of inliers to count and start again
			num_inliers = 0;
		}
	}
	// Output: inlier_output[0] = X  and inlier_output[1] = x
	return inlier_output;
}
//===================================================================
Mat to_homo(Mat x)
{
	// Map a vector in non-homogeneous coordinates (dimension m)
	//  to a vector in homogeneous coordinates     (dimension m+1)

	Mat x_ = Mat::ones(x.rows + 1, x.cols, CV_64FC1);
	for (int i = 0; i < x.rows; ++i)
		for (int j = 0; j < x.cols; ++j)
			x_.at<double>(i, j) = x.at<double>(i, j);

	return x_;
}
//===================================================================
Mat from_homo(Mat x_)
{
	// Map a vector in homogeneous coordinates     (dimension m+1)
	//  to a vector in non-homogeneous coordinates (dimension m)

	const size_t rows = x_.rows;
	const size_t cols = x_.cols;

	Mat x = Mat::ones(rows - 1, cols, CV_64FC1);
	for (int i = 0; i < rows - 1; ++i)
		for (int j = 0; j < cols; ++j)
			x.at<double>(i, j) = x_.at<double>(i, j) / x_.at<double>(rows - 1, j);

	return x;
}
//===================================================================