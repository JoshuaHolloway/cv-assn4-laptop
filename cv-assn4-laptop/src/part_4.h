#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
using std::vector;
using namespace cv;
void part_4(Mat img_0, Mat img_1, Mat img_2, Mat H1, Mat H2, 
	vector<Point2f> X1, vector<Point2f> x1,
	vector<Point2f> X2, vector<Point2f> x2)
{
	// Step 1: Change variables to work with 1 to 2 then do 0 to 1

	// Are the sizes of all the images the same???
	const int height1 = img_1.rows, width1 = img_1.cols;
	const int height2 = img_2.rows, width2 = img_2.cols;

	Mat T1 = cv::getPerspectiveTransform(X1, x1);
	Mat T2 = cv::getPerspectiveTransform(X2, x2);
	//cout << "\n\nT1 = \n" << T1 << "\n\n";
	//cout << "\n\nT2 = \n" << T2 << "\n\n";
	//cout << "\n\nX1" << X1 << "\n\n\n";
	//cout << "\n\nx1" << x1 << "\n\n\n";
	//cout << "\n\nX2" << X2 << "\n\n\n";
	//cout << "\n\nx2" << x2 << "\n\n\n";

	// Find max value of x and y and pass to the size for the perspective warp
	int maxCols1(0), maxRows1(0);
	int maxCols2(0), maxRows2(0);
	for (int i = 0; i < X2.size(); i++)
	{
		// Max's form 1 <-> 2 correspondances
		cout << "\ni = " << i << "\n";
		if (maxRows2 < X2.at(i).y)
			maxRows2 = X2.at(i).y;
		if (maxCols2 < X2.at(i).x)
			maxCols2 = X2.at(i).x;

		// Max's form 0 <-> 1 correspondances
		if (maxRows1 < X1.at(i).y)
			maxRows1 = X1.at(i).y;
		if (maxCols1 < X1.at(i).x)
			maxCols1 = X1.at(i).x;
	}
	//auto size = Size(maxCols, maxRows);
	auto size2 = Size(width2 + maxCols2, height2+maxRows2);
	auto size1 = Size(width1 + maxCols1, height1 + maxRows1);
//	cout << "\n\nsize" << size2 << "\n\n";
	Size dsize1(height1, width1);
	Size dsize2(height2, width2);

	Mat img_perspective_opencv_H2, img_perspective_custom_H2;
	Mat img_perspective_opencv_H1, img_perspective_custom_H1;

	cout << "\n\nH1 = \n" << H1 << "\n\n";
	cout << "\n\nH2 = \n" << H2 << "\n\n";
	//cv::warpPerspective(img_2, img_perspective_opencv_H1, T2, size2, cv::INTER_LINEAR, cv::BORDER_CONSTANT); // Use homogr from opencv
	cv::warpPerspective(img_0, img_perspective_custom_H1, H1, size1, cv::INTER_LINEAR, cv::BORDER_CONSTANT); // Use homogr from custom matlab function

	cv::warpPerspective(img_2, img_perspective_custom_H2, H2, size2, cv::INTER_LINEAR, cv::BORDER_CONSTANT); // Use homogr from custom matlab function
	

	Mat img_c1;              cvtColor(img_1, img_c1, cv::COLOR_GRAY2BGR);
	Mat img_c2;              cvtColor(img_2, img_c2, cv::COLOR_GRAY2BGR);
	//Mat img_perspective_c2_opencv;  cvtColor(img_perspective_opencv_H2, img_perspective_c2_opencv, cv::COLOR_GRAY2BGR);
	Mat img_perspective_c1;  cvtColor(img_perspective_custom_H1, img_perspective_c1, cv::COLOR_GRAY2BGR);
	Mat img_perspective_c2;  cvtColor(img_perspective_custom_H2, img_perspective_c2, cv::COLOR_GRAY2BGR);

	// Draw points on original image
	int radius = 2;
	Scalar color(255, 0, 0);
	vector<Scalar> colors{ Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 0) };
	int thickness;

	// Look at mapping of 4-points for homographic transformation: X -> x
	int i = 0;
	for (auto itter : X2)
	{
		//cv::circle(img_c1, itter, radius, colors[i++], thickness = 8);
		cout << "*itter = " << itter << " \n";
	}
	
	imshow("Original with 4-points", img_c1);


	// 1 <-> 2 correspondances
	i = 0;
	for (auto itter : x2)
		cv::circle(img_perspective_c2, itter, radius, colors[i], thickness = 8);

	// 0 <-> 1 correspondances
	i = 0;
	for (auto itter : x1)
		cv::circle(img_perspective_c1, itter, radius, colors[i], thickness = 8);
		

	// look at images with points on them
	
	imshow("Perspective-Transf - Custom H1", img_perspective_c1);
	imshow("Perspective-Transf - Custom H2", img_perspective_c2);
	waitKey(0);
}