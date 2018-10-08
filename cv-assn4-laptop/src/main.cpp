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
	// NOTE - the feature matching section assumes there are less features in image-2 than image-3

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
	matlab.command("matlab_script");



	imshow("test", imgs[0]);
	waitKey(0);

	return 0;
}