/*

This code was written by Tim Ranson for the master's dissertation 'Noise-robust motion detection for low-light videos' by Tim Ranson.

*/


#ifndef DENOISING_
#define DENOISING_

#include <iostream>
using std::cout;
using std::endl;
#include <vector>
using std::vector;
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::cuda;

// #include "mcwf.h"

namespace dn {

	const short DEN_BLUR = 0;
	const short DEN_GAUSS_BLUR = 1;
	const short DEN_MEDIAN_BLUR = 2;
	const short DEN_BILATERAL_FILTER = 3;
	const short DEN_NL_MEANS = 4;
	// const short DEN_ZLOKOLICA = 5;

	const int ksize = 7;

	void blur(const Mat& src, Mat& dst){
		Size kernelSize = Size(ksize,ksize);
		blur(src,dst,kernelSize);
	}

	void gaussianBlur(const Mat& src, Mat& dst){
		Size kernelSize = Size(ksize,ksize);
		short stddev = 100;
		GaussianBlur(src,dst,kernelSize,stddev);
	}

	void medianBlur(const Mat& src, Mat& dst){
		medianBlur(src,dst,ksize);
	}

	void bilateralFilter(const Mat& src, Mat& dst){
		double sigmaColor = 75;
		double sigmaSpace = sigmaColor;
		bilateralFilter(src,dst,ksize,sigmaColor,sigmaSpace);
	}

	void nonLocalMeansDenoising(const Mat& src, Mat& dst){
		float h = 15.; //Filter sigma regulating filter strength for color.
		fastNlMeansDenoising(src,dst,h);
		// Alternatively temporal non-local means is available
	}

	// void zlokolica(const Mat& src, Mat& dst){
		// Size kernelSize = Size(ksize,ksize);

	// }

	void denoise(const Mat& src, Mat& dst, short type){
		switch(type){
			case DEN_BLUR : blur(src,dst); break;
			case DEN_GAUSS_BLUR : gaussianBlur(src,dst); break;
			case DEN_MEDIAN_BLUR : medianBlur(src,dst); break;
			case DEN_BILATERAL_FILTER : bilateralFilter(src,dst); break;
			case DEN_NL_MEANS : nonLocalMeansDenoising(src,dst); break;
			// case DEN_ZLOKOLICA : zlokolica(src,dst); break;
			default : std::cerr<<"Error selecting denoising method, invalid type: "<<type<<endl;exit(1);
		}
	}
}

#endif