#ifndef _IMAGE_PROCESSING
#define _IMAGE_PROCESSING

#include <iostream>
using std::endl;
using std::ostream;
using std::stringstream;
#include <cmath>
#include <vector>
using std::vector;
#include <string>
using std::string;
#include <opencv2/opencv.hpp>
using namespace cv;

typedef Vec<int,4> Point4i;

const double NOISE_STDDEV_MIN = 0.0;
const double NOISE_STDDEV_MAX = 30.0;
const double NOISE_STDDEV_INC = 10.0;

/* Unused */
template <typename T>
void printHistogram(ostream& os, vector<T>& hist, string name){
	os<<"--- Histogram for "<<name<<" ---"<<endl;
	for(int i=0;i<hist.size();i++){
		os<<hist[i]<<";";
		if( (i+1)%30==0 ) os<<endl;
	}
	os<<endl;
}

/*
* Creates histogram for src
*/
template <typename _Tp>
void _histogram(const Mat& src, vector<int>& hist, int numBins, int maxVal) {
    hist = vector<int>(numBins);
    int binWidth = maxVal/numBins;
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            int bin = src.at<_Tp>(i,j)/binWidth;
            hist[bin] += 1;
        }
    }
}

/* Unused */
/* Wrapper function for histogram_ */
/* Based on code from https://github.com/bytefish/opencv/tree/master/lbp */
void histogram(const Mat& src, vector<int>& hist, int numBins, int maxVal) {
	const int CHAR 		= CV_8SC(src.channels());
	const int UCHAR 	= CV_8UC(src.channels());
	const int SHORT 	= CV_16SC(src.channels());
	const int USHORT 	= CV_16UC(src.channels());
	const int INT 		= CV_32SC(src.channels());
	const int FLOAT 	= CV_32FC(src.channels());
	const int DOUBLE 	= CV_64FC(src.channels());
	if(src.type()==CHAR){
		_histogram<char>(src, hist, numBins, maxVal);
	}else if(src.type()==UCHAR){
		_histogram<unsigned char>(src, hist, numBins, maxVal);
	}else if(src.type()==SHORT){
		_histogram<short>(src, hist, numBins, maxVal);
	}else if(src.type()==USHORT){
		_histogram<unsigned short>(src, hist, numBins, maxVal);
	}else if(src.type()==INT){
		_histogram<int>(src, hist, numBins, maxVal);
	}else if(src.type()==FLOAT){
		_histogram<float>(src, hist, numBins, maxVal);
	}else if(src.type()==DOUBLE){
		_histogram<double>(src, hist, numBins, maxVal);
	}else{
		cerr<<"Error in histogram : convert src to grayscale with valid type"<<endl;throw;
	}
}

void postProcessing(Mat& src, Mat& dst, int strucSize, int morphOp=MORPH_OPEN){
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
	morphologyEx(src,dst,morphOp,struc);
}

typedef Vec<short,3> Vec3;
void addNoise(Mat& src, Mat& dst, double stddev=30.0){
	int chan = src.channels();
	int typeNoise = CV_16SC(chan);

	Mat noise = Mat::zeros(src.rows,src.cols,typeNoise);
	double mean = 0.0;
	randn(noise,Scalar::all(mean),Scalar::all(stddev));

	Mat src_16S;
	src.convertTo(src_16S,typeNoise);
	addWeighted(src_16S,1.0,noise,1.0,0,src_16S);

	src_16S.convertTo(dst,src.type());
}

void threshold(const Mat& src,Mat& dst,double threshval=-1){
	double maxval = 255;
	int type = CV_THRESH_BINARY;
	if(threshval==-1) type += CV_THRESH_OTSU;
	threshold(src,dst,threshval,maxval,type);
}

/* Unused */
void imageSmoothing(Mat& src, Mat& dst){
	int d = 9; // Neighbour max distance, 5 for real-time, 9 for offline with heavy noise
	double sigmaColor, sigmaSpace;
	sigmaColor = sigmaSpace = 50; // Small (<10) barely has effect, large (>150) has a very strong effect
	bilateralFilter(src,dst,d,sigmaColor,sigmaSpace);
}

void binMat2Vec(const Mat& src, vector<Point2f>& pts){
    for(int i=0;i<src.rows;i++){
    	for(int j=0;j<src.cols;j++){
    		if(src.at<unsigned char>(i,j) > 0){
    			pts.push_back(Point2f(j,i));
    		}
    	}
    }
}

void boundingBoxes(const Mat& src, Mat& dst){
	Mat labels;
	int n = connectedComponents(src,labels);
	int width=src.cols,height=src.rows;
	vector<Point4i> cBBC(n,Point4i(-1,-1,-1,-1)); // component bounding boxes coordinates, (x,y) for upper left corner (smallest x and y) and (x,y) for lower right corner (largest x and y)
	for(int x=0;x<width;x++){
	for(int y=0;y<height;y++){
		int label = labels.at<int>(y,x);
		Point4i pt = cBBC[label];
		if(pt[0]==-1){ // First occurrence of label
			pt[0] = pt[2] = x;
			pt[1] = pt[3] = y;
		}else{
					if(x < pt[0]) pt[0] = x;
			else 	if(x > pt[2]) pt[2] = x;
					if(y < pt[1]) pt[1] = y;
			else 	if(y > pt[3]) pt[3] = y;
		}
		cBBC[label] = pt;
	}
	}
	src.copyTo(dst);
	vector<Mat> mv(3,dst);
	merge(mv,dst);
	Scalar red(0,0,255);
	for(int i=0;i<n;i++){
		Point4i pt = cBBC[i];
		rectangle(dst,Point(pt[0],pt[1]),Point(pt[2],pt[3]),red);
	}
	stringstream ss;
	ss<<n;
	putText(dst,ss.str(),Point(30,30),FONT_HERSHEY_SIMPLEX,.75,red);
}

float vectorLengthSquared(float deltaX, float deltaY){
	return deltaX*deltaX + deltaY*deltaY;
}

float vectorLength(float deltaX, float deltaY){
	return sqrt(vectorLengthSquared(deltaX,deltaY));
}

// Checks radii for one quarter of the circle, sets values for corresponding pixels in other quarters
void getCircularStructuringElement(short radius, Mat& struc){
	short size = 2*radius+1;
	struc = Mat::zeros(Size(size,size),CV_8UC1);
	int cX = radius, cY = radius; // center pixel index
	int rSq = radius*radius;
	for(int i=0;i<=radius;i++){
		for(int j=0;j<=radius;j++){
			int sqSum = i*i+j*j;
			if(sqSum <= rSq){
				struc.at<uchar>(cY+j,cX+i) = 1;
				struc.at<uchar>(cY-j,cX+i) = 1;
				struc.at<uchar>(cY+j,cX-i) = 1;
				struc.at<uchar>(cY-j,cX-i) = 1;
			}
		}
	}
}

bool hasHighValue(const Mat& bin){
	for(int i=0;i<bin.total();i++){
		if(bin.at<uchar>(i) > 0) return true;
	}
	return false;
}

#endif