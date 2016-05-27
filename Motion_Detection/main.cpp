#include <string>
#include <iostream>
#include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/ml.hpp>
#include "Headers/io.hpp"

using namespace std;
using namespace cv;
// using namespace cv::ml;

void postProcessing(Mat& src, Mat& dst,int strucSize){
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
	morphologyEx(src,dst,MORPH_OPEN,struc);
}

void addNoise(Mat& src, Mat& dst){
	Mat noise = Mat::zeros(src.rows,src.cols,src.type());
	int mean = 128;
	int stddev = 100;
	cv::randn(noise,mean,stddev);
	addWeighted(src,1,noise,0.1,0,dst);
}

void otsuThreshold(Mat& img,Mat& dst){
	double threshval = 0;
	double maxval = 255;
	int type = CV_THRESH_BINARY + CV_THRESH_OTSU;
	threshold(img,dst,threshval,maxval,type);
	// postProcessing(dst,dst,3);
}

void gradient(Mat& src, Mat& dst, bool filterGaussian=false){
	src.copyTo(dst);

	if(filterGaussian){
		int gkSize = 7;
		double sigma = -1;
		Mat gausKern = getGaussianKernel(gkSize,sigma);
		filter2D(dst,dst,-1,gausKern);
	}

	Mat gx,gy;
	int sobelSize = 3;
	Sobel(dst,gx,-1,1,0,sobelSize);
	Sobel(dst,gy,-1,0,1,sobelSize);
	dst = gx + gy;
}

void opticalFlow(Mat& src1, Mat& src2, Mat& optFlow, vector<Point2f>& prevPts, vector<Point2f>& nxtPts){
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(31,31);

    vector<uchar> status;
    vector<float> err;

    calcOpticalFlowPyrLK(src1, src2, prevPts, nxtPts, status, err, winSize, 3, termcrit, 0, 0.001);

    src2.copyTo(optFlow);
    if(optFlow.channels()==1) cvtColor(optFlow,optFlow,CV_GRAY2BGR);
    Scalar red(0,0,255), blue(255,0,0), green(0,255,0);
    for(int i=0;i<nxtPts.size();i++){
    	line(optFlow,prevPts[i],nxtPts[i],blue);
    	circle(optFlow,prevPts[i],1,green);
    	circle(optFlow,nxtPts[i],1,red);
    }

    // src2.copyTo(optFlow);
    // cvtColor(optFlow,optFlow,CV_GRAY2BGR);

    // // Convert to YUV heatmap
    // vector<Mat> yuv(3);
    // src2.copyTo(yuv[0]);
    // merge(&yuv[0],yuv.size(),optFlow);

}

void motionDetection(Mat& src1, Mat& src2, Mat& motMask, string name, bool showSrc=true){
	Mat srcDiff,srcThresh,grad1,grad2,gradDiff,gradThresh;

	string srcThreshStr = " src diff + thresh";
	srcDiff = src1-src2;
	otsuThreshold(srcDiff,srcThresh);

	string grad1Str=" grad 1",grad2Str=" grad 2",gradDiffStr=" grad diff";
	gradient(src1,grad1);
	gradient(src2,grad2);
	gradDiff = grad1-grad2;

	string gradThreshStr = " grad diff + thresh";
	otsuThreshold(gradDiff,gradThresh);

	motMask = srcThresh + gradThresh;

	string motMaskStr = " post processing";
	int strucSize = 5;
	postProcessing(motMask,motMask,strucSize);

	string optFlowStr = " optical flow";
	string preOptFlowStr = " pre optical flow";
	vector<Point2f> points[2];
	Mat preOptFlow,optFlow;
	preOptFlow = Mat(srcThresh);
	postProcessing(srcThresh,preOptFlow,5);
	// gradient(preOptFlow,preOptFlow);
	for(int i=0;i<preOptFlow.rows;i++){
		for(int j=0;j<preOptFlow.cols;j++){
			if(preOptFlow.at<unsigned char>(i,j)!=0)
				points[0].push_back(Point2f(j,i));
		}
	}
	opticalFlow(src1,src2,optFlow,points[0],points[1]);
	// opticalFlow(grad1,grad2,optFlow,points[0],points[1]);

	if(showSrc) io::showImage(name,src1,true);
	io::showImage(name+srcThreshStr, srcThresh,true);
	// io::showImage(name+grad1Str, grad1,true);
	// io::showImage(name+grad2Str, grad2,true);
	// io::showImage(name+gradDiffStr, gradDiff,true);
	// io::showImage(name+gradThreshStr, gradThresh,true);
	// io::showImage(name+motMaskStr, motMask,true);
 //    io::showImage(name+preOptFlowStr,preOptFlow,true);
    io::showImage(name+optFlowStr,optFlow,true);

	io::saveImage(name,src1);
	io::saveImage(name+" 1 "+srcThreshStr, srcThresh);
	// io::saveImage(name+" 2 "+grad1Str, grad1);
	// io::saveImage(name+" 3 "+grad2Str, grad2);
	// io::saveImage(name+" 4 "+gradDiffStr, gradDiff);
	// io::saveImage(name+" 5 "+gradThreshStr, gradThresh);
	// io::saveImage(name+" 6 "+motMaskStr, motMask);
    io::saveImage(name+" 7 "+optFlowStr,optFlow);
}

int main (int argc, char** argv){
	io::checkDir(io::dirOutput);
	
	vector< string* > imgSets;
	string street_corner_1[] = {"Street corner 1","in000080.jpg","in000081.jpg"};
	string street_corner_2[] = {"Street corner 2","in000970.jpg","in000971.jpg"};
	string pedestrians[] = {"Pedestrians","in000500.jpg","in000501.jpg"};
	string fluid_highway[] = {"Fluid highway","in000460.jpg","in000461.jpg"};

	// imgSets.push_back(street_corner_1);
	// imgSets.push_back(street_corner_2);
	// imgSets.push_back(pedestrians);
	imgSets.push_back(fluid_highway);

	for(int i=0;i<imgSets.size();i++){
		string name = imgSets[i][0];
		string fnImg1 = imgSets[i][1];
		string fnImg2 = imgSets[i][2];

		Mat img1,img2,img1Noisy,img2Noisy;
		img1 = imread(fnImg1,CV_LOAD_IMAGE_GRAYSCALE);
		img2 = imread(fnImg2,CV_LOAD_IMAGE_GRAYSCALE);

		Mat motMask;
		motionDetection(img1,img2,motMask,name);

		addNoise(img1,img1Noisy);
		addNoise(img2,img2Noisy);

		motionDetection(img1Noisy,img2Noisy,motMask,name+" noisy",true);
	}

	waitKey(0);
	destroyAllWindows();
	return 0;
}