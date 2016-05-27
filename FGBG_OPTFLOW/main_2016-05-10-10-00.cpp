#include <string>
#include <iostream>
#include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/ml.hpp>
#include "Headers/io.hpp"
#include <opencv2/video.hpp>
#include <cmath>

using namespace std;
using namespace cv;
// using namespace cv::ml;

const int ESCAPE_NL = 537919515;
const int ESCAPE_EN = 1048603;

const string dirOutput = "output";

// class motionVector{
// public:
// 	motionVector(){}

// 	motionVector(int _x1,int _y1,int _x2,int _y2):x1(_x1),y1(_y1),x2(_x2),y2(_y2){
// 		calc();
// 	}

// 	motionVector(Point2f _p1,Point2f _p2):x1(_p1.x),y1(_p1.y),x2(_p2.x),y2(_p2.y){
// 		calc();
// 	}

// 	int square(int x){
// 		return x*x;
// 	}
// 	void calc(){
// 		angle = atan2( (double) y1 - y2, (double) x2 - x1 );
//  		size = sqrt( square(y1 - y2) + square(x2 - x1) );
// 	}
// private:
// 	int x1,y1,x2,y2;
// 	double angle,size;
// };

int square(int x){
	return x*x;
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

void postProcessing(Mat& src, Mat& dst,int strucSize){
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
	morphologyEx(src,dst,MORPH_OPEN,struc);
}

void addNoise(Mat& src, Mat& dst){
	Mat noise = Mat::zeros(src.rows,src.cols,src.type());
	int mean = 128;
	int stddev = 100;
	cv::randn(noise,mean,stddev);
	addWeighted(src,1,noise,0.3,0,dst);
}

void otsuThreshold(Mat& img,Mat& dst){
	double threshval = 0;
	double maxval = 255;
	int type = CV_THRESH_BINARY + CV_THRESH_OTSU;
	threshold(img,dst,threshval,maxval,type);
	// postProcessing(dst,dst,3);
}

void backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures=true){
	double learningRate = -1;
    pMOG2->apply(frame, fgMask,learningRate);
    // postProcessing(fgMask,fgMask,2);

    if(!goodFeaturesToTrack.empty()) goodFeaturesToTrack.clear();
    if(useForegroundFeatures)
	    for(int i=0;i<frame.rows;i++){
	    	for(int j=0;j<frame.cols;j++){
	    		if(fgMask.at<unsigned char>(i,j) > 0){
	    			goodFeaturesToTrack.push_back(Point2f(j,i));
	    		}
	    	}
	    }

	 // If foreground consists of more than 80% of the images pixels, ignore it for being too much data.
	 if(goodFeaturesToTrack.size() > 0.8 * (frame.cols*frame.rows) )
	 	goodFeaturesToTrack.clear();
}

struct motionVector{
	int x1,y1,x2,y2;
	double angle,size;
	motionVector(int _x1,int _y1,int _x2,int _y2):x1(_x1),y1(_y1),x2(_x2),y2(_y2){
		angle = atan2( (double) y1 - y2, (double) x2 - x1 );
 		size = sqrt( (y1 - y2)*(y1 - y2) + (x2 - x1)*(x2 - x1) );
	}
	motionVector(Point2f _p1,Point2f _p2):x1(_p1.x),y1(_p1.y),x2(_p2.x),y2(_p2.y){
		angle = atan2( (double) y1 - y2, (double) x2 - x1 );
 		size = sqrt( (y1 - y2)*(y1 - y2) + (x2 - x1)*(x2 - x1) );
	}
};

// void printHistogram(ostream& os, vector<int>* hist, string name){
template <typename T>
void printHistogram(ostream& os, vector<T>& hist, string name){
	os<<"--- Histogram for "<<name<<" ---"<<endl;
	for(int i=0;i<hist.size();i++){
		os<<hist[i]<<";";
		if( (i+1)%30==0 ) cout<<endl;
	}
	os<<endl;
}

void opticalFlow(const Mat& src1, const Mat& src2, Mat& optFlow, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask){
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21);

    vector<uchar> status;
    vector<float> err;

    if(prvPts.empty()){
    	// cout<<"Fg Mask"<<endl;
    	goodFeaturesToTrack(src1, prvPts, 500, 0.01, 10, fgMask, 3, 0, 0.04);
    }
    if(prvPts.empty()){
    	// cout<<"Empty prvPts"<<endl;
    	goodFeaturesToTrack(src1, prvPts, 500, 0.01, 10, Mat(), 3, 0, 0.04);
    }
    calcOpticalFlowPyrLK(src1, src2, prvPts, nxtPts, status, err, winSize, 3, termcrit, 0, 0.001);

    src2.copyTo(optFlow);
    if(optFlow.channels()==1) cvtColor(optFlow,optFlow,CV_GRAY2BGR);
    Scalar red(0,0,255), blue(255,0,0), green(0,255,0);
    for(int i=0;i<nxtPts.size();i++){
    	if(status[i]==0) continue;
    	line(optFlow,prvPts[i],nxtPts[i],blue);
    	// circle(optFlow,prvPts[i],1,green);
    	circle(optFlow,nxtPts[i],1,red);
    }

    const int width=src1.cols, height=src1.rows;
    const int blkSize = 32, blksInWidth=width/blkSize,blksInHeight=height/blkSize;

    // vector<int>* grid[width][height];
    vector<int> grid[blksInWidth][blksInHeight];

    for(int i=0;i<status.size();i++){
    	if(status[i]){
    		Point2f p1=prvPts[i];//, p2=nxtPts[i];
    		int blkX = p1.x/blkSize, blkY = p1.y/blkSize;
    		// vector<int>* blkList = grid[blkX][blkY];
    		// if (blkList == NULL) blkList = new vector<int>();
    		// cout<<blkList->empty()<<endl;
    		// blkList->push_back(i);
    		if(grid[blkX][blkY].empty()) grid[blkX][blkY] = vector<int>(blkSize*blkSize,0);
    		grid[blkX][blkY].push_back(i);
    	}
    }

    // Mat* hists[width][height];
    for(int blkX=0;blkX<width;blkX+=blkSize){
    for(int blkY=0;blkY<height;blkY+=blkSize){
    	vector<int> blkList = grid[blkX][blkY];
    	cout<<"("<<blkX<<","<<blkY<<") "<<endl;
    	cout<<!blkList.empty()<<endl;
    	if(!blkList.empty()){
    		// hists[blkX][blkY] = new Mat(blkSize,blkSize,CV_16UC2);
    		// Mat angles(blkSize,blkSize,CV_16UC1);
    		// Mat sizes(blkSize,blkSize,CV_32SC1);
    		vector<int> hist_angle(360,0),hist_size(500,0);
    		cout<<"blkList size "<<blkList.size()<<", "<<blkSize*blkSize<<endl;
    		if(blkList.size()<blkSize*blkSize){
    			cout<<"check"<<endl;
		    	for(int mV=0;mV<blkList.size();mV++){
		    		cout<<"x1 "<<endl;
		    		int x1=prvPts[mV].x,y1=prvPts[mV].y,x2=nxtPts[mV].x,y2=nxtPts[mV].y;
		    		unsigned short angle = atan2( (double) y1 - y2, (double) x2 - x1 );
		 			int size = sqrt( square(y1 - y2) + square(x2 - x1) );
		 			// int histX = x1-blkX*blkSize;
		 			// int histY = y1-blkX*blkSize;
		 			// angles.at<usigned short>(histY,histX)=angle;
		 			// sizes.at<int>(histY,histX)=size;
		 			cout<<blkList.size()<<" "<<mV<<" "<<angle<<" "<<size<<endl;
		 			hist_angle[angle]++;
		 			if(size>=0 && size<500) hist_size[size]++;
		    	}
		    	// delete hists[blkX][blkY];
		    	printHistogram(cout,hist_angle,"angle");
		    	printHistogram(cout,hist_size,"size");
		    }
    	}
    }
    }
 //    for(int i=0li<width;i++){
 //    for(int j=0;j<height;j++){
 //    	vector<int>* blkList = grid[blkX][blkY];
 //    	if(blkList != NULL) delete blkList;
 //    }
	// }
}

void regularization_block(Mat& src, string name){

}

void regularization(vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& src, string name){

}

void motionDetection(Mat& prvFr, Mat& nxtFr, Ptr<BackgroundSubtractor> pMOG2, string name, bool showSrc=true, bool useForegroundFeatures=true){
	vector<Point2f> prvPts,nxtPts;
	Mat optFlow,fgMask;

	if(nxtFr.channels() > 1) cvtColor(nxtFr,nxtFr,CV_BGR2GRAY);
	backgroundSubtraction(prvFr,fgMask,pMOG2,prvPts,useForegroundFeatures);
	opticalFlow(prvFr,nxtFr,optFlow,prvPts,nxtPts,fgMask);
	// regularization();

	if(showSrc) io::showImage(name+" Src",nxtFr);
	io::showImage(name+" Foreground Mask",fgMask);
	io::showImage(name+" OptFlow",optFlow);

	nxtFr.copyTo(prvFr);
}

int main (int argc, char** argv){
	int keyboard = -1;

	// string dirInput = "input/streetcorner/";
	string dirInput = "input/tramstation/";
	string regex = "in%06d.jpg";

	// BGS Configuration
	int history = 500;
	double varThreshold = 16;
	bool detectShadows=false;

	Mat prvFr,nxtFr,prvFrNoisy,nxtFrNoisy;

	VideoCapture cap(dirInput+regex);
	if(cap.isOpened() && cap.read(prvFr)){
		cvtColor(prvFr,prvFr,CV_BGR2GRAY);
		Ptr<BackgroundSubtractor> pMOG2Orig = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
		Ptr<BackgroundSubtractor> pMOG2Noisy = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);

		// Ptr<DenseOpticalFlow> denOptFlo = createOptFlow_DualTVL1();

		while(cap.read(nxtFr)){
			string nameOrig="Original",nameNoisy="Noisy";
			// addNoise(prvFr,prvFrNoisy);
			// addNoise(nxtFr,nxtFrNoisy);

			// motionDetection(prvFr,nxtFr,pMOG2Orig,nameOrig,true,false);
			motionDetection(prvFr,nxtFr,pMOG2Orig,nameOrig+" FGFeat");
			// motionDetection(prvFrNoisy,nxtFrNoisy,pMOG2Noisy,nameNoisy,true,false);
			// motionDetection(prvFrNoisy,nxtFrNoisy,pMOG2Noisy,nameNoisy+" FGFeat");

			if((keyboard=waitKey(0)) == ESCAPE_NL || keyboard == ESCAPE_EN)
				break;
		}
	}else{
		cerr<<"Problem reading dataset."<<endl;
	}

	destroyAllWindows();
	return 0;
}