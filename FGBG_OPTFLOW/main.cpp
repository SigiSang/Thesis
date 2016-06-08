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

typedef Vec<unsigned short,3> DataVec;

const int ESCAPE_NL = 537919515;
const int ESCAPE_EN = 1048603;
const double PI = 3.14159265358979323846;
const string dirOutput = "output";

template <typename T>
void printHistogram(ostream& os, vector<T>& hist, string name){
	os<<"--- Histogram for "<<name<<" ---"<<endl;
	for(int i=0;i<hist.size();i++){
		os<<hist[i]<<";";
		if( (i+1)%30==0 ) cout<<endl;
	}
	os<<endl;
}

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

void postProcessing(Mat& src, Mat& dst, int strucSize, int morphOp=MORPH_OPEN){
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
	morphologyEx(src,dst,morphOp,struc);
}

void addNoise(Mat& src, Mat& dst){
	Mat noise = Mat::zeros(src.rows,src.cols,src.type());
	int mean = 128;
	int stddev = 100;
	cv::randn(noise,mean,stddev);
	addWeighted(src,1,noise,0.3,0,dst);
}

void threshold(const Mat& img,Mat& dst,double threshval=-1){
	double maxval = 255;
	int type = CV_THRESH_BINARY;
	if(threshval==-1) type += CV_THRESH_OTSU;
	threshold(img,dst,threshval,maxval,type);
	// postProcessing(dst,dst,3);
}

void imageSmoothing(Mat& src, Mat& dst){
	int d = 9; // Neighbour max distance, 5 for real-time, 9 for offline with heavy noise
	double sigmaColor, sigmaSpace;
	sigmaColor = sigmaSpace = 50; // Small (<10) barely has effect, large (>150) has a very strong effect
	bilateralFilter(src,dst,d,sigmaColor,sigmaSpace);
	// adaptiveBilateralFilter(frame,dst,d,sigmaSpace);
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

void backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2){
	double learningRate = -1;
    pMOG2->apply(frame, fgMask,learningRate);
    // postProcessing(fgMask,fgMask,2);
}

void getOptFlowFeatures(const Mat& motionCompMask, const Mat& fgMask, Mat& combMask, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures=true){
	const int width = motionCompMask.cols, height = motionCompMask.rows;
	if(!goodFeaturesToTrack.empty()) goodFeaturesToTrack.clear();

	// Combine previous motion mask and current foreground mask
	fgMask.copyTo(combMask);
	// bitwise_or(motionCompMask,fgMask,combMask);

	// Add points from combined mask
    if(useForegroundFeatures)
    	binMat2Vec(combMask,goodFeaturesToTrack);

	 // If foreground consists of more than 80% of the images pixels, ignore it for being too much data, opencv function will be used instead.
	 if(goodFeaturesToTrack.size() > 0.8 * (width*height) )
	 	goodFeaturesToTrack.clear();
}

void opticalFlow(const Mat& prvFr, const Mat& nxtFr, Mat& optFlow, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask){
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21);
    const int width=prvFr.cols, height=prvFr.rows;
    const int blkSize = 32, blksInWidth=ceil((float)width/blkSize),blksInHeight=ceil((float)height/blkSize);

    vector<float> err;

    if(prvPts.empty()){
    	// cout<<"Fg Mask"<<endl;
    	goodFeaturesToTrack(prvFr, prvPts, 500, 0.01, 10, fgMask, 3, 0, 0.04);
    }
    if(prvPts.empty()){
    	// cout<<"Empty prvPts"<<endl;
    	goodFeaturesToTrack(prvFr, prvPts, 500, 0.01, 10, Mat(), 3, 0, 0.04);
    }
    calcOpticalFlowPyrLK(prvFr, nxtFr, prvPts, nxtPts, status, err, winSize, 3, termcrit, 0, 0.001);

    nxtFr.copyTo(optFlow);
    if(optFlow.channels()==1) cvtColor(optFlow,optFlow,CV_GRAY2BGR);
    Scalar red(0,0,255), blue(255,0,0), green(0,255,0);
    for(int i=0;i<nxtPts.size();i++){
    	if(status[i]==0) continue;
    	line(optFlow,prvPts[i],nxtPts[i],blue);
    	// circle(optFlow,prvPts[i],1,green);
    	circle(optFlow,nxtPts[i],1,red);
    }
}

void similarNeighbourWeights(const Mat& src, const Mat& data, Mat& weights){
	//Temps  for debugging
	int blkX = 50,blkY = 150;

	short r = 1; //Radius
	for(int x=0;x<src.cols;x++){
	for(int y=0;y<src.rows;y++){
		short neighbours = 0;
		// TODO
		/* Aanpassen naar deltaX en deltaY, als deze gelijk zijn is dat correcter dan hoek&&grootte.
		*  Zal nu wss exactere maatstaf zijn, intervallen rond delta's kan nuttig zijn
		*/
		int angle = data.at<DataVec>(y,x)[1];
		int size = data.at<DataVec>(y,x)[2];
		if(angle > 0 && size > 0){
			for(int i=-1; i<=r; i+=r){
			if(x+i>0 && x+i < src.cols){
				for(int j=-1; j<=r; j+=r){
				if(!(i==0 && j==0) && y+j>0 && y+j < src.rows){
					if(angle == data.at<DataVec>(y+j,x+i)[1]
						&&
						size == data.at<DataVec>(y+j,x+i)[2]){
					// if(angle == data.at<DataVec>(y+j,x+i)[1]){
					// if(size == data.at<DataVec>(y+j,x+i)[2]){
						neighbours++;
					}
				}
				}
			}
			}
		}
		// TODO weight veranderen naar float
		unsigned char weight = (1<<neighbours)-1;
		// unsigned char weight = neighbours*32;
		// if(neighbours!=0) weight--;
		// if(x>blkX && x<blkX+32 && y>blkY && y<blkY+32) cout<<neighbours<<","<<(int)weight<<endl;
		weights.at<unsigned char>(y,x) = weight;
	}
	}
}

void regularization(const Mat& src, Mat& dst, Mat& data, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights){

    data = Mat::zeros(src.size(),CV_16UC3);
	weights = Mat::zeros(src.size(),CV_8UC1);

    for(int ptsIdx=0;ptsIdx<status.size();ptsIdx++){
    	if(status[ptsIdx]){
			int x1=prvPts[ptsIdx].x,y1=prvPts[ptsIdx].y,x2=nxtPts[ptsIdx].x,y2=nxtPts[ptsIdx].y;
			double angle = ( atan2( (double)y1-y2, (double)x2-x1) )*180/PI;
			int size = sqrt( square(y1 - y2) + square(x2 - x1) );
			if(size>=500) continue;
			if(angle<0) angle += 360;
			data.at<DataVec>(y1,x1)[0] = ptsIdx;
			// TODO veranderen naar deltaX en deltaY
			data.at<DataVec>(y1,x1)[1] = (unsigned short)angle;
			data.at<DataVec>(y1,x1)[2] = (unsigned short)size;
    	}
    }
    similarNeighbourWeights(src,data,weights);
   	threshold(weights,dst,(1<<4)-1);
   	dst.copyTo(weights);
}

void expandByList(const Mat& src, const Mat& mask, Mat& weights, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, ushort angle, ushort size){
    const int width=mask.cols, height=mask.rows, r=1;
    const ushort x = size * cos(angle);
    const ushort y = size * sin(angle);
	for(int i=0; i<toExpand.size();i++){
    	int idx = toExpand[i];
    	if(!expanded[idx]){
    		expanded[idx] = true;
    		weights.at<unsigned char>(idx)=(int)-1; // Max value
    		// regulData.at<DataVec>(idx+y*width+x) = DataVec(0,angle,size);
			for(int i=-1; i<=r; i+=r){
			for(int j=-1; j<=r; j+=r){
				int idx2 = idx+j*width+i;
				if(!(i==0 && j==0) && idx2>0 && idx2 < width*height){
					if( (short)(mask.at<unsigned char>(idx))>0){
						toExpand.push_back(idx2);
					}
				}
			}
			}
    	}
	}
}

void expandRegions(const Mat& src, const Mat& mask, const Mat& data, Mat& regulData, Mat& weights){
    const int width=src.cols, height=src.rows;
    regulData = Mat::zeros(data.size(),data.type());
   	vector<bool> expanded(width*height,false);
    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	// TODO Change to threshold value variable
    	if((int)(weights.at<unsigned char>(y,x))>127){
    		int idx = y*width+x;
	    	if(!expanded[idx]){
				vector<int> toExpand;
				ushort angle = data.at<DataVec>(y,x)[1];
				ushort size = data.at<DataVec>(y,x)[2];

				toExpand.push_back(idx);
				expandByList(src,mask,weights,expanded,toExpand,regulData,angle,size);
	    	}
	    }
    }
    }
}

void morphologicalReconstruction(const Mat& mask, Mat& dst, Mat& weights){
    // TODO Change to threshold value variable
   	Mat dilation,maskedDilation,prevMaskedDilation;
   	bool hasChanged;
   	int strucSize = 5;
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
   	threshold(weights,maskedDilation,(1<<4)-1);
   	do{
   		maskedDilation.copyTo(prevMaskedDilation);
   		dilate(prevMaskedDilation,dilation,struc);
   		min(dilation,mask,maskedDilation);
   		// equal = (sum(maskedDilation!=dilation) == Scalar(0,0,0,0));
   		hasChanged = (countNonZero(maskedDilation!=prevMaskedDilation) != 0);
   	}while(hasChanged);
   	maskedDilation.copyTo(dst);
}

void motionDetection(Mat& prvFr, Mat& nxtFr, Mat& motionCompMask, Mat& motionMask, Ptr<BackgroundSubtractor> pMOG2, string name, bool showSrc=true, bool useForegroundFeatures=true, bool useMorphRec=true){
	bool secondIter = false;

	vector<uchar> status;
	vector<Point2f> prvPts,nxtPts;
	Mat data, regulData;
	Mat smoothing,fgMask,combMask,weights,optFlow1,optFlow2;

	// imageSmoothing(prvFr,smoothing);
	// backgroundSubtraction(smoothing,fgMask,pMOG2,prvPts,useForegroundFeatures);
	backgroundSubtraction(prvFr,fgMask,pMOG2);
	getOptFlowFeatures(motionCompMask,fgMask,combMask,prvPts,useForegroundFeatures);
	opticalFlow(prvFr,nxtFr,optFlow1,status,prvPts,nxtPts,fgMask);
	regularization(prvFr,motionMask,data,status,prvPts,nxtPts,weights);
   	// if(useMorphRec){
   	// 	morphologicalReconstruction(fgMask,motionMask,motionMask);
   	// }else{
   	// 	expandRegions(prvFr,fgMask,data,regulData,motionMask);
   	// }

   	// Morphological reconstruction
   	morphologicalReconstruction(fgMask,motionMask,motionMask);

   	// Morph + Exp
 //   	uchar threshold = 127;
 //   	Mat motMaskMorph,motMaskExp;
 //   	morphologicalReconstruction(fgMask,motMaskMorph,motionMask);
 //   	motionMask.copyTo(motMaskExp);
 //   	expandRegions(prvFr,fgMask,data,regulData,motMaskExp);
 //   	for(int x=0;x<motMaskMorph.cols;x++){
 //   	for(int y=0;y<motMaskMorph.rows;y++){
 //   		if(motMaskMorph.at<uchar>(y,x) > threshold
 //   			|| motMaskExp.at<uchar>(y,x) > threshold){
 //   			motionMask.at<uchar>(y,x) = -1;
 //   		}
	// }
 //   	}

	bool resize = true;
	if(showSrc) io::showImage(name+" Src",nxtFr,resize);
	// io::showImage(name+" Smoothing",smoothing,resize);
	io::showImage(name+" Foreground Mask",fgMask,resize);
	// io::showImage(name+" Comb Mask",combMask,resize);
	// io::showImage(name+" OptFlow1",optFlow1,resize);
	io::showImage(name+" Weights",weights,resize);
	io::showImage(name+" Morph Rec",motMaskMorph,resize);
	io::showImage(name+" Expansion",motMaskExp,resize);
	if(secondIter) io::showImage(name+" OptFlow2",optFlow2,resize);
	io::showImage(name+" MotionMask",motionMask,resize);
}

int main (int argc, char** argv){
	int keyboard = -1;

	// string dirInput = "input/streetcorner/";
	string dirInput = "input/tramstation/";
	// string dirInput = "input/blizzard/";
	// string dirInput = "input/pedestrians/";
	string regex = "%06d.jpg";

	// BGS Configuration
	int history = 500;
	double varThreshold = 16;
	bool detectShadows=false;

	Mat prvFr,nxtFr,motionCompMask,motionMask;
	Mat prvFrNoisy,nxtFrNoisy,motionCompMaskNoisy,motionMaskNoisy;

	VideoCapture cap(dirInput+regex);
	if(cap.isOpened() && cap.read(prvFr)){
		cvtColor(prvFr,prvFr,CV_BGR2GRAY);
		Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
		Ptr<BackgroundSubtractor> pMOG2Noisy = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
		// Ptr<BackgroundSubtractor> pMOG2Exp = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
		// Ptr<BackgroundSubtractor> pMOG2NoisyExp = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
		motionCompMask = Mat::zeros(prvFr.size(),CV_8UC1);
		motionCompMaskNoisy = Mat::zeros(prvFr.size(),CV_8UC1);
		while(cap.read(nxtFr)){
			string name="Original",nameNoisy="Noisy";
			if(nxtFr.channels() > 1) cvtColor(nxtFr,nxtFr,CV_BGR2GRAY);
			addNoise(prvFr,prvFrNoisy);
			addNoise(nxtFr,nxtFrNoisy);

			motionDetection(prvFr,nxtFr,motionCompMask,motionMask,pMOG2,name+" Morph Rec");
			// motionDetection(prvFr,nxtFr,motionCompMask,motionMask,pMOG2Exp,name+" Expansion",true,true,false);
			motionDetection(prvFrNoisy,nxtFrNoisy,motionCompMaskNoisy,motionMaskNoisy,pMOG2Noisy,nameNoisy+" Morph Rec");
			// motionDetection(prvFrNoisy,nxtFrNoisy,motionCompMaskNoisy,motionMaskNoisy,pMOG2NoisyExp,nameNoisy+" Expansion",true,true,false);

			if((keyboard=waitKey(0)) == ESCAPE_NL || keyboard == ESCAPE_EN)
				break;
			nxtFr.copyTo(prvFr);
		}
	}else{
		cerr<<"Problem reading dataset."<<endl;
	}

	destroyAllWindows();
	return 0;
}