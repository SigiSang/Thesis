#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "Headers/io.hpp"
#include <opencv2/video.hpp>
#include <cmath>

using namespace std;
using namespace cv;
// using namespace cv::ml;

typedef Vec<short,3> DataVec;

const int ESCAPE_NL = 537919515;
const int ESCAPE_EN = 1048603;
const string dirOutput = "output";

/* Unused */
template <typename T>
void printHistogram(ostream& os, vector<T>& hist, string name){
	os<<"--- Histogram for "<<name<<" ---"<<endl;
	for(int i=0;i<hist.size();i++){
		os<<hist[i]<<";";
		if( (i+1)%30==0 ) cout<<endl;
	}
	os<<endl;
}

void stop(){
	destroyAllWindows();
	exit(0);
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

void addNoise(Mat& src, Mat& dst){
	Mat noise = Mat::zeros(src.rows,src.cols,src.type());
	int mean = 128;
	int stddev = 100;
	cv::randn(noise,mean,stddev);
	addWeighted(src,1,noise,0.3,0,dst);
}

void threshold(const Mat& src,Mat& dst,double threshval=-1){
	double maxval = 255;
	int type = CV_THRESH_BINARY;
	if(threshval==-1) type += CV_THRESH_OTSU;
	threshold(src,dst,threshval,maxval,type);
	// postProcessing(dst,dst,3);
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

void checkRegulData(const Mat& mask, const Mat& regulData){
	Mat regulMask = Mat::zeros(mask.size(),mask.type()); // will get high values if pixel is sourcepoint for motion vector in regulData
    const int width=mask.cols, height=mask.rows;
    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	DataVec dv = regulData.at<DataVec>(y*width+x); // relative motion vector representation
    	if(dv[1]!=0 || dv[2]!=0){ // relative motion vector is non-zero
    		regulMask.at<uchar>(y,x) = -1;
    	}
    }
    }
    cout<<"Mask vs RegulMask ";
    io::calculateScores(mask,regulMask);
    io::showMaskOverlap(mask,"Mask",regulMask,"RegulMask");
}

void backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2){
	double learningRate = -1;
    pMOG2->apply(frame, fgMask,learningRate);
}

// TODO remove combinedMask as parameter, not needed as output parameter. Placeholder for data, transferred to goodFeaturesToTrack
void getOptFlowFeatures(const Mat& motCompMask, const Mat& fgMask, Mat& combinedMask, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures=true){
	const int width = motCompMask.cols, height = motCompMask.rows;
	if(!goodFeaturesToTrack.empty()) goodFeaturesToTrack.clear();

	// Combine previous motion mask and current foreground mask
	fgMask.copyTo(combinedMask);
	bitwise_or(motCompMask,fgMask,combinedMask);

	// Add points from combined mask
    if(useForegroundFeatures)
    	binMat2Vec(combinedMask,goodFeaturesToTrack);

	 // If foreground consists of more than 80% of the images pixels, ignore it for being too much data, opencv function will be used instead.
	 if(goodFeaturesToTrack.size() > 0.8 * (width*height) )
	 	goodFeaturesToTrack.clear();
}

// TODO When performing optical flow on the entire frames (all pixels) instead of goodFeatureToTrack, consider using a larger winSize
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

void similarNeighbourWeighting(const Mat& data, Mat& weights){
	short r = 1; //Radius
	// windowWidth = 2r+1
	// maxNeighbours = windowWidth²-1 = 4r²+4r = 4r(r+1)
	// maxWeight = 2^(maxNeighbours) - 1
	float maxWeight = (1<<(4*r*(r+1)))-1;
	for(int x=0;x<data.cols;x++){
	for(int y=0;y<data.rows;y++){
		short neighbours = 0;
		short deltaX = data.at<DataVec>(y,x)[1];
		short deltaY = data.at<DataVec>(y,x)[2];
		if(deltaX != 0 && deltaY != 0){
			for(int i=-1; i<=r; i+=r){
			if(x+i>0 && x+i < data.cols){
				for(int j=-1; j<=r; j+=r){
				if(!(i==0 && j==0) && y+j>0 && y+j < data.rows){
					if(deltaX == data.at<DataVec>(y+j,x+i)[1]
						&&
						deltaY == data.at<DataVec>(y+j,x+i)[2]){
						neighbours++;
					}
				}
				}
			}
			}
		}
		float weight = ((1<<neighbours)-1)/maxWeight;
		weights.at<float>(y,x) = weight;
	}
	}
}

void optFlowRegularization(const Size& size, Mat& dst, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& data){

    data = Mat::zeros(size,CV_16UC3);
	weights = Mat::zeros(size,CV_32FC1);

	// Optical flow to data
    for(int ptsIdx=0;ptsIdx<status.size();ptsIdx++){
    	if(status[ptsIdx]){
			int x1=prvPts[ptsIdx].x,y1=prvPts[ptsIdx].y,x2=nxtPts[ptsIdx].x,y2=nxtPts[ptsIdx].y;
			short deltaX = x2-x1;
			short deltaY = y2-y1;
			// if(deltaX != 0 && deltaY != 0)
			// 	cout<<deltaX<<","<<deltaY<<","<<x2<<","<<x1<<","<<y2<<","<<y1<<endl;
			data.at<DataVec>(y1,x1)[0] = ptsIdx;
			data.at<DataVec>(y1,x1)[1] = deltaX;
			data.at<DataVec>(y1,x1)[2] = deltaY;
    	}
    }

    // Weighting for regularization
	similarNeighbourWeighting(data,weights);

    dst = Mat::ones(size,CV_8UC1);
    weights.convertTo(dst,CV_8UC1,255);
   	threshold(dst,dst,127);
}

void expandByList(const Mat& mask, Mat& marker, vector<bool>& expanded, vector<int> toExpand, ushort deltaX, ushort deltaY){
    const int width=mask.cols, height=mask.rows, r=2;
	for(int i=0; i<toExpand.size();i++){
    	int idx = toExpand[i];
    	if(!expanded[idx]){
    		expanded[idx] = true;
    		marker.at<uchar>(idx)=(uchar)(-1); // Max value
			for(int i=-r; i<=r; i++){
			for(int j=-r; j<=r; j++){
				int idxNb = idx+j*width+i;
				if(idxNb != idx && idxNb>0 && idxNb < width*height){
					if( (short)(mask.at<uchar>(idxNb))>0){
						toExpand.push_back(idxNb);
					}
				}
			}
			}
    	}
	}
}

void expandMarker(const Mat& mask, const Mat& data, Mat& marker){
    const int width=mask.cols, height=mask.rows;
   	vector<bool> expanded(width*height,false);
    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	if((int)(marker.at<uchar>(y,x))>127){
    		int idx = y*width+x;
	    	if(!expanded[idx]){
				vector<int> toExpand;
				short deltaX = data.at<DataVec>(y,x)[1];
				short deltaY = data.at<DataVec>(y,x)[2];

				toExpand.push_back(idx);
				expandByList(mask,marker,expanded,toExpand,deltaX,deltaY);
	    	}
	    }
    }
    }
}

void morphologicalReconstruction(Mat& dst, const Mat& mask, const Mat& marker){
   	Mat dilation,maskedDilation,prevMaskedDilation;
   	bool hasChanged;
   	int strucSize = 7;
   	int numOfIterations = 0;
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
    // TODO Change threshold value to variable
   	threshold(marker,maskedDilation,127);
   	do{
   		numOfIterations++;
   		maskedDilation.copyTo(prevMaskedDilation);
   		dilate(prevMaskedDilation,dilation,struc);
   		min(dilation,mask,maskedDilation);
   		hasChanged = (countNonZero(maskedDilation!=prevMaskedDilation) != 0);
   	}while(hasChanged);
   	// cout<<"Morph Reconstr Iterations: "<<numOfIterations<<endl;
   	maskedDilation.copyTo(dst);
}

void regularizeDataByList(const Mat& mask, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, short deltaX, short deltaY){
    const int width=mask.cols, height=mask.rows, r=1;
	for(int i=0; i<toExpand.size();i++){
    	int idx = toExpand[i];
    	if(!expanded[idx]){
    		expanded[idx] = true;
    		regulData.at<DataVec>(idx) = DataVec(0,deltaX,deltaY);
			for(int i=-r; i<=r; i++){
			for(int j=-r; j<=r; j++){
				int idxNb = idx+j*width+i;
				if(idxNb != idx && idxNb>0 && idxNb < width*height){
					if( (short)(mask.at<uchar>(idxNb))>0 ){
						toExpand.push_back(idxNb);
					}
				}
			}
			}
    	}
	}
}

void regularizeData(const Mat& marker, const Mat& mask, const Mat& data, Mat& regulData){
    const int width=marker.cols, height=marker.rows;
    regulData = Mat::zeros(data.size(),data.type());
   	vector<bool> expanded(width*height,false);
    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	if((short)(marker.at<uchar>(y,x))>0){
    		int idx = y*width+x;
	    	if(!expanded[idx]){
				vector<int> toExpand;
				short deltaX = data.at<DataVec>(y,x)[1];
				short deltaY = data.at<DataVec>(y,x)[2];

				toExpand.push_back(idx);
				regularizeDataByList(mask,expanded,toExpand,regulData,deltaX,deltaY);
	    	}
	    }
    }
    }
    // checkRegulData(mask,regulData);
}

void applyMotion(const Mat& src, const Mat& regulData, Mat& dst){
    const int width=src.cols, height=src.rows;
    dst = Mat::zeros(src.size(),src.type());

    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	if((short)(src.at<uchar>(y,x))>0){
			short newX = x + regulData.at<DataVec>(y,x)[1];
			short newY = y + regulData.at<DataVec>(y,x)[2];
			if(newX >= 0 && newY >= 0 && newX < width && newY < height)
				dst.at<uchar>(newY,newX) = (uchar)(-1);
    	}
    }
	}
}

// TODO nu code zelfde als applyMotion. Motion vectors zitten nu in 'data' op coordinaten van pixels in prvFrame (dus volgens motionMask). Voor eenvoudig toepassen moet voor elke pixel in motionMask 2 keer de verplaatsing toegepast worden, om te stroken met informatie in 'data'. Of in applyMotion een Mat& postMotionData meegeven om de data ook te verplaatsen, en vervolgens applyMotion nog eens uit te voeren voor motion compensation.
// void motionCompensation(const Mat& motionMask, Mat& motCompMask, const Mat& data){
//     const int width=motionMask.cols, height=motionMask.rows;
//     motCompMask = Mat::zeros(motionMask.size(),motionMask.type());
//     for(int x=0;x<width;x++){
//     for(int y=0;y<height;y++){
//     	if((short)(motionMask.at<uchar>(y,x))>0){
// 			short newX = x + data.at<DataVec>(y,x)[1];
// 			short newY = y + data.at<DataVec>(y,x)[2];
// 			if(newX >= 0 && newY >= 0 && newX < width && newY < height){
//     			motCompMask.at<uchar>(newY,newX) = motionMask.at<uchar>(y,x);
// 			}
//     	}
//     }
// 	}
//     io::showMaskOverlap(motionMask,"Mask",motCompMask,"Motion Compensated Mask");
// }

void motionDetection(Mat& prvFr, Mat& nxtFr, Mat& motCompMask, Mat& motionMask, Ptr<BackgroundSubtractor> pMOG2, string name, bool onlyUpdateBGModel, bool useRegExpansion, bool showSrc=true, bool useForegroundFeatures=true){
	bool secondIter = false;

	vector<uchar> status;
	vector<Point2f> prvPts,nxtPts;
	Mat data, regulData;
	Mat fgMask,combMask,maskReg,weights,optFlow1,optFlow2,postMotionMask;

	backgroundSubtraction(prvFr,fgMask,pMOG2);

	if(!onlyUpdateBGModel){

		getOptFlowFeatures(motCompMask,fgMask,combMask,prvPts,useForegroundFeatures);
		opticalFlow(prvFr,nxtFr,optFlow1,status,prvPts,nxtPts,fgMask);
		optFlowRegularization(prvFr.size(),maskReg,status,prvPts,nxtPts,weights,data);

		motionMask = Mat::zeros(prvFr.size(),CV_8UC1);
		/**/
		if(!useRegExpansion){
		   	// Morphological reconstruction
		   	morphologicalReconstruction(motionMask,fgMask,maskReg);
	   	/**/
	   	}else{
	   		// Region Expansion
		   	// uchar threshold = 127;
		   	// Mat maskMorph,maskExp;
		   	// morphologicalReconstruction(maskMorph,fgMask,maskReg);
		   	maskReg.copyTo(motionMask);
		   	expandMarker(fgMask,data,motionMask);
		 //   	for(int x=0;x<maskMorph.cols;x++){
		 //   	for(int y=0;y<maskMorph.rows;y++){
		 //   		if(maskMorph.at<uchar>(y,x) > threshold
		 //   			|| maskExp.at<uchar>(y,x) > threshold){
		 //   			motionMask.at<uchar>(y,x) = -1;
		 //   		}
			// }
		 //   	}
	   	/**/
	   	}

		// cout<<"Morph vs Exp"<<endl;
		// Mat maskMorph,maskExp;
		// morphologicalReconstruction(maskMorph,fgMask,maskReg);
		// maskReg.copyTo(maskExp);
		// expandMarker(fgMask,data,maskExp);
		// io::calculateScores(maskMorph,maskExp);
	 //    io::showMaskOverlap(maskMorph,"Morph",maskExp,"Exp");

		// maskMorph.copyTo(motionMask);

	 //   	int strucSize = 3;
		// Mat struc = getStructuringElement(MORPH_RECT,Size(strucSize,strucSize));
		// morphologyEx(motionMask,motionMask,MORPH_CLOSE,struc);
	   	postProcessing(motionMask,motionMask,3,MORPH_CLOSE);

	   	regularizeData(maskReg,motionMask,data,regulData);

	   	applyMotion(motionMask,regulData,postMotionMask);
	   	
		// motionCompensation(motionMask,motCompMask,regulData);

		bool resize = true;
	   	/**/
		if(showSrc) io::showImage(name+" Src",nxtFr,resize);
		io::showImage(name+" Foreground Mask",fgMask,resize);
		// io::showImage(name+" Comb Mask",combMask,resize);
		io::showImage(name+" OptFlow1",optFlow1,resize);
		io::showImage(name+" Weights",weights,resize);
		// io::showImage(name+" Morph Rec",maskMorph,resize);
		// io::showImage(name+" Expansion",maskExp,resize);
		if(secondIter) io::showImage(name+" OptFlow2",optFlow2,resize);
		/**/
		io::showImage(name+" MotionMask",motionMask,resize);
    	io::showMaskOverlap(motionMask,name+" MotionMask",postMotionMask,"PostMotionMask");
	}
}

int main (int argc, char** argv){
	int keyboard = -1;
	char buffer[30];

	// string dirInput = "input/streetcorner/"; int skipToFrame = 150;
	string dirInput = "input/tramStation/"; int skipToFrame = 0; //93;
	// string dirInput = "input/blizzard/"; int skipToFrame = 30;
	// string dirInput = "input/pedestrians/"; int skipToFrame = 30;
	string regexGt = "groundtruth/%06d.png";
	string regexIn = "input/%06d.jpg";
	string fnFrameGt,fnFrameIn, fnRoi="ROI.bmp";

	string name="Original",nameNoisy="Noisy";

	// BGS Configuration
	int history = 500;
	double varThreshold = 16;
	bool detectShadows=false;

	Mat prvFr,nxtFr,nxtGt,motCompMask,motionMask;
	Mat prvFrNoisy,nxtFrNoisy,motCompMaskNoisy,motionMaskNoisy;
	Mat ROI = imread(dirInput+fnRoi,CV_LOAD_IMAGE_GRAYSCALE);

	sprintf(buffer,(dirInput+regexIn).c_str(),0); fnFrameIn = string(buffer);
	if(!io::fileExists(fnFrameIn)){
		cerr<<"Problem reading dataset first frame: "<<fnFrameIn<<endl;
		stop();
	}
	prvFr = imread(fnFrameIn,CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
	Ptr<BackgroundSubtractor> pMOG2Noisy = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
	Ptr<BackgroundSubtractor> pMOG2Exp = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
	Ptr<BackgroundSubtractor> pMOG2NoisyExp = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
	motCompMask = Mat::zeros(prvFr.size(),CV_8UC1);
	motCompMask.copyTo(motCompMaskNoisy);

	const int MAX_FRAME_IDX = 2500;
	for(int frameIdx = 1;frameIdx<MAX_FRAME_IDX;frameIdx++){

		bool onlyUpdateBGModel = frameIdx<skipToFrame;

		sprintf(buffer,(dirInput+regexGt).c_str(),frameIdx); fnFrameGt = string(buffer);
		sprintf(buffer,(dirInput+regexIn).c_str(),frameIdx); fnFrameIn = string(buffer);
		if(!io::fileExists(fnFrameIn)){
			cout<<"Stopped at frame index "<<frameIdx<<endl;
			stop();
		}
		nxtGt = imread(fnFrameGt,CV_LOAD_IMAGE_GRAYSCALE);
		if(!nxtGt.data){
			cerr<<"Problem reading GT: "<<fnFrameGt<<endl;
			throw;
		}
		nxtFr = imread(fnFrameIn,CV_LOAD_IMAGE_GRAYSCALE);

		addNoise(prvFr,prvFrNoisy);
		addNoise(nxtFr,nxtFrNoisy);

		// Use only morphological reconstruction
		motionDetection(prvFr,nxtFr,motCompMask,motionMask,pMOG2,name+"",onlyUpdateBGModel,false);
		motionDetection(prvFrNoisy,nxtFrNoisy,motCompMaskNoisy,motionMaskNoisy,pMOG2Noisy,nameNoisy+"",onlyUpdateBGModel,false);

		io::showImage(name+" GT",nxtGt,true);
		bitwise_and(nxtGt,ROI,nxtGt);
		threshold(nxtGt,nxtGt,1);
		if(!onlyUpdateBGModel){
			Mat motMaskROI,motMaskROINoisy;
			bitwise_and(motionMask,ROI,motMaskROI);
			bitwise_and(motionMaskNoisy,ROI,motMaskROINoisy);

	    	cout<<"Frame "<<frameIdx<<", with Morph"<<endl;
	  //   	cout<<"Comp full inter: ";
			// io::calculateScores(motionMask,motionMaskNoisy);
	    	cout<<"Comp ROI gt-normal: ";
			io::calculateScores(nxtGt,motionMask);
	    	cout<<"Comp ROI gt-noisy: ";
			io::calculateScores(nxtGt,motionMaskNoisy);
			cout<<endl;
		}

		// Use region expansion
		// motionDetection(prvFr,nxtFr,motCompMask,motionMask,pMOG2Exp,name+" Exp",onlyUpdateBGModel,true);
		// motionDetection(prvFrNoisy,nxtFrNoisy,motCompMaskNoisy,motionMaskNoisy,pMOG2NoisyExp,nameNoisy+" Exp",onlyUpdateBGModel,true);

		// if(!onlyUpdateBGModel){
		// 	Mat motMaskROI,motMaskROINoisy;
		// 	bitwise_and(motionMask,ROI,motMaskROI);
		// 	bitwise_and(motionMaskNoisy,ROI,motMaskROINoisy);

	 //    	cout<<"Frame "<<frameIdx<<", with Exp"<<endl;
	 //  //   	cout<<"Comp full inter: ";
		// 	// io::calculateScores(motionMask,motionMaskNoisy);
	 //    	cout<<"Comp ROI gt-normal: ";
		// 	io::calculateScores(nxtGt,motionMask);
	 //    	cout<<"Comp ROI gt-noisy: ";
		// 	io::calculateScores(nxtGt,motionMaskNoisy);
		// 	cout<<endl;
		// }
		
		if(!onlyUpdateBGModel){
			if((keyboard=waitKey(0)) == ESCAPE_NL || keyboard == ESCAPE_EN){
				stop();
			}
		}
		nxtFr.copyTo(prvFr);
	}

	stop();
}