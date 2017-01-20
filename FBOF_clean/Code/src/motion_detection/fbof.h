/*

This code was written by Tim Ranson for the master's dissertation 'Noise-robust motion detection for low-light videos' by Tim Ranson.

*/


#ifndef _FBOF
#define _FBOF

#include <opencv2/opencv.hpp>
using namespace cv;

#include "image_processing.h"
#include "io.h"
#include "denoising.h"
#include "scores.h"

class Fbof{
public:
	Fbof(){}
	Fbof(string _name, float _minVecLen_axis=1.0, short _r_sn=1, float _t_sv=0.03, double _t_sn=0.375, short _r_mr=2,bool _showResults=false,bool _useDenoising=false):
		name(_name),minVecLen_axis(_minVecLen_axis),r_sn(_r_sn),t_sv(_t_sv),t_sn(_t_sn),r_mr(_r_mr),showResults(_showResults),useDenoising(_useDenoising)
		{ init(); }
	~Fbof(){}

void motionDetection(Mat& prvFr,Mat& nxtFr,Mat& motCompMask,Mat& motionMask,bool usePostProcessing,bool onlyUpdateBGModel,bool useRegExpansion);

protected:
	typedef float DataVecType;
	typedef Vec<DataVecType,3> DataVec;
	typedef Vec<float,2> OptFlowVec;

	string name;
	Ptr<BackgroundSubtractor> pMOG2;
	bool showResults;
	bool useDenoising;

	/* Algorithm parameters */
	float minVecLen_axis; // Minimum vector size along an axis, e.g. 1 will set threshold at length of vector (1,1)
	float t_sv; // similarity threshold for similar vector estimation: similarity if difference is below threshold
	short r_sn; // Neighbour radius for similar neighbour weighting
	float t_sn; // percentage threshold for similar neighbour weights
	short r_mr; // radius of structuring element for dilation during morphological reconstruction

	/* Algorithm parameters set in init() */
	float minVecLenSquared;
	float t_sv_squared;
	float maxNeighbours;
	uchar t_sn_converted; 

	void init();
	void denoise(const Mat& src, Mat& dst);
	void backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2);
	void getOptFlowFeatures(const Mat& motCompMask, const Mat& fgMask, Mat& combinedMask, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures);
	void opticalFlow(const Mat& prvFr, const Mat& nxtFr, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask, Mat& flow);
	bool similarVectorEstimation(const DataVec& data1, const DataVec& data2);
	void similarNeighbourWeighting(const Mat& data, Mat& weights);
	void optFlowRegularization(const Size& size, const Mat& fgMask, Mat& dst, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& data);
	void morphologicalReconstruction(const Mat& mask, const Mat& marker, Mat& dst);
	void regularizeDataByList(const Mat& mask, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, DataVecType deltaX, DataVecType deltaY);
	void regularizeData(const Mat& marker, const Mat& mask, const Mat& data, Mat& regulData);
	void applyMotion(const Mat& src, const Mat& regulData, Mat& dst);
	void postProcessing(Mat& img);

	bool isNonZeroVector(DataVec& d){ return ( d[1]!=0 && d[2]!=0 ); }
};

void Fbof::init(){
	t_sv_squared = t_sv*t_sv;
	float windowWidth = 2*r_sn+1;
	maxNeighbours = windowWidth*windowWidth - 1;
	t_sn_converted = 255*t_sn; //153
	minVecLenSquared = vectorLengthSquared(minVecLen_axis,minVecLen_axis);
	/* Background subtraction parameters */
	int history = 500;
	double varThreshold = 16;
	bool detectShadows=false;
	pMOG2 = createBackgroundSubtractorMOG2(history,varThreshold,detectShadows);
}

/*
Perform Non-local means denoising
*/
void Fbof::denoise(const Mat& src, Mat& dst){
	dn::denoise(src,dst,dn::DEN_NL_MEANS);
}

/*
Background subtraction using OpenCV implementation cv::BackgroundSubtractorMOG2 of Zivkovic's adaptation of Gaussian Mixture Model
*/
void Fbof::backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2){
	double learningRate = -1;
    pMOG2->apply(frame, fgMask,learningRate);
}

/*
Optical flow using OpenCV implementation of Farneback's method
*/
void Fbof::opticalFlow(const Mat& prvFr, const Mat& nxtFr, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask, Mat& flow){
	Mat prvFrGrey, nxtFrGrey;
	prvFr.copyTo(prvFrGrey);
	nxtFr.copyTo(nxtFrGrey);
	if(prvFr.channels()>1){
		cvtColor(prvFrGrey, prvFrGrey, COLOR_BGR2GRAY);
	}
	if(prvFr.channels()>1){
		cvtColor(nxtFrGrey, nxtFrGrey, COLOR_BGR2GRAY);
	}
	
	// Mat flow;
	double pyr_scale = 0.5;
	int levels = 1;
	int winsize = 15; // Larger window size is more robust to noise, too large can give faulty results?
	int iterations = 3;
	// int poly_n = 5; double poly_sigma = 1.1;
	int poly_n = 7; double poly_sigma = 1.5;
	int flags = 0;

	calcOpticalFlowFarneback(prvFrGrey,nxtFrGrey,flow,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma,flags);

	prvPts.clear();
	nxtPts.clear();
	for(int x=0;x<flow.cols;x++){
	for(int y=0;y<flow.rows;y++){
		DataVecType deltaX = flow.at<OptFlowVec>(y,x)[0];
		DataVecType deltaY = flow.at<OptFlowVec>(y,x)[1];
		if( abs(deltaX)>1 || abs(deltaY)>1 ){
			prvPts.push_back(Point2f(x,y));
			nxtPts.push_back(Point2f(x+deltaX,y+deltaY));
		}else{
			flow.at<OptFlowVec>(y,x)[0] = 0;
			flow.at<OptFlowVec>(y,x)[1] = 0;
		}
	}
	}

	vector<Mat> mv;
	split(flow,mv);
	mv.push_back(Mat::zeros(mv[0].size(),mv[0].type()));
	merge(mv,flow);
}


/*
Similar neighbour estimation as described in master's dissertation
*/
bool Fbof::similarVectorEstimation(const DataVec& dataVec1, const DataVec& dataVec2){
	DataVecType deltaX1 = dataVec1[1];
	DataVecType deltaY1 = dataVec1[2];
	DataVecType deltaX2 = dataVec2[1];
	DataVecType deltaY2 = dataVec2[2];

	DataVecType deltaXd = deltaX2 - deltaX1;
	DataVecType deltaYd = deltaY2 - deltaY1;

	float dLengthSquared = vectorLengthSquared(deltaXd,deltaYd);
	float v1LengthSquared = vectorLengthSquared(deltaX1,deltaY1);

	return ( dLengthSquared/v1LengthSquared < t_sv_squared );
}

/*
Similar neighbour weighting as described in master's dissertation
*/
void Fbof::similarNeighbourWeighting(const Mat& data, Mat& weights){
	for(int x=0;x<data.cols;x++){
	for(int y=0;y<data.rows;y++){
		short neighbours = 0;
		DataVec dataVec = data.at<DataVec>(y,x);
		if(isNonZeroVector(dataVec)){ // Only perform for non-zero vectors
			for(int i=-1; i<=r_sn; i+=r_sn){
			if(x+i>0 && x+i < data.cols){
				for(int j=-1; j<=r_sn; j+=r_sn){
				if(!(i==0 && j==0) && y+j>0 && y+j < data.rows){
					DataVec dataVec2 = data.at<DataVec>(y+j,x+i);
					if(	isNonZeroVector(dataVec2)
						&&
						similarVectorEstimation(dataVec,dataVec2)){
							neighbours++;
					}
				}
				}
			}
			}
		}
		float weight = neighbours/maxNeighbours;
		weights.at<float>(y,x) = weight;
	}
	}
}

/*
Performs optical flow regularisation, resulting in the regularised motion mask stored in 'dst'.
Starts by converting the optical flow data in 'prvPts' and 'nxtPts' to a different format and removing small vector simultaneously (replaced by zero-vector).
Similar neighbour weighting is performed on the optical flow data and finally thresholds the similar neighbour weights, result is stored in 'dst'.
*/
void Fbof::optFlowRegularization(const Size& size, const Mat& fgMask, Mat& dst, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& data){

	if(std::is_same<DataVecType,short>::value){
    	data = Mat::zeros(size,CV_16SC3);
	}else{
    	data = Mat::zeros(size,CV_32FC3);
    }
	weights = Mat::zeros(size,CV_32FC1);

	/* Optical flow data reformatting */
    for(int ptsIdx=0;ptsIdx<prvPts.size();ptsIdx++){
		DataVecType x1=prvPts[ptsIdx].x,y1=prvPts[ptsIdx].y,x2=nxtPts[ptsIdx].x,y2=nxtPts[ptsIdx].y;
		Point srcPt = Point((int)x1,(int)y1);
		DataVecType deltaX = x2-x1;
		DataVecType deltaY = y2-y1;
		/* Remove small vectors */
		if(vectorLengthSquared(deltaX,deltaY) < minVecLenSquared){
			deltaX = 0;
			deltaY = 0;
		}
		if(fgMask.at<uchar>(srcPt) > 0){
			data.at<DataVec>(srcPt)[0] = ptsIdx;
			data.at<DataVec>(srcPt)[1] = deltaX;
			data.at<DataVec>(srcPt)[2] = deltaY;
		}
    }

    /* Perform similar neighbour weighting and threshold the results */
	similarNeighbourWeighting(data,weights);
    dst = Mat::ones(size,CV_8UC1);
    weights.convertTo(dst,CV_8UC1,255);
   	threshold(dst,dst,t_sn_converted);
}

/*
Performs typical morphological reconstruction.
'marker' is dilated repeatedly without protruding 'mask', until 'marker' no longer changes.
Result is stored in 'dst'
*/
void Fbof::morphologicalReconstruction(const Mat& mask, const Mat& marker, Mat& dst){
   	Mat dilation,maskedDilation,prevMaskedDilation;
   	bool hasChanged;
   	int r=r_mr;
   	int strucSize = 2*r+1;
    Mat struc;
    struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
   	threshold(marker,maskedDilation,127);
   	do{
   		maskedDilation.copyTo(prevMaskedDilation);
   		dilate(prevMaskedDilation,dilation,struc);
   		min(dilation,mask,maskedDilation);
   		hasChanged = (countNonZero(maskedDilation!=prevMaskedDilation) != 0);
   	}while(hasChanged);
   	maskedDilation.copyTo(dst);
}

/*
Two-part optical flow data regularisation.
@see Fbof::regularizeData()
*/
void Fbof::regularizeDataByList(const Mat& mask, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, DataVecType deltaX, DataVecType deltaY){
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

/*
Two-part optical flow data regularisation.
Regularises optical flow data (motion vectors) in 'data' and stores the regularised data in 'regulData'.
Binary mask 'marker' marks all pixels of which motion vectors are consistent (similar).
Each region in 'marker' receives the same motion vector.
These regions are expanded within 'mask', resulting in regularised data where each region in 'mask' has a uniform motion vector.
*/
void Fbof::regularizeData(const Mat& marker, const Mat& mask, const Mat& data, Mat& regulData){
    const int width=marker.cols, height=marker.rows;
    regulData = Mat::zeros(data.size(),data.type());
   	vector<bool> expanded(width*height,false);
    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	if((short)(marker.at<uchar>(y,x))>0){
    		int idx = y*width+x;
	    	if(!expanded[idx]){
				vector<int> toExpand;
				DataVecType deltaX = data.at<DataVec>(y,x)[1];
				DataVecType deltaY = data.at<DataVec>(y,x)[2];

				toExpand.push_back(idx);
				regularizeDataByList(mask,expanded,toExpand,regulData,deltaX,deltaY);
	    	}
	    }
    }
    }
}

/*
Performs motion compensation on the binary mask 'src', using the (regularised) optical flow data in 'regulData'.
The result is stored in 'dst'.
*/
void Fbof::applyMotion(const Mat& src, const Mat& regulData, Mat& dst){
    const int width=src.cols, height=src.rows;
    dst = Mat::zeros(src.size(),src.type());

    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	if((short)(src.at<uchar>(y,x))>0){
			DataVecType newX = x - regulData.at<DataVec>(y,x)[1];
			DataVecType newY = y - regulData.at<DataVec>(y,x)[2];
			if(newX >= 0 && newY >= 0 && newX < width && newY < height)
				dst.at<uchar>(newY,newX) = (uchar)(-1);
    	}
    }
	}
}

/*
Performs post-processing:
-three times dilation followed by three times erosion with a 7x7 circular structuring element
*/
void Fbof::postProcessing(Mat& img){
   	int sizeMedBlur = 3;
   	int sizeMorph = 7;
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(sizeMorph,sizeMorph));
   	medianBlur(img,img,sizeMedBlur);
   	medianBlur(img,img,sizeMedBlur);
   	dilate(img,img,struc);
   	dilate(img,img,struc);
   	dilate(img,img,struc);
   	erode(img,img,struc);
   	erode(img,img,struc);
}

void Fbof::motionDetection(Mat& prvFr, Mat& nxtFr, Mat& motCompMask, Mat& motionMask, bool usePostProcessing, bool onlyUpdateBGModel=false, bool useRegExpansion=true){

	vector<uchar> status;
	vector<Point2f> prvPts,nxtPts;
	Mat data, regulData;
	Mat nxtFrDenoised,fgMask,combMask,optFlow,weights,maskReg,morphRecMask,morphRecMarker,morphRecResult,postMotionMask;
	Mat flowFarneback = Mat::zeros(prvFr.size(),CV_32FC2);

	if(useDenoising)
		denoise(nxtFr,nxtFrDenoised);
	else
		nxtFr.copyTo(nxtFrDenoised);

	/* Background subtraction */
	backgroundSubtraction(nxtFrDenoised,fgMask,pMOG2);

	// When not presenting or saving any results, only the background model needs to be updated and the following operations are unnecessary
	// Allows for skipping to a certain frame more quickly
	if(!onlyUpdateBGModel){

		/* Optical flow */
		binMat2Vec(Mat::ones(prvFr.size(),CV_8UC1),prvPts);
		opticalFlow(nxtFr,prvFr,status,prvPts,nxtPts,fgMask,flowFarneback);

		/* Optical flow regularization */
		optFlowRegularization(prvFr.size(),fgMask,maskReg,prvPts,nxtPts,weights,data);

		if(!hasHighValue(maskReg)){ // If the regularised mask is empty (only low values), use the foreground mask for motion mask and skip morphological reconstruction
			fgMask.copyTo(motionMask);
			threshold(motionMask,motionMask,250); // MOG2 can detect shadows, which are removed here (values below 250)
		}else{
			/* The regularised mask is the marker for morphological reconstruction */
			maskReg.copyTo(morphRecMarker);

			/* Combination of the foreground mask and regularised mask is the mask for  morphological reconstruction */
			bitwise_or(fgMask,maskReg,morphRecMask);

		   	/* Morphological reconstruction */
		   	morphologicalReconstruction(morphRecMask,morphRecMarker,morphRecResult);

		   	morphRecResult.copyTo(motionMask);
	   	}

	   	if(usePostProcessing){
	   		postProcessing(motionMask);
		}

	   	/* Motion compensation for possible use in next iteration, currently not implemented *
	   	regularizeData(maskReg,motionMask,data,regulData);

	   	applyMotion(motionMask,regulData,motCompMask);
	   	/**/

   		if(showResults){
   			/* Switch for saving the shown images, @see io::showImage() */
		    bool saveResults = false;
   			/* Switch for resizing the shown images, @see io::showImage() */
			bool resize = true;
			io::showImage(name+" Source",nxtFr,resize,saveResults);
			if(useDenoising) io::showImage(name+" Denoised",nxtFrDenoised,resize,saveResults);
			if(!fgMask.empty()) io::showImage(name+" Foreground Mask",fgMask,resize,saveResults);
			if(!flowFarneback.empty()) io::showImage(name+" Farneback",flowFarneback,resize,saveResults);
			if(!weights.empty()) io::showImage(name+" Weights",weights,resize,saveResults);
			if(!maskReg.empty()) io::showImage(name+" Weights thresholded",maskReg,resize,saveResults);
			if(!morphRecMarker.empty()) io::showImage(name+" MR Marker",morphRecMarker,resize,saveResults);
			if(!morphRecMask.empty()) io::showImage(name+" MR Mask",morphRecMask,resize,saveResults);
			if(!morphRecResult.empty()) io::showImage(name+" MR Result",morphRecResult,resize,saveResults);
			if(!motionMask.empty()) io::showImage(name+" MotionMask",motionMask,resize,saveResults);
			/* Shows motion compensated mask, if motion compensation is enabled. */
			if(!motCompMask.empty()) io::showImage(name+" MotComp Mask",motCompMask,resize,saveResults);
		}
	}
}

#endif