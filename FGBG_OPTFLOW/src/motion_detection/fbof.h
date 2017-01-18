#ifndef _FBOF
#define _FBOF

#include <opencv2/opencv.hpp>
using namespace cv;

#include "image_processing.h"
#include "io.h"
#include "denoising.h"
#include "scores.h"

#define FARNEBACK

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
	void checkRegulData(const Mat& mask, const Mat& regulData);
	void denoise(const Mat& src, Mat& dst);
	void backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2);
	void getOptFlowFeatures(const Mat& motCompMask, const Mat& fgMask, Mat& combinedMask, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures);
	void opticalFlow(const Mat& prvFr, const Mat& nxtFr, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask, Mat& flow);
	bool similarVectorEstimation(const DataVec& data1, const DataVec& data2);
	void similarNeighbourWeighting(const Mat& data, Mat& weights);
	void optFlowRegularization(const Size& size, const Mat& fgMask, Mat& dst, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& data, Mat& weightsEntireFrameMask);
	void expandByList(const Mat& data, const Mat& mask, Mat& marker, vector<bool>& expanded, vector<int> toExpand);
	void expandMarker(const Mat& data, const Mat& mask, const Mat& marker, Mat& dst);
	void morphologicalReconstruction(const Mat& mask, const Mat& marker, Mat& dst);
	void regularizeDataByList(const Mat& mask, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, DataVecType deltaX, DataVecType deltaY);
	void regularizeData(const Mat& marker, const Mat& mask, const Mat& data, Mat& regulData);
	void applyMotion(const Mat& src, const Mat& regulData, Mat& dst);
	// void motionCompensation(const Mat& motionMask, Mat& motCompMask, const Mat& data);
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

void Fbof::checkRegulData(const Mat& mask, const Mat& regulData){
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
    io::printScores(mask,regulMask);
    io::showMaskOverlap(mask,"Mask",regulMask,"RegulMask");
}

void Fbof::denoise(const Mat& src, Mat& dst){
	dn::denoise(src,dst,dn::DEN_NL_MEANS);
	// dn::denoise(src,dst,dn::DEN_MEDIAN_BLUR);
	// dn::denoise(src,dst,dn::DEN_BILATERAL_FILTER);
}

void Fbof::backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2){
	double learningRate = -1;
    pMOG2->apply(frame, fgMask,learningRate);
}

/* Unused */
// TODO remove combinedMask as parameter, not needed as output parameter. Placeholder for data, transferred to goodFeaturesToTrack
void Fbof::getOptFlowFeatures(const Mat& motCompMask, const Mat& fgMask, Mat& combinedMask, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures=true){
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
	
	#ifndef FARNEBACK
		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
		// int ws = 7;
		int ws = 21;
		Size winSize(ws,ws);
		vector<float> err;

		if(prvPts.empty()){
			// cout<<"Fg Mask"<<endl;
			goodFeaturesToTrack(prvFrGrey, prvPts, 500, 0.01, 10, fgMask, 3, 0, 0.04);
		}
		if(prvPts.empty()){
			// cout<<"Empty prvPts"<<endl;
			goodFeaturesToTrack(prvFrGrey, prvPts, 500, 0.01, 10, Mat(), 3, 0, 0.04);
		}
		calcOpticalFlowPyrLK(prvFrGrey, nxtFrGrey, prvPts, nxtPts, status, err, winSize, 3, termcrit, 0, 0.001);
	#else
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
	#endif
}

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
* TODO thresholding at half the weight == 8 equal neighbours >> 7 equal neighbours == (2⁷-1)/255 = 0.49!
* Remove exponential weighting in similarNeighbourWeighting
* TODO add squared vector length to Datavec
*/
void Fbof::optFlowRegularization(const Size& size, const Mat& fgMask, Mat& dst, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& dataEntireFrame, Mat& weightsEntireFrameMask){

	Mat data;
	if(std::is_same<DataVecType,short>::value){
    	dataEntireFrame = Mat::zeros(size,CV_16SC3);
	}else{
    	dataEntireFrame = Mat::zeros(size,CV_32FC3);
    }
    dataEntireFrame.copyTo(data);
	weights = Mat::zeros(size,CV_32FC1);
	Mat weightsEntireFrame = Mat::zeros(size,CV_32FC1);

	// Optical flow to data
    for(int ptsIdx=0;ptsIdx<prvPts.size();ptsIdx++){
    	#ifndef FARNEBACK
    	if(status[ptsIdx]){
    	#endif
			DataVecType x1=prvPts[ptsIdx].x,y1=prvPts[ptsIdx].y,x2=nxtPts[ptsIdx].x,y2=nxtPts[ptsIdx].y;
			Point srcPt = Point((int)x1,(int)y1);
			DataVecType deltaX = x2-x1;
			DataVecType deltaY = y2-y1;
	// 		cout<<deltaX<<" "<<deltaY<<endl;
			if(vectorLengthSquared(deltaX,deltaY) < minVecLenSquared){
				deltaX = 0;
				deltaY = 0;
			}
			/* Disable entire frame optical flow *
			dataEntireFrame.at<DataVec>(srcPt)[0] = ptsIdx;
			dataEntireFrame.at<DataVec>(srcPt)[1] = deltaX;
			dataEntireFrame.at<DataVec>(srcPt)[2] = deltaY; /**/
			if(fgMask.at<uchar>(srcPt) > 0){
				data.at<DataVec>(srcPt)[0] = ptsIdx;
				data.at<DataVec>(srcPt)[1] = deltaX;
				data.at<DataVec>(srcPt)[2] = deltaY;
			}
    	#ifndef FARNEBACK
    	}
    	#endif
    }

    // Weighting for regularization
	similarNeighbourWeighting(data,weights);
    dst = Mat::ones(size,CV_8UC1);
    weights.convertTo(dst,CV_8UC1,255);
   	threshold(dst,dst,t_sn_converted);

   	/* Disable entire frame optical flow *
	similarNeighbourWeighting(dataEntireFrame,weightsEntireFrame);
    weightsEntireFrameMask = Mat::ones(size,CV_8UC1);
    weightsEntireFrame.convertTo(weightsEntireFrameMask,CV_8UC1,255);
   	// threshold(weightsEntireFrameMask,weightsEntireFrameMask,255*0.8);
   	threshold(weightsEntireFrameMask,weightsEntireFrameMask,t_sn_converted);
   	/**/
}

/*
Two-part alternative implementation for morphological reconstruction
@see expandMarker
*/
void Fbof::expandByList(const Mat& data, const Mat& mask, Mat& marker, vector<bool>& expanded, vector<int> toExpand){
    const int width=mask.cols, height=mask.rows, r=r_mr;
   	int strucSize = 2*r+1;
   	int c=r; // centre of struc
    Mat struc;
    struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
    // getCircularStructuringElement(r,struc);
	for(int k=0; k<toExpand.size();k++){
    	int idx = toExpand[k];
    	if(!expanded[idx]){
    		expanded[idx] = true;
    		marker.at<uchar>(idx)=(uchar)(-1); // Max value
			for(int i=0; i<strucSize; i++){
			for(int j=0; j<strucSize; j++){
				if(struc.at<uchar>(j,i)>0){
					// (i,j) are the coordinates in the structuring element
					// (i-c,j-c) are the coordinates of the neighbouring pixel, relative to the current centring pixel
					int idxNb = idx+(j-c)*width+(i-c);
					if(idxNb != idx && idxNb>0 && idxNb < width*height){
						if( (short)(mask.at<uchar>(idxNb))>0
							// ||
							// similarVectorEstimation(data.at<DataVec>(idx),data.at<DataVec>(idxNb))
							){
							toExpand.push_back(idxNb);
						}
					}
				}
			}
			}
    	}
	}
}

/*
Alternative implementation for morphological reconstruction to be less redundant.
Iterates all pixels in the mask, expands all high valued pixels of 'marker'.
expandByList() iterates all pixels in 'toExpand', sets their value in 'marker' to high and iterates all connecting pixels, adds them to 'toExpand' if high-valued in 'mask'.
Boolean array 'expanded' keeps track of all expanded pixels, so that each pixel is processed at most once.
*/
void Fbof::expandMarker(const Mat& data, const Mat& mask, const Mat& marker, Mat& dst){
    const int width=mask.cols, height=mask.rows;
    marker.copyTo(dst);
   	vector<bool> expanded(width*height,false);
    for(int x=0;x<width;x++){
    for(int y=0;y<height;y++){
    	if((int)(dst.at<uchar>(y,x))>127){
    		int idx = y*width+x;
	    	if(!expanded[idx]){
				vector<int> toExpand;
				toExpand.push_back(idx);
				expandByList(data,mask,dst,expanded,toExpand);
	    	}
	    }
    }
    }
}

void Fbof::morphologicalReconstruction(const Mat& mask, const Mat& marker, Mat& dst){
   	Mat dilation,maskedDilation,prevMaskedDilation;
   	bool hasChanged;
   	int r=r_mr;
   	int strucSize = 2*r+1;
    Mat struc;
    struc = getStructuringElement(MORPH_ELLIPSE,Size(strucSize,strucSize));
    // getCircularStructuringElement(r,struc);
    // TODO Change threshold value to variable
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
* Nu mogelijk dat er een memory overload is doordat buren van een pixel gepushd worden terwijl ze er al inzitten (veeel redundantie).
* Mogelijks efficienter: 2e set array die unieke indexen van toExpand bijhoudt, gebruiken voor check. toExpand behouden om volgorde van expanderen te bepalen.
* TODO --^ 
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
* TODO Volgorde aanpassen zodat niet van links naar rechts overschreven wordt: eerst alle high values in marker in toExpand toevoegen,
* dan pas 1 keer regularizeDataByList oproepen.
* Op deze manier worden alle zekere MVs al gezet en pas daarna worden deze geëxpandeerd.
* Expansie verloopt dan ook per groeiende radius (eerst alle r=1 buren van alle marker pixels, daarna pas alle r=2,...).
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
    checkRegulData(mask,regulData);
}

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

// TODO nu code zelfde als applyMotion. Motion vectors zitten nu in 'data' op coordinaten van pixels in prvFrame (dus volgens motionMask). Voor eenvoudig toepassen moet voor elke pixel in motionMask 2 keer de verplaatsing toegepast worden, om te stroken met informatie in 'data'. Of in applyMotion een Mat& postMotionData meegeven om de data ook te verplaatsen, en vervolgens applyMotion nog eens uit te voeren voor motion compensation.
// void Fbof::motionCompensation(const Mat& motionMask, Mat& motCompMask, const Mat& data){
//     const int width=motionMask.cols, height=motionMask.rows;
//     motCompMask = Mat::zeros(motionMask.size(),motionMask.type());
//     for(int x=0;x<width;x++){
//     for(int y=0;y<height;y++){
//     	if((short)(motionMask.at<uchar>(y,x))>0){
//			// short newX = x + data.at<DataVec>(y,x)[1];
//			// short newY = y + data.at<DataVec>(y,x)[2];
//			float newX = x + regulData.at<DataVec>(y,x)[1];
//			float newY = y + regulData.at<DataVec>(y,x)[2];
// 			if(newX >= 0 && newY >= 0 && newX < width && newY < height){
//     			motCompMask.at<uchar>(newY,newX) = motionMask.at<uchar>(y,x);
// 			}
//     	}
//     }
// 	}
//     io::showMaskOverlap(motionMask,"Mask",motCompMask,"Motion Compensated Mask");
// }

void Fbof::postProcessing(Mat& img){
	/* Small window *
   	int sizeMedBlur = 3;
   	int sizeMorph = 5;
   	/* Large window *
   	int sizeMedBlur = 7;
   	int sizeMorph = 9;
   	/* Large closing */
   	int sizeMedBlur = 3;
   	int sizeMorph = 21;
   	/**/
	Mat struc = getStructuringElement(MORPH_ELLIPSE,Size(sizeMorph,sizeMorph));
   	medianBlur(img,img,sizeMedBlur);
	morphologyEx(img,img,MORPH_CLOSE,struc);
	// morphologyEx(img,img,MORPH_OPEN,struc);
	// morphologyEx(img,img,MORPH_CLOSE,struc);
}

void Fbof::motionDetection(Mat& prvFr, Mat& nxtFr, Mat& motCompMask, Mat& motionMask, bool usePostProcessing, bool onlyUpdateBGModel=false, bool useRegExpansion=true){

	// vector<bool> lOptFlowOnEntireImage={true,false};

	vector<uchar> status;
	vector<Point2f> prvPts,nxtPts;
	Mat data, regulData;
	Mat nxtFrDenoised,fgMask,combMask,optFlow,weights,maskReg,morphRecMask,morphRecMarker,morphRecResult,postMotionMask;
	Mat flowFarneback = Mat::zeros(prvFr.size(),CV_32FC2);

	/* TMP for checking workflow *
	if(motionMask.empty()) motCompMask = Mat::zeros(prvFr.size(),CV_8UC1);
	else motionMask.copyTo(motCompMask);
	motionMask = Mat::zeros(prvFr.size(),CV_8UC1);
	/* TMP END */

	if(useDenoising)
		denoise(nxtFr,nxtFrDenoised);
	else
		nxtFr.copyTo(nxtFrDenoised);

	// Background subtraction
	backgroundSubtraction(nxtFrDenoised,fgMask,pMOG2);

	if(!onlyUpdateBGModel){

		// fgMask.copyTo(motionMask);
	   	/* TR START Checking workflow*/
		Mat maskRegEntireFrame;

		// Optical flow
		// binMat2Vec(fgMask,prvPts);
		binMat2Vec(Mat::ones(prvFr.size(),CV_8UC1),prvPts);
		// opticalFlow(prvFr,nxtFr,status,prvPts,nxtPts,fgMask,flowFarneback); // switch prvFr & nxtFr
		opticalFlow(nxtFr,prvFr,status,prvPts,nxtPts,fgMask,flowFarneback);

		// Optical flow regularization
		// optFlowRegularization(prvFr.size(),motCompMask,maskReg,status,prvPts,nxtPts,weights,data,maskRegEntireFrame);
		optFlowRegularization(prvFr.size(),fgMask,maskReg,status,prvPts,nxtPts,weights,data,maskRegEntireFrame);

		if(!hasHighValue(maskReg)){
			fgMask.copyTo(motionMask);
			threshold(motionMask,motionMask,250);
		}else{
			// Foreground reduce
			maskReg.copyTo(morphRecMarker);

			// Foreground expand
			// fgMask.copyTo(morphRecMask);
			
			// Still a lot of blocklike artefacts in maskRegEntireFrame
			bitwise_or(fgMask,maskReg,morphRecMask);
			/* Disable entire frame optical flow *
			bitwise_or(fgMask,maskRegEntireFrame,morphRecMask); /**/
			if(!useRegExpansion){
			   	// Morphological reconstruction
			   	morphologicalReconstruction(morphRecMask,morphRecMarker,morphRecResult);
		   	}else{
		   		// Morphological reconstruction by region expansion
			   	expandMarker(data,morphRecMask,morphRecMarker,morphRecResult);
		   	}


		   	/* TR START Checking workflow*
			Mat motMaskMorph,motMaskExp;
			morphologicalReconstruction(morphRecMask,morphRecMarker,motMaskMorph);
			expandMarker(data,morphRecMask,morphRecMarker,motMaskExp);
			// cout<<"Morph vs Exp"<<endl;
			// io::printScores(motMaskMorph,motMaskExp);
		    // io::showMaskOverlap(motMaskMorph,"Morph",motMaskExp,"Exp"); /**/

		   	morphRecResult.copyTo(motionMask);
	   	}

	   	if(usePostProcessing){
	   		postProcessing(motionMask);
		}

		/* TR OFF Motion compensation *
	   	regularizeData(maskReg,motionMask,data,regulData);

	   	applyMotion(motionMask,regulData,motCompMask);
	   	/**/

   		if(showResults){
			/* Draw optical flow motion vectors */
			Mat optFlow2;
    		prvFr.copyTo(optFlow);
    		prvFr.copyTo(optFlow2);
		    if(optFlow.channels()==1){
		    	cvtColor(optFlow,optFlow,CV_GRAY2BGR);
		    	cvtColor(optFlow2,optFlow2,CV_GRAY2BGR);
		    }
		    Scalar red(0,0,255), blue(255,0,0), green(0,255,0);
			int thickness=1, lineType=8, shift=0;
			double tipLength=0.1;
			int width = data.cols, height = data.rows;

			int skip;
			bool arrows = true;
			if(!arrows) skip = 1; else skip = 5;
		    for(int x=0;x<width;x+=skip){
		    for(int y=0;y<height;y+=skip){
		    	if(data.at<DataVec>(y,x)[1] > 0 || data.at<DataVec>(y,x)[2] > 0){
					short newX = x + data.at<DataVec>(y,x)[1];
					short newY = y + data.at<DataVec>(y,x)[2];
					if(newX >= 0 && newY >= 0 && newX < width && newY < height){
		    			if(arrows) arrowedLine(optFlow,Point2f(x,y),Point2f(newX,newY),green,thickness,lineType,shift,tipLength);
		    			else{
			    			line(optFlow,Point2f(x,y),Point2f(newX,newY),blue);
			    			circle(optFlow,Point2f(newX,newY),3,red);
		    			}
					}
				}
		    }
		    }
		  //   for(int x=0;x<width;x+=skip){
		  //   for(int y=0;y<height;y+=skip){
		  //   	if(motionMask.at<uchar>(y,x)>0){
		  //   		if(data.at<DataVec>(y,x)[1] > 0 || data.at<DataVec>(y,x)[2] > 0){
				// 		short newX = x + data.at<DataVec>(y,x)[1];
				// 		short newY = y + data.at<DataVec>(y,x)[2];
				// 		if(newX >= 0 && newY >= 0 && newX < width && newY < height){
			 //    			if(arrows) arrowedLine(optFlow2,Point2f(x,y),Point2f(newX,newY),green,thickness,lineType,shift,tipLength);
			 //    			else{
				//     			line(optFlow2,Point2f(x,y),Point2f(newX,newY),blue);
				//     			circle(optFlow2,Point2f(newX,newY),3,red);
			 //    			}
				// 		}
				// 	}
				// }
		  //   }
		  //   }
		  	/**/

		    bool saveResults = true;
			bool resize = true;
			string postfix = "";
			// postfix = useDenoising? " w den" : " wo den";-
			io::showImage(name+" Source"+postfix,nxtFr,resize,saveResults);
			if(useDenoising) io::showImage(name+" Denoised"+postfix,nxtFrDenoised,resize,saveResults);
			if(!fgMask.empty()) io::showImage(name+" Foreground Mask"+postfix,fgMask,resize,saveResults);
			// if(!combMask.empty()) io::showImage(name+" Comb Mask",combMask,resize,saveResults);
			#ifdef FARNEBACK
			if(!flowFarneback.empty()) io::showImage(name+" Farneback"+postfix,flowFarneback,resize,saveResults);
			#else
			if(!optFlow.empty()) io::showImage(name+" OptFlow1"+postfix,optFlow,resize,saveResults);
			// if(!optFlow2.empty()) io::showImage(name+" OptFlow2"+postfix,optFlow2,resize,saveResults);
			#endif
			if(!weights.empty()) io::showImage(name+" Weights"+postfix,weights,resize,saveResults);
			if(!maskReg.empty()) io::showImage(name+" Weights thresholded"+postfix,maskReg,resize,saveResults);
			// if(!maskRegEntireFrame.empty()) io::showImage(name+" Wei Ent Fr thresholded"+postfix,maskRegEntireFrame,resize,saveResults);
			if(!morphRecMarker.empty()) io::showImage(name+" MR Marker"+postfix,morphRecMarker,resize,saveResults);
			if(!morphRecMask.empty()) io::showImage(name+" MR Mask"+postfix,morphRecMask,resize,saveResults);
			if(!morphRecResult.empty()) io::showImage(name+" MR Result"+postfix,morphRecResult,resize,saveResults);
			// if(!motMaskMorph.empty()) io::showImage(name+" Morph Rec"+postfix,motMaskMorph,resize,saveResults);
			// if(!motMaskExp.empty()) io::showImage(name+" Expansion"+postfix,motMaskExp,resize,saveResults);
			if(!motionMask.empty()) io::showImage(name+" MotionMask"+postfix,motionMask,resize,saveResults);
			// if(!motCompMask.empty()) io::showImage(name+" MotComp Mask"+postfix,motCompMask,resize,saveResults);
			
			// io::showMaskOverlap(fgMask,name+" Foreground Mask",motionMask,name+" MotionMask");
	    	// io::showMaskOverlap(motionMask,name+" MotionMask",postMotionMask,"PostMotionMask"+postfix);
		}
	}
}

#endif