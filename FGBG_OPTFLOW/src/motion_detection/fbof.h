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
	Fbof(string _name,bool _showResults=false,bool _useDenoising=false,bool _useForegroundFeatures=true):name(_name),showResults(_showResults),useDenoising(_useDenoising),useForegroundFeatures(_useForegroundFeatures){ init(); }
	~Fbof(){}

void motionDetection(Mat& prvFr,Mat& nxtFr,Mat& motCompMask,Mat& motionMask,bool usePostProcessing,bool onlyUpdateBGModel,bool useRegExpansion);

protected:
	typedef Vec<short,3> DataVec;
	// typedef Vec<float,3> DataVec;
	typedef Vec<float,2> OptFlowVec;

	string name;
	Ptr<BackgroundSubtractor> pMOG2;
	bool showResults;
	bool useDenoising;
	bool useForegroundFeatures;

	void init();
	void checkRegulData(const Mat& mask, const Mat& regulData);
	void denoise(const Mat& src, Mat& dst);
	void backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2);
	void getOptFlowFeatures(const Mat& motCompMask, const Mat& fgMask, Mat& combinedMask, vector<Point2f>& goodFeaturesToTrack, bool useForegroundFeatures);
	void opticalFlow(const Mat& prvFr, const Mat& nxtFr, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask);
	void similarNeighbourWeighting(const Mat& data, Mat& weights);
	void optFlowRegularization(const Size& size, Mat& dst, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& data);
	void expandByList(const Mat& mask, Mat& marker, vector<bool>& expanded, vector<int> toExpand, short deltaX, short deltaY);
	void expandMarker(const Mat& mask, const Mat& data, Mat& marker);
	void morphologicalReconstruction(Mat& dst, const Mat& mask, const Mat& marker);
	void regularizeDataByList(const Mat& mask, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, short deltaX, short deltaY);
	void regularizeData(const Mat& marker, const Mat& mask, const Mat& data, Mat& regulData);
	void applyMotion(const Mat& src, const Mat& regulData, Mat& dst);
	// void motionCompensation(const Mat& motionMask, Mat& motCompMask, const Mat& data);
	void postProcessing(Mat& img);
};

void Fbof::init(){
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
}

void Fbof::backgroundSubtraction(const Mat& frame, Mat& fgMask, Ptr<BackgroundSubtractor> pMOG2){
	double learningRate = -1;
    pMOG2->apply(frame, fgMask,learningRate);
}

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
void Fbof::opticalFlow(const Mat& prvFr, const Mat& nxtFr, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& fgMask){
	
	bool usePyrLK = true;
	if(usePyrLK){
		TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
		Size winSize(21,21);
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
	}else{
		Mat flow;
		double pyr_scale = 0.5;
		int levels = 3;
		int winsize = 15;
		int iterations = 3;
		int poly_n = 5; double poly_sigma = 1.1;
		// int poly_n = 7; double poly_sigma = 1.5;
		int flags = 0;

		calcOpticalFlowFarneback(prvFr,nxtFr,flow,pyr_scale,levels,winsize,iterations,poly_n,poly_sigma,flags);
		for(int x=0;x<flow.cols;x++){
		for(int y=0;y<flow.rows;y++){
			float deltaX = flow.at<OptFlowVec>(y,x)[0];
			float deltaY = flow.at<OptFlowVec>(y,x)[1];
			if( abs(deltaX)>1 || abs(deltaY)>1 ){
				prvPts.push_back(Point2f(y,x));
				nxtPts.push_back(Point2f(y+deltaY,x+deltaX));
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
		io::showImage("Farneback",flow,true);
	}
}

void Fbof::similarNeighbourWeighting(const Mat& data, Mat& weights){
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
		// float deltaX = data.at<DataVec>(y,x)[1];
		// float deltaY = data.at<DataVec>(y,x)[2];
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
		// float weight = ((1<<neighbours)-1)/maxWeight;
		float maxNeighbours = 8;
		float weight = neighbours/maxNeighbours;
		weights.at<float>(y,x) = weight;
	}
	}
}

/*
* TODO thresholding at half the weight == 8 equal neighbours >> 7 equal neighbours == (2⁷-1)/255 = 0.49!
* Remove exponential weighting in similarNeighbourWeighting
*/
void Fbof::optFlowRegularization(const Size& size, Mat& dst, vector<uchar>& status, vector<Point2f>& prvPts, vector<Point2f>& nxtPts, Mat& weights, Mat& data){

    data = Mat::zeros(size,CV_16SC3);
    // data = Mat::zeros(size,CV_32FC3);
	weights = Mat::zeros(size,CV_32FC1);

	bool usePyrLK = true;

	// Optical flow to data
    for(int ptsIdx=0;ptsIdx<prvPts.size();ptsIdx++){
    	if(!usePyrLK || status[ptsIdx]){
			int x1=prvPts[ptsIdx].x,y1=prvPts[ptsIdx].y,x2=nxtPts[ptsIdx].x,y2=nxtPts[ptsIdx].y;
			short deltaX = x2-x1;
			short deltaY = y2-y1;
			// float x1=prvPts[ptsIdx].x,y1=prvPts[ptsIdx].y,x2=nxtPts[ptsIdx].x,y2=nxtPts[ptsIdx].y;
			// float deltaX = x2-x1;
			// float deltaY = y2-y1;
			// if(abs(deltaX) > 1 || abs(deltaY) > 1){
				data.at<DataVec>(y1,x1)[0] = ptsIdx;
				data.at<DataVec>(y1,x1)[1] = deltaX;
				data.at<DataVec>(y1,x1)[2] = deltaY;
			// }
    	}
    }

    // Weighting for regularization
	similarNeighbourWeighting(data,weights);

    dst = Mat::ones(size,CV_8UC1);
    weights.convertTo(dst,CV_8UC1,255);
   	threshold(dst,dst,127);
}

/* Main difference in result with morphological reconstruction:
* the implemented MR used a circuler structuring element,
* while this method searches withing a window around a pixel
* (pixel at the center, window is square with edge size = 2r+1 for a set radius r)
*/
void Fbof::expandByList(const Mat& mask, Mat& marker, vector<bool>& expanded, vector<int> toExpand, short deltaX, short deltaY){
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

void Fbof::expandMarker(const Mat& mask, const Mat& data, Mat& marker){
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
				// float deltaX = data.at<DataVec>(y,x)[1];
				// float deltaY = data.at<DataVec>(y,x)[2];

				toExpand.push_back(idx);
				expandByList(mask,marker,expanded,toExpand,deltaX,deltaY);
	    	}
	    }
    }
    }
}

void Fbof::morphologicalReconstruction(Mat& dst, const Mat& mask, const Mat& marker){
   	Mat dilation,maskedDilation,prevMaskedDilation;
   	bool hasChanged;
   	int strucSize = 7;
   	// int strucSize = 5; // @see expandByList : Corresponds to radius r=2 (diameter 2r+1)
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

/*
* Nu mogelijk dat er een memory overload is doordat buren van een pixel gepushd worden terwijl ze er al inzitten (veeel redundantie).
* Mogelijks efficienter: 2e set array die unieke indexen van toExpand bijhoudt, gebruiken voor check. toExpand behouden om volgorde van expanderen te bepalen.
* TODO --^ 
*/
void Fbof::regularizeDataByList(const Mat& mask, vector<bool>& expanded, vector<int> toExpand, Mat& regulData, short deltaX, short deltaY){
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
* Op deze manier worden alle zeker MVs al gezet en pas daarna worden deze geëxpandeerd.
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
				short deltaX = data.at<DataVec>(y,x)[1];
				short deltaY = data.at<DataVec>(y,x)[2];
				// float deltaX = data.at<DataVec>(y,x)[1];
				// float deltaY = data.at<DataVec>(y,x)[2];

				toExpand.push_back(idx);
				regularizeDataByList(mask,expanded,toExpand,regulData,deltaX,deltaY);
	    	}
	    }
    }
    }
    // checkRegulData(mask,regulData);
}

void Fbof::applyMotion(const Mat& src, const Mat& regulData, Mat& dst){
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
// void Fbof::motionCompensation(const Mat& motionMask, Mat& motCompMask, const Mat& data){
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

void Fbof::postProcessing(Mat& img){
   	int strucSize = 3;
   	medianBlur(img,img,strucSize);
}

void Fbof::motionDetection(Mat& prvFr, Mat& nxtFr, Mat& motCompMask, Mat& motionMask, bool usePostProcessing, bool onlyUpdateBGModel=false, bool useRegExpansion=false){

	// vector<bool> lOptFlowOnEntireImage={true,false};

	vector<uchar> status;
	vector<Point2f> prvPts,nxtPts;
	Mat data, regulData;
	Mat prvFrDenoised,fgMask,combMask,maskReg,morphRecMask,morphRecMarker,weights,optFlow,postMotionMask;

	Mat motionMask_wo_OFEI;

	if(useDenoising)
		denoise(prvFr,prvFrDenoised);
	else
		prvFr.copyTo(prvFrDenoised);

	backgroundSubtraction(prvFrDenoised,fgMask,pMOG2);

	if(!onlyUpdateBGModel){
		string namePrefix = name;
		// for(bool optFlowOnEntireImage : {false,true}){
		for(bool optFlowOnEntireImage : {false}){
			name = namePrefix+" w"+(optFlowOnEntireImage?"":"o")+"_OFEI";
			if(optFlowOnEntireImage){
				binMat2Vec(Mat::ones(prvFr.size(),CV_8UC1),prvPts);
				opticalFlow(prvFr,nxtFr,status,prvPts,nxtPts,fgMask);
				optFlowRegularization(prvFr.size(),maskReg,status,prvPts,nxtPts,weights,data);
				bitwise_or(fgMask,maskReg,morphRecMask);
			}else{
				fgMask.copyTo(morphRecMask);
			}
			getOptFlowFeatures(motCompMask,fgMask,combMask,prvPts,useForegroundFeatures);
			opticalFlow(prvFr,nxtFr,status,prvPts,nxtPts,fgMask);
			optFlowRegularization(prvFr.size(),maskReg,status,prvPts,nxtPts,weights,data);
			maskReg.copyTo(morphRecMarker);
			// io::showMaskOverlap(fgMask,"fgMask",morphRecMarker,"morphRecMarker",morphRecMask,"morphRecMask");
			// io::showMaskOverlap(morphRecMarker,"morphRecMarker",morphRecMask,"morphRecMask");

			motionMask = Mat::zeros(prvFr.size(),CV_8UC1);
			if(!useRegExpansion){
			   	// Morphological reconstruction
			   	morphologicalReconstruction(motionMask,morphRecMask,morphRecMarker);
		   	}else{
		   		// Morphological reconstruction by region expansion
			   	morphRecMarker.copyTo(motionMask);
			   	expandMarker(morphRecMask,data,motionMask);
		   	}

		   	if(!optFlowOnEntireImage) motionMask.copyTo(motionMask_wo_OFEI);

		   	/* Compare morphological reconstruction by iterative dilation and differencing, and by marker expansion.
		   	See notes at expandMarker, correct implementation must give same results as first MR method, but more efficiently. *
			Mat maskMorph,maskExp;
			morphologicalReconstruction(maskMorph,fgMask,maskReg);
			maskReg.copyTo(maskExp);
			expandMarker(fgMask,data,maskExp);
			cout<<"Morph vs Exp"<<endl;
			io::printScores(maskMorph,maskExp);
		    io::showMaskOverlap(maskMorph,"Morph",maskExp,"Exp");
			maskMorph.copyTo(motionMask); /**/

		   	if(usePostProcessing){
		   		postProcessing(motionMask);
			}

		   	// regularizeData(maskReg,motionMask,data,regulData);

		   	// applyMotion(motionMask,regulData,postMotionMask);
		   	
			// motionCompensation(motionMask,motCompMask,regulData);


		   	if(showResults){
				// Draw optical flow motion vectors
	    		prvFr.copyTo(optFlow);
			    if(optFlow.channels()==1) cvtColor(optFlow,optFlow,CV_GRAY2BGR);
			    Scalar red(0,0,255), blue(255,0,0), green(0,255,0);
				int thickness=1, lineType=8, shift=0;
				double tipLength=0.1;
				int width = data.cols, height = data.rows;
				// int skip = 1;
				int skip = 5;
			    for(int x=0;x<width;x+=skip){
			    for(int y=0;y<height;y+=skip){
			    	if(data.at<DataVec>(y,x)[1] > 0 || data.at<DataVec>(y,x)[2] > 0){
						short newX = x + data.at<DataVec>(y,x)[1];
						short newY = y + data.at<DataVec>(y,x)[2];
						if(newX >= 0 && newY >= 0 && newX < width && newY < height){
			    			arrowedLine(optFlow,Point2f(x,y),Point2f(newX,newY),green,thickness,lineType,shift,tipLength);
						}
					}
			    }
			    }

			    bool saveResults = false;
				bool resize = true;
				string postfix = "";
				// postfix = useDenoising? " w den" : " wo den";-
				// if(useDenoising) io::showImage(name+" Denoised"+postfix,prvFrDenoised,resize,saveResults);
				io::showImage(name+" Foreground Mask"+postfix,fgMask,resize,saveResults);
				// io::showImage(name+" Comb Mask",combMask,resize,saveResults);
				io::showImage(name+" OptFlow1"+postfix,optFlow,resize,saveResults);
				io::showImage(name+" Weights"+postfix,weights,resize,saveResults);
				io::showImage(name+" Weights thresholded"+postfix,maskReg,resize,saveResults);
				// io::showImage(name+" Morph Rec"+postfix,maskMorph,resize,saveResults);
				// io::showImage(name+" Expansion"+postfix,maskExp,resize,saveResults);
				io::showImage(name+" MR Marker"+postfix,morphRecMarker,resize,saveResults);
				io::showImage(name+" MR Mask"+postfix,morphRecMask,resize,saveResults);
				io::showImage(name+" MotionMask"+postfix,motionMask,resize,saveResults);
		    	// io::showMaskOverlap(motionMask,name+" MotionMask",postMotionMask,"PostMotionMask"+postfix,saveResults);
		    }
		    name = namePrefix;
		}
		// io::showMaskOverlap(motionMask_wo_OFEI,"motMasWoOFEI",motionMask,"motMas");
	}
}

#endif