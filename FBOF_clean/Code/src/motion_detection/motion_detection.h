/*

This code was written by Tim Ranson for the master's dissertation 'Noise-robust motion detection for low-light videos' by Tim Ranson.

*/


#ifndef MOTION_DETECTION_
#define MOTION_DETECTION_

#include <opencv2/opencv.hpp>

#include "fbof.h"
#include "libvibe++/ViBe.h"
#include "libvibe++/distances/Manhattan.h"
#include "BackgroundSubtractorLOBSTER.h"
// #include "BackgroundSubtractorPAWCS.h"
#include "BackgroundSubtractorSuBSENSE.h"
// #include "KDE.h"

const string FBOF = "fbof";
const string LOBSTER = "lobster";
const string SUBSENSE = "subsense";
const string VIBE = "vibe";
const string EFIC = "efic";

class MotionDetection{
public:
	virtual void next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing)=0;
	virtual void init(Mat& firstFr)=0;

	string getName() const { return name; }
protected:
	string name;

	virtual void destroy()=0;
};

///////// FBOF //////////////////////////////////////////////////////

class MdFbof : public MotionDetection{
public:
	MdFbof(){}
	MdFbof(Mat& firstFr){ init(firstFr); }
	~MdFbof(){ destroy(); }

	void next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing);
	void init(Mat& firstFr);

protected:
	Fbof* fbof;
	Mat prvFr;

	void destroy();
};

void MdFbof::init(Mat& firstFr){
	name = FBOF;
	fbof = new Fbof("FgBg-OptFlow");
	firstFr.copyTo(prvFr);
}

void MdFbof::destroy(){
	if(fbof!=nullptr) delete fbof;
}

void MdFbof::next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing){
	Mat motCompMask = Mat::zeros(nxtFr.size(),nxtFr.depth());
	fbof->motionDetection(prvFr,nxtFr,motCompMask,motionMask,applyPostProcessing);
	nxtFr.copyTo(prvFr);
}

///////// LOBSTER //////////////////////////////////////////////////////

class MdLobster : public MotionDetection{
public:
	MdLobster(){ 
		// cout<<"Lobster ctor"<<endl;
	}
	MdLobster(Mat& firstFr){ init(firstFr); }
	~MdLobster(){ destroy(); }

	void next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing);
	void init(Mat& firstFr);

protected:
    BackgroundSubtractorLOBSTER bgsLobster;
    short frameIdx;

	void destroy();
};

void MdLobster::init(Mat& firstFr){
	name = LOBSTER;
	frameIdx = 1;
	Mat sequenceROI = Mat(firstFr.size(),CV_8UC1,cv::Scalar_<uchar>(255));
	bgsLobster.initialize(firstFr,sequenceROI);
}

void MdLobster::destroy(){
	// Nothing to see here, keep scrolling.
}

void MdLobster::next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing){
	bgsLobster.setApplyPostProcessing(applyPostProcessing);
    bgsLobster.apply(nxtFr,motionMask,double(frameIdx<=100?1:BGSLOBSTER_DEFAULT_LEARNING_RATE));
    frameIdx++;
}

///////// SUBSENSE //////////////////////////////////////////////////////

class MdSubsense : public MotionDetection{
public:
	MdSubsense(){}
	MdSubsense(Mat& firstFr){ init(firstFr); }
	~MdSubsense(){ destroy(); }

	void next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing);
	void init(Mat& firstFr);

protected:
    BackgroundSubtractorSuBSENSE bgsSubsense;
    short frameIdx;

	void destroy();
};

void MdSubsense::init(Mat& firstFr){
	name = SUBSENSE;
	frameIdx = 1;
	Mat sequenceROI = Mat(firstFr.size(),CV_8UC1,cv::Scalar_<uchar>(255));
	bgsSubsense.initialize(firstFr,sequenceROI);
}

void MdSubsense::destroy(){
	// Nothing to see here, keep scrolling.
}

void MdSubsense::next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing){
	bgsSubsense.setApplyPostProcessing(applyPostProcessing);
    bgsSubsense.apply(nxtFr,motionMask,double(frameIdx<=100));
    frameIdx++;
}

///////// VIBE //////////////////////////////////////////////////////

class MdVibe : public MotionDetection{
	typedef ViBe::ViBeSequential<1, ViBe::Manhattan<1> > ViBe; // both 1's = num of channels
public:
	MdVibe(){}
	MdVibe(Mat& firstFr){ init(firstFr); }
	~MdVibe(){ destroy(); }

	void next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing);
	void init(Mat& firstFr);

protected:
	ViBe* vibe;

	void destroy();
};

void MdVibe::init(Mat& firstFr){
	name = VIBE;
    int32_t height = firstFr.rows;
    int32_t width = firstFr.cols;
    const uint8_t* buffer = firstFr.data;
	vibe = new ViBe(height,width,buffer);
}

void MdVibe::destroy(){
	if(vibe!=nullptr) delete vibe;
}

void MdVibe::next(Mat& nxtFr, Mat& motionMask, bool applyPostProcessing){
	if(!motionMask.data){
		motionMask = Mat::zeros(nxtFr.size(),CV_8UC1);
	}
    vibe->segmentation(nxtFr.data, motionMask.data);
    vibe->update(nxtFr.data, motionMask.data);
    if(applyPostProcessing)
    	cv::medianBlur(motionMask,motionMask,3);
}

///////// FACTORY //////////////////////////////////////////////////////

void loadMotionDetection(MotionDetection*& md, string& name){
	if(name==FBOF){
		md = new MdFbof();
	}else if(name==LOBSTER){
		md = new MdLobster();
	// }else if(name==PAWCS){
	// 	md = new MdPawcs();
	}else if(name==SUBSENSE){
		md = new MdSubsense();
	}else if(name==VIBE){
		md = new MdVibe();
	}else{
		cerr<<"Warning: motion_detection.h : loadMotionDetection : invalid name:"<<name<<endl;
	}
}

#endif