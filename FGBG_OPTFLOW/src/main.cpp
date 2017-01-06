#include <string>
using std::string;
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <opencv2/video.hpp>

#include "dataset.h"
#include "fbof.h"
#include "io.h"
#include "scores.h"
// #include "motion_detection.h"

void stop(){
	destroyAllWindows();
	exit(0);
}

int main (int argc, char** argv){

	/* Keyboard input configuration */
	char keyboard;
	const short waitInterval = 100;
	int wait = 0;
	// int wait = waitInterval;

	/* Motion detection configuration */
	ds::Dataset dataset;
	// const int VID_ID = ds::CD_BRIDGE_ENTRY , skipToFrame = 0;
	// const int VID_ID = ds::CD_BUSY_BOULEVARD , skipToFrame = 745/;
	// const int VID_ID = ds::CD_STREETCORNER , skipToFrame = 80; // 865;
	const int VID_ID = ds::CD_TRAMSTATION, skipToFrame = 30;
	// const int VID_ID = ds::CD_TRAMSTATION, skipToFrame = 590;
	// const int VID_ID = ds::CD_WINTERSTREET, skipToFrame = 0;
	// const int VID_ID = ds::AVSS_SHORT, skipToFrame = 0;
	// const int VID_ID = ds::MISC_3, skipToFrame = 200;
	// const int VID_ID = ds::MISC_test2, skipToFrame = 0;
	bool grayscale = true;

	ds::loadDataset(dataset,VID_ID,grayscale);
	bool hasGT = dataset.hasGroundTruth();
	string dirInput = dataset.getIOPathFrames();
	string fnRoi = "ROI.bmp";

	string name="Original",nameNoisy="Noisy";

	short morpRecRadius = 2; // radius for morphological reconstruction
	bool useRegionExpansion;
	bool applyPostProcessing = true;
	bool showResults = true;
	bool useDenoising = true;
	bool useForegroundFeatures = true;

	bool showSrc = true;
	/**/
	bool withOriginal = false;
	bool withNoisy = true;
	/**
	bool withOriginal = true;
	bool withNoisy = false;
	/**
	bool withOriginal = true;
	bool withNoisy = true;
	/**/

	Mat prvFr,nxtFr,nxtGt,motCompMask,motionMask;
	Mat prvFrNoisy,nxtFrNoisy,motCompMaskNoisy,motionMaskNoisy;
	Mat ROI;
	io::readInputImage(dirInput,fnRoi,true,ROI);

	dataset.next(prvFr);
	Fbof fbof(name,morpRecRadius,showResults,useDenoising,useForegroundFeatures);
	Fbof fbofNoisy(nameNoisy,morpRecRadius,showResults,useDenoising,useForegroundFeatures);
	// Fbof fbofExp(name+" Exp",morpRecRadius,showResults,useDenoising,useForegroundFeatures);
	// Fbof fbofExpNoisy(nameNoisy+" Exp",morpRecRadius,showResults,useDenoising,useForegroundFeatures);

	motCompMask = Mat::zeros(prvFr.size(),CV_8UC1);
	motCompMask.copyTo(motCompMaskNoisy);

	int frameIdx = 0;
	while(dataset.hasNext()){
		frameIdx++;
		bool onlyUpdateBGModel = frameIdx<skipToFrame;

		dataset.next(nxtFr,nxtGt);

		addNoise(prvFr,prvFrNoisy);
		addNoise(nxtFr,nxtFrNoisy);

		// Use only morphological reconstruction
		useRegionExpansion = true;

		if(showSrc) io::showImage("Src",prvFr,true);
		if(withOriginal) fbof.motionDetection(prvFr,nxtFr,motCompMask,motionMask,applyPostProcessing,onlyUpdateBGModel,useRegionExpansion);
		if(withNoisy) fbofNoisy.motionDetection(prvFrNoisy,nxtFrNoisy,motCompMaskNoisy,motionMaskNoisy,applyPostProcessing,onlyUpdateBGModel,useRegionExpansion);

		if(hasGT && !onlyUpdateBGModel){
			io::showImage(name+" GT",nxtGt,true);
			bitwise_and(nxtGt,ROI,nxtGt);
			threshold(nxtGt,nxtGt,1);

			Mat motMaskROI,motMaskROINoisy;
	    	cout<<"Frame "<<frameIdx<<", with Morph"<<endl;

	    	if(withOriginal){
				bitwise_and(motionMask,ROI,motMaskROI);
				io::showImage(name+" motMaskROI",motMaskROI,true);
		    	cout<<"Comp ROI gt-normal: "<<endl;
				io::printScores(nxtGt,motionMask);
			}

			if(withNoisy){
		    	cout<<"Comp ROI gt-noisy: "<<endl;
				io::printScores(nxtGt,motionMaskNoisy);
				bitwise_and(motionMaskNoisy,ROI,motMaskROINoisy);
				io::showImage(nameNoisy+" motMaskROI",motMaskROINoisy,true);
			}

			cout<<endl;
		}

		// Use region expansion
		// useRegionExpansion = true;
		// fbofExp.motionDetection(prvFr,nxtFr,motCompMask,motionMask,applyPostProcessing,onlyUpdateBGModel,useRegionExpansion);
		// fbofExpNoisy.motionDetection(prvFrNoisy,nxtFrNoisy,motCompMaskNoisy,motionMaskNoisy,applyPostProcessing,onlyUpdateBGModel,useRegionExpansion);

		// if(!onlyUpdateBGModel){
		// 	Mat motMaskROI,motMaskROINoisy;
		// 	bitwise_and(motionMask,ROI,motMaskROI);
		// 	bitwise_and(motionMaskNoisy,ROI,motMaskROINoisy);

	 //    	cout<<"Frame "<<frameIdx<<", with Exp"<<endl;
	 //  //   	cout<<"Comp full inter: ";
		// 	// io::printScores(motionMask,motionMaskNoisy);
	 //    	cout<<"Comp ROI gt-normal: ";
		// 	io::printScores(nxtGt,motionMask);
	 //    	cout<<"Comp ROI gt-noisy: ";
		// 	io::printScores(nxtGt,motionMaskNoisy);
		// 	cout<<endl;
		// }
		
		if(!onlyUpdateBGModel){
			keyboard = waitKey(wait);
			if(keyboard == -1 || keyboard == '\n'){
				// Nothing to do here, keep scrolling
			}else if(keyboard == '\e'){
				stop();
			// }else if(keyboard == 'n'){
			// 	applyNoise = !applyNoise;
			}else if(keyboard == 'b'){
				applyPostProcessing = !applyPostProcessing;
			}else if(keyboard == 'p'){
				wait = (wait+waitInterval)%(2*waitInterval);
			}else{
				cout<<"Pressing unknown key: "<<keyboard<<endl;
			}
		}
		nxtFr.copyTo(prvFr);
	}

	stop();
}