#include "dataset.h"
#include "io.h"
#include "motion_detection.h"

MotionDetection* md;
MotionDetection* md2;

void stop(){
	if(md==nullptr) delete md;
	if(md2==nullptr) delete md2;
	destroyAllWindows();
	exit(0);
}

int main(int argc,char **argv){
	char keyboard;
	const short waitInterval = 100;
	int wait = waitInterval;

	Mat img,gt;
	Mat motMask;
	Mat motMask2;
	int DS_ID = ds::CD_BRIDGE_ENTRY;
	// int DS_ID = ds::MISC_test2;
	bool grayscale = true;
	ds::Dataset d;
	ds::loadDataset(d,DS_ID,grayscale);
	bool hasGT = d.hasGroundTruth();

	bool firstFrame = true;
	bool applyPostProcessing = false;
	bool applyNoise = true;
	bool resize = false;
	bool runAlgorithm = true;
	if(runAlgorithm)
	while(d.hasNext()){
		if(hasGT){
			d.next(img,gt);
			io::showImage("Next",img,resize);
			io::showImage("GT",gt,resize);
		}else{
			d.next(img);
			io::showImage("Next",img,resize);
		}
		if(applyNoise) addNoise(img,img);
		io::showImage("Next",img,resize);
		if(firstFrame){
			md = new MdFbof(img);
			// md = new MdLobster(img);
			// md = new MdPawcs(img);
			// md2 = new MdSubsense(img);
			md2 = new MdVibe(img);

			firstFrame = false;
		}else{
			md->next(img,motMask,applyPostProcessing);
			io::showImage("Motion mask",motMask,resize);
			if(md2){
				md2->next(img,motMask2,applyPostProcessing);
				io::showImage("Motion mask 2",motMask2,resize);
			}

			// d.saveOutput(motMask);
		}

		/*
		* ENTER : next frame
		* ESC	: stop immediately
		* B 	: switch post-processing
		* E 	: end (calculate scores and stop)
		* N 	: switch added noise
		* P 	: pause/play
 		*/
		keyboard = waitKey(wait);
		if(keyboard == -1 || keyboard == '\n'){
			// Nothing to do here, keep scrolling
		}else if(keyboard == '\e'){
			stop();
		}else if(keyboard == 'b'){
			applyPostProcessing = !applyPostProcessing;
		}else if(keyboard == 'e'){
			// d.calculateScores();
			stop();
		}else if(keyboard == 'n'){
			applyNoise = !applyNoise;
		}else if(keyboard == 'p'){
			wait = (wait+waitInterval)%(2*waitInterval);
		}else{
			cout<<"Pressing unknown key: "<<keyboard<<endl;
		}
	}

	// d.calculateScores();
	stop();
}