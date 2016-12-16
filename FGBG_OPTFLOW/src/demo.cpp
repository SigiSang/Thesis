#include "dataset.h"
#include "io.h"
#include "motion_detection.h"

MotionDetection* md;
MotionDetection* md2;

void stop(){
	if(md) delete md;
	if(md2) delete md2;
	destroyAllWindows();
	exit(0);
}

int main(int argc,char **argv){
	char keyboard;
	const short waitInterval = 100;
	int wait = 0;
	// int wait = waitInterval;

	Mat img,gt;
	Mat motMask;
	Mat motMask2;
	// int DS_ID = ds::CD_BRIDGE_ENTRY;
	int DS_ID = ds::MISC_test;
	bool grayscale = true;
	ds::Dataset d;
	ds::loadDataset(d,DS_ID,grayscale);
	bool hasGT = d.hasGroundTruth();

	bool firstFrame = true;
	bool applyPostProcessing = false;
	bool applyNoise = true;
	bool resize = false;
	string nameMd1,nameMd2;
	while(d.hasNext()){
		if(hasGT){
			d.next(img,gt);
			io::showImage("Input",img,resize);
			io::showImage("GT",gt,resize);
		}else{
			d.next(img);
			io::showImage("Input",img,resize);
		}
		if(applyNoise) addNoise(img,img);
		io::showImage("Input",img,resize);
		if(firstFrame){
			md = new MdFbof(img); nameMd1 = "FBOF";
			// md = new MdLobster(img); nameMd1 = "Lobster";
			// md = new MdPawcs(img); nameMd2 = "PAWCS";
			md2 = new MdSubsense(img); nameMd2 = "SuBSENSE";
			// md2 = new MdVibe(img); nameMd2 = "ViBe+";

			firstFrame = false;
		}else{
			md->next(img,motMask,applyPostProcessing);
			io::showImage(nameMd1,motMask,resize);
			if(md2){
				md2->next(img,motMask2,applyPostProcessing);
				io::showImage(nameMd2,motMask2,resize);
			}
		}

		keyboard = waitKey(wait);
		if(keyboard == -1 || keyboard == '\n'){
			// Nothing to do here, keep scrolling
		}else if(keyboard == '\e'){
			stop();
		}else if(keyboard == 'n'){
			applyNoise = !applyNoise;
		}else if(keyboard == 'b'){
			applyPostProcessing = !applyPostProcessing;
		}else if(keyboard == 'p'){
			wait = (wait+waitInterval)%(2*waitInterval);
		}else{
			cout<<"Pressing unknown key: "<<keyboard<<endl;
		}
	}

	stop();
}