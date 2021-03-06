#include <iomanip>
#include <pthread.h>
#include <sstream>
using std::stringstream;
#include <vector>
using std::vector;
#include <queue>
using std::queue;
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "image_processing.h"
#include "io.h"
#include "motion_detection.h"

/** Multithreading helpers **/
const int MAX_THREADS = 5;
queue<int> isAvailable;
#define EVAL_MULTITHREADING 0

/** Initialise iteration parameters **/
vector<string> lMd = {
	// FBOF
	LOBSTER
	// ,PAWCS
	,SUBSENSE
	,VIBE
};
vector<int> lDs = { // Dataset IDs
	 ds::CD_BRIDGE_ENTRY
	,ds::CD_BUSY_BOULEVARD
	,ds::CD_FLUID_HIGHWAY
	,ds::CD_STREETCORNER
	,ds::CD_TRAMSTATION
	,ds::CD_WINTERSTREET
};
vector<bool> lPp = {false,true}; // Post-processing

/** Static parameters **/
bool grayscale = true;
bool runMotionDetection = false;
bool runCalculateScores = true;

struct thread_data{
	string mdName;
	int DS_ID;
	double noiseStddev;
	bool applyPostProcessing;
	int idx;
};

void buildPathsOutput(string& pathImgs, string& pathScores, string mdName, string dsName, int noiseStddev, bool applyPostProcessing){
	stringstream ss;
	ss << ds::DIR_DS;
	ss << mdName<<"/";
	ss << (applyPostProcessing?"w-":"wo-")<<"pospro";
	ss << "_";
	ss << "noise-"<<std::setfill('0')<<std::setw(3)<<noiseStddev;
	ss << "/";

	pathScores = ss.str();

	ss << dsName<<"/";

	pathImgs = ss.str();
}

void endThread(int idx, MotionDetection* m){
	cout<<"endThread "<<idx<<endl;
	isAvailable.push(idx);
	#ifdef EVAL_MULTITHREADING
		delete m;
		pthread_exit(NULL);
	#endif
}

void *run (void* arg){
	struct thread_data* td = (struct thread_data*) arg;

	string mdName = td->mdName;
	int DS_ID = td->DS_ID;
	double noiseStddev = td->noiseStddev;
	bool applyPostProcessing = td->applyPostProcessing;
	int threadIdx = td->idx;
	cout<<"Go for "<<threadIdx<<endl;

	MotionDetection* m;
	loadMotionDetection(m,mdName);
	ds::Dataset d;
	ds::loadDataset(d,DS_ID,grayscale,noiseStddev);

	Mat frame,gt,motionMask;

	if(!d.hasGroundTruth()){
		cerr<<"Using dataset without GT : "<<d.getName()<<endl;
		endThread(threadIdx,m);
	}

	int idx = io::IDX_OUTPUT_START;
	d.next(frame);
	m->init(frame);

	string pathImgs, pathScores;
	buildPathsOutput(pathImgs,pathScores,m->getName(),d.getName(),noiseStddev,applyPostProcessing);

	if(!io::isDirExist(io::DIR_OUTPUT+pathImgs)){
		if(runMotionDetection){
			cout<<"Running motion detection for: "<<pathImgs<<endl;
			// io::clearOutput(pathImgs);
			for(; d.hasNext(); idx++){
				d.next(frame,gt);
				m->next(frame,motionMask,applyPostProcessing);

				// Build img filename
				char bufferImgName[10];
				sprintf(bufferImgName,io::REGEX_IMG_OUTPUT.c_str(),idx);
				io::saveImage(pathImgs,string(bufferImgName),motionMask);
			}
			cout<<"Finished motion detection for: "<<pathImgs<<endl;
		}
	}

	if(runCalculateScores){
		cout<<"Running score calculation for: "<<pathImgs<<endl;
		d.calculateScores(pathImgs,pathScores);
		cout<<"Finished score calculation for: "<<pathImgs<<endl;
	}

	endThread(threadIdx,m);
}

int main(){
	pthread_t threads[MAX_THREADS];
	struct thread_data td[MAX_THREADS];
	for(int i=0;i<MAX_THREADS;i++){
		isAvailable.push(i);
	}

	for(int mdIdx=0;mdIdx<lMd.size();mdIdx++){
	for(int dsIdx=0;dsIdx<lDs.size();dsIdx++){
	// for(int ns=NOISE_STDDEV_MIN;ns<=NOISE_STDDEV_MIN;ns+=NOISE_STDDEV_INC){
	for(int ns=NOISE_STDDEV_MIN;ns<=NOISE_STDDEV_MAX;ns+=NOISE_STDDEV_INC){
	for(int ppIdx=0;ppIdx<lPp.size();ppIdx++){

		while(isAvailable.empty())
			sleep(1);

		int idx = isAvailable.front();
		isAvailable.pop();

		string mdName = lMd[mdIdx];
		int DS_ID = lDs[dsIdx];
		double noiseStddev = ns;
		bool applyPostProcessing = lPp[ppIdx];

		td[idx].mdName = mdName;
		td[idx].DS_ID = DS_ID;
		td[idx].noiseStddev = noiseStddev;
		td[idx].applyPostProcessing = applyPostProcessing;
		td[idx].idx = idx;

		#ifdef EVAL_MULTITHREADING
			pthread_create(&threads[idx],NULL,run,(void*)&td[idx]);
		#else
			run((void*)&td[idx]);
		#endif
		
	}
	}
	}
	}

	pthread_exit(NULL);
	return 0;
}