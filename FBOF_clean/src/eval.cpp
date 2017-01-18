#include <chrono>
#include <iomanip>
#include <pthread.h>
#include <random>
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
#define EVAL_MULTITHREADING
ofstream osLog;

/** Initialise iteration parameters **/
vector<string> lMd = {
	FBOF
	// ,LOBSTER // done
	// ,PAWCS
	// ,SUBSENSE // done
	// ,VIBE // done
	// ,EFIC // done
};
vector<int> lDs = { // Dataset IDs
	 ds::CD_BRIDGE_ENTRY
	,ds::CD_BUSY_BOULEVARD
	,ds::CD_FLUID_HIGHWAY
	,ds::CD_STREETCORNER
	,ds::CD_TRAMSTATION
	,ds::CD_WINTERSTREET
};
vector<bool> lPp = {false};
// vector<bool> lPp = {false,true}; // Post-processing

/** Static parameters **/
bool grayscale = true;
bool runMotionDetection = true;
bool runCalculateScores = true;

struct thread_data{
	string mdName;
	int DS_ID;
	double noiseStddev;
	bool applyPostProcessing;
	int idx;
};

/* Printing and logging */
void println(ostream& os, const char *line){
	os<<line<<endl;
}

void printAndLogln(const char *line){
	println(cout,line);
	println(osLog,line);
}

void printAndLogln(string line){
	printAndLogln(line.c_str());
}
/*  */

void waitRandom(){
	float sleepTime = 10*rand()/RAND_MAX;        
    usleep(sleepTime);
}

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
		waitRandom();
		pthread_exit(NULL);
	#endif
}

void *run (void* arg){
	stringstream ss;

	struct thread_data* td = (struct thread_data*) arg;

	string mdName = td->mdName;
	int DS_ID = td->DS_ID;
	double noiseStddev = td->noiseStddev;
	bool applyPostProcessing = td->applyPostProcessing;
	int threadIdx = td->idx;
	ss.str("");	ss<<"Go for "<<threadIdx;	printAndLogln(ss.str());

	MotionDetection* m;
	if(mdName!=EFIC) loadMotionDetection(m,mdName);
	ds::Dataset d;
	ds::loadDataset(d,DS_ID,grayscale,noiseStddev);

	Mat frame,gt,motionMask;

	if(!d.hasGroundTruth()){
		cerr<<"Using dataset without GT : "<<d.getName()<<endl;
		endThread(threadIdx,m);
	}

	int idx = io::IDX_OUTPUT_START;

	d.next(frame);
	if(mdName!=EFIC) m->init(frame);

	string pathImgs, pathScores;
	buildPathsOutput(pathImgs,pathScores,mdName,d.getName(),noiseStddev,applyPostProcessing);

	if( runMotionDetection && (mdName==FBOF || !io::isDirExist(io::DIR_OUTPUT+pathImgs)) ){
		printAndLogln("Running motion detection for: "+pathImgs);
		for(; d.hasNext(); idx++){
			d.next(frame,gt);
			m->next(frame,motionMask,applyPostProcessing);

			// Build img filename
			char bufferImgName[10];
			sprintf(bufferImgName,io::REGEX_IMG_OUTPUT.c_str(),idx);
			io::saveImage(pathImgs,string(bufferImgName),motionMask);
		}
		printAndLogln("Finished motion detection for: "+pathImgs);
	}

	if(runCalculateScores){
		if(!io::isDirExist(io::DIR_OUTPUT+pathImgs)){
			printAndLogln("WARNING: No output for "+pathImgs);
		}else{
			printAndLogln("Running score calculation for: "+pathImgs);
			d.calculateScores(pathImgs,pathScores);
			printAndLogln("Finished score calculation for: "+pathImgs);
		}
	}

	endThread(threadIdx,m);
}

int main(){
	pthread_t threads[MAX_THREADS];
	struct thread_data td[MAX_THREADS];
	for(int i=0;i<MAX_THREADS;i++){
		isAvailable.push(i);
	}
	if(runCalculateScores) io::openLogFile("Calculate_scores", osLog);

	for(int mdIdx=0;mdIdx<lMd.size();mdIdx++){
	for(int dsIdx=0;dsIdx<lDs.size();dsIdx++){
	for(int ns=30;ns<=NOISE_STDDEV_MAX;ns+=NOISE_STDDEV_INC){
	// for(int ns=NOISE_STDDEV_MIN;ns<=NOISE_STDDEV_MAX;ns+=NOISE_STDDEV_INC){
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
	osLog.close();
	return 0;
}