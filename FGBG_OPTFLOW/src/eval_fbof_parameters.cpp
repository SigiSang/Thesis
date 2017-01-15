#include <chrono>
#include <iomanip>
#include <pthread.h>
#include <random>
#include <sstream>
using std::stringstream;
#include <map>
using std::map;
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

const string P_MIN_VEC = "minVec";
const string P_R_SN = "r_sn";
const string P_T_SV = "t_sv";
const string P_T_SN = "t_sn";
const string P_R_MR = "r_mr";

map<string, vector<float> > paramValues;

/** Initialise iteration parameters **/
const vector<int> lDs = { // Dataset IDs
	 ds::CD_BRIDGE_ENTRY
	// ,ds::CD_BUSY_BOULEVARD
	// ,ds::CD_FLUID_HIGHWAY
	,ds::CD_STREETCORNER
	,ds::CD_TRAMSTATION
	// ,ds::CD_WINTERSTREET
};

map<int,int> temporalROIs; // For each dataset, index of ROI starting frame
int sampleSize = 300; // Number of frames with ground truth motion detection will be performed on for each dataset

/** Static parameters **/
bool grayscale = true;

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

void writeScoresPerNoiseHeader(ostream& os){
	os<< "paramVal" <<io::SC_DELIM;
	os<< "noise" <<io::SC_DELIM;
	os<< "dsName" <<io::SC_DELIM;
	os<< "TP" <<io::SC_DELIM;
	os<< "TN" <<io::SC_DELIM;
	os<< "FP" <<io::SC_DELIM;
	os<< "FN" <<io::SC_DELIM;
	os<< "RE" <<io::SC_DELIM;
	os<< "FPR" <<io::SC_DELIM;
    os<< endl;
}

void writeScoresRow(ostream& os, const vector<double>& data, float paramVal, int noiseStddev, string dsName){
	using namespace scores;

	if(!isZeroData(data)){
		os<< paramVal <<io::SC_DELIM;
		os<< noiseStddev <<io::SC_DELIM;
		os<< dsName <<io::SC_DELIM;
		os<< data[TP] <<io::SC_DELIM;
		os<< data[TN] <<io::SC_DELIM;
		os<< data[FP] <<io::SC_DELIM;
		os<< data[FN] <<io::SC_DELIM;
		os<< data[RE] <<io::SC_DELIM;
		os<< data[FPR] <<io::SC_DELIM;
	    os<< endl;
	}
}

void writeScoresRowPerDataset(ostream& os, const vector<double>& data, float paramVal, int noiseStddev, string dsName){
	writeScoresRow(os,data,paramVal,noiseStddev,dsName);
}

void writeScoresRowPerNoise(ostream& os, const vector<double>& data, float paramVal, int noiseStddev){
	vector<double> avgs(data.size());
	for(int i=0;i<data.size()-1;i++){
		avgs[i] = data[i]/data[scores::TOTAL_SAMPLES];
	}
	writeScoresRow(os,avgs,paramVal,noiseStddev,"all");
}

struct thread_data{
	string paramName;
	vector<float> values;
	float minVecLen_axis,t_sv,t_sn;
	short r_sn,r_mr;
	int idx;
};

void waitRandom(){
	float sleepTime = 10*rand()/RAND_MAX;        
    usleep(sleepTime);
}

void endThread(int idx){
	stringstream ss;
	ss<<"endThread "<<to_string(idx);
	printAndLogln(ss.str());
	isAvailable.push(idx);
	#ifdef EVAL_MULTITHREADING
		waitRandom();
		pthread_exit(NULL);
	#endif
}

void *run (void* arg){
	struct thread_data* td = (struct thread_data*) arg;

	string paramName = td->paramName;
	vector<float> values = td->values;
	float minVecLen_axis = td->minVecLen_axis;
	short r_sn = td->r_sn;
	float t_sv = td->t_sv;
	float t_sn = td->t_sn;
	short r_mr = td->r_mr;
	int threadIdx = td->idx;

	stringstream ss;
	ss<<"Go for "<<to_string(threadIdx)<<" ("<<paramName<<")";
	printAndLogln(ss.str());

	string csvName = paramName;
	ofstream csv;
	io::openCsvFile(io::DIR_EVAL_FBOF_PARAMS,csvName,csv);

	vector<double> data, datasetTotals, noiseTotals;

	for(int vIdx=0;vIdx<values.size();vIdx++){
		/**/
		float paramVal = values[vIdx];
		if(paramName==P_MIN_VEC){
			minVecLen_axis = paramVal;
		}else if(paramName==P_R_SN){
			r_sn = paramVal;
		}else if(paramName==P_T_SV){
			t_sv = paramVal;
		}else if(paramName==P_T_SN){
			t_sn = paramVal;
		}

		Fbof *f = new Fbof("FgBg-OptFlow",minVecLen_axis,r_sn,t_sv,t_sn,r_mr);

		for(int ns=NOISE_STDDEV_MIN;ns<=NOISE_STDDEV_MAX;ns+=NOISE_STDDEV_INC){
			
			writeScoresPerNoiseHeader(csv);
			for(int dsIdx=0;dsIdx<lDs.size();dsIdx++){

				int DS_ID = lDs[dsIdx];
				ds::Dataset d;
				ds::loadDataset(d,DS_ID,grayscale,ns);

				string dsName = d.getName();
				Mat prvFr,nxtFr,motCompMask,motionMask,gt;

				cout<< paramName << ": ";
				cout<< paramVal <<io::SC_DELIM;
				cout<< ns <<io::SC_DELIM;
				cout<< dsName <<io::SC_DELIM;
				cout<< endl;

				if(!d.hasGroundTruth()){
					printAndLogln("Using dataset without GT : "+d.getName());
					delete f;
					endThread(threadIdx);
				}

				int idx = io::IDX_OUTPUT_START;
				int temporalROI = temporalROIs[DS_ID];
				bool applyPostProcessing = false;
				bool onlyUpdateBGModel = true;

				d.next(prvFr,gt);

				for(; d.hasNext() && idx<temporalROI; idx++){
					d.next(nxtFr,gt);
					f->motionDetection(prvFr, nxtFr, motCompMask, motionMask, applyPostProcessing, onlyUpdateBGModel);
					nxtFr.copyTo(prvFr);
				}

				onlyUpdateBGModel = false;
				for(; d.hasNext() && idx<temporalROI+sampleSize; idx++){
					d.next(nxtFr,gt);
					f->motionDetection(prvFr, nxtFr, motCompMask, motionMask, applyPostProcessing, onlyUpdateBGModel);
					nxtFr.copyTo(prvFr);

					scores::calculateConfusionMatrix(gt,motionMask,data);
					scores::updateTotals(data,datasetTotals);
					data.clear();
				}
				scores::calculateMetrics(datasetTotals);
				writeScoresRowPerDataset(csv,datasetTotals,paramVal,ns,dsName);
				datasetTotals[scores::TOTAL_SAMPLES] = 1;
				scores::updateTotals(datasetTotals,noiseTotals);
				datasetTotals.clear();
			}	
			writeScoresRowPerNoise(csv,noiseTotals,paramVal,ns);
			csv<<endl;
			noiseTotals.clear();
		}
		delete f;
		/**/
	}
	io::closeFile(csv);
	endThread(threadIdx);
}

int main(){
	/* Set up multithreading */
	pthread_t threads[MAX_THREADS];
	struct thread_data td[MAX_THREADS];
	for(int i=0;i<MAX_THREADS;i++){
		isAvailable.push(i);
	}

	/* Set up logging */
	io::openLogFile("Eval_fbof_parameters", osLog);

	/* Load dataset ROI starting frame index */
	temporalROIs[ds::CD_BRIDGE_ENTRY] = 1000;
	temporalROIs[ds::CD_BUSY_BOULEVARD] = 730;
	temporalROIs[ds::CD_FLUID_HIGHWAY] = 400;
	temporalROIs[ds::CD_STREETCORNER] = 800;
	temporalROIs[ds::CD_TRAMSTATION] = 500;
	temporalROIs[ds::CD_WINTERSTREET] = 900;

	/* Iterable algorithm parameters */
	float minVecLen_axis = 1.0; // Minimum vector size along an axis, e.g. 1 will set threshold at length of vector (1,1)
	float t_sv = 0.05; // similarity threshold for similar vector estimation: similarity if difference is below threshold
	short r_sn = 1; // Neighbour radius for similar neighbour weighting
	float t_sn = 0.5; // percentage threshold for similar neighbour weights
	short r_mr = 2; // radius of structuring element for dilation during morphological reconstruction

	vector<float> values;

	// values.clear();
	// for(float v=1; v<=5; v+=1) values.push_back(v);
	// paramValues[P_MIN_VEC] = vector<float>(values);

	// values.clear();
	// for(float v=1; v<=3; v+=1) values.push_back(v);
	// paramValues[P_R_SN] = vector<float>(values);

	// values.clear();
	// for(float v=0.01; v<=0.19; v+=0.02) values.push_back(v);
	// paramValues[P_T_SV] = vector<float>(values);

	// values.clear();
	// for(float v=0.1; v<1.1; v+=0.1) values.push_back(v);
	// paramValues[P_T_SN] = vector<float>(values);

	// values.clear();
	// for(float v=1; v<=5; v+=1) values.push_back(v);
	// paramValues[P_R_MR] = vector<float>(values);

	for(auto itP=paramValues.begin();itP!=paramValues.end();itP++){

		while(isAvailable.empty())
			sleep(1);

		int idx = isAvailable.front();
		isAvailable.pop();

		td[idx].paramName = itP->first;
		td[idx].values = itP->second;
		td[idx].minVecLen_axis = minVecLen_axis;
		td[idx].t_sv = t_sv;
		td[idx].r_sn = r_sn;
		td[idx].t_sn = t_sn;
		td[idx].r_mr = r_mr;
		td[idx].idx = idx;

		#ifdef EVAL_MULTITHREADING
			pthread_create(&threads[idx],NULL,run,(void*)&td[idx]);
		#else
			run((void*)&td[idx]);
		#endif
	}

	pthread_exit(NULL);
	osLog.close();
	return 0;
}