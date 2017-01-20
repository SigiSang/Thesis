/*

This code was written by Tim Ranson for the master's dissertation 'Noise-robust motion detection for low-light videos' by Tim Ranson.

*/


#ifndef DATASETS_
#define DATASETS_

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <opencv2/opencv.hpp>
using namespace cv;

#include <glob.h>

#include "io.h"

namespace ds{

	const string DIR_DS = "datasets/";
	// const string DIR_SCORES = "scores/";
	const string DS_AVSS = "AVSS/";
	const string DS_CD = "CD.NET/";
	const string DS_MISC = "MISC/";

	const int AVSS_SHORT = 0;
	const int CD_BRIDGE_ENTRY = 1;
	const int CD_BUSY_BOULEVARD = 2;
	const int CD_FLUID_HIGHWAY = 3;
	const int CD_STREETCORNER = 4;
	const int CD_TRAMSTATION = 5;
	const int CD_WINTERSTREET = 6;
	const int MISC_1 = 7;
	const int MISC_2 = 8;
	const int MISC_3 = 9;
	const int MISC_test = 999;
	const int MISC_test2 = 998;

	const string REGEX_IMG_INPUT = "in%06d.jpg";
	const string REGEX_GT_INPUT = "gt%06d.png";
	const string REGEX_NOISE_STDDEV = "%03d_";

	class Dataset{
	public:
		Dataset(){};
		Dataset(string _relPath, string _inOrig, bool _grayscale):relPath(_relPath),inOrig(_inOrig),grayscale(_grayscale){ init(); }
		Dataset(string _relPath, string _inOrig, string _inGT, bool _grayscale, double _noiseStddev):relPath(_relPath),inOrig(_inOrig),inGT(_inGT),grayscale(_grayscale){ init(_noiseStddev); }
		~Dataset(){
			vcOrig.release();
		}

		int getWidth(const VideoCapture& vc){ return vc.get(CAP_PROP_FRAME_WIDTH); }
		int getHeight(const VideoCapture& vc){ return vc.get(CAP_PROP_FRAME_HEIGHT); }
		int getIdxNext(const VideoCapture& vc){ return vc.get(CAP_PROP_POS_FRAMES); }
		int getFrameCount(const VideoCapture& vc){ return vc.get(CAP_PROP_FRAME_COUNT); }

		bool hasNext(const VideoCapture& vc){ return getIdxNext(vc)<getFrameCount(vc); }
		bool hasNext(){ return hasNext(vcOrig); }
		bool hasGroundTruth(){ return !inGT.empty(); }

		string getName(){ return dsName; }
		string getRegexInput(){ return inOrig; }
		string getSrcPath(){ return io::DIR_INPUT+DIR_DS+relPath; }
		string getIOPathFrames(){ return DIR_DS+relPath+dsName+"/"; }

		string getRegexInputNoisy(int noiseStddev){
			string inOrigNoisy = inOrig;
			string postfixDirNoisy = "_noisy/";
			if(inOrig.find(postfixDirNoisy) == string::npos){
				char buffer[10];
				sprintf(buffer,REGEX_NOISE_STDDEV.c_str(),noiseStddev);

				string regexNoisy = postfixDirNoisy+REGEX_IMG_INPUT;
				size_t pos = regexNoisy.find_first_of('%');
				regexNoisy.insert(pos,buffer);

				pos = inOrigNoisy.find_last_of('/');
				inOrigNoisy.replace(pos,(size_t)regexNoisy.size(),regexNoisy);
			}
			return inOrigNoisy;
		}

		void next(Mat& img){
			Mat emptyMat;
			next(img,emptyMat);
		}

		void next(Mat& img,Mat& gt){
			next(vcOrig,img);
			if(hasGroundTruth())
				nextBinary(vcGt,gt);
		}

		void saveOutput(Mat& img){
			string img_path = getIOPathFrames();

			if(currIdxOutput==io::IDX_OUTPUT_START){
				io::clearOutput(img_path);
			}

			char buffer[10];
			sprintf(buffer,io::REGEX_IMG_OUTPUT.c_str(),currIdxOutput++);
			string img_name(buffer);
			io::saveImage(img_path,img_name,img);
		}

		void calculateScores(string& pathImgs, string& pathScores){
			if(!hasGroundTruth()){
				cerr<<"datasets.h : Dataset.calculateScores() : Trying to calculate scores for dataset without GT!"<<endl;
				throw;
			}

			// string pattern=io::DIR_OUTPUT+pathScores+dsName+"*.csv";
			// glob_t glob_result;
			// int ret = glob(pattern.c_str(),GLOB_PERIOD,NULL,&glob_result);
			// if(glob_result.gl_pathc>0){
			// 	cout<<"Skipping                      "<<pathScores+dsName<<endl;
			// 	globfree(&glob_result);
			// 	return;
			// }
			// globfree(&glob_result);

			VideoCapture vcOutput(io::DIR_OUTPUT+pathImgs+io::REGEX_IMG_OUTPUT);

			if(vcGt.isOpened()) vcGt.release();
			vcGt = VideoCapture(getSrcPath()+inGT);

			vector<double> data;
			vector<double> dataTotals;

			ofstream csv;
			io::openCsvFile(pathScores,dsName,csv);
			io::writeScoresHeader(csv,dsName);

			Mat out,gt;
			while(hasNext(vcOutput)){
				nextBinary(vcOutput,out);
				nextBinary(vcGt,gt);
				scores::calculateConfusionMatrix(gt,out,data);
				io::writeScoresRow(csv,getIdxNext(vcOutput),data);
				scores::updateTotals(data,dataTotals);
			}

			io::writeScoresFooter(csv,dsName,dataTotals);
			io::closeFile(csv);
		}

	protected:
		int currIdxOutput;
		string relPath;
		string inOrig;
		string inGT;
		string dsName;
		bool grayscale;
		VideoCapture vcOrig;
		VideoCapture vcGt;

		void init(double noiseStddev=0){
			currIdxOutput = io::IDX_OUTPUT_START;

			if(noiseStddev>0){
				inOrig = getRegexInputNoisy(noiseStddev);
			}

			size_t pos = inOrig.find_first_of("/");
			if(pos == string::npos){
				pos = inOrig.find_first_of(".");
			}
			dsName = inOrig.substr(0,pos);

			vcOrig = VideoCapture(getSrcPath()+inOrig);
			if(hasGroundTruth())
				vcGt = VideoCapture(getSrcPath()+inGT);
		}

		void next(VideoCapture& vc, Mat& img){
			vc.read(img);
			if(grayscale)
				cvtColor(img,img,COLOR_BGR2GRAY);
		}

		void nextBinary(VideoCapture& vc, Mat& img){
			vc.read(img);
			if(img.channels()>1){
				cvtColor(img,img,COLOR_BGR2GRAY);
			}
		}
	};

	void loadDataset(Dataset& d, const int id, bool grayscale, double noiseStddev=0){
		switch(id){
			case AVSS_SHORT : d=Dataset(DS_AVSS,"AVSS_PV_Night_Short.avi",grayscale);break;
			case CD_BRIDGE_ENTRY : d=Dataset(DS_CD+"nightVideos/","bridgeEntry/input/"+REGEX_IMG_INPUT,"bridgeEntry/groundtruth/"+REGEX_GT_INPUT,grayscale,noiseStddev);break;
			case CD_BUSY_BOULEVARD : d=Dataset(DS_CD+"nightVideos/","busyBoulvard/input/"+REGEX_IMG_INPUT,"busyBoulvard/groundtruth/"+REGEX_GT_INPUT,grayscale,noiseStddev);break;
			case CD_FLUID_HIGHWAY : d=Dataset(DS_CD+"nightVideos/","fluidHighway/input/"+REGEX_IMG_INPUT,"fluidHighway/groundtruth/"+REGEX_GT_INPUT,grayscale,noiseStddev);break;
			case CD_STREETCORNER : d=Dataset(DS_CD+"nightVideos/","streetCornerAtNight/input/"+REGEX_IMG_INPUT,"streetCornerAtNight/groundtruth/"+REGEX_GT_INPUT,grayscale,noiseStddev);break;
			case CD_TRAMSTATION : d=Dataset(DS_CD+"nightVideos/","tramStation/input/"+REGEX_IMG_INPUT,"tramStation/groundtruth/"+REGEX_GT_INPUT,grayscale,noiseStddev);break;
			case CD_WINTERSTREET : d=Dataset(DS_CD+"nightVideos/","winterStreet/input/"+REGEX_IMG_INPUT,"winterStreet/groundtruth/"+REGEX_GT_INPUT,grayscale,noiseStddev);break;
			case MISC_1 : d=Dataset(DS_MISC,"M2U00006_edit.mpeg",grayscale);break;
			case MISC_2 : d=Dataset(DS_MISC,"M2U00007.MPG",grayscale);break;
			case MISC_3 : d=Dataset(DS_MISC,"M2U00009_edit.mpeg",grayscale);break;
			case MISC_test : d=Dataset(DS_MISC,"M2U00009_edit_quick_test.mpeg",grayscale);break;
			case MISC_test2 : d=Dataset(DS_MISC,"2016-11-17-180945.webm",grayscale);break;
			// case MISC_test2 : d=Dataset(DS_MISC,"MOV_0013.MP4",grayscale);break;
			default: cerr<<"Failed to load dataset with id: "<<id<<endl;throw;
		}
	}
}

#endif