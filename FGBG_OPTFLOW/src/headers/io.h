#ifndef _IO
#define _IO

#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <map>

#include "scores.h"

#if defined(_WIN32)
#include <direct.h>   // _mkdir
#endif

using namespace std;
using namespace cv;

namespace io {

	/** Directories **/
	const string DIR_INPUT = "/media/tim/Game drive/Data_thesis/input/";
	const string DIR_OUTPUT = "/media/tim/Game drive/Data_thesis/output/";

	/** Constants for Scores IO **/
	const char SC_DELIM = ',';

	/** Constants for Image IO **/
	const short IDX_OUTPUT_START = 1;
	const string REGEX_IMG_OUTPUT = "%06d.png";

	/** Shown image properties and helper values **/
	int imX=5,imY=5
		,shownImages=0
		,consecutiveImages=4
		,normalXSize=320
		,normalYSize=261;

	map<string,vector<int> > imgPos;

	const string currentDateTime() {
	    time_t     now = time(0);
	    struct tm  tstruct;
	    char       buf[80];
	    tstruct = *localtime(&now);
	    strftime(buf, sizeof(buf), "%Y-%m-%d.%R", &tstruct);

	    return buf;
	}

	bool fileExists(const string& name) {
		struct stat buffer;   
		return (stat (name.c_str(), &buffer) == 0); 
	}

	bool isDirExist(const std::string& path){
	#if defined(_WIN32)
	    struct _stat info;
	    if (_stat(path.c_str(), &info) != 0)
	    {
	        return false;
	    }
	    return (info.st_mode & _S_IFDIR) != 0;
	#else 
	    struct stat info;
	    if (stat(path.c_str(), &info) != 0)
	    {
	        return false;
	    }
	    return (info.st_mode & S_IFDIR) != 0;
	#endif
	}

	bool makePath(const std::string& path){
	#if defined(_WIN32)
	    int ret = _mkdir(path.c_str());
	#else
	    mode_t mode = 0755;
	    int ret = mkdir(path.c_str(), mode);
	#endif
	    if (ret == 0)
	        return true;

	    switch (errno)
	    {
	    case ENOENT:
	        // parent didn't exist, try to create it
	        {
	            int pos = path.find_last_of('/');
	            if (pos == std::string::npos)
	#if defined(_WIN32)
	                pos = path.find_last_of('\\');
	            if (pos == std::string::npos)
	#endif
	                return false;
	            if (!makePath( path.substr(0, pos) ))
	                return false;
	        }
	        // now, try to create again
	#if defined(_WIN32)
	        return 0 == _mkdir(path.c_str());
	#else 
	        return 0 == mkdir(path.c_str(), mode);
	#endif

	    case EEXIST:
	        // done!
	        return isDirExist(path);

	    default:
	        return false;
	    }
	}

	int clearDirectory(string& dirPath){
		DIR* dir;
		struct dirent* ent;
		if( (dir=opendir(dirPath.c_str())) != NULL ){
			while( (ent=readdir(dir)) != NULL ){
				string file( dirPath + ent->d_name);
				remove( file.c_str() );
			}
			closedir(dir);
			return 0;
		}else{
			return EXIT_FAILURE;
		}
	}

	void clearOutput(string& relPath){
		string dir = DIR_OUTPUT+relPath;
		clearDirectory(dir);
	}

	void readInputImage(string relPath, string fn, bool grayscale, Mat& img){
		string path = DIR_INPUT+relPath;
		if(!isDirExist(path)){
			cerr<<"Error reading image: invalid path: "<<path<<endl;
			return;
		}
		string fn_full = path+fn;
		if(!fileExists(fn_full)){
			cerr<<"Error reading image: invalid filename: "<<fn<<" at "<<path<<endl;
			return;
		}
		int flagColor = (grayscale? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
	   	img = imread(fn_full,flagColor);
	}

	void saveImage(string relPath, string fn, const Mat& img){
		string path = DIR_OUTPUT+relPath;
		string fn_full = path+fn;
		makePath(path);
	   	imwrite(fn_full,img);
	}

	void showImage(string windowName,const Mat& img, bool resize=false, bool save=false){
	    int windowFlag, xSize, ySize, margin=3;
	    if(resize){
	        windowFlag = WINDOW_NORMAL;
	        xSize = normalXSize;
	        ySize = normalYSize;
	    } else {
	        windowFlag = WINDOW_AUTOSIZE;
	        xSize = img.cols;
	        ySize = img.rows;
	    }

	    namedWindow(windowName,windowFlag);
	    imshow(windowName, img);
	    if(save) saveImage("images/",windowName+".png",img);

	    if(imgPos[windowName].empty()){
	    	imgPos[windowName] = vector<int>(2);
	    	imgPos[windowName][0]=imX;
	    	imgPos[windowName][1]=imY;
	    	moveWindow(windowName,imX,imY);
		    imX += xSize+margin;
		    if(++shownImages%consecutiveImages == 0){
		    	imX = 5;
		    	imY += ySize;
		    }
	    }
	}

	/*
	* Show overlap between two binary/grayscale images, resulting in an BGR image.
	* The first image is loaded into the green layer of the resulting image.
	* The second image is loaded into the red layer of the resulting image.
	* Overlap is colored to yellow.
	*/
	void showMaskOverlap(const Mat& m1, string strM1, const Mat& m2, string strM2){
		Mat dst,emptyMat = Mat::zeros(m1.size(),m1.type());
		vector<Mat> mv;
		mv.push_back(emptyMat);
		mv.push_back(m1); // green layer
		mv.push_back(m2); // red layer
		merge(mv,dst);
		showImage(string(strM1+" vs "+strM2),dst,true);
	}
	void showMaskOverlap(const Mat& m1, string strM1, const Mat& m2, string strM2, const Mat& m3, string strM3){
		Mat dst,emptyMat = Mat::zeros(m1.size(),m1.type());
		vector<Mat> mv;
		mv.push_back(m1); // blue layer
		mv.push_back(m2); // green layer
		mv.push_back(m3); // red layer
		merge(mv,dst);
		showImage(string(strM1+" vs "+strM2+" vs "+strM3),dst,true);
	}

	template <typename T>
	void printVector(ostream& os, const vector<T>& v){
		int i=0;
	    while(i<v.size()-1){
	        os<<v[i]<<";";
	        i++;
	    }
	    os<<v[i];
	}

	void openLogFile(const string& name, ofstream& os){
		string path = DIR_OUTPUT+"log/";
		makePath(path);
		string fileName = path+name+"_"+currentDateTime()+".log";
		os.open(fileName.c_str(),ios::out);
	}

	void openCsvFile(const string& relPath, const string& name, ofstream& os){
		string path = DIR_OUTPUT+relPath;
		makePath(path);
		string fileName = path+name+"_"+currentDateTime()+".csv";
		os.open(fileName.c_str(),ios::out);
	}

	void closeFile(ofstream& os){
		os.close();
	}

	void writeScoresHeader(ostream& os, const string& name){
		os<<"START OF "<<name<<endl;
		string header = "row,TP,TN,FP,FN,Re,Sp,FPR,FNR,PWC,F,Pr";
		if(SC_DELIM != ','){ // Quick workaround to keep header and delimiter seperately and easily changeable
			size_t pos = -1;
			while((pos=header.find_first_of(',',pos+1)) != string::npos )
				header = header.replace(pos,1,&SC_DELIM);
		}
		os<<header<<endl;
	}

	void writeScoresRow(ostream& os, const int rowIdx, const vector<double>& data){
		using namespace scores;

		if(!isZeroData(data)){
			os<< rowIdx <<SC_DELIM;
			os<< data[TP] <<SC_DELIM;
			os<< data[TN] <<SC_DELIM;
			os<< data[FP] <<SC_DELIM;
			os<< data[FN] <<SC_DELIM;
			os<< data[RE] <<SC_DELIM;
			os<< data[SP] <<SC_DELIM;
			os<< data[FPR] <<SC_DELIM;
			os<< data[FNR] <<SC_DELIM;
			os<< data[PWC] <<SC_DELIM;
			os<< data[FS] <<SC_DELIM;
			os<< data[PR] <<SC_DELIM;
		    os<< endl;
		}
	}

	void writeScoresFooter(ostream& os, const string& name, const vector<double>& totals){
		using namespace scores;
		os<<"END OF "<<name<<endl;
		os<<"\"Total samples:\""<<endl;
		writeScoresRow(os,(int)totals[TOTAL_SAMPLES],totals);
		vector<double> avgs(totals.size());
		for(int i=0;i<totals.size()-1;i++){
			avgs[i] = totals[i]/totals[TOTAL_SAMPLES];
		}
		os<<"\"Averages:\""<<endl;
		writeScoresRow(os,0,avgs);
		os<<endl;
	}

	void printScores(const Mat& gt, const Mat& mask){
		using namespace scores;
		vector<double> data;
		calculateConfusionMatrix(gt,mask,data);

	    // cout<<"Re : "<<getRecall(data)<<endl;
	    // cout<<"Sp : "<<getSpecificity(data)<<endl;
	    // cout<<"FPR: "<<getFalsePositiveRate(data)<<endl;
	    // cout<<"FNR: "<<getFalseNegativeRate(data)<<endl;
	    // cout<<"PWC: "<<getPercentageOfWrongClassifications(data)<<endl;
	    cout<<"F  : "<<getFScore(data)<<endl;
	    // cout<<"Pr : "<<getPrecision(data)<<endl;
	}
}

#endif