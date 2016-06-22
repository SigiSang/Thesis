#ifndef IO_
#define IO_

#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include <map>

using namespace std;
using namespace cv;

namespace io {

	/** Output directory **/
	string dirOutput = "output/";

	/** Shown image properties and helper values **/
	int imX=5,imY=5
		,shownImages=0
		,consecutiveImages=3
		,normalXSize=320
		,normalYSize=261;

	map<string,vector<int> > imgPos;

	void showImage(string windowName,const Mat& img, bool resize=false){
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
	    if(imgPos[windowName].empty()){
	    	imgPos[windowName] = vector<int>(2);
	    	imgPos[windowName][0]=imX;
	    	imgPos[windowName][1]=imY;
	    	moveWindow(windowName,imX,imY);
	    }
	    imX += xSize+margin;
	    if(++shownImages%consecutiveImages == 0){
	    	imX = 5;
	    	imY += ySize;
	    }
	}

	void saveImage(string fn, Mat& img){
	    imwrite(dirOutput+fn+".png",img);
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

	void checkDir(string dir){
	    struct stat st;
	    const char* cDir = dir.c_str();
	    if(stat(cDir,&st) == -1){
	        if(mkdir(cDir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH ) != 0){
	            cerr<<"Error in checkOutputDir : failed to create dir "<<dir<<endl;
	            throw;
	        }
	    }
	}

	bool fileExists(const string& name) {
		struct stat buffer;   
		return (stat (name.c_str(), &buffer) == 0); 
	}

	void calculateScores(const Mat& gt, const Mat& mask) {
	    if(gt.type() != CV_8UC1 || mask.type() != CV_8UC1){
	    	cerr<<"Invalid Mat type in io::calculateScores! Must be CV_8UC1."<<endl;
	    	throw;
	    }
	    Mat confusion = Mat::zeros(2,2, CV_32S);
		for(int y=0; y<gt.rows;y++) {
		for(int x=0; x<gt.cols;x++) {
		    short label = gt.at<uchar>(y,x);
		    short predicted = mask.at<uchar>(y,x);
		    if(label==255 && predicted==255) {	//TP
		        confusion.at<int>(0,0)++;
		    }
		    else if (label==0 && predicted==0) {	//TN
		        confusion.at<int>(1,1)++;
		    }
		    else if (label==0 && predicted==255) {	//FP
		        confusion.at<int>(1,0)++;
		    }
		    else if (label==255 && predicted==0) {	//FN
		        confusion.at<int>(0,1)++;
		    // }else{
		    	// cout<<"EUNKKKKKK; l:"<<label<<",p:"<<predicted<<endl;
		    }
		}
		}
		double precision = ((double)confusion.at<int>(0,0)/(confusion.at<int>(0,0)+confusion.at<int>(1,0))*100);
		double recall = ((double)confusion.at<int>(0,0)/(confusion.at<int>(0,0)+confusion.at<int>(0,1)))*100;	
		double F = 2*(precision*recall)/(precision+recall);
	    // cout<<"Confusion matrix: "<<endl<<confusion<<endl;
	    // cout<<"Precision: "<<precision<<"%"<<endl;
	    // cout<<"Recall: "<<recall<<"%"<<endl;
	    cout<<"F: "<<F<<"%"<<endl;
	}

	void showMaskOverlap(const Mat& m1, string strM1, const Mat& m2, string strM2){
		Mat dst,emptyMat = Mat::zeros(m1.size(),m1.type());
		vector<Mat> mv;
		mv.push_back(emptyMat);
		mv.push_back(m1);
		mv.push_back(m2);
		merge(mv,dst);
		showImage(string(strM1+" vs "+strM2),dst);
	}
}

#endif