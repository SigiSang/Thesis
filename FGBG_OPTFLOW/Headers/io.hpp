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
		,consecutiveImages=5
		,normalXSize=320
		,normalYSize=261;

	map<string,vector<int> > imgPos;

	void showImage(string windowName,Mat& img, bool resize=false){
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

	void readImages(String folder, String regex, vector<Mat>& images) {
		VideoCapture cap(folder+"/"+regex);
		images.clear();
		while( cap.isOpened() )
		{
		    Mat img;
		    if(!cap.read(img)) {
		    	return;
		    }
		   	images.push_back(img);
		}
	}

	void calculateScores(Mat& gt, Mat& mask) {
	    if(gt.type() != CV_8UC1 || mask.type() != CV_8UC1){
	    	cerr<<"Invalid Mat type in io::calculateScores! Must be CV_8UC1."<<endl;
	    	throw;
	    }
	    Mat confusion = Mat::zeros(2,2, CV_32S);
	    int size = gt.rows*gt.cols;
		for(int i=0; i<size;i++) {
		    uchar label = gt.at<uchar>(i);
		    uchar predicted = mask.at<uchar>(i);
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
		    }
		}
		// double accuracy = ((double)(confusion.at<int>(0,0)+confusion.at<int>(1,1)))/((double)gt.rows*gt.cols)*100;
		double precision = ((double)confusion.at<int>(0,0)/(confusion.at<int>(0,0)+confusion.at<int>(1,0))*100);
		double recall = ((double)confusion.at<int>(0,0)/(confusion.at<int>(0,0)+confusion.at<int>(0,1)))*100;	
		double F = 2*(precision*recall)/(precision+recall);
	    // cout<<"Confusion matrix: "<<endl<<confusion<<endl;
	    // cout<<"Precision: "<<precision<<"%"<<endl;
	    // cout<<"Recall: "<<recall<<"%"<<endl;
	    // cout<<"Accuracy: "<<accuracy<<"%"<<endl;   
	    cout<<"F: "<<F<<"%"<<endl;
	}
}

#endif