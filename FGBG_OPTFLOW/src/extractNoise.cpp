#include <string>
using std::string;
#include <vector>
using std::vector;
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "denoising.h"
#include "image_processing.h"
#include "io.h"

void stop(){
	destroyAllWindows();
	exit(0);
}

void addNoiseLocal(Mat& src, Mat& dst){
	int chan = src.channels();
	int typeNoise = CV_16SC(chan);

	Mat noise = Mat::zeros(src.rows,src.cols,typeNoise), noiseConverted;
	double mean = 127.0;
	double stddev = 11.26942767; // sqrt(mean=127)
	randn(noise,Scalar::all(mean),Scalar::all(stddev));
	cv::subtract(noise,Scalar::all(mean),noise);

	noise.convertTo(noiseConverted,src.type());
	io::showImage("Noise orig",noiseConverted,true);

	Mat src_16S;
	src.convertTo(src_16S,typeNoise);
	addWeighted(src_16S,1.0,noise,1.0,0,src_16S);

	src_16S.convertTo(dst,src.type());
}

int main(){
	/* Keyboard input configuration */
	char keyboard;
	const short waitInterval = 100;
	int wait = 0;

	const int VID_ID = ds::CD_TRAMSTATION;
	// const int VID_ID = ds::MISC_test2;
	bool grayscale = true;

	ds::Dataset dataset;
	ds::loadDataset(dataset,VID_ID,grayscale);

	Mat frame,frameDenoised,noise;
	bool resize = true;
	
	int frameIdx = 0;
	while(dataset.hasNext()){
		dataset.next(frame);

		Mat gauss;
		addNoiseLocal(frame,frame); // simulate Poisson
		Mat emptyMat = Mat::zeros(frame.size(),frame.type());
		addNoise(emptyMat,gauss,11.26); // Gaussian

		dn::denoise(frame,frameDenoised,dn::DEN_NL_MEANS);
		// dn::denoise(frame,frameDenoised,dn::DEN_GAUSS_BLUR);
		cv::subtract(frame,frameDenoised,noise);

		io::showImage("Frame",frame,resize);
		io::showImage("Denoised",frameDenoised,resize);
		io::showImage("Noise from subtraction",noise,resize);
		io::showImage("Noise Gauss",gauss,resize);

		keyboard = waitKey(wait);
		if(keyboard == -1 || keyboard == '\n'){
			// Nothing to do here, keep scrolling
		}else if(keyboard == '\e'){
			stop();
		}else if(keyboard == 'p'){
			wait = (wait+waitInterval)%(2*waitInterval);
		}else{
			cout<<"Pressing unknown key: "<<keyboard<<endl;
		}
	}

	stop();
}