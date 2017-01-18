#include <string>
using std::string;
#include <vector>
using std::vector;
#include <opencv2/opencv.hpp>

#include "dataset.h"
#include "image_processing.h"
#include "io.h"

int main(){
	vector<int> lDs = { // Dataset IDs
		 ds::CD_BRIDGE_ENTRY
		,ds::CD_BUSY_BOULEVARD
		,ds::CD_FLUID_HIGHWAY
		,ds::CD_STREETCORNER
		,ds::CD_TRAMSTATION
		,ds::CD_WINTERSTREET
	};

	bool grayscale = false;
	Mat frame;
	ds::Dataset d;

	for(int dsIdx=0;dsIdx<lDs.size();dsIdx++){
		int noiseStddev=NOISE_STDDEV_MIN+NOISE_STDDEV_INC;
		for(;noiseStddev<=NOISE_STDDEV_MAX;noiseStddev+=NOISE_STDDEV_INC){
			ds::loadDataset(d,lDs[dsIdx],grayscale);
			// Rebuild img path and filename for noisy images
			string fnImg = d.getSrcPath()+d.getRegexInputNoisy(noiseStddev);

			cout<<"Adding noise for "<<fnImg<<endl;

			size_t pos = fnImg.find_last_of('/');
			io::makePath(fnImg.substr(0,pos+1));

			char* bufferImgName = (char*)malloc(fnImg.size()*sizeof(char));
			for(int idx = 1; d.hasNext(); idx++){
				d.next(frame);
				addNoise(frame,frame,noiseStddev);

				sprintf(bufferImgName,fnImg.c_str(),idx);
				cv::imwrite(string(bufferImgName),frame, (vector<int>){CV_IMWRITE_JPEG_QUALITY,50});
			}
			delete bufferImgName;
		}
		cout<<endl<<endl;
	}
	
	return 0;
}