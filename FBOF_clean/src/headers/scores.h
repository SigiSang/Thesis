#ifndef _FSCORES
#define _FSCORES

#include <cmath>
#include <opencv2/opencv.hpp>

namespace scores{

	/* Data array indexes */
	static const short TP 	= 0;
	static const short TN 	= 1;
	static const short FP	= 2;
	static const short FN	= 3;
	static const short RE	= 4;
	static const short SP	= 5;
	static const short FPR	= 6;
	static const short FNR	= 7;
	static const short PWC	= 8;
	static const short PR	= 9;
	static const short FS	= 10;
	static const short TOTAL_SAMPLES = 11; // total number of samples in the data array
	/*  */

	static const short DATA_SIZE = 12; // size of a data array

	// The proportion of actual positives which are predicted positive.
	// Of all values actually positive, this portion was correctly predicted positive.
	double getRecall(const vector<double>& data){
		int tp = data[TP];
		int fn = data[FN];
		return ((double) tp / (tp+fn) );
	}

	// The proportion of actual negatives which are predicted negative.
	// This can be considered the recall for negative values (inverse recall),
	// so of all values actually negative, this portion was correctly predicted negative.
	double getSpecificity(const vector<double>& data){
		int fp = data[FP];
		int tn = data[TN];
		return ((double) tn / (tn+fp) );
	}

	// The proportion of actual negatives which are predicted positive.
	// Of all values actually negative, this portion was falsely predicted positive.
	double getFalsePositiveRate(const vector<double>& data){
		int fp = data[FP];
		int tn = data[TN];
		return ((double) fp / (fp+tn) );
	}

	// The proportion of actual positives which are predicted negative.
	// Of all values actually positive, this portion was falsely predicted negative.
	double getFalseNegativeRate(const vector<double>& data){
		int tp = data[TP];
		int fn = data[FN];
		return ((double) fn / (tp+fn) );
	}

	// The percentage of wrongly predicted values of the total number of predicted values.
	double getPercentageOfWrongClassifications(const vector<double>& data){
		int tp = data[TP];
		int fn = data[FN];
		int fp = data[FP];
		int tn = data[TN];
		return ((double) 100 * (fn+fp) / (tp+fn+fp+tn) );
	}

	// The proportion of predicted positives which are correctly positives.
	// Of all values predicted positive, this portion was correctly predicted positive.
	// This is a measure for the accuracy in identifying foreground pixels as foreground.
	double getPrecision(const vector<double>& data){
		int tp = data[TP];
		int fp = data[FP];
		return ((double) tp / (tp+fp) );
	}

	// The harmonic mean of precision and recall.
	double getFScore(const vector<double>& data){
		double precision = getPrecision(data);
		double recall = getRecall(data);
		return 2*(precision*recall) / (precision+recall);
	}

	void calculateMetrics(vector<double>& data){
		data[RE] = getRecall(data);
		data[SP] = getSpecificity(data);
		data[FPR] = getFalsePositiveRate(data);
		data[FNR] = getFalseNegativeRate(data);
		data[PWC] = getPercentageOfWrongClassifications(data);
		data[PR] = getPrecision(data);
		data[FS] = getFScore(data);
		for(int i=0;i<data.size();i++){
			if(isnan(data[i])) data[i] = 0;
		}
	}

	void calculateConfusionMatrix(const Mat& gt, const Mat& bin, vector<double>& data) {
		const uchar STATIC = 0;
		const uchar HARD_SHADOW = 50;
		const uchar OUTSIDE_ROI = 85;
		const uchar UNKNOWN_MOTION = 170;
		const uchar MOTION = 255;
	    if(gt.type() != CV_8UC1 || bin.type() != CV_8UC1){
	    	cerr<<"scores.h : Invalid Mat type of "<<((gt.type()!=CV_8UC1)?"gt":"bin")<<"! Must be CV_8UC1."<<endl;
	    	throw;
	    }
	    if(gt.empty() || bin.empty()){
	    	cerr<<"scores.h : Empty Mat "<<(gt.empty()?"gt":"bin")<<" !"<<endl;
	    	throw;
	    }
	    data = vector<double>(DATA_SIZE,0);
		for(int y=0; y<gt.rows;y++) {
		for(int x=0; x<gt.cols;x++) {
		    short label = gt.at<uchar>(y,x);
		    short predicted = bin.at<uchar>(y,x);
		    if(label==STATIC){ // N
		    	if(predicted==STATIC){
		    		data[TN]++;
		    	}else if(predicted==MOTION){
		    		data[FN]++;
		    	}
		    }else if(label==MOTION){ // P
		    	if(predicted==STATIC){
		    		data[FP]++;
		    	}else if(predicted==MOTION){
		        	data[TP]++;
		    	}
		    }
		}
		}
		data[TOTAL_SAMPLES] = 1;
		calculateMetrics(data);
	}

	bool isZeroData(const vector<double>& data){
		for(int i=0;i<4;i++)
			if(data[i]>0) return false;
		return true;
	}

	void updateTotals(const vector<double>& data, vector<double>& totals){
		if(totals.empty()){
			totals = vector<double>(DATA_SIZE,0);
		}
		if(!isZeroData(data)){
			for(int i=0;i<DATA_SIZE;i++)
				totals[i] += data[i];
		}
	}
}

#endif