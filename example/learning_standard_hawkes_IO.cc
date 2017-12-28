#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include <chrono>
#include <fstream>
#include <string>

void saveParameters(Eigen::VectorXd params, std::string estimatedParamFileName){
	std::ofstream estimatedParamFile(estimatedParamFileName);

	if(estimatedParamFile.is_open()){
		estimatedParamFile << params.transpose() << std::endl;
	}
	estimatedParamFile.close();
}

double computeAP(const int& eventID, const int &num_dims, Eigen::VectorXd& predictIntensity_dim){
	double AP = 0.0;
	std::vector<std::pair<double, int>> intensity_list;
	for(int i=0; i<num_dims; i++){
		intensity_list.push_back(std::make_pair(predictIntensity_dim[i], i));
	}
	std::sort(intensity_list.begin(), intensity_list.end(), std::greater<std::pair<double, int>>());

	for(int i=0; i<num_dims; i++){
		//std::cout << "prediction" << intensity_list[i].second << std::endl;
		if(intensity_list[i].second == eventID){
			AP = 1.0/(i+1);
			return AP;
		}
	}

	return AP;
}

void predictNextEvent(PlainHawkes& hawkesObj, const std::vector<Sequence>& data, double validRatio, int num_dims){
	int num_sequences = data.size();
	double MAP = 0;
	double AP = 0;
	double totalTestLen = 0;
	for(int k=0; k<num_sequences; k++){
		Sequence seq = data[k];
		const std::vector<Event>& seqEvents = seq.GetEvents();
		int seqLen = seqEvents.size();
		int validLen = (int) (seqLen*validRatio);
		int testLen = seqLen - validLen;

		for(int eventIndex=validLen; eventIndex<seqLen; ++eventIndex){
			double eventTime = seqEvents[eventIndex].time;
			int eventID = seqEvents[eventIndex].EventID;
			//std::cout << "truth event id" << eventID;
			Eigen::VectorXd predictIntensity_dim = Eigen::VectorXd::Zero(num_dims);
			hawkesObj.Intensity(eventTime, seq, predictIntensity_dim);
			AP=computeAP(eventID, num_dims, predictIntensity_dim);
			MAP += AP;
		}
		totalTestLen += testLen;

	}
	MAP = MAP/totalTestLen;
	std::cout << "totalTestLen\t" << totalTestLen << std::endl;
	std::cout << "MAP \t" << MAP << std::endl;

}

int main(const int argc, const char** argv)
{
	// unsigned dim = 2, num_params = dim * (dim + 1);

	// OgataThinning ot(dim);

	// Eigen::VectorXd params(num_params);
	// params << 0.1, 0.2, 0.5, 0.5, 0.5, 0.5; 

	// Eigen::MatrixXd beta(dim,dim);
	// beta << 1, 1, 1, 1;

	// PlainHawkes hawkes(num_params, dim, beta);
	// hawkes.SetParameters(params);

	// std::vector<Sequence> sequences;

	// unsigned num_events = 1000, num_sequences = 10;
	// std::cout << "1. Simulating " << num_sequences << " sequences with " << num_events << " events each " << std::endl;

	// ot.Simulate(hawkes, num_events, num_sequences, sequences);
	std::vector<Sequence> trainSequences;

	unsigned dim = 2527;

	std::string trainTimeFileName = "../data/walmartTrainSeqTime_popular.txt"; 
//	std::string timeFileName = "../data/fullTrainSeqTime.txt";
//	std::string timeFileName = "../data/lowrankTrainSeqTime.txt";
	// std::string timeFileName = "../data/trainSeqTime.txt";

	std::string trainEventFileName = "../data/walmartTrainSeqAction_popular.txt";
//	std::string eventFileName = "../data/fullTrainSeqAction.txt";
//	std::string eventFileName = "../data/lowrankTrainSeqAction.txt";
	// std::string eventFileName = "../data/trainSeqAction.txt";
	ImportFromExistingTimeEventsSequences(trainTimeFileName, trainEventFileName, trainSequences);
	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1);

	unsigned num_params = dim*(dim+1);

	PlainHawkes hawkes_new(num_params, dim, beta);
	PlainHawkes::OPTION options;
	options.method = PlainHawkes::PLBFGS;
	options.base_intensity_regularizer = PlainHawkes::NONE;
	options.excitation_regularizer = PlainHawkes::NONE;

	std::cout << "2. Fitting Parameters " << std::endl << std::endl;  
	hawkes_new.fit(trainSequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	// std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	// std::cout << "True Parameters : " << std::endl;
	std::string estimatedParamFile = "standard_hawkes_estimatedParam_popular.txt";
	saveParameters(hawkes_new.GetParameters().transpose(), estimatedParamFile);
	// std::cout << params.transpose() << std::endl;

	std::vector<Sequence> testSequences;
	std::string testTimeFileName = "../data/walmartTestSeqTime_popular.txt";
	std::string testEventFileName = "../data/walmartTestSeqAction_popular.txt";
	ImportFromExistingTimeEventsSequences(testTimeFileName, testEventFileName, testSequences);

	double validRatio = 0.8;
	predictNextEvent(hawkes_new, testSequences, validRatio, dim);
	return 0;
}
