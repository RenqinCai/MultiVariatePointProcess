#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include <chrono>
#include <fstream>
#include <string>

//using namespace std

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

	std::vector<Sequence> sequences;

	unsigned dim = 716;

	std::string timeFileName = "../data/walmartTrainSeqTime.txt"; 
//	std::string timeFileName = "../data/fullTrainSeqTime.txt";
//	std::string timeFileName = "../data/lowrankTrainSeqTime.txt";
	// std::string timeFileName = "../data/trainSeqTime.txt";

	std::string eventFileName = "../data/walmartTrainSeqAction.txt";
//	std::string eventFileName = "../data/fullTrainSeqAction.txt";
//	std::string eventFileName = "../data/lowrankTrainSeqAction.txt";
	// std::string eventFileName = "../data/trainSeqAction.txt";

	ImportFromExistingTimeEventsSequences(timeFileName, eventFileName, sequences);
	
	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1.0);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	unsigned num_params = dim*(dim+1);

	PlainHawkes hawkes_new(num_params, dim, beta);
	
	PlainHawkes::OPTION options;
	options.base_intensity_regularizer = PlainHawkes::NONE;
	options.excitation_regularizer = PlainHawkes::NUCLEAR;
	options.coefficients[PlainHawkes::BETA] = 0.1;
	options.ini_learning_rate = 5e-5;
	options.rho = 1;
	options.ub_nuclear = 1;
	options.ini_max_iter = 10000;

	Eigen::VectorXd params(num_params);

	hawkes_new.fit(sequences, options, params);

	std::cout << "Estimated Parameters : " << std::endl;
	// std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	// std::cout << "True Parameters : " << std::endl;
	// std::cout << params.transpose() << std::endl;

	// double testRatio = 0.8;
	// std::pair<double, double> metric = hawkes_new.EvaluateTestSeqs(testSequences, testRatio);
	// double err_cnt = metric.first*1.0/metric.second;
	// std::cout << metric.first << "\t" << metric.second << "\terr_cnt\t" << err_cnt << std::endl;

	std::string estimatedParamFile = "lowrank_hawkes_estimatedParam.txt";
	saveParameters(hawkes_new.GetParameters().transpose(), estimatedParamFile);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << duration / 1000000.0 << " secs." << std::endl;
	
	
	std::vector<Sequence> testSequences;
	std::string testTimeFileName = "../data/walmartTestSeqTime.txt";
	std::string testEventFileName = "../data/walmartTestSeqAction.txt";
	ImportFromExistingTimeEventsSequences(testTimeFileName, testEventFileName, testSequences);

	double validRatio = 0.8;
	predictNextEvent(hawkes_new, testSequences, validRatio, dim);

	return 0;
}
