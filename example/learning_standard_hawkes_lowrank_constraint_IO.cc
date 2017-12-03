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


int main(const int argc, const char** argv)
{

	std::vector<Sequence> sequences;

	unsigned dim = 30;

	std::string timeFileName = "../data/sparseTrainSeqTime"; 
	// std::string timeFileName = "../data/fullTrainSeqTime";
	// std::string timeFileName = "../data/lowrankTrainSeqTime";

	std::string eventFileName = "../data/sparseTrainSeqAction";
	// std::string eventFileName = "../data/fullTrainSeqAction";
	// std::string eventFileName = "../data/lowrankTrainSeqAction";

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
	options.ini_max_iter = 1000;

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

	return 0;
}
