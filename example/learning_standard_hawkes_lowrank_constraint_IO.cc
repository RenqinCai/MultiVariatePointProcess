#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include <chrono>
#include <fstream>
#include <string>

//using namespace std

int main(const int argc, const char** argv)
{
	// unsigned dim = 5, num_params = dim * (dim + 1);
	// unsigned dim = 100, num_params = dim * (dim + 1);

/**
 * Generate a 5-by-5 matrix with rank 2.
 */
	// Eigen::MatrixXd B1 = Eigen::MatrixXd::Zero(dim,9).array();
	// Eigen::MatrixXd B2 = Eigen::MatrixXd::Zero(dim,9).array();

	// int lowRank = 9;
	// int lowNode = dim/10;
	// for(int i=1; i<=lowRank; i++){
	// 	for(int j=lowNode*(i-1); j<lowNode*(i+1); j++){
	// 		B1(j, i-1) = rand()*0.1;
	// 		B2(j, i-1) = rand()*0.1;
	// 	}
	// }
/**
 * Simply guarantee the stationary condition of the mulivariate Hawkes process.
 */
	// Eigen::MatrixXd B = B1 * B2.transpose()/9;

	// Eigen::EigenSolver<Eigen::MatrixXd> es(B);

	// OgataThinning ot(dim);

	// Eigen::VectorXd params(num_params);
	
	// Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	// Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);

	// Lambda0 = Eigen::VectorXd::Constant(dim, 0.1);
	// A = B;

	std::vector<Sequence> sequences;

	unsigned dim = 100;

	ImportFromExistingUserItemSequences("data/timeFileName", "data/eventFileName", sequences);
	unsigned num_params = dim*(dim+1);

	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1.0);

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	unsigned n = 40;
	unsigned num_sequences = 6000;
	ot.Simulate(hawkes, n, num_sequences, sequences);

	std::vector<Sequence> trainSequences;
	std::vector<Sequence> testSequences;

	int crossVal = 10;
	for(std::vector<Sequence>::iterator it=sequences.begin(); it!=sequences.end(); it++){
		int crossIndex = rand()%crossVal;
		if(crossIndex == 8){
			testSequences.push_back(*it);
		}else{
			trainSequences.push_back(*it);
		}
	}

	std::string trainSeqActionFileName = "trainSeqAction.txt";
	std::string trainSeqTimeFileName = "trainSeqTime.txt";

	std::string testSeqActionFileName = "testSeqAction.txt";
	std::string testSeqTimeFileName = "testSeqTime.txt";
	outputSeq(trainSequences, trainSeqActionFileName, trainSeqTimeFileName, testSequences, testSeqActionFileName, testSeqTimeFileName);

	// PlainHawkes hawkes_new(num_params, dim, beta);
	
	// PlainHawkes::OPTION options;
	// options.base_intensity_regularizer = PlainHawkes::NONE;
	// options.excitation_regularizer = PlainHawkes::NUCLEAR;
	// options.coefficients[PlainHawkes::BETA] = 0.1;
	// options.ini_learning_rate = 5e-5;
	// options.rho = 1;
	// options.ub_nuclear = 1;
	// options.ini_max_iter = 1000;
	// hawkes_new.fit(trainSequences, options, params);
	
	// std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	// auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	// std::cout << "Simulating" << num_sequences << "sequences *" << n << "events," << dim << "nodes" << duration / 1000000.0 << " secs." << std::endl;

	// std::cout << "Estimated Parameters : " << std::endl;
	// std::cout << hawkes_new.GetParameters().transpose() << std::endl;
	// std::cout << "True Parameters : " << std::endl;
	// std::cout << params.transpose() << std::endl;

	// double testRatio = 0.8;
	// std::pair<double, double> metric = hawkes_new.EvaluateTestSeqs(testSequences, testRatio);
	// double err_cnt = metric.first*1.0/metric.second;
	// std::cout << metric.first << "\t" << metric.second << "\terr_cnt\t" << err_cnt << std::endl;

	return 0;
}
