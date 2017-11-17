#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"
#include <chrono>
#include <fstream>
#include <string>

void outputSeq(std::vector<Sequence> & trainSequences, std::string trainSeqActionFileName, std::string trainSeqTimeFileName, std::vector<Sequence> & testSequences, std::string testSeqActionFileName, std::string testSeqTimeFileName){
	
	std::ofstream trainActionFile(trainSeqActionFileName);
	std::ofstream trainTimeFile(trainSeqTimeFileName);
	if(trainActionFile.is_open() && trainTimeFile.is_open()){
		for(std::vector<Sequence>::iterator it=trainSequences.begin(); it!=trainSequences.end(); it++){
			std::vector<Event> seqEvents = it->GetEvents();
			for(unsigned k=0; k<seqEvents.size(); k++){
				trainActionFile << seqEvents[k].DimentionID << "\t";
				trainTimeFile << seqEvents[k].time << "\t";
			}
			trainActionFile << "\n";
			trainTimeFile << "\n";
		}
		trainActionFile.close();
		trainTimeFile.close();
	}else
		std::cout << "unable to train seq open file";
	
	std::ofstream testActionFile(testSeqActionFileName);
	std::ofstream testTimeFile(testSeqTimeFileName);
	if(testActionFile.is_open() && testTimeFile.is_open()){
		for(std::vector<Sequence>::iterator it=testSequences.begin(); it!=testSequences.end(); it++){
			std::vector<Event> seqEvents = it->GetEvents();
			for(unsigned k=0; k<seqEvents.size(); k++){
				testActionFile << seqEvents[k].DimentionID << "\t";
				testTimeFile << seqEvents[k].time << "\t";
			}
			testActionFile << "\n";
			testTimeFile << "\n";
		}
		testActionFile.close();
		testTimeFile.close();
	}else
		std::cout << "unable to test seq open file";
	
}

int main(const int argc, const char** argv)
{
	unsigned dim = 20, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);

	Eigen::MatrixXd B1 = (Eigen::MatrixXd::Random(dim,9).array()+1.0)/2.0;
	Eigen::MatrixXd B2 = (Eigen::MatrixXd::Random(dim,9).array()+1.0)/2.0;

	int lowRank = 9;
	int lowNode = dim/9;
	// for(int i=1; i<=lowRank; i++){
	// 	for(int j=lowNode*(i-1); j<lowNode*(i+1); j++){
	// 		B1(j, i-1) = 0.0;
	// 		B2(j, i-1) = 0.0;
	// 	}
	// }

	Eigen::MatrixXd B = B1 * B2.transpose()/9;

	for(int i=1; i<lowRank; i++){
		for(int j=(lowNode-1)*i; j<lowNode*(i+1); j++){
			B(j, i) = 0.0;
		}
	}

	for(int i=0; i<dim; i++){
		for(int j=0; j<dim; j++){
			std::cout << B(i, j) << "\t";
		}
		std::cout << "\n";
	}
	
	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);

	Lambda0 = Eigen::VectorXd::Constant(dim, 0.1);
	A = B;

	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1.0);
	
	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::vector<Sequence> sequences;
	OgataThinning ot(dim);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	unsigned num_events = 10, num_sequences = 10;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << num_events << " events each " << std::endl;
	ot.Simulate(hawkes, num_events, num_sequences, sequences);

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

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Simulating" << num_sequences << "sequences *" << num_events << "events," << dim << "nodes" << duration / 1000000.0 << " secs." << std::endl;
	return 0;
}