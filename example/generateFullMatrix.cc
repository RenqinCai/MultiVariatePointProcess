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

void saveParameters(unsigned dim, Eigen::MatrixXd A, Eigen::VectorXd Lambda0, std::string lambdaFileName, std::string alphaFileName){
	std::ofstream lambdaFile(lambdaFileName);
	std::ofstream alphaFile(alphaFileName);

	if(lambdaFile.is_open() && alphaFile.is_open()){
		for(int i=0; i<dim; i++){
			for(int j=0; j<dim; j++){
				alphaFile << A(i, j) << "\t";
			}
			alphaFile << "\n";
			lambdaFile << Lambda0[i] << "\n";
		}
	}
	alphaFile.close();
	lambdaFile.close();
}

int main(const int argc, const char** argv)
{
	unsigned dim = 30, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);

	// int lowRank = 15;
	// int lowNode = dim/lowRank;

	// Eigen::MatrixXd B1 = (Eigen::MatrixXd::Random(dim,lowRank).array()+1.0)/2.0;
	// Eigen::MatrixXd B2 = (Eigen::MatrixXd::Random(dim,lowRank).array()+1.0)/2.0;

	// Eigen::MatrixXd B = B1 * B2.transpose()/lowRank;

	// for(int i=1; i<lowRank; i++){
	// 	for(int j=(lowNode-1)*i; j<lowNode*(i+1); j++){
	// 		B(j, i) = 0.0;
	// 	}
	// }

	Eigen::MatrixXd B = (Eigen::MatrixXd::Random(dim,dim).array()+1.0)/2.0;

	// for(int i=1; i<lowRank; i++){
	// 	for(int j=(lowNode-1)*i; j<lowNode*(i+1); j++){
	// 		B(j, i) = 0.0;
	// 	}
	// }
	
	Eigen::Map<Eigen::VectorXd> Lambda0 = Eigen::Map<Eigen::VectorXd>(params.segment(0, dim).data(), dim);
	
	Eigen::Map<Eigen::MatrixXd> A = Eigen::Map<Eigen::MatrixXd>(params.segment(dim, dim * dim).data(), dim, dim);

	Lambda0 = Eigen::VectorXd::Constant(dim, 0.1);
	A = B;

	Eigen::MatrixXd beta = Eigen::MatrixXd::Constant(dim,dim,1.0);
	std::string lambdaFileName = "fullLambda.txt";
	std::string alphaFileName = "fullAlpha.txt";
	saveParameters(dim, A, Lambda0, lambdaFileName, alphaFileName);

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::vector<Sequence> sequences;
	OgataThinning ot(dim);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	unsigned num_events = 50, num_sequences = 4000;
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

	std::string trainSeqActionFileName = "fullTrainSeqAction.txt";
	std::string trainSeqTimeFileName = "fullTrainSeqTime.txt";

	std::string testSeqActionFileName = "fullTestSeqAction.txt";
	std::string testSeqTimeFileName = "fullTestSeqTime.txt";
	outputSeq(trainSequences, trainSeqActionFileName, trainSeqTimeFileName, testSequences, testSeqActionFileName, testSeqTimeFileName);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Simulating" << num_sequences << "sequences *" << num_events << "events," << dim << "nodes" << duration / 1000000.0 << " secs." << std::endl;
	return 0;
}