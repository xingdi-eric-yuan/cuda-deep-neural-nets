/*
 ============================================================================
 Name        : cudacnn.cu
 Author      : eric_yuan
 Version     : 0.1.0
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include "general_settings.h"

float momentum_w_init = 0.5;
float momentum_d2_init = 0.5;
float momentum_w_adjust = 0.95;
float momentum_d2_adjust = 0.90;
float lrate_w = 0.0;
float lrate_b = 0.0;

bool is_gradient_checking = false;
bool use_log = false;
int training_epochs = 0;
int iter_per_epo = 0;

void run(){

    std::vector<cpuMat*> trainX;
    std::vector<cpuMat*> testX;
    cpuMat* trainY = NULL;
    cpuMat* testY = NULL;
    read_CIFAR10_data(trainX, testX, trainY, testY);

    std::vector<network_layer*> flow;
    buildNetworkFromConfigFile("config.txt", flow);

    trainNetwork(trainX, trainY, testX, testY, flow);

    flow.clear();
    std::vector<network_layer*>().swap(flow);

    releaseVector(trainX);
    releaseVector(testX);
    trainY -> release();
    testY -> release();
    trainX.clear();
    std::vector<cpuMat*>().swap(trainX);
    testX.clear();
    std::vector<cpuMat*>().swap(testX);

}



int main(void){

//	showGpuProperty();

//	runAllTest();

	run();




	return 0;
}
