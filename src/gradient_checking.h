#pragma once
#include "general_settings.h"

using namespace std;

void gradient_checking(const std::vector<cpuMat*>&, const cpuMat*, std::vector<network_layer*>&, Mat*, Mat*);

void gradientChecking_SoftmaxLayer(std::vector<network_layer*>&, const std::vector<cpuMat*>&, const cpuMat*);
void gradientChecking_FullyConnectedLayer(std::vector<network_layer*>&, const std::vector<cpuMat*>&, const cpuMat*);
void gradientChecking_ConvolutionalLayer(std::vector<network_layer*>&, const std::vector<cpuMat*>&, const cpuMat*);

void gradient_checking_network_layers(std::vector<network_layer*>&, const std::vector<cpuMat*>&, const cpuMat*);
