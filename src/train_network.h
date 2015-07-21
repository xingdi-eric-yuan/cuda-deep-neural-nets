#pragma once
#include "general_settings.h"

using namespace std;


void forwardPassInit(const std::vector<cpuMat>&, const cpuMat&, std::vector<network_layer*>&);
void forwardPass(const std::vector<cpuMat>&, const cpuMat&, std::vector<network_layer*>&);
void forwardPassTest(const std::vector<cpuMat>&, const cpuMat&, std::vector<network_layer*>&);

void backwardPass(std::vector<network_layer*>&);
void updateNetwork(std::vector<network_layer*>&, int);

void printNetwork(std::vector<network_layer*>&);
void testNetwork(const std::vector<cpuMat>&, const cpuMat&, std::vector<network_layer*>&);
void trainNetwork(const std::vector<cpuMat>&, const cpuMat&, const std::vector<cpuMat>&, const cpuMat&, std::vector<network_layer*>&);
