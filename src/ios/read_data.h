#pragma once
#include "../general_settings.h"

#include <fstream>
#include <sstream>
using namespace std;
class Mat;
class cpuMat;
class vector2i;
class vector3f;

void read_batch(std::string, std::vector<cpuMat>&, cpuMat&);
void read_CIFAR10_data(std::vector<cpuMat>&, std::vector<cpuMat>&, cpuMat&, cpuMat&);
void preProcessing(std::vector<cpuMat>&, std::vector<cpuMat>&);
cpuMat concat(const std::vector<cpuMat>&);
