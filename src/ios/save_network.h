#pragma once
#include "../general_settings.h"
#include <iostream>
#include <fstream>
using namespace std;

class Mat;
class cpuMat;
class vector2i;
class vector3f;

void save(const std::string&, const Mat*);
void save(const std::string&, const cpuMat*);
void save(const std::string&, const vector3f*);
void save(const std::string&, const vector2i*);
void saveNetwork(const std::string&, const std::vector<network_layer*>&);
void saveNetworkConfig(const std::string&, const std::vector<network_layer*>&);
