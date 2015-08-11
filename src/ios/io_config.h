#pragma once
#include "../general_settings.h"
using namespace std;

class network_layer;
class Mat;
class cpuMat;
class vector2i;
class vector3f;

std::string read_2_string(std::string);
bool get_word_bool(std::string&, std::string);
int get_word_int(std::string&, std::string);
std::string get_word_string(std::string&, std::string);
double get_word_double(std::string&, std::string);
int get_word_type(std::string&, std::string);
void delete_comment(std::string&);
void delete_space(std::string &);
void get_layers_config(std::string&, std::vector<network_layer*> &);
void buildNetworkFromConfigFile(const std::string&, std::vector<network_layer*>&, const std::vector<cpuMat*>&, const cpuMat*);
void buildNetworkFromSavedData(const std::string&, std::vector<network_layer*>&, const std::vector<cpuMat*>&, const cpuMat*);
void saveNetworkConfig(const std::string&, const std::vector<network_layer*>&);


//*/
