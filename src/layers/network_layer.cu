#include "layer_bank.h"

using namespace std;

// kernel
network_layer::network_layer(){
    output_matrix = new Mat();
    delta_matrix = new Mat();
    d2_matrix = new Mat();
}
network_layer::~network_layer(){
    output_vector.clear();
    std::vector<std::vector<Mat*> >().swap(output_vector);
    delta_vector.clear();
    std::vector<std::vector<Mat*> >().swap(delta_vector);
    d2_vector.clear();
    std::vector<std::vector<Mat*> >().swap(d2_vector);
}


//*/
