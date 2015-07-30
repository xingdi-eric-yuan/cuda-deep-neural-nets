#include "layer_bank.h"

using namespace std;

pooling_layer::pooling_layer(){
    stride = 1;
    window_size = 0;
    method = POOL_MAX;
    overlap = false;
}
pooling_layer::~pooling_layer(){
	releaseVector(location);
	location.clear();
    std::vector<std::vector<Mat*> >().swap(location);
}
// with overlap
void pooling_layer::init_config(string namestr, int _method, string outputformat, int _stride, int windowsize){
    layer_type = "pooling";
    layer_name = namestr;
    output_format = outputformat;
    stride = _stride;
    window_size = windowsize;
    method = _method;
    overlap = true;
}
// without overlap
void pooling_layer::init_config(string namestr, int _method, string outputformat, int _stride){
    layer_type = "pooling";
    layer_name = namestr;
    output_format = outputformat;
    stride = _stride;
    method = _method;
    overlap = false;
    window_size = 0;
}

void pooling_layer::forwardPass(int nsamples, network_layer* previous_layer){

	if(previous_layer -> output_format == "matrix"){
        cout<<"??? Can not do pooling with matrix... give me an image..."<<endl;
        return;
	}
	std::vector<std::vector<Mat*> > input;
	copyVector(previous_layer -> output_vector, input);
	vector2i *_size = NULL;
	if(overlap) _size = new vector2i(window_size, window_size);

    releaseVector(output_vector);
    releaseVector(location);
    output_vector.clear();
    output_vector.resize(input.size());
    location.clear();
    location.resize(input.size());
    for(int i = 0; i < input.size(); i++){
    	location[i].clear();
        location[i].resize(input[i].size());
    	output_vector[i].clear();
        output_vector[i].resize(input[i].size());
        for(int j = 0; j < input[i].size(); ++j){
        	output_vector[i][j] = new Mat();
        	location[i][j] = new Mat();
        	if(overlap){
				safeGetPt(output_vector[i][j], pooling_with_overlap(input[i][j], _size, stride, method, location[i][j]));
        	}else{
				safeGetPt(output_vector[i][j], pooling(input[i][j], stride, method, location[i][j]));
        	}
        }
    }
    if(overlap) _size -> release();
    releaseVector(input);
	input.clear();
	std::vector<std::vector<Mat*> >().swap(input);
}

void pooling_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
	pooling_layer::forwardPass(nsamples, previous_layer);
}

void pooling_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    std::vector<std::vector<Mat*> > input;
    if(previous_layer -> output_format == "image"){
    	copyVector(previous_layer -> output_vector, input);
    }else{
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    std::vector<std::vector<Mat*> > derivative;
    std::vector<std::vector<Mat*> > deriv2;
	vector2i *up_size = new vector2i(input[0][0] -> rows, input[0][0] -> cols);
	vector2i *_size = NULL;
	if(overlap) _size = new vector2i(window_size, window_size);
    if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0] -> rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0] -> rows);
    }else{
    	copyVector(next_layer -> delta_vector, derivative);
    	copyVector(next_layer -> d2_vector, deriv2);
    }
    releaseVector(delta_vector);
    releaseVector(d2_vector);
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(derivative.size());
    d2_vector.resize(derivative.size());
    for(int i = 0; i < delta_vector.size(); i++){
        delta_vector[i].resize(derivative[i].size());
        d2_vector[i].resize(derivative[i].size());
        for(int j = 0; j < derivative[i].size(); ++j){
        	delta_vector[i][j] = new Mat();
        	d2_vector[i][j] = new Mat();
        	if(overlap){
        		safeGetPt(delta_vector[i][j], unpooling_with_overlap(derivative[i][j], _size, stride, method, location[i][j], up_size));
        		safeGetPt(d2_vector[i][j], unpooling_with_overlap(deriv2[i][j], _size, stride, method, location[i][j], up_size));
        	}else{
        		safeGetPt(delta_vector[i][j], unpooling(derivative[i][j], stride, method, location[i][j], up_size));
        		safeGetPt(d2_vector[i][j], unpooling(deriv2[i][j], stride, method, location[i][j], up_size));
        	}
        }
    }
    if(overlap) _size -> release();
    up_size -> release();
    releaseVector(derivative);
    derivative.clear();
    std::vector<std::vector<Mat*> >().swap(derivative);
    releaseVector(deriv2);
    deriv2.clear();
    std::vector<std::vector<Mat*> >().swap(deriv2);
    releaseVector(input);
    input.clear();
    std::vector<std::vector<Mat*> >().swap(input);
}


