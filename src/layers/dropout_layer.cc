#include "layer_bank.h"

using namespace std;

// dropout layer
dropout_layer::dropout_layer(){
	dropout_rate = 0.0;
	bernoulli_matrix = NULL;
}
dropout_layer::~dropout_layer(){
	bernoulli_matrix -> release();
	releaseVector(bernoulli_vector);
	bernoulli_vector.clear();
    std::vector<std::vector<Mat*> >().swap(bernoulli_vector);
}

void dropout_layer::init_config(string namestr, string outputformat, float dor){
    layer_type = "dropout";
    layer_name = namestr;
    output_format = outputformat;
    dropout_rate = dor;
}

void dropout_layer::forwardPass(int nsamples, network_layer* previous_layer){

    if(output_format == "matrix"){
        Mat *input = new Mat();
        if(previous_layer -> output_format == "matrix"){
            previous_layer -> output_matrix -> copyTo(*input);
        }else{
            convert(previous_layer -> output_vector, input);
        }
        safeGetPt(bernoulli_matrix, getBernoulliMatrix(input -> rows, input -> cols, input -> channels, dropout_rate));
        safeGetPt(output_matrix, multiply_elem(bernoulli_matrix, input));
        input -> release();
    }else{ // output_format == "image"
        std::vector<std::vector<Mat*> > input;
        if(previous_layer -> output_format == "image"){
        	copyVector(previous_layer -> output_vector, input);
        }else{
            cout<<"??? image after matrix??? I can't do that for now..."<<endl;
            return;
        }
        releaseVector(output_vector);
        releaseVector(bernoulli_vector);
        output_vector.clear();
        output_vector.resize(input.size());
        bernoulli_vector.clear();
        bernoulli_vector.resize(input.size());
        for(int i = 0; i < output_vector.size(); i++){
            output_vector[i].resize(input[i].size());
            bernoulli_vector[i].resize(input[i].size());
        }
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
            	output_vector[i][j] = new Mat();
            	bernoulli_vector[i][j] = new Mat();
                safeGetPt(bernoulli_vector[i][j], getBernoulliMatrix(input[i][j] -> rows, input[i][j] -> cols, input[i][j] -> channels, dropout_rate));
                safeGetPt(output_vector[i][j], multiply_elem(bernoulli_vector[i][j], input[i][j]));
            }
        }
        releaseVector(input);
        input.clear();
        std::vector<std::vector<Mat*> >().swap(input);
    }
}


void dropout_layer::forwardPassTest(int nsamples, network_layer* previous_layer){

    if(output_format == "matrix"){
        Mat *input = new Mat();
        if(previous_layer -> output_format == "matrix"){
            previous_layer -> output_matrix -> copyTo(*input);
        }else{
            convert(previous_layer -> output_vector, input);
        }
        safeGetPt(output_matrix, multiply_elem(input, dropout_rate));
        input -> release();
    }else{ // output_format == "image"
        std::vector<std::vector<Mat*> > input;
        if(previous_layer -> output_format == "image"){
        	copyVector(previous_layer -> output_vector, input);
        }else{
            cout<<"??? image after matrix??? I can't do that for now..."<<endl;
            return;
        }
        releaseVector(output_vector);
        releaseVector(bernoulli_vector);
        output_vector.clear();
        output_vector.resize(input.size());
        bernoulli_vector.clear();
        bernoulli_vector.resize(input.size());
        for(int i = 0; i < output_vector.size(); i++){
            output_vector[i].resize(input[i].size());
            bernoulli_vector[i].resize(input[i].size());
        }
        for(int i = 0; i < input.size(); i++){
            for(int j = 0; j < input[i].size(); j++){
            	output_vector[i][j] = new Mat();
                safeGetPt(output_vector[i][j], multiply_elem(input[i][j], dropout_rate));
            }
        }
        releaseVector(input);
        input.clear();
        std::vector<std::vector<Mat*> >().swap(input);
    }
}

void dropout_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    if(output_format == "matrix"){
        Mat *derivative = new Mat();
        Mat *deriv2 = new Mat();
        if(next_layer -> output_format == "matrix"){
            next_layer -> delta_matrix -> copyTo(*derivative);
            next_layer -> d2_matrix -> copyTo(*deriv2);
        }else{
            convert(next_layer -> delta_vector, derivative);
            convert(next_layer -> d2_vector, deriv2);
        }
        Mat *input = new Mat();
        if(previous_layer -> output_format == "matrix"){
            previous_layer -> output_matrix -> copyTo(*input);
        }else{
            convert(previous_layer -> output_vector, input);
        }
        Mat *tmp = new Mat();
        safeGetPt(tmp, square(bernoulli_matrix));
        safeGetPt(delta_matrix, multiply_elem(derivative, bernoulli_matrix));
        safeGetPt(d2_matrix, multiply_elem(deriv2, tmp));
        tmp -> release();
        input -> release();
        derivative -> release();
        deriv2 -> release();
    }else{
        std::vector<std::vector<Mat*> > input;
        if(previous_layer -> output_format == "image"){
        	copyVector(previous_layer -> output_vector, input);
        }else{
            cout<<"??? image after matrix??? I can't do that for now..."<<endl;
            return;
        }
        std::vector<std::vector<Mat*> > derivative;
        std::vector<std::vector<Mat*> > deriv2;
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
        }
    	Mat *tmp = new Mat();
        safeGetPt(tmp, square(bernoulli_matrix));
        for(int i = 0; i < derivative.size(); i++){
            for(int j = 0; j < derivative[i].size(); j++){
            	delta_vector[i][j] = new Mat();
            	d2_vector[i][j] = new Mat();
                safeGetPt(delta_vector[i][j], multiply_elem(derivative[i][j], bernoulli_matrix));
                safeGetPt(d2_vector[i][j], multiply_elem(deriv2[i][j], tmp));
            }
        }
        tmp -> release();
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
}
