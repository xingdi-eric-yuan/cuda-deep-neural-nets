#include "layer_bank.h"

using namespace std;

fully_connected_layer::fully_connected_layer(){
    size = 0;
    weight_decay = 0.0;
    momentum_derivative = 0.0;
    momentum_second_derivative = 0.0;
    iter = 0;
    mu = 0.0;
    w = new Mat();
    b = new Mat();
    wgrad = new Mat();
    bgrad = new Mat();
    wd2 = new Mat();
    bd2 = new Mat();
    velocity_w = new Mat();
    velocity_b = new Mat();
    second_derivative_w = new Mat();
    second_derivative_b = new Mat();
    learning_rate = new Mat();
}
fully_connected_layer::~fully_connected_layer(){
}

void fully_connected_layer::init_config(string namestr, int hiddensize, float weightDecay, string outputformat){
    layer_type = "fully_connected";
    layer_name = namestr;
    output_format = outputformat;
    size = hiddensize;
    weight_decay = weightDecay;
}

void fully_connected_layer::init_weight(network_layer* previous_layer){

    int inputsize = 0;
    if(previous_layer -> output_format == "image"){
        inputsize = previous_layer -> output_vector[0].size() * previous_layer -> output_vector[0][0] -> rows * previous_layer -> output_vector[0][0] -> cols * 3;
    }else{
        inputsize = previous_layer -> output_matrix -> rows;
    }
    float epsilon = 0.12;
    w -> setSize(size, inputsize, 1);
    w -> randu();
    (*w) *= epsilon;
    b -> setSize(size, 1, 1);
    wgrad -> setSize(size, inputsize, 1);
    wd2 -> setSize(size, inputsize, 1);
    bgrad -> setSize(size, 1, 1);
    bd2 -> setSize(size, 1, 1);

    // updater
    velocity_w -> setSize(size, inputsize, 1);
    velocity_b -> setSize(size, 1, 1);
    second_derivative_w -> setSize(size, inputsize, 1);
    second_derivative_b -> setSize(size, 1, 1);
    iter = 0;
    mu = 1e-2;
    fully_connected_layer::setMomentum();
}

void fully_connected_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void fully_connected_layer::update(int iter_num){
    iter = iter_num;
    if(iter == 30) fully_connected_layer::setMomentum();
    *second_derivative_w = (*second_derivative_w) * momentum_second_derivative + (*wd2) * (1.0 - momentum_second_derivative);
    *learning_rate = divide(lrate_w, (*second_derivative_w + mu));
    *velocity_w = (*velocity_w) * momentum_derivative + wgrad -> mul(*learning_rate) * (1.0 - momentum_derivative);
    *w -= *velocity_w;

    *second_derivative_b = (*second_derivative_b) * momentum_second_derivative + (*bd2) * (1.0 - momentum_second_derivative);
    *learning_rate = divide(lrate_b, (*second_derivative_b + mu));
    *velocity_b = (*velocity_b) * momentum_derivative + bgrad -> mul(*learning_rate) * (1.0 - momentum_derivative);
    *b -= *velocity_b;
}

void fully_connected_layer::forwardPass(int nsamples, network_layer* previous_layer){
    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat tmpacti = (*w) * (*input);
    tmpacti += (*repmat(b, 1, nsamples));
    tmpacti.copyTo(*output_matrix);
}

void fully_connected_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    fully_connected_layer::forwardPass(nsamples, previous_layer);
}

void fully_connected_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){
    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    if(next_layer -> output_format == "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
    }else{
        Mat derivative;
        Mat deriv2;
        next_layer -> delta_matrix -> copyTo(derivative);
        next_layer -> d2_matrix -> copyTo(deriv2);

        *wgrad = derivative * input -> t() / nsamples + (*w) * weight_decay;
        *bgrad = reduce(derivative, REDUCE_TO_SINGLE_COL, REDUCE_SUM) / nsamples;
        *wd2 = deriv2 * square(input -> t()) / nsamples + weight_decay;
        *bd2 = reduce(deriv2, REDUCE_TO_SINGLE_COL, REDUCE_SUM) / nsamples;

        Mat tmp = w -> t() * derivative;
        tmp.copyTo(*delta_matrix);
        tmp = square(w -> t()) * deriv2;
        tmp.copyTo(*d2_matrix);
    }
}

//*/


