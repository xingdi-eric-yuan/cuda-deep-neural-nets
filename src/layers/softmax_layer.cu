#include "layer_bank.h"

using namespace std;

softmax_layer::softmax_layer(){
    network_cost = 0.0;
    output_size = 0;
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
softmax_layer::~softmax_layer(){
}

void softmax_layer::init_config(string namestr, int numclasses, float weightDecay, string outputformat){
    layer_type = "softmax";
    layer_name = namestr;
    output_format = outputformat;
    output_size = numclasses;
    weight_decay = weightDecay;
}

void softmax_layer::init_weight(network_layer* previous_layer){

    int inputsize = 0;
    if(previous_layer -> output_format == "image"){
        inputsize = previous_layer -> output_vector[0].size() * previous_layer -> output_vector[0][0] -> rows * previous_layer -> output_vector[0][0] -> cols * 3;
    }else{
        inputsize = previous_layer -> output_matrix -> rows;
    }
    float epsilon = 0.12;
    w -> setSize(output_size, inputsize, 1);
    w -> randu();
    (*w) *= epsilon;
    b -> setSize(output_size, 1, 1);
    wgrad -> setSize(output_size, inputsize, 1);
    wd2 -> setSize(output_size, inputsize, 1);
    bgrad -> setSize(output_size, 1, 1);
    bd2 -> setSize(output_size, 1, 1);

    // updater
    velocity_w -> setSize(output_size, inputsize, 1);
    velocity_b -> setSize(output_size, 1, 1);
    second_derivative_w -> setSize(output_size, inputsize, 1);
    second_derivative_b -> setSize(output_size, 1, 1);
    iter = 0;
    mu = 1e-2;
    softmax_layer::setMomentum();
}

void softmax_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void softmax_layer::update(int iter_num){
    iter = iter_num;
    if(iter == 30) softmax_layer::setMomentum();

    *second_derivative_w = (*second_derivative_w) * momentum_second_derivative + (*wd2) * (1.0 - momentum_second_derivative);
    *learning_rate = divide(lrate_w, (*second_derivative_w + mu));
    *velocity_w = (*velocity_w) * momentum_derivative + wgrad -> mul(*learning_rate) * (1.0 - momentum_derivative);
    *w -= *velocity_w;

    *second_derivative_b = (*second_derivative_b) * momentum_second_derivative + (*bd2) * (1.0 - momentum_second_derivative);
    *learning_rate = divide(lrate_b, (*second_derivative_b + mu));
    *velocity_b = (*velocity_b) * momentum_derivative + bgrad -> mul(*learning_rate) * (1.0 - momentum_derivative);
    *b -= *velocity_b;
}

void softmax_layer::forwardPass(int nsamples, network_layer* previous_layer){
    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat M = (*w) * (*input) + (*repmat(b, 1, nsamples));
    M -= repmat(reduce(M, REDUCE_TO_SINGLE_ROW, REDUCE_MAX), M.rows, 1);
    M = exp(M);
    Mat p = divide(M, repmat(reduce(M, REDUCE_TO_SINGLE_ROW, REDUCE_SUM), M.rows, 1));
    p.copyTo(*output_matrix);
}

void softmax_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat M = (*w) * (*input) + (*repmat(b, 1, nsamples));
    M.copyTo(*output_matrix);
}

void softmax_layer::backwardPass(int nsamples, network_layer* previous_layer, Mat& groundTruth){
    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat derivative = groundTruth - *output_matrix;
    *wgrad = (derivative * input -> t()).mul(-1.0) / nsamples + (*w) * weight_decay;
    *bgrad = reduce(derivative, REDUCE_TO_SINGLE_COL, REDUCE_SUM).mul(-1.0) / nsamples;
    *wd2 = pow(derivative, 2.0) * pow(input -> t(), 2.0) / nsamples + weight_decay;
    *bd2 = reduce(pow(derivative, 2.0), REDUCE_TO_SINGLE_COL, REDUCE_SUM) / nsamples;

    Mat tmp = (w -> t() * derivative).mul(-1);
    tmp.copyTo(*delta_matrix);
    tmp = pow(w -> t(), 2.0) * pow(derivative, 2.0);
    tmp.copyTo(*d2_matrix);
}
//*/



