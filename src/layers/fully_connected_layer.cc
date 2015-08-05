#include "layer_bank.h"

using namespace std;

// fully connected layer
fully_connected_layer::fully_connected_layer(){
    size = 0;
    weight_decay = 0.0;
    momentum_derivative = 0.0;
    momentum_second_derivative = 0.0;
    iter = 0;
    mu = 0.0;
    w = NULL;
    b = NULL;
    wgrad = NULL;
    bgrad = NULL;
    wd2 = NULL;
    bd2 = NULL;
    velocity_w = NULL;
    velocity_b = NULL;
    second_derivative_w = NULL;
    second_derivative_b = NULL;
    learning_rate = NULL;
}
fully_connected_layer::~fully_connected_layer(){

    w -> release();
    b -> release();
    wgrad -> release();
    bgrad -> release();
    wd2 -> release();
    bd2 -> release();
    velocity_w -> release();
    velocity_b -> release();
    second_derivative_w -> release();
    second_derivative_b -> release();
    learning_rate -> release();
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
    w = new Mat(size, inputsize, 1);
    w -> randn();
    safeGetPt(w, multiply_elem(w, epsilon));
    b = new Mat(size, 1, 1);
    b -> randn();
    safeGetPt(b, multiply_elem(b, epsilon));
    wgrad = new Mat(size, inputsize, 1);
    wd2 = new Mat(size, inputsize, 1);
    bgrad = new Mat(size, 1, 1);
    bd2 = new Mat(size, 1, 1);

    // updater
    velocity_w = new Mat(size, inputsize, 1);
    velocity_b = new Mat(size, 1, 1);
    second_derivative_w = new Mat(size, inputsize, 1);
    second_derivative_b = new Mat(size, 1, 1);

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
    Mat *tmp = new Mat();
    safeGetPt(second_derivative_w, multiply_elem(second_derivative_w, momentum_second_derivative));
    safeGetPt(tmp, multiply_elem(wd2, 1.0 - momentum_second_derivative));
    safeGetPt(second_derivative_w, add(second_derivative_w, tmp));
    safeGetPt(tmp, add(second_derivative_w, mu));
    safeGetPt(learning_rate, divide(lrate_w, tmp));
    safeGetPt(velocity_w, multiply_elem(velocity_w, momentum_derivative));
    safeGetPt(tmp, multiply_elem(wgrad, learning_rate));
    safeGetPt(tmp, multiply_elem(tmp, 1.0 - momentum_derivative));
    safeGetPt(velocity_w, add(tmp, velocity_w));
    safeGetPt(w, subtract(w, velocity_w));

    safeGetPt(second_derivative_b, multiply_elem(second_derivative_b, momentum_second_derivative));
    safeGetPt(tmp, multiply_elem(bd2, 1.0 - momentum_second_derivative));
    safeGetPt(second_derivative_b, add(second_derivative_b, tmp));
    safeGetPt(tmp, add(second_derivative_b, mu));
    safeGetPt(learning_rate, divide(lrate_b, tmp));
    safeGetPt(velocity_b, multiply_elem(velocity_b, momentum_derivative));
    safeGetPt(tmp, multiply_elem(bgrad, learning_rate));
    safeGetPt(tmp, multiply_elem(tmp, 1.0 - momentum_derivative));
    safeGetPt(velocity_b, add(tmp, velocity_b));
    safeGetPt(b, subtract(b, velocity_b));

    tmp -> release();
}

void fully_connected_layer::forwardPass(int nsamples, network_layer* previous_layer){

    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat *tmp = new Mat();
    safeGetPt(output_matrix, multiply(w, input));
    safeGetPt(tmp, repmat(b, 1, nsamples));
    safeGetPt(output_matrix, add(output_matrix, tmp));

    input -> release();
    tmp -> release();

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
        exit(0);
    }
    Mat *derivative = new Mat();
    Mat *deriv2 = new Mat();
    next_layer -> delta_matrix -> copyTo(*derivative);
    next_layer -> d2_matrix -> copyTo(*deriv2);
    Mat *tmp = new Mat();
    Mat *tmp2 = new Mat();

    safeGetPt(tmp, t(input));
    safeGetPt(wgrad, multiply(derivative, tmp));
    safeGetPt(wgrad, divide(wgrad, nsamples));
    safeGetPt(tmp2, multiply_elem(w, weight_decay));
    safeGetPt(wgrad, add(wgrad, tmp2));
    safeGetPt(bgrad, reduce(derivative, REDUCE_TO_SINGLE_COL, REDUCE_SUM));
    safeGetPt(bgrad, divide(bgrad, nsamples));

    safeGetPt(tmp, square(tmp));
    safeGetPt(wd2, multiply(deriv2, tmp));
    safeGetPt(wd2, divide(wd2, nsamples));
    safeGetPt(wd2, add(wd2, weight_decay));
    safeGetPt(bd2, reduce(deriv2, REDUCE_TO_SINGLE_COL, REDUCE_SUM));
    safeGetPt(bd2, divide(bd2, nsamples));

    safeGetPt(tmp, t(w));
    safeGetPt(delta_matrix, multiply(tmp, derivative));

    safeGetPt(tmp, square(tmp));
    safeGetPt(d2_matrix, multiply(tmp, deriv2));

    input -> release();
    derivative -> release();
    deriv2 -> release();
    tmp -> release();
    tmp2 -> release();
}

//*/


