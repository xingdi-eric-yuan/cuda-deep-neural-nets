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
softmax_layer::~softmax_layer(){
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
    w = new Mat(output_size, inputsize, 1);
    w -> randu();
    safeGetPt(w, multiply_elem(w, epsilon));
    b = new Mat(output_size, 1, 1);
    b -> randu();
    safeGetPt(b, multiply_elem(b, epsilon));
    wgrad = new Mat(output_size, inputsize, 1);
    wd2 = new Mat(output_size, inputsize, 1);
    bgrad = new Mat(output_size, 1, 1);
    bd2 = new Mat(output_size, 1, 1);

    // updater
    velocity_w = new Mat(output_size, inputsize, 1);
    velocity_b = new Mat(output_size, 1, 1);
    second_derivative_w = new Mat(output_size, inputsize, 1);
    second_derivative_b = new Mat(output_size, 1, 1);
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

void softmax_layer::forwardPass(int nsamples, network_layer* previous_layer){

    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat *tmp = new Mat();
    Mat *M = new Mat();
    safeGetPt(M, multiply(w, input));
    safeGetPt(tmp, repmat(b, 1, nsamples));
    safeGetPt(M, add(M, tmp));
    safeGetPt(tmp, reduce(M, REDUCE_TO_SINGLE_ROW, REDUCE_MAX));
    safeGetPt(tmp, repmat(tmp, M -> rows, 1));
    safeGetPt(M, subtract(M, tmp));
    safeGetPt(M, exp(M));
    safeGetPt(tmp, reduce(M, REDUCE_TO_SINGLE_ROW, REDUCE_SUM));
    safeGetPt(tmp, repmat(tmp, M -> rows, 1));
    safeGetPt(output_matrix, divide(M, tmp));

    M -> release();
    input -> release();
    tmp -> release();
}

void softmax_layer::forwardPassTest(int nsamples, network_layer* previous_layer){

    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat *tmp = new Mat();
    Mat *M = new Mat();
    safeGetPt(M, multiply(w, input));
    safeGetPt(tmp, repmat(b, 1, nsamples));
    safeGetPt(output_matrix, add(M, tmp));
    M -> release();
    input -> release();
    tmp -> release();
}

void softmax_layer::backwardPass(int nsamples, network_layer* previous_layer, Mat* groundTruth){

    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat *tmp1 = new Mat();
    Mat *tmp2 = new Mat();
    Mat *derivative = new Mat();
    groundTruth -> copyTo(*derivative);

    safeGetPt(derivative, subtract(derivative, output_matrix));
    safeGetPt(tmp2, t(input));
    safeGetPt(tmp2, multiply_elem(tmp2, -1));
    safeGetPt(tmp1, multiply(derivative, tmp2));
    safeGetPt(wgrad, divide(tmp1, nsamples));
    safeGetPt(tmp1, multiply_elem(w, weight_decay));
    safeGetPt(wgrad, add(wgrad, tmp1));

    safeGetPt(tmp1, reduce(derivative, REDUCE_TO_SINGLE_COL, REDUCE_SUM));
    safeGetPt(bgrad, divide(tmp1, -nsamples));

    safeGetPt(tmp1, square(derivative));
    safeGetPt(tmp2, square(tmp2));
    safeGetPt(wd2, multiply(tmp1, tmp2));
    safeGetPt(wd2, divide(wd2, nsamples));
    safeGetPt(wd2, add(wd2, weight_decay));
    safeGetPt(bd2, reduce(tmp1, REDUCE_TO_SINGLE_COL, REDUCE_SUM));
    safeGetPt(bd2, divide(bd2, nsamples));

    safeGetPt(tmp1, t(w));
    safeGetPt(tmp1, multiply_elem(tmp1, -1));
    safeGetPt(delta_matrix, multiply(tmp1, derivative));
    safeGetPt(tmp1, square(tmp1));
    safeGetPt(tmp2, square(derivative));
    safeGetPt(d2_matrix, multiply(tmp1, tmp2));

    tmp1 -> release();
    tmp2 -> release();
    input -> release();
    derivative -> release();
}
//*/



