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
    Mat tmp;
    cout<<"fcud    ----     1"<<endl;
    second_derivative_w -> mul(momentum_second_derivative).moveTo(*second_derivative_w);
    cout<<"fcud    ----     1.1"<<endl;
    wd2 -> mul(1.0 - momentum_second_derivative).moveTo(tmp);
    cout<<"fcud    ----     2"<<endl;
    (*second_derivative_w) += tmp;
    (*second_derivative_w + mu).moveTo(tmp);
    cout<<"fcud    ----     3"<<endl;
    divide(lrate_w, tmp).moveTo(*learning_rate);
    velocity_w -> mul(momentum_derivative).moveTo(*velocity_w);
    cout<<"fcud    ----     4"<<endl;
    wgrad -> mul(*learning_rate).moveTo(tmp);
    tmp.mul(1.0 - momentum_derivative).moveTo(tmp);
    cout<<"fcud    ----     5"<<endl;
    (*velocity_w) += tmp;
    (*w) -= (*velocity_w);

    cout<<"fcud    ----     6"<<endl;
    second_derivative_b -> mul(momentum_second_derivative).moveTo(*second_derivative_b);
    bd2 -> mul(1.0 - momentum_second_derivative).moveTo(tmp);
    cout<<"fcud    ----     7"<<endl;
    (*second_derivative_b) += tmp;
    (*second_derivative_b + mu).moveTo(tmp);
    cout<<"fcud    ----     8"<<endl;
    divide(lrate_b, tmp).moveTo(*learning_rate);
    velocity_b -> mul(momentum_derivative).moveTo(*velocity_b);
    cout<<"fcud    ----     9"<<endl;
    bgrad -> mul(*learning_rate).moveTo(tmp);
    tmp.mul(1.0 - momentum_derivative).moveTo(tmp);
    cout<<"fcud    ----     #"<<endl;
    (*velocity_b) += tmp;
    (*b) -= (*velocity_b);
    cout<<"fcud    ----     @"<<endl;

    tmp.release();
}

void fully_connected_layer::forwardPass(int nsamples, network_layer* previous_layer){
    Mat *input = new Mat();
    if(previous_layer -> output_format == "image"){
        convert(previous_layer -> output_vector, input);
    }else{
        previous_layer -> output_matrix -> copyTo(*input);
    }
    Mat tmp;
    ((*w) * (*input)).moveTo(*output_matrix);
    repmat(b, 1, nsamples) -> moveTo(tmp);
    (*output_matrix) += tmp;
    input -> release();
    tmp.release();
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
    Mat derivative;
    Mat deriv2;
    next_layer -> delta_matrix -> copyTo(derivative);
    next_layer -> d2_matrix -> copyTo(deriv2);
    Mat tmp;

    input -> t().moveTo(tmp);
    (derivative * tmp).moveTo(*wgrad);
    (*wgrad) /= nsamples;
    ((*w) * weight_decay).moveTo(tmp);
    (*wgrad) += tmp;
    reduce(derivative, REDUCE_TO_SINGLE_COL, REDUCE_SUM).moveTo(*bgrad);
    (*bgrad) /= nsamples;

    input -> t().moveTo(tmp);
    square(tmp).moveTo(tmp);
    (deriv2 * tmp).moveTo(*wd2);
    (*wd2) /= nsamples;
    (*wd2) += weight_decay;
    reduce(deriv2, REDUCE_TO_SINGLE_COL, REDUCE_SUM).moveTo(*bd2);
    (*bd2) /= nsamples;


    w -> t().moveTo(tmp);
    (tmp * derivative).moveTo(*delta_matrix);
    square(tmp).moveTo(tmp);
    (tmp * deriv2).moveTo(*d2_matrix);

    input -> release();
    derivative.release();
    deriv2.release();
    tmp.release();
}

//*/


