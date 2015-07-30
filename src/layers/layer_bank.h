#pragma once
#include "../general_settings.h"

using namespace std;
class Mat;
class cpuMat;
class vector2i;
class vector3f;

class convolutional_kernel{
public:
    convolutional_kernel();
    ~convolutional_kernel();
    void init_config(int, float);
    void release();

    Mat* w;
    vector3f* b;
    Mat* wgrad;
    vector3f* bgrad;
    Mat* wd2;
    vector3f* bd2;
    int kernel_size;
    float weight_decay;
};

class network_layer{
public:
    network_layer();
    virtual ~network_layer();

    Mat* output_matrix;
    std::vector<std::vector<Mat*> > output_vector;
    Mat* delta_matrix;
    std::vector<std::vector<Mat*> > delta_vector;
    Mat* d2_matrix;
    std::vector<std::vector<Mat*> > d2_vector;

    std::string layer_name;
    std::string layer_type;
    std::string output_format;
    /// layer type:
    // input
    // convolutional
    // full_connected
    // pooling
    // softmax
    // local_response_normalization
    // dropout
    // non_linearity
    // branch
    // combine
};

class input_layer : public network_layer{
public:
    input_layer();
    ~input_layer();
    void init_config(string, int, string);
    void forwardPass(int, const std::vector<cpuMat*>&, const cpuMat*);
    void forwardPassTest(int, const std::vector<cpuMat*>&, const cpuMat*);
    void getSample(const std::vector<cpuMat*>&, std::vector<std::vector<Mat*> >&, const cpuMat*, Mat*);
    void backwardPass();
    Mat *label;
    int batch_size;
};

class convolutional_layer : public network_layer{
public:
    convolutional_layer();
    ~convolutional_layer();
    void init_config(string, int, int, int, int, int, float, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    std::vector<convolutional_kernel*> kernels;
    Mat* combine_weight;
    Mat* combine_weight_grad;
    Mat* combine_weight_d2;
    int padding;
    int stride;
    int combine_feature_map;

    // updater
    void setMomentum();
    void update(int);
    float momentum_derivative;
    float momentum_second_derivative;
    int iter;
    float mu;
    std::vector<Mat*> velocity_w;
    std::vector<vector3f*> velocity_b;
    std::vector<Mat*> second_derivative_w;
    std::vector<vector3f*> second_derivative_b;
    Mat* velocity_combine_weight;
    Mat* second_derivative_combine_weight;
    Mat* learning_rate_w;
    vector3f* learning_rate_b;
};

class pooling_layer : public network_layer{
public:
    pooling_layer();
    ~pooling_layer();

    void init_config(string, int, string, int, int);
    void init_config(string, int, string, int);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    int stride;
    int window_size;
    int method;
    bool overlap;
    std::vector<std::vector<Mat*> > location;
    //std::vector<std::vector<std::vector<vector3f*> > > location;
};

class fully_connected_layer : public network_layer{
public:
    fully_connected_layer();
    ~fully_connected_layer();
    void init_config(string, int, float, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    Mat* w;
    Mat* b;
    Mat* wgrad;
    Mat* bgrad;
    Mat* wd2;
    Mat* bd2;

    int size;
    float weight_decay;

    // updater
    void setMomentum();
    void update(int);
    float momentum_derivative;
    float momentum_second_derivative;
    int iter;
    float mu;
    Mat* velocity_w;
    Mat* velocity_b;
    Mat* second_derivative_w;
    Mat* second_derivative_b;
    Mat* learning_rate;
};

class softmax_layer : public network_layer{
public:
    softmax_layer();
    ~softmax_layer();
    void init_config(string, int, float, string);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void init_weight(network_layer*);
    void backwardPass(int, network_layer*, const Mat*);

    Mat* w;
    Mat* b;
    Mat* wgrad;
    Mat* bgrad;
    Mat* wd2;
    Mat* bd2;
    float network_cost;
    int output_size;
    float weight_decay;

    // updater
    void setMomentum();
    void update(int);

    float momentum_derivative;
    float momentum_second_derivative;
    int iter;
    float mu;
    Mat* velocity_w;
    Mat* velocity_b;
    Mat* second_derivative_w;
    Mat* second_derivative_b;
    Mat* learning_rate;
};
class local_response_normalization_layer : public network_layer{
public:

    local_response_normalization_layer();
    ~local_response_normalization_layer();
    void init_config(string, string, float, float, float, int);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);
    Mat* local_response_normalization(std::vector<Mat*>&, int);
    Mat* dlocal_response_normalization(std::vector<Mat*>&, int);

    float alpha;
    float beta;
    float k;
    int n;
};

class dropout_layer : public network_layer{
public:
    dropout_layer();
    ~dropout_layer();
    void init_config(string, string, float);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

    float dropout_rate;
    Mat* bernoulli_matrix;
    std::vector<std::vector<Mat*> > bernoulli_vector;
};

class non_linearity_layer : public network_layer{
public:
    non_linearity_layer();
    ~non_linearity_layer();
    void init_config(string, int, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);
    int method;
};

class branch_layer : public network_layer{
public:
    branch_layer();
    ~branch_layer();
    void init_config(string, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);

};

class combine_layer : public network_layer{
public:
    combine_layer();
    ~combine_layer();
    void init_config(string, string);
    void init_weight(network_layer*);
    void forwardPass(int, network_layer*);
    void forwardPassTest(int, network_layer*);
    void backwardPass(int, network_layer*, network_layer*);
};





