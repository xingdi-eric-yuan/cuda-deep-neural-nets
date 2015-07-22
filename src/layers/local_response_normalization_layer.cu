#include "layer_bank.h"

using namespace std;

local_response_normalization_layer::local_response_normalization_layer(){
    alpha = 0.000125;
    beta = 0.75;
    k = 2.0;
    n = 5;
}
local_response_normalization_layer::~local_response_normalization_layer(){
}

void local_response_normalization_layer::init_config(string namestr, string outputformat, float _alpha, float _beta, float _k, int _n){
    layer_type = "local_response_normalization";
    layer_name = namestr;
    output_format = outputformat;
    alpha = _alpha;
    beta = _beta;
    k = _k;
    n = _n;
}

void local_response_normalization_layer::forwardPass(int nsamples, network_layer* previous_layer){

    if(output_format == "matrix"){
        cout<<"Can't use LRN for matrix, give me an image..."<<endl;
        return;
    }
    std::vector<std::vector<Mat*> > input;
    if(previous_layer -> output_format == "image"){
    	copyVector(previous_layer -> output_vector, input);
    }else{
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    output_vector.clear();
    output_vector.resize(input.size());
    for(int i = 0; i < output_vector.size(); i++){
    	output_vector[i].clear();
        output_vector[i].resize(input[i].size());
    }
    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[i].size(); j++){
        	output_vector[i][j] = new Mat();
            Mat res = local_response_normalization(input[i], j);
            res.copyTo(*(output_vector[i][j]));
        }
    }
    input.clear();
    std::vector<std::vector<Mat*> >().swap(input);
}

void local_response_normalization_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    local_response_normalization_layer::forwardPass(nsamples, previous_layer);
}

void local_response_normalization_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    if(output_format == "matrix"){
        cout<<"Can't use LRN for matrix, give me an image..."<<endl;
        return;
    }
    if(previous_layer -> output_format != "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    std::vector<std::vector<Mat*> > derivative;
    std::vector<std::vector<Mat*> > deriv2;
    std::vector<std::vector<Mat*> > input;
	copyVector(previous_layer -> output_vector, input);

	if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0] -> rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0] -> rows);
    }else{
    	copyVector(next_layer -> delta_vector, derivative);
    	copyVector(next_layer -> d2_vector, deriv2);
    }
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(derivative.size());
    d2_vector.resize(derivative.size());
    for(int i = 0; i < delta_vector.size(); i++){
        delta_vector[i].clear();
        d2_vector[i].clear();
        delta_vector[i].resize(derivative[i].size());
        d2_vector[i].resize(derivative[i].size());
        for(int j = 0; j < derivative[i].size(); ++j){
        	delta_vector[i][j] = new Mat();
        	d2_vector[i][j] = new Mat();
        }
    }
    Mat tmp, tmp2;
    for(int i = 0; i < derivative.size(); i++){
        for(int j = 0; j < derivative[i].size(); j++){
            tmp = dlocal_response_normalization(input[i], j);
            tmp2 = derivative[i][j] -> mul(tmp);
            tmp2.copyTo(*(delta_vector[i][j]));
            tmp2 = deriv2[i][j] -> mul(pow(tmp, 2.0));
            tmp2.copyTo(*(d2_vector[i][j]));
        }
    }
    derivative.clear();
    std::vector<std::vector<Mat*> >().swap(derivative);
    deriv2.clear();
    std::vector<std::vector<Mat*> >().swap(deriv2);
    input.clear();
    std::vector<std::vector<Mat*> >().swap(input);
}

Mat local_response_normalization_layer::local_response_normalization(std::vector<Mat*> &vec, int which){

	Mat res;
	vec[which] -> copyTo(res);
	Mat sum(res.rows, res.cols, 3);
	int from, to;
	if(vec.size() < n){
		from = 0;
		to = vec.size() - 1;
	}else{
		int half = n >> 1;
		from = (k - half) >= 0 ? (k - half) : 0;
		to = (k + half) <= (vec.size() - 1) ? (k + half) : (vec.size() - 1);
	}
	for(int i = from; i <= to; ++i){
		sum += pow(*(vec[i]), 2.0);
	}
	float scale = alpha / (to - from + 1);
	sum *= scale;
	sum += k;
	res = divide(res, pow(sum, beta));
	return res;
}

Mat local_response_normalization_layer::dlocal_response_normalization(std::vector<Mat*> &vec_input, int which){

	Mat input;
	vec_input[which] -> copyTo(input);
	Mat sum(input.rows, input.cols, 3);
	int from, to;
	if(vec_input.size() < n){
		from = 0;
		to = vec_input.size() - 1;
	}else{
		int half = n >> 1;
		from = (k - half) >= 0 ? (k - half) : 0;
		to = (k + half) <= (vec_input.size() - 1) ? (k + half) : (vec_input.size() - 1);
	}
	for(int i = from; i <= to; ++i){
		sum += pow(*(vec_input[i]), 2.0);
	}
	float scale = alpha / (to - from + 1);
	sum *= scale;
	sum += k;

	Mat t1 = pow(sum, beta - 1);	// pow(sum, beta - 1)
	Mat t2 = t1.mul(sum); 			// pow(sum, beta)
	Mat t3 = t2.mul(t2); 			// pow(sum, 2*beta)

	float tmp = beta * alpha / (to - from + 1) * 2;
	Mat tmp2 = input.mul(input).mul(t1);
	tmp2 *= tmp;
	Mat res = t2 - tmp2;
	res = divide(res, t3);
    return res;
}



/*
void local_response_normalization_layer::update(){}

void local_response_normalization_layer::init_weight(network_layer* previous_layer){}

*/



