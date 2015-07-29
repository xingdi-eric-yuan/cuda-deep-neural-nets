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
    releaseVector(output_vector);
    output_vector.clear();
    output_vector.resize(input.size());
    for(int i = 0; i < output_vector.size(); i++){
    	output_vector[i].clear();
        output_vector[i].resize(input[i].size());
    }
    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[i].size(); j++){
        	output_vector[i][j] = new Mat();
        	safeGetPt(output_vector[i][j], local_response_normalization(input[i], j));
        }
    }
    releaseVector(input);
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
    Mat *tmp = NULL;
    Mat *tmp2 = NULL;
    std::vector<std::vector<Mat*> > input;
	copyVector(previous_layer -> output_vector, input);
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
        delta_vector[i].clear();
        d2_vector[i].clear();
        delta_vector[i].resize(derivative[i].size());
        d2_vector[i].resize(derivative[i].size());
        for(int j = 0; j < derivative[i].size(); ++j){
        	delta_vector[i][j] = new Mat();
        	d2_vector[i][j] = new Mat();
        	safeGetPt(tmp, dlocal_response_normalization(input[i], j));
        	safeGetPt(delta_vector[i][j], multiply_elem(derivative[i][j], tmp));
        	safeGetPt(tmp2, square(tmp));
        	safeGetPt(d2_vector[i][j], multiply_elem(deriv2[i][j], tmp2));
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

Mat* local_response_normalization_layer::local_response_normalization(std::vector<Mat*> &vec, int which){
	Mat *res = new Mat();
	vec[which] -> copyTo(*res);
	Mat *sum = new Mat(res -> rows, res -> cols, 3);
	Mat *tmp = NULL;
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
		safeGetPt(tmp, square(vec[i]));
		safeGetPt(sum, add(sum, tmp));
	}
	float scale = alpha / (to - from + 1);
	safeGetPt(sum, multiply_elem(sum, scale));
	safeGetPt(sum, add(sum, k));
	safeGetPt(tmp, pow(sum, beta));
	safeGetPt(res, divide(res, tmp));
	sum -> release();
	tmp -> release();
	return res;
}

Mat* local_response_normalization_layer::dlocal_response_normalization(std::vector<Mat*> &vec_input, int which){
	Mat *input = new Mat();
	vec_input[which] -> copyTo(*input);
	Mat *sum = new Mat(input -> rows, input -> cols, 3);
	Mat *tmp2 = NULL;
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
		safeGetPt(tmp2, square(vec_input[i]));
		safeGetPt(sum, add(sum, tmp2));
	}
	float scale = alpha / (to - from + 1);
	safeGetPt(sum, multiply_elem(sum, scale));
	safeGetPt(sum, add(sum, k));
	Mat *t1 = NULL;
	Mat *t2 = NULL;
	Mat *t3 = NULL;
	safeGetPt(t1, pow(sum, beta - 1));			// pow(sum, beta - 1)
	safeGetPt(t2, multiply_elem(t1, sum));		// pow(sum, beta)
	safeGetPt(t3, multiply_elem(t2, t2));		// pow(sum, 2*beta)
	float tmp = beta * alpha / (to - from + 1) * 2;
	Mat *res = NULL;
	safeGetPt(tmp2, multiply_elem(input, input));
	safeGetPt(tmp2, multiply_elem(tmp2, t1));
	safeGetPt(tmp2, multiply_elem(tmp2, tmp));
	safeGetPt(res, subtract(t2, tmp2));
	safeGetPt(res, divide(res, t3));
	t1 -> release();
	t2 -> release();
	t3 -> release();
	tmp2 -> release();
	input -> release();
	sum -> release();
    return res;
}
