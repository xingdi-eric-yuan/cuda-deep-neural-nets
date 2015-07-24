#include "matrix_maths.h"



Mat* exp(const Mat* src){
	Mat *res = new Mat();
	exp(*src).moveTo(*res);
	return res;
}

Mat* log(const Mat* src){
	Mat *res = new Mat();
	log(*src).moveTo(*res);
	return res;
}

Mat* pow(const Mat* src, float power){
	Mat *res = new Mat();
	pow(*src, power).moveTo(*res);
	return res;
}

Mat* square(const Mat* src){
	Mat *res = new Mat();
	square(*src).moveTo(*res);
	return res;
}

Mat* divide(const Mat* numerator, float denominator){
	Mat *res = new Mat();
	divide(*numerator, denominator).moveTo(*res);
	return res;
}

Mat*divide(float numerator, const Mat* denominator){
	Mat *res = new Mat();
	divide(numerator, *denominator).moveTo(*res);
	return res;
}

vector3f* divide(const vector3f* numerator, float denominator){
	vector3f* res = new vector3f();
	*res = divide(*numerator, denominator);
	return res;
}

vector3f* divide(float numerator, const vector3f* denominator){
	vector3f* res = new vector3f();
	*res = divide(numerator, *denominator);
	return res;
}

Mat* divide(const Mat* numerator, const vector3f* denominator){
    Mat *res = new Mat();
    divide(*numerator, *denominator).moveTo(*res);
    return res;
}

Mat* divide(const vector3f* numerator, const Mat* denominator){
    Mat *res = new Mat();
    divide(*numerator, *denominator).moveTo(*res);
    return res;
}

Mat* divide(const Mat* numerator, const Mat* denominator){
    Mat *res = new Mat();
    divide(*numerator, *denominator).moveTo(*res);
    return res;
}

vector3f* divide(const vector3f* numerator, const vector3f* denominator){
    vector3f *res = new vector3f();
    *res = divide(*numerator, *denominator);
    return res;
}

cpuMat*divide(const cpuMat* numerator, const vector3f* denominator){
    cpuMat *res = new cpuMat();
    divide(*numerator, *denominator).moveTo(*res);
    return res;
}

cpuMat*divide(const cpuMat* numerator, float denominator){
    cpuMat *res = new cpuMat();
    divide(*numerator, denominator).moveTo(*res);
    return res;
}

float sum(const vector3f* src){
    float res = sum(*src);
    return res;
}

vector3f* sum(const Mat* src){
    vector3f *res = new vector3f();
    *res = sum(*src);
    return res;
}

vector3f* average(const Mat* src){
    vector3f *res = new vector3f();
    *res = average(*src);
    return res;
}

vector3f* average(const cpuMat* src){
    vector3f *res = new vector3f();
    *res = average(*src);
    return res;
}

vector3f* stddev(const cpuMat* src, const vector3f* avg){
    vector3f *res = new vector3f();
    *res = stddev(*src, *avg);
    return res;
}

vector3f* stddev(const Mat* src, const vector3f* avg){
    vector3f *res = new vector3f();
    *res = stddev(*src, *avg);
    return res;
}

float max(const vector3f* src){
    float res = max(*src);
    return res;
}

vector3f* max(const Mat* src){
    vector3f *res = new vector3f();
    *res = max(*src);
    return res;
}

void max(const Mat* src, vector3f* max_val, vector3f* max_loc){
    max(*src, *max_val, *max_loc);
}

float min(const vector3f*src){
    float res = min(*src);
    return res;
}

vector3f* min(const Mat* src){
    vector3f *res = new vector3f();
    *res = min(*src);
    return res;
}

void min(const Mat* src, vector3f* min_val, vector3f* min_loc){
    min(*src, *min_val, *min_loc);
}

void minMaxLoc(const Mat* src, vector3f* max_val, vector3f* max_loc, vector3f* min_val, vector3f* min_loc){
    minMaxLoc(*src, *max_val, *max_loc, *min_val, *min_loc);
}

Mat*greaterThan(const Mat*src, float val){
    Mat *res = new Mat();
    greaterThan(*src, val).moveTo(*res);
    return res;
}

Mat*lessThan(const Mat*src, float val){
    Mat *res = new Mat();
    lessThan(*src, val).moveTo(*res);
    return res;
}

// convert from vector of img to matrix
// vec.size() == nsamples
void convert(std::vector<std::vector<Mat*> >& vec, Mat*M){
    std::vector<std::vector<Mat> > tmp;
    for(int i = 0; i < vec.size(); ++i){
        std::vector<Mat> tmp1;
        for(int j = 0; j < vec[i].size(); ++j){
            tmp1.push_back(*(vec[i][j]));
        }
        tmp.push_back(tmp1);
    }
    convert(tmp, *M);
    releaseVector(tmp);
    tmp.clear();
    std::vector<std::vector<Mat> >().swap(tmp);
}

// convert from matrix to vector of img
// vec.size() == nsamples
void convert(Mat*M, std::vector<std::vector<Mat*> >& vec, int nsamples, int imagesize){

    std::vector<std::vector<Mat> > tmp;
    convert(*M, tmp, nsamples, imagesize);
    vec.clear();
    vec.resize(tmp.size());
    for(int i = 0; i < vec.size(); ++i){
        vec[i].clear();
        vec[i].resize(tmp[i].size());
    }
    for(int i = 0; i < vec.size(); ++i){
        for(int j = 0; j < vec[i].size(); ++j){
            vec[i][j] = new Mat();
            tmp[i][j].copyTo(*(vec[i][j]));
        }
    }
    releaseVector(tmp);
    tmp.clear();
    std::vector<std::vector<Mat> >().swap(tmp);
}

// non-linearity
Mat* sigmoid(const Mat*src){
    Mat *res = new Mat();
    sigmoid(*src).moveTo(*res);
    return res;
}

Mat* dsigmoid(const Mat*src){
    Mat *res = new Mat();
    dsigmoid(*src).moveTo(*res);
    return res;
}

Mat* dsigmoid_a(const Mat*src){
    Mat *res = new Mat();
    dsigmoid_a(*src).moveTo(*res);
    return res;
}

Mat* ReLU(const Mat* M){
    Mat *res = new Mat();
    ReLU(*M).moveTo(*res);
    return res;
}

Mat* dReLU(const Mat* M){
    Mat *res = new Mat();
    dReLU(*M).moveTo(*res);
    return res;
}

Mat* LeakyReLU(const Mat* M){
    Mat *res = new Mat();
    LeakyReLU(*M).moveTo(*res);
    return res;
}

Mat* dLeakyReLU(const Mat* M){
    Mat *res = new Mat();
    dLeakyReLU(*M).moveTo(*res);
    return res;
}

Mat* Tanh(const Mat*src){
    Mat *res = new Mat();
    Tanh(*src).moveTo(*res);
    return res;
}

Mat* dTanh(const Mat*src){
    Mat *res = new Mat();
    dTanh(*src).moveTo(*res);
    return res;
}

Mat* nonLinearity(const Mat*M, int method){
    Mat *res = new Mat();
    nonLinearity(*M, method).moveTo(*res);
    return res;
}

Mat* dnonLinearity(const Mat*M, int method){
    Mat *res = new Mat();
    dnonLinearity(*M, method).moveTo(*res);
    return res;
}

// convolution and pooling
Mat* fliplr(const Mat*src){
    Mat *res = new Mat();
    fliplr(*src).moveTo(*res);
    return res;
}

Mat* rot90(const Mat*src, int k){
    Mat *res = new Mat();
    rot90(*src, k).moveTo(*res);
    return res;
}

Mat* dopadding(const Mat*src, int pad){
    Mat *res = new Mat();
    dopadding(*src, pad).moveTo(*res);
    return res;
}

Mat* depadding(const Mat*src, int pad){
    Mat *res = new Mat();
    depadding(*src, pad).moveTo(*res);
    return res;
}

Mat* reduce(const Mat* src, int direction, int mode){
    Mat *res = new Mat();
    reduce(*src, direction, mode).moveTo(*res);
    return res;
}

Mat* interpolation(const Mat* src, int _size){
    Mat *res = new Mat();
    interpolation(*src, _size).moveTo(*res);
    return res;
}

Mat* repmat(const Mat*src, int vert, int hori){
    Mat *res = new Mat();
    repmat(*src, vert, hori).moveTo(*res);
    return res;
}

Mat* kron(const Mat*a, const Mat*b){
    Mat *res = new Mat();
    kron(*a, *b).moveTo(*res);
    return res;
}

Mat* conv2(const Mat*m, const Mat*kernel){
    Mat *res = new Mat();
    conv2(*m, *kernel).moveTo(*res);
    return res;
}

Mat* conv2(const Mat*m, const Mat*kernel, int convtype, int pad, int stride){
    Mat *res = new Mat();
    conv2(*m, *kernel, convtype, pad, stride).moveTo(*res);
    return res;
}

Mat* getRange(const Mat* src, int xstart, int xend, int ystart, int yend){
    Mat *res = new Mat();
    getRange(*src, xstart, xend, ystart, yend).moveTo(*res);
    return res;
}

Mat* downSample(const Mat* src, int y_stride, int x_stride){
    Mat *res = new Mat();
    downSample(*src, y_stride, x_stride).moveTo(*res);
    return res;
}

Mat*copyMakeBorder(const Mat* src, int up, int down, int left, int right, const vector3f* val){
    Mat *res = new Mat();
    copyMakeBorder(*src, up, down, left, right, *val).moveTo(*res);
    return res;
}

// Pooling with overlap
// Max pooling and stochastic pooling supported
// output size = (input size - window size) / stride + 1
Mat* pooling_with_overlap(const Mat*src, vector2i window_size, int stride, int poolingMethod, std::vector<vector3f*> &locat){
    std::vector<vector3f> tmplocat;
    Mat *res = new Mat();
    pooling_with_overlap(*src, window_size, stride, poolingMethod, tmplocat).moveTo(*res);
    locat.resize(tmplocat.size());
    for(int i = 0; i < locat.size(); ++i){
        locat[i] = new vector3f();
        tmplocat[i].copyTo(*(locat[i]));
    }
    return res;
}

// Max pooling and stochastic pooling supported
Mat* unpooling_with_overlap(const Mat*src, vector2i window_size, int stride, int poolingMethod, std::vector<vector3f*> &locat, vector2i& up_size){
    std::vector<vector3f> tmplocat;
    Mat *res = new Mat();
    for(int i = 0; i < locat.size(); ++i){
        vector3f tmp;
        locat[i] -> copyTo(tmp);
        tmplocat.push_back(tmp);
    }
    unpooling_with_overlap(*src, window_size, stride, poolingMethod, tmplocat, up_size).moveTo(*res);
    tmplocat.clear();
    std::vector<vector3f>().swap(tmplocat);
    return res;
}

Mat* pooling(const Mat* src, int stride, int poolingMethod, std::vector<vector3f*> &locat){
    std::vector<vector3f> tmplocat;
    Mat *res = new Mat();
    pooling(*src, stride, poolingMethod, tmplocat).moveTo(*res);
    locat.resize(tmplocat.size());
    for(int i = 0; i < locat.size(); ++i){
        locat[i] = new vector3f();
        tmplocat[i].copyTo(*(locat[i]));
    }
    return res;
}

Mat* unpooling(const Mat* src, int stride, int poolingMethod, std::vector<vector3f*>& locat, vector2i& up_size){
    std::vector<vector3f> tmplocat;
    for(int i = 0; i < locat.size(); ++i){
        vector3f tmp;
        locat[i] -> copyTo(tmp);
        tmplocat.push_back(tmp);
    }
    Mat *res = new Mat();
    unpooling(*src, stride, poolingMethod, tmplocat, up_size).moveTo(*res);
    tmplocat.clear();
    std::vector<vector3f>().swap(tmplocat);
    return res;
}
//*/
