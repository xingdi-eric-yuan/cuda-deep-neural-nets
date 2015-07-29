#pragma once
#include "general_settings.h"

class Mat;
class cpuMat;
class vector3f;
class vector2i;

/////////// pointer version
void safeGetPt(Mat*&, Mat*);
void safeGetPt(cpuMat*&, cpuMat*);

vector3f* add(const vector3f*, float);
vector3f* add(const vector3f*, const vector3f*);
Mat* add(const Mat*, float);
Mat* add(const Mat*, const vector3f*);
Mat* add(const Mat*, const Mat*);
vector3f* subtract(const vector3f*, float);
vector3f* subtract(const vector3f*, const vector3f*);
Mat* subtract(const Mat*, float);
Mat* subtract(const Mat*, const vector3f*);
Mat* subtract(const Mat*, const Mat*);
vector3f* multiply_elem(const vector3f*, float);
vector3f* multiply_elem(const vector3f*, const vector3f*);
Mat* multiply_elem(const Mat*, float);
Mat* multiply_elem(const Mat*, const vector3f*);
Mat* multiply(const Mat*, const Mat*);
Mat* multiply_elem(const Mat*, const Mat*);
Mat* t(const Mat*);
vector3f* div_rem(vector3f*, int);
vector3f* div_no_rem(vector3f*, int);

// basic maths
Mat* exp(const Mat*);
Mat* log(const Mat*);
Mat* pow(const Mat*, float);
Mat* square(const Mat*);
vector3f* square(const vector3f*);
Mat* sqrt(const Mat*);
vector3f* sqrt(const vector3f*);
Mat* divide(const Mat*, float);
Mat* divide(float, const Mat*);
vector3f* divide(const vector3f*, float);
vector3f* divide(float, const vector3f*);
Mat* divide(const Mat*, const vector3f*);
Mat* divide(const vector3f*, const Mat*);
Mat* divide(const Mat*, const Mat*);
vector3f* divide(const vector3f*, const vector3f*);
float sum(const vector3f*);
vector3f* sum(const Mat*);
vector3f* average(const Mat*);
vector3f* stddev(const Mat*, const vector3f*);
float max(const vector3f*);
float min(const vector3f*);
vector3f* max(const Mat*);
vector3f* min(const Mat*);
void max(const Mat*, vector3f*, vector3f*);
void min(const Mat*, vector3f*, vector3f*);
void minMaxLoc(const Mat*, vector3f*, vector3f*, vector3f*, vector3f*);

Mat* greaterThan(const Mat*, float);
Mat* lessThan(const Mat*, float);
Mat* equalTo(const Mat*, float);
Mat* findMax(const Mat*);
int sameValuesInMat(const Mat*, const Mat*);

void convert(std::vector<std::vector<Mat*> >&, Mat*);
void convert(Mat*, std::vector<std::vector<Mat*> >&, int, int);

// non-linearity
Mat* sigmoid(const Mat*);
Mat* dsigmoid(const Mat*);
Mat* dsigmoid_a(const Mat*);
Mat* ReLU(const Mat*);
Mat* dReLU(const Mat*);
Mat* LeakyReLU(const Mat*);
Mat* dLeakyReLU(const Mat*);
Mat* Tanh(const Mat*);
Mat* dTanh(const Mat*);
Mat* nonLinearity(const Mat*, int);
Mat* dnonLinearity(const Mat*, int);

// convolution and pooling
Mat* fliplr(const Mat*);
Mat* rot90(const Mat*, int);
Mat* dopadding(const Mat*, int);
Mat* depadding(const Mat*, int);

Mat* reduce(const Mat*, int, int);
Mat* interpolation(const Mat*, int);
Mat* repmat(const Mat*, int, int);
Mat* kron(const Mat*, const Mat*);

Mat* conv2(const Mat*, const Mat*);
Mat* conv2(const Mat*, const Mat*, int, int, int);
Mat* getRange(const Mat*, int, int, int, int);
Mat* downSample(const Mat*, int, int);
Mat* copyMakeBorder(const Mat*, int, int, int, int, const vector3f*);

Mat* pooling_with_overlap(const Mat*, vector2i*, int, int, std::vector<vector3f*>&);
Mat* unpooling_with_overlap(const Mat*, vector2i*, int, int, std::vector<vector3f*>&, vector2i*);
Mat* pooling(const Mat*, int, int, std::vector<vector3f*>&);
Mat* unpooling(const Mat*, int, int, std::vector<vector3f*>&, vector2i*);



// cpu math
vector3f* average(const cpuMat*);
vector3f* stddev(const cpuMat*, const vector3f*);
cpuMat* divide(const cpuMat*, const vector3f*);
cpuMat* divide(const cpuMat*, float);
cpuMat* subtract(const cpuMat*, float);
cpuMat* subtract(const cpuMat*, const vector3f*);
cpuMat* subtract(const cpuMat*, const cpuMat*);



