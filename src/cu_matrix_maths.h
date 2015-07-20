#pragma once
#include "general_settings.h"

__global__ void cu_plus(float*, float*, const int);
__global__ void cu_plus(float*, float*, float*, const int);
__global__ void cu_plus(float*, float, const int);
__global__ void cu_plus(float*, float*, const float, const int);
__global__ void cu_minus(float*, float*, int);
__global__ void cu_minus(float*, float*, float*, const int);
__global__ void cu_minus(float*, float, const int);
__global__ void cu_minus(float*, float*, const float, const int);
__global__ void cu_square(float*, float*, const int);
__global__ void cu_elementWiseMultiply(float*, float*, const int);
__global__ void cu_elementWiseMultiply(float*, float*, float*, const int);
__global__ void cu_setAll(float*, const float, const int);
__global__ void cu_exp(const float*, float*, const int);
__global__ void cu_log(const float*, float*, const int);
__global__ void cu_pow(const float*, float*, const int, const int);
__global__ void cu_divide(const float*, float*, const float, const int);
__global__ void cu_divide(const float, const float*, float*, const int);
__global__ void cu_divide(const float*, const float*, float*, const int);
__global__ void cu_sum(const float*, float*, const int);
//__global__ void cu_stddev(const float*, float*, const float, const int);
__global__ void cu_minMaxLoc(const float*, float*, float*, int*, int*, const int);
__global__ void cu_greaterThan(const float*, float*, const double, const int);
__global__ void cu_greaterThanOrEqualTo(const float*, float*, const double, const int);
__global__ void cu_lessThan(const float*, float*, const double, const int);
__global__ void cu_lessThanOrEqualTo(const float*, float*, const double, const int);
__global__ void cu_equalTo(const float*, float*, const double, const int);
__global__ void cu_tanh(const float*, float*, const int);
__global__ void cu_fliplr(const float*, float*, const int, const int, const int);
__global__ void cu_padding(const float*, float*, const int, const int, const int, const int);
__global__ void cu_depadding(const float*, float*, const int, const int, const int, const int);
__global__ void cu_repmat(const float*, float*, const int, const int, const int, const int, const int);
__global__ void cu_kron(const float*, const float*, float*, const int, const int, const int, const int, const int);
__global__ void cu_downSample(const float*, float*, const int, const int, const int, const int);
__global__ void cu_getRange(const float*, float*, const int, const int, const int, const int, const int, const int);
__global__ void cu_copyMakeBorder(const float*, float*, const int, const int, const int, const int, const int, const int, const int);









