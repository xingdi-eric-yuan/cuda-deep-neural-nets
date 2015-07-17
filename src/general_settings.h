#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>

#include "data_structure.h"
#include "cu_matrix_maths.h"
#include "matrix_maths.h"
#include "memory_helper.h"

class Mat;
class vector2i;
class vector3f;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

#define elif else if
#define threadsPerBlock 512
#define Point2i vector2i;
#define Size2i vector2i;

// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
#define NL_LEAKY_RELU 3

using namespace std;

static double leaky_relu_alpha = 100.0;

