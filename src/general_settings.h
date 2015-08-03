#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <cufft.h>
#include <time.h>

#include "check_cuda_error.h"
#include "data_structures/data_structure.h"
#include "layers/layer_bank.h"
#include "ios/read_data.h"
#include "ios/read_config.h"
#include "train_network.h"
#include "gradient_checking.h"
#include "cu_matrix_maths.h"
#include "matrix_maths.h"
#include "memory_helper.h"
#include "helper.h"
#include "unit_test/unit_test.h"
#include "convolutionFFT2D/convolutionFFT2D_common.h"

class Mat;
class cpuMat;
class vector2i;
class vector3f;

#define RC2IDX(R,C,COLS) (((R)*(COLS))+(C))

#define elif else if
#define threadsPerBlock 256

// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
#define NL_LEAKY_RELU 3
// convolution
#define CONV_SAME 0
#define CONV_VALID 1
#define CONV_FULL 2
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1
#define POOL_STOCHASTIC 2
// reduce
#define REDUCE_SUM 0
#define REDUCE_AVG 1
#define REDUCE_MAX 2
#define REDUCE_MIN 3
#define REDUCE_TO_SINGLE_ROW 0
#define REDUCE_TO_SINGLE_COL 1

using namespace std;

const size_t Mb = 1 << 20; // Assuming a 1Mb page size here
extern bool is_gradient_checking;
extern bool use_log;
extern int training_epochs;
extern int iter_per_epo;
extern float lrate_w;
extern float lrate_b;

extern float momentum_w_init;
extern float momentum_d2_init;
extern float momentum_w_adjust;
extern float momentum_d2_adjust;

