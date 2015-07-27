#pragma once
#include "../general_settings.h"
using namespace std;

const float test_tolerance  = 1e-4;

void runAllTest();
// f represents float
// v represents vector3f
// m represents Mat/cpuMat
// vv represents std::vector<std::vector<Mat*> >

Mat* getTestMatrix_5();
Mat* getTestMatrix_3();
vector3f* getTestVector3f();
float getTestFloat();

bool hostEqualToDevice(const Mat*);
bool areIdentical(const Mat*, const Mat*);
bool areIdentical(const cpuMat*, const cpuMat*);
bool areIdentical(const vector3f*, const vector3f*);
bool areIdentical(float, float);
bool areApproximatelyIdentical(const Mat*, const Mat*);
bool areApproximatelyIdentical(const cpuMat*, const cpuMat*);
bool areApproximatelyIdentical(const vector3f*, const vector3f*);
bool areApproximatelyIdentical(float, float);



//

bool test_add_v_f();
bool test_add_v_v();
bool test_add_m_f();
bool test_add_m_v();
bool test_add_m_m();
bool test_subtract_v_f();
bool test_subtract_v_v();
bool test_subtract_m_f();
bool test_subtract_m_v();
bool test_subtract_m_m();
bool test_multiply_elem_v_f();
bool test_multiply_elem_v_v();
bool test_multiply_elem_m_f();
bool test_multiply_elem_m_v();
bool test_multiply_elem_m_m();
bool test_multiply();
bool test_t();
bool test_div_rem();
bool test_div_no_rem();
bool test_divide_m_f();
bool test_divide_f_m();
bool test_divide_v_f();
bool test_divide_f_v();
bool test_divide_m_f();
bool test_divide_f_m();
bool test_divide_m_m();
bool test_divide_v_v();

bool test_exp();
bool test_log();
bool test_pow();
bool test_square_m();
bool test_square_v();
bool test_sqrt_m();
bool test_sqrt_v();
bool test_sum_v();
bool test_sum_m();
bool test_average();
bool test_stddev();
bool test_max_v();
bool test_max_m();
bool test_min_v();
bool test_min_m();
bool test_maxLoc();
bool test_minLoc();
bool test_minMaxLoc();
bool test_greaterThan();
bool test_lessThan();

bool test_convert_vv();
bool test_convert_m();
bool test_sigmoid();
bool test_dsigmoid();
bool test_dsigmoid_a();
bool test_ReLU();
bool test_dReLU();
bool test_LeakyReLU();
bool test_dLeakyReLU();
bool test_Tanh();
bool test_dTanh();
bool test_nonLinearity();
bool test_dnonLinearity();

bool test_fliplr();
bool test_rot90();
bool test_dopadding();
bool test_depadding();
bool test_reduce();
bool test_interpolation();
bool test_repmat();
bool test_kron();
bool test_conv2();
bool test_getRange();
bool test_downSample();
bool test_copyMakeBorder();
bool test_pooling_with_overlap();
bool test_unpooling_with_overlap();
bool test_pooling();
bool test_unpooling();

bool test_cpu_average();
bool test_cpu_stddev();
bool test_cpu_divide_m_v();
bool test_cpu_divide_m_f();
bool test_cpu_subtract_m_f();
bool test_cpu_subtract_m_v();
bool test_cpu_subtract_m_m();


