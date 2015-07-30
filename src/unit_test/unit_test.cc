#include "unit_test.h"


void runAllTest(){
	int counter = 0;
	int success = 0;

	if(test_add_v_f()){ ++ success;} ++ counter;
	if(test_add_v_v()){ ++ success;} ++ counter;
	if(test_add_m_f()){ ++ success;} ++ counter;
	if(test_add_m_v()){ ++ success;} ++ counter;
	if(test_add_m_m()){ ++ success;} ++ counter;
	if(test_subtract_v_f()){ ++ success;} ++ counter;
	if(test_subtract_v_v()){ ++ success;} ++ counter;
	if(test_subtract_m_f()){ ++ success;} ++ counter;
	if(test_subtract_m_v()){ ++ success;} ++ counter;
	if(test_subtract_m_m()){ ++ success;} ++ counter;
	if(test_multiply_elem_v_f()){ ++ success;} ++ counter;
	if(test_multiply_elem_v_v()){ ++ success;} ++ counter;
	if(test_multiply_elem_m_f()){ ++ success;} ++ counter;
	if(test_multiply_elem_m_v()){ ++ success;} ++ counter;
	if(test_multiply_elem_m_m()){ ++ success;} ++ counter;
	if(test_multiply()){ ++ success;} ++ counter;
	if(test_t()){ ++ success;} ++ counter;
	if(test_div_rem()){ ++ success;} ++ counter;
	if(test_div_no_rem()){ ++ success;} ++ counter;
	if(test_divide_m_f()){ ++ success;} ++ counter;
	if(test_divide_f_m()){ ++ success;} ++ counter;
	if(test_divide_v_f()){ ++ success;} ++ counter;
	if(test_divide_f_v()){ ++ success;} ++ counter;
	if(test_divide_m_v()){ ++ success;} ++ counter;
	if(test_divide_v_m()){ ++ success;} ++ counter;
	if(test_divide_m_m()){ ++ success;} ++ counter;
	if(test_divide_v_v()){ ++ success;} ++ counter;

	if(test_exp()){ ++ success;} ++ counter;
	if(test_log()){ ++ success;} ++ counter;
	if(test_pow()){ ++ success;} ++ counter;
	if(test_square_m()){ ++ success;} ++ counter;
	if(test_square_v()){ ++ success;} ++ counter;
	if(test_sqrt_m()){ ++ success;} ++ counter;
	if(test_sqrt_v()){ ++ success;} ++ counter;
	if(test_sum_v()){ ++ success;} ++ counter;
	if(test_sum_m()){ ++ success;} ++ counter;
	if(test_average()){ ++ success;} ++ counter;
	if(test_stddev()){ ++ success;} ++ counter;
	if(test_max_v()){ ++ success;} ++ counter;
	if(test_max_m()){ ++ success;} ++ counter;
	if(test_min_v()){ ++ success;} ++ counter;
	if(test_min_m()){ ++ success;} ++ counter;
	if(test_maxLoc()){ ++ success;} ++ counter;
	if(test_minLoc()){ ++ success;} ++ counter;
	if(test_minMaxLoc()){ ++ success;} ++ counter;
	if(test_greaterThan()){ ++ success;} ++ counter;
	if(test_lessThan()){ ++ success;} ++ counter;
	if(test_equalTo()){ ++ success;} ++ counter;
	if(test_findMax()){ ++ success;} ++ counter;
	if(test_sameValuesInMat()){ ++ success;} ++ counter;


	if(test_convert_vv()){ ++ success;} ++ counter;
	if(test_convert_m()){ ++ success;} ++ counter;
	if(test_sigmoid()){ ++ success;} ++ counter;
	if(test_dsigmoid()){ ++ success;} ++ counter;
	if(test_dsigmoid_a()){ ++ success;} ++ counter;
	if(test_ReLU()){ ++ success;} ++ counter;
	if(test_dReLU()){ ++ success;} ++ counter;
	if(test_LeakyReLU()){ ++ success;} ++ counter;
	if(test_dLeakyReLU()){ ++ success;} ++ counter;
	if(test_Tanh()){ ++ success;} ++ counter;
	if(test_dTanh()){ ++ success;} ++ counter;

	if(test_fliplr()){ ++ success;} ++ counter;
	if(test_rot90()){ ++ success;} ++ counter;
	if(test_dopadding()){ ++ success;} ++ counter;
	if(test_depadding()){ ++ success;} ++ counter;
	if(test_reduce()){ ++ success;} ++ counter;
	if(test_getRange()){ ++ success;} ++ counter;
	if(test_interpolation()){ ++ success;} ++ counter;
	if(test_repmat()){ ++ success;} ++ counter;
	if(test_downSample()){ ++ success;} ++ counter;
	if(test_copyMakeBorder()){ ++ success;} ++ counter;
	if(test_kron()){ ++ success;} ++ counter;
	if(test_conv2_kernel()){ ++ success;} ++ counter;
	if(test_conv2()){ ++ success;} ++ counter;
	if(test_pooling_max()){ ++ success;} ++ counter;
	if(test_pooling_mean()){ ++ success;} ++ counter;
	if(test_pooling_overlap_max()){ ++ success;} ++ counter;



	cout<<"test success : "<<success<<", failed : "<<counter - success<<", total : "<<counter<<endl;
}

// get data
Mat* getTestMatrix_5(){
	Mat *m = new Mat(5, 5, 3);
	for(int i = 0; i < m -> getLength(); ++i){
		m -> set(i, (float)i / 10.0);
	}
	return m;
}

Mat* getTestMatrix_3(){
	Mat *m = new Mat(3, 3, 3);
	for(int i = 0; i < m -> getLength(); ++i){
		m -> set(i, (float)i / 10.0);
	}
	return m;
}

Mat *getTestMatrix_3_rand(){
	const float array_data[27] = {
			 0.2916, -1.1480, -1.7947,
			 0.1978,  0.1049,  0.8404,
			 1.5877,  0.7223, -0.8880,
			-0.8045,  2.5855,  0.1001,
			 0.6966, -0.6669, -0.5445,
			 0.8351,  0.1873,  0.3035,
			-0.2437, -0.0825, -0.6003,
			 0.2157, -1.9330,  0.4900,
			-1.1658, -0.4390,  0.7394};
	Mat *res = new Mat(3, 3, 3);
	memcpy(res -> hostData, array_data, res -> getLength() * sizeof(float));
	res -> hostToDevice();
	return res;
}

Mat *getTestMatrix_5_rand(){
	const float array_data[75] = {
		-0.7648, -1.1658,  0.3035, -1.2078,  0.2820, 
		-1.4023, -1.1480, -0.6003,  2.9080,  0.0335, 
		-1.4224,  0.1049,  0.4900,  0.8252, -1.3337, 
		 0.4882,  0.7223,  0.7394,  1.3790,  1.1275, 
		-0.1774,  2.5855,  1.7119, -1.0582,  0.3502, 
		-0.1961, -0.6669, -0.1941, -0.4686, -0.2991, 
		 1.4193,  0.1873, -2.1384, -0.2725,  0.0229, 
		 0.2916, -0.0825, -0.8396,  1.0984, -0.2620, 
		 0.1978, -1.9330,  1.3546, -0.2779, -1.7502, 
		 1.5877, -0.4390, -1.0722,  0.7015, -0.2857, 
		-0.8045, -1.7947,  0.9610, -2.0518, -0.8314, 
		 0.6966,  0.8404,  0.1240, -0.3538, -0.9792, 
		 0.8351, -0.8880,  1.4367, -0.8236, -1.1564, 
		-0.2437,  0.1001, -1.9609, -1.5771, -0.5336, 
		 0.2157, -0.5445, -0.1977,  0.5080, -2.0026};
	Mat *res = new Mat(5, 5, 3);
	memcpy(res -> hostData, array_data, res -> getLength() * sizeof(float));
	res -> hostToDevice();
	return res;
}

Mat *getTestMatrix_10_rand(){
	const float array_data[300] = {
			 0.5377,  0.8884, -1.0891, -1.1480,  2.9080,  0.5201, -1.3617, -0.5320, -1.0667,  0.3271,
			 1.8339, -1.1471,  0.0326,  0.1049,  0.8252, -0.0200,  0.4550,  1.6821,  0.9337,  1.0826,
			-2.2588, -1.0689,  0.5525,  0.7223,  1.3790, -0.0348, -0.8487, -0.8757,  0.3503,  1.0061,
			 0.8622, -0.8095,  1.1006,  2.5855, -1.0582, -0.7982, -0.3349, -0.4838, -0.0290, -0.6509,
			 0.3188, -2.9443,  1.5442, -0.6669, -0.4686,  1.0187,  0.5528, -0.7120,  0.1825,  0.2571,
			-1.3077,  1.4384,  0.0859,  0.1873, -0.2725, -0.1332,  1.0391, -1.1742, -1.5651, -0.9444,
			-0.4336,  0.3252, -1.4916, -0.0825,  1.0984, -0.7145, -1.1176, -0.1922, -0.0845, -1.3218,
			 0.3426, -0.7549, -0.7423, -1.9330, -0.2779,  1.3514,  1.2607, -0.2741,  1.6039,  0.9248,
			 3.5784,  1.3703, -1.0616, -0.4390,  0.7015, -0.2248,  0.6601,  1.5301,  0.0983,  0.0000,
			 2.7694, -1.7115,  2.3505, -1.7947, -2.0518, -0.5890, -0.0679, -0.2490,  0.0414, -0.0549,
			-1.3499, -0.1022, -0.6156,  0.8404, -0.3538, -0.2938, -0.1952, -1.0642, -0.7342,  0.9111,
			 3.0349, -0.2414,  0.7481, -0.8880, -0.8236, -0.8479, -0.2176,  1.6035, -0.0308,  0.5946,
			 0.7254,  0.3192, -0.1924,  0.1001, -1.5771, -1.1201, -0.3031,  1.2347,  0.2323,  0.3502,
			-0.0631,  0.3129,  0.8886, -0.5445,  0.5080,  2.5260,  0.0230, -0.2296,  0.4264,  1.2503,
			 0.7147, -0.8649, -0.7648,  0.3035,  0.2820,  1.6555,  0.0513, -1.5062, -0.3728,  0.9298,
			-0.2050, -0.0301, -1.4023, -0.6003,  0.0335,  0.3075,  0.8261, -0.4446, -0.2365,  0.2398,
			-0.1241, -0.1649, -1.4224,  0.4900, -1.3337, -1.2571,  1.5270, -0.1559,  2.0237, -0.6904,
			 1.4897,  0.6277,  0.4882,  0.7394,  1.1275, -0.8655,  0.4669,  0.2761, -2.2584, -0.6516,
			 1.4090,  1.0933, -0.1774,  1.7119,  0.3502, -0.1765, -0.2097, -0.2612,  2.2294,  1.1921,
			 1.4172,  1.1093, -0.1961, -0.1941, -0.2991,  0.7914,  0.6252,  0.4434,  0.3376, -1.6118,
			 0.6715, -0.8637,  1.4193, -2.1384,  0.0229, -1.3320,  0.1832,  0.3919,  1.0001, -0.0245,
			-1.2075,  0.0774,  0.2916, -0.8396, -0.2620, -2.3299, -1.0298, -1.2507, -1.6642, -1.9488,
			 0.7172, -1.2141,  0.1978,  1.3546, -1.7502, -1.4491,  0.9492, -0.9480, -0.5900,  1.0205,
			 1.6302, -1.1135,  1.5877, -1.0722, -0.2857,  0.3335,  0.3071, -0.7411, -0.2781,  0.8617,
			 0.4889, -0.0068, -0.8045,  0.9610, -0.8314,  0.3914,  0.1352, -0.5078,  0.4227,  0.0012,
			 1.0347,  1.5326,  0.6966,  0.1240, -0.9792,  0.4517,  0.5152, -0.3206, -1.6702, -0.0708,
			 0.7269, -0.7697,  0.8351,  1.4367, -1.1564, -0.1303,  0.2614,  0.0125,  0.4716, -2.4863,
			-0.3034,  0.3714, -0.2437, -1.9609, -0.5336,  0.1837, -0.9415, -3.0292, -1.2128,  0.5812,
			 0.2939, -0.2256,  0.2157, -0.1977, -2.0026, -0.4762, -0.1623, -0.4570,  0.0662, -2.1924,
			-0.7873,  1.1174, -1.1658, -1.2078,  0.9642,  0.8620, -0.1461,  1.2424,  0.6524, -2.3193};
	Mat *res = new Mat(10, 10, 3);
	memcpy(res -> hostData, array_data, res -> getLength() * sizeof(float));
	res -> hostToDevice();
	return res;
}


vector3f* getTestVector3f_0(){
	vector3f *v = new vector3f();
	for(int i = 0; i < 3; ++i){
		v -> set(i, (float)(i + 1) / 4.0);
	}
	return v;
}

vector3f* getTestVector3f_1(){
	vector3f *v = new vector3f();
	for(int i = 0; i < 3; ++i){
		v -> set(i, (float)(i + 1) / 4.0 + 3.0);
	}
	return v;
}

void getTestVectorVectorMat(std::vector<std::vector<Mat*> >& vec){
	vec.clear();
	vec.resize(3);
	for(int i = 0; i < vec.size(); ++i){
		vec[i].clear();
		vec[i].resize(2);
		for(int j = 0; j < vec[i].size(); ++j){
			vec[i][j] = getTestMatrix_3();
		}
	}
}

float getTestFloat(){
	return 0.5;
}

int getTestInt(){
	return 2;
}

bool hostEqualToDevice(const Mat* a){
	if(NULL == a -> hostData || NULL == a -> devData) return false;
	float *tmpMemory = (float*)MemoryMonitor::instance()->cpuMalloc(a -> getLength() * sizeof(float));
	cudaMemcpy(tmpMemory, a -> devData, a -> getLength() * sizeof(float), cudaMemcpyDeviceToHost);
	int n = memcmp(tmpMemory, a -> hostData, a -> getLength() * sizeof(float));
	free(tmpMemory);
	return 0 == n ? true : false;
}

bool areIdentical(const std::vector<std::vector<Mat*> >& a, const std::vector<std::vector<Mat*> >& b){
	if(a.size() != b.size()) return false;
	for(int i = 0; i < a.size(); ++i){
		if(a[i].size() != b[i].size()) return false;
		for(int j = 0; j < a[i].size(); ++j){
			if(!areIdentical(a[i][j], b[i][j])) return false;
		}
	}
	return true;
}

bool areIdentical(const std::vector<vector3f*>& a, const std::vector<vector3f*>& b){
	if(a.size() != b.size()) return false;
	for(int i = 0; i < a.size(); ++i){
		if(!areIdentical(a[i], b[i])) return false;
	}
	return true;
}

bool areIdentical(const Mat* a, const Mat* b){
	if(NULL == a || NULL == b) return false;
	if(NULL == a -> hostData && NULL == a -> devData && NULL == b -> hostData && NULL == b -> devData){
		return true;
	}
	if(NULL == a -> hostData || NULL == a -> devData || NULL == b -> hostData || NULL == b -> devData){
		return false;
	}
	if(a -> getLength() != b -> getLength()) return false;
	if(!hostEqualToDevice(a) || !hostEqualToDevice(b)) return false;
	int n = memcmp(b -> hostData, a -> hostData, a -> getLength() * sizeof(float));
	return 0 == n ? true : false;
}

bool areIdentical(const cpuMat* a, const cpuMat* b){
	if(NULL == a || NULL == b) return false;
	if(NULL == a -> Data && NULL == b -> Data){
		return true;
	}
	if(NULL == a -> Data || NULL == b -> Data){
		return false;
	}
	if(a -> getLength() != b -> getLength()) return false;
	int n = memcmp(b -> Data, a -> Data, a -> getLength() * sizeof(float));
	return 0 == n ? true : false;
}

bool areIdentical(const vector3f* a, const vector3f* b){
	if(NULL == a || NULL == b) return false;
	for(int i = 0; i < 3; ++i){
		if(a -> get(i) != b -> get(i)) return false;
	}
	return true;
}

bool areIdentical(float a, float b){
	return a == b ? true : false;
}

bool areApproximatelyIdentical(const std::vector<std::vector<Mat*> >& a, const std::vector<std::vector<Mat*> >& b){
	if(a.size() != b.size()) {
		return false;
	}
	for(int i = 0; i < a.size(); ++i){
		if(a[i].size() != b[i].size()) {
			return false;
		}
		for(int j = 0; j < a[i].size(); ++j){
			if(!areApproximatelyIdentical(a[i][j], b[i][j])) {
				return false;
			}
		}
	}
	return true;
}

bool areApproximatelyIdentical(const std::vector<vector3f*>& a, const std::vector<vector3f*>& b){
	if(a.size() != b.size()) {
		return false;
	}
	for(int i = 0; i < a.size(); ++i){
		if(!areApproximatelyIdentical(a[i], b[i])) {
			return false;
		}
	}
	return true;
}

bool areApproximatelyIdentical(const Mat* a, const Mat* b){
	if(NULL == a || NULL == b) return false;
	if(NULL == a -> hostData && NULL == a -> devData && NULL == b -> hostData && NULL == b -> devData){
		return true;
	}
	if(NULL == a -> hostData || NULL == a -> devData || NULL == b -> hostData || NULL == b -> devData){
		return false;
	}
	if(a -> getLength() != b -> getLength()) return false;
	if(!hostEqualToDevice(a) || !hostEqualToDevice(b)) return false;
	for(int i = 0; i < a -> getLength(); ++i){
		float n = b -> hostData[i] - a -> hostData[i];
		if(fabsf(n) > test_tolerance) return false;
	}
	return true;
}

bool areApproximatelyIdentical(const cpuMat* a, const cpuMat* b){
	if(NULL == a || NULL == b) return false;
	if(NULL == a -> Data && NULL == b -> Data){
		return true;
	}
	if(NULL == a -> Data || NULL == b -> Data){
		return false;
	}
	if(a -> getLength() != b -> getLength()) return false;
	for(int i = 0; i < a -> getLength(); ++i){
		float n = b -> Data[i] - a -> Data[i];
		if(fabsf(n) > test_tolerance) return false;
	}
	return true;
}

bool areApproximatelyIdentical(const vector3f* a, const vector3f* b){
	if(NULL == a || NULL == b) return false;
	for(int i = 0; i < 3; ++i){
		float n = a -> get(i) - b -> get(i);
		if(fabsf(n) > test_tolerance) return false;
	}
	return true;
}

bool areApproximatelyIdentical(float a, float b){
	float n = a - b;
	return fabsf(n) <= test_tolerance ? true : false;
}

// tests
bool test_add_v_f(){
	cout<<"testing add_v_f --- ";
	vector3f *a = NULL;
	vector3f *res = NULL;
	safeGetPt(a, getTestVector3f_0());
	float b = getTestFloat();
	safeGetPt(res, add(a, b));
	vector3f *expect = new vector3f(0.75, 1.0, 1.25);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_add_v_v(){
	cout<<"testing add_v_v --- ";
	vector3f *a = NULL;
	vector3f *b = NULL;
	safeGetPt(a, getTestVector3f_0());
	safeGetPt(b, getTestVector3f_0());
	vector3f *res = NULL;
	safeGetPt(res, add(a, b));
	vector3f *expect = new vector3f(0.50, 1.0, 1.5);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_add_m_f(){
	const float array_add_m_f[27] = {
		0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 
		1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 
		2.3000, 2.4000, 2.5000, 2.6000, 2.7000, 2.8000, 2.9000, 3.0000, 3.1000};
	cout<<"testing add_m_f --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, add(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_add_m_f, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_add_m_v(){
	const float array_add_m_v[27] = {	
		0.2500, 0.3500, 0.4500, 0.5500, 0.6500, 0.7500, 0.8500, 0.9500, 1.0500,
		1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 2.1000, 2.2000,
		2.5500, 2.6500, 2.7500, 2.8500, 2.9500, 3.0500, 3.1500, 3.2500, 3.3500};
	cout<<"testing add_m_v --- ";
	Mat *a = getTestMatrix_3();
	vector3f *b = getTestVector3f_0();
	Mat *res = NULL;
	safeGetPt(res, add(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_add_m_v, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	b -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_add_m_m(){
	const float array_add_m_m[27] = {	
		0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 1.2000, 1.4000, 1.6000, 
		1.8000, 2.0000, 2.2000, 2.4000, 2.6000, 2.8000, 3.0000, 3.2000, 3.4000, 
		3.6000, 3.8000, 4.0000, 4.2000, 4.4000, 4.6000, 4.8000, 5.0000, 5.2000};
	cout<<"testing add_m_m --- ";
	Mat *a = getTestMatrix_3();
	Mat *b = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, add(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_add_m_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_subtract_v_f(){
	cout<<"testing subtract_v_f --- ";
	vector3f *a = getTestVector3f_0();
	float b = getTestFloat();
	vector3f *res = NULL;
	safeGetPt(res, subtract(a, b));
	vector3f *expect = new vector3f(-0.25, 0.0, 0.25);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_subtract_v_v(){
	cout<<"testing subtract_v_v --- ";
	vector3f *a = getTestVector3f_0();
	vector3f *b = getTestVector3f_0();
	vector3f *res = NULL;
	safeGetPt(res, subtract(a, b));
	vector3f *expect = new vector3f(0.00, 0.00, 0.0);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_subtract_m_f(){
	const float array_subtract_m_f[27] = {
		-0.5000, -0.4000, -0.3000, -0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000, 
		 0.4000,  0.5000,  0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000, 
		 1.3000,  1.4000,  1.5000,  1.6000,  1.7000,  1.8000,  1.9000,  2.0000,  2.1000};
	cout<<"testing subtract_m_f --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, subtract(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_subtract_m_f, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_subtract_m_v(){
	const float array_subtract_m_v[27] = {	
		-0.2500, -0.1500, -0.0500,  0.0500,  0.1500,  0.2500,  0.3500,  0.4500,  0.5500,
		 0.4000,  0.5000,  0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000,
		 1.0500,  1.1500,  1.2500,  1.3500,  1.4500,  1.5500,  1.6500,  1.7500,  1.8500};
	cout<<"testing subtract_m_v --- ";
	Mat *a = getTestMatrix_3();
	vector3f *b = getTestVector3f_0();
	Mat *res = NULL;
	safeGetPt(res, subtract(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_subtract_m_v, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_subtract_m_m(){
	const float array_subtract_m_m[27] = {	
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
	cout<<"testing subtract_m_m --- ";
	Mat *a = getTestMatrix_3();
	Mat *b = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, subtract(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_subtract_m_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_multiply_elem_v_f(){
	cout<<"testing multiply_elem_v_f --- ";
	vector3f *a = getTestVector3f_0();
	float b = getTestFloat();
	vector3f *res = NULL;
	safeGetPt(res, multiply_elem(a, b));
	vector3f *expect = new vector3f(0.125, 0.25, 0.375);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_multiply_elem_v_v(){
	cout<<"testing multiply_elem_v_v --- ";
	vector3f *a = getTestVector3f_0();
	vector3f *b = getTestVector3f_0();
	vector3f *res = NULL;
	safeGetPt(res, multiply_elem(a, b));
	vector3f *expect = new vector3f(0.0625, 0.25, 0.5625);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_multiply_elem_m_f(){
	const float array_multiply_elem_m_f[27] = {
		0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000, 
		0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 
		0.9000, 0.9500, 1.0000, 1.0500, 1.1000, 1.1500, 1.2000, 1.2500, 1.3000};
	cout<<"testing multiply_elem_m_f --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, multiply_elem(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_multiply_elem_m_f, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_multiply_elem_m_v(){
	const float array_multiply_elem_m_v[27] = {	
		0.0000, 0.0250, 0.0500, 0.0750, 0.1000, 0.1250, 0.1500, 0.1750, 0.2000, 
		0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 
		1.3500, 1.4250, 1.5000, 1.5750, 1.6500, 1.7250, 1.8000, 1.8750, 1.9500};
	cout<<"testing multiply_elem_m_v --- ";
	Mat *a = getTestMatrix_3();
	vector3f *b = getTestVector3f_0();
	Mat *res = NULL;
	safeGetPt(res, multiply_elem(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_multiply_elem_m_v, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_multiply_elem_m_m(){
	const float array_multiply_elem_m_m[27] = {	
		0.0000, 0.0100, 0.0400, 0.0900, 0.1600, 0.2500, 0.3600, 0.4900, 0.6400, 
		0.8100, 1.0000, 1.2100, 1.4400, 1.6900, 1.9600, 2.2500, 2.5600, 2.8900, 
		3.2400, 3.6100, 4.0000, 4.4100, 4.8400, 5.2900, 5.7600, 6.2500, 6.7600};
	cout<<"testing multiply_elem_m_m --- ";
	Mat *a = getTestMatrix_3();
	Mat *b = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, multiply_elem(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_multiply_elem_m_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_multiply(){
	const float array_multiply[27] = {	
		0.1500,  0.1800,  0.2100,  0.4200,  0.5400,  0.6600,  0.6900,  0.9000,  1.1100, 
		3.6600,  3.9600,  4.2600,  4.7400,  5.1300,  5.5200,  5.8200,  6.3000,  6.7800, 
		12.0300, 12.6000, 13.1700, 13.9200, 14.5800, 15.2400, 15.8100, 16.5600, 17.3100};
	cout<<"testing multiply --- ";
	Mat *a = getTestMatrix_3();
	Mat *b = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, multiply(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_multiply, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_t(){
	const float array_t[27] = {	
		0.0000, 0.3000, 0.6000, 0.1000, 0.4000, 0.7000, 0.2000, 0.5000, 0.8000, 
		0.9000, 1.2000, 1.5000, 1.0000, 1.3000, 1.6000, 1.1000, 1.4000, 1.7000, 
		1.8000, 2.1000, 2.4000, 1.9000, 2.2000, 2.5000, 2.0000, 2.3000, 2.6000};
	cout<<"testing transpose --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, t(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_t, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_div_rem(){
	cout<<"testing div_rem --- ";
	vector3f *a = getTestVector3f_1();
	int b = getTestInt();
	vector3f *res = NULL;
	safeGetPt(res, div_rem(a, b));
	vector3f *expect = new vector3f(1, 1, 1);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_div_no_rem(){
	cout<<"testing div_no_rem --- ";
	vector3f *a = getTestVector3f_1();
	int b = getTestInt();
	vector3f *res = NULL;
	safeGetPt(res, div_no_rem(a, b));
	vector3f *expect = new vector3f(1, 1, 1);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_divide_m_f(){
	const float array_divide_m_f[27] = {
		0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 1.2000, 1.4000, 1.6000, 
		1.8000, 2.0000, 2.2000, 2.4000, 2.6000, 2.8000, 3.0000, 3.2000, 3.4000, 
		3.6000, 3.8000, 4.0000, 4.2000, 4.4000, 4.6000, 4.8000, 5.0000, 5.2000};
	cout<<"testing divide_m_f --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, divide(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_divide_m_f, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

// edit zero-element in matrix
bool test_divide_f_m(){
	const float array_divide_f_m[27] = {
		5.0000, 5.0000, 2.5000, 1.6667, 1.2500, 1.0000, 0.8333, 0.7143, 0.6250, 
		0.5556, 0.5000, 0.4545, 0.4167, 0.3846, 0.3571, 0.3333, 0.3125, 0.2941, 
		0.2778, 0.2632, 0.2500, 0.2381, 0.2273, 0.2174, 0.2083, 0.2000, 0.1923};
	cout<<"testing divide_m_f --- ";
	Mat *a = getTestMatrix_3();
	a -> set(0, 0, 0, 0.1);
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, divide(b, a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_divide_f_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_divide_v_f(){
	cout<<"testing divide_v_f --- ";
	vector3f *a = getTestVector3f_0();
	float b = getTestFloat();
	vector3f *res = NULL;
	safeGetPt(res, divide(a, b));
	vector3f *expect = new vector3f(0.5, 1.0, 1.5);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_divide_f_v(){
	cout<<"testing divide_f_v --- ";
	vector3f *a = getTestVector3f_0();
	float b = getTestFloat();
	vector3f *res = NULL;
	safeGetPt(res, divide(b, a));
	vector3f *expect = new vector3f(2.0000, 1.0000, 0.6667);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_divide_m_v(){
	const float array_divide_m_v[27] = {
		0.0000, 0.4000, 0.8000, 1.2000, 1.6000, 2.0000, 2.4000, 2.8000, 3.2000, 
		1.8000, 2.0000, 2.2000, 2.4000, 2.6000, 2.8000, 3.0000, 3.2000, 3.4000, 
		2.4000, 2.5333, 2.6667, 2.8000, 2.9333, 3.0667, 3.2000, 3.3333, 3.4667};
	cout<<"testing divide_m_v --- ";
	Mat *a = getTestMatrix_3();
	vector3f *b = getTestVector3f_0();
	Mat *res = NULL;
	safeGetPt(res, divide(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_divide_m_v, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

// edit zero-element in matrix
bool test_divide_v_m(){
	const float array_divide_v_m[27] = {
		2.5000, 2.5000, 1.2500, 0.8333, 0.6250, 0.5000, 0.4167, 0.3571, 0.3125, 
		0.5556, 0.5000, 0.4545, 0.4167, 0.3846, 0.3571, 0.3333, 0.3125, 0.2941, 
		0.4167, 0.3947, 0.3750, 0.3571, 0.3409, 0.3261, 0.3125, 0.3000, 0.2885};
	cout<<"testing divide_v_m --- ";
	Mat *a = getTestMatrix_3();
	a -> set(0, 0, 0, 0.1);
	vector3f *b = getTestVector3f_0();
	Mat *res = NULL;
	safeGetPt(res, divide(b, a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_divide_v_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

// edit zero-element in matrix
bool test_divide_m_m(){
	const float array_divide_m_m[27] = {
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	cout<<"testing divide_m_m --- ";
	Mat *a = getTestMatrix_3();
	a -> set(0, 0, 0, 0.1);
	Mat *b = getTestMatrix_3();
	b -> set(0, 0, 0, 0.1);
	Mat *res = NULL;
	safeGetPt(res, divide(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_divide_m_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_divide_v_v(){
	cout<<"testing divide_v_v --- ";
	vector3f *a = getTestVector3f_0();
	vector3f *b = getTestVector3f_0();
	vector3f *res = NULL;
	safeGetPt(res, divide(a, b));
	vector3f *expect = new vector3f(1.0, 1.0, 1.0);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_exp(){
	const float array_exp[27] = {	
			 1.3385,  0.3173,  0.1662,
			 1.2187,  1.1106,  2.3172,
			 4.8925,  2.0591,  0.4115,
			 0.4473, 13.2699,  1.1053,
			 2.0070,  0.5133,  0.5801,
			 2.3050,  1.2060,  1.3546,
			 0.7837,  0.9208,  0.5486,
			 1.2407,  0.1447,  1.6323,
			 0.3117,  0.6447,  2.0946};
	cout<<"testing exp --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, exp(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_exp, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

// edit zero-element in matrix
bool test_log(){
	const float array_log[27] = {	
		-2.3026, -2.3026, -1.6094, -1.2040, -0.9163, -0.6931, -0.5108, -0.3567, -0.2231, 
		-0.1054,  0.0000,  0.0953,  0.1823,  0.2624,  0.3365,  0.4055,  0.4700,  0.5306, 
		 0.5878,  0.6419,  0.6931,  0.7419,  0.7885,  0.8329,  0.8755,  0.9163,  0.9555};
	cout<<"testing log --- ";
	Mat *a = getTestMatrix_3();
	a -> set(0, 0, 0, 0.1);
	Mat *res = NULL;
	safeGetPt(res, log(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_log, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_pow(){
	const float array_pow[27] = {
			0.0000, 0.3162, 0.4472, 0.5477, 0.6325, 0.7071, 0.7746, 0.8367, 0.8944,
			0.9487, 1.0000, 1.0488, 1.0954, 1.1402, 1.1832, 1.2247, 1.2649, 1.3038,
			1.3416, 1.3784, 1.4142, 1.4491, 1.4832, 1.5166, 1.5492, 1.5811, 1.6125};
	cout<<"testing pow --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, pow(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_pow, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_square_m(){
	const float array_square_m[27] = {	
			 0.0850, 1.3179, 3.2209,
			 0.0391, 0.0110, 0.7062,
			 2.5208, 0.5217, 0.7886,
			 0.6472, 6.6848, 0.0100,
			 0.4853, 0.4447, 0.2965,
			 0.6974, 0.0351, 0.0921,
			 0.0594, 0.0068, 0.3604,
			 0.0465, 3.7365, 0.2401,
			 1.3591, 0.1927, 0.5467};
	cout<<"testing square_m --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, square(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_square_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_square_v(){
	cout<<"testing square_v --- ";
	vector3f *a = getTestVector3f_0();
	vector3f *res = NULL;
	safeGetPt(res, square(a));
	vector3f *expect = new vector3f(0.0625, 0.2500, 0.5625);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_sqrt_m(){
	const float array_sqrt_m[27] = {	
		0.0000, 0.3162, 0.4472, 0.5477, 0.6325, 0.7071, 0.7746, 0.8367, 0.8944, 
		0.9487, 1.0000, 1.0488, 1.0954, 1.1402, 1.1832, 1.2247, 1.2649, 1.3038, 
		1.3416, 1.3784, 1.4142, 1.4491, 1.4832, 1.5166, 1.5492, 1.5811, 1.6125};
	cout<<"testing sqrt_m --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, sqrt(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_sqrt_m, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_sqrt_v(){
	cout<<"testing sqrt_v --- ";
	vector3f *a = getTestVector3f_0();
	vector3f *res = NULL;
	safeGetPt(res, sqrt(a));
	vector3f *expect = new vector3f(0.5000, 0.7071, 0.8660);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	res -> release();
	expect -> release();
	return result;
}

bool test_sum_v(){
	cout<<"testing sum_v --- ";
	vector3f *a = getTestVector3f_0();
	float res = sum(a);
	float expect = 1.5;
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	return result;
}

bool test_sum_m(){
	cout<<"testing sum_m --- ";
	Mat *a = getTestMatrix_3();
	vector3f *res = NULL;
	res = sum(a);
	vector3f *expect = new vector3f(3.6, 11.7, 19.8);
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	cout<<(result ? "success" : "failed")<<endl;
	res -> release();
	expect -> release();
	return result;
}

bool test_average(){
	cout<<"testing average --- ";
	Mat *a = getTestMatrix_3();
	vector3f *res = NULL;
	res = average(a);
	vector3f *expect = new vector3f(0.4, 1.3, 2.2);
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_stddev(){
	cout<<"testing standard deviation --- ";
	Mat *a = getTestMatrix_3();
	vector3f *res = NULL;
	vector3f *avg = new vector3f(0.4, 1.3, 2.2);
	res = stddev(a, avg);
	vector3f *expect = new vector3f(0.2582, 0.2582, 0.2582);
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	avg -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_max_v(){
	cout<<"testing max_v --- ";
	vector3f *a = getTestVector3f_0();
	float res = max(a);
	float expect = 0.75;
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_min_v(){
	cout<<"testing max_v --- ";
	vector3f *a = getTestVector3f_0();
	float res = min(a);
	float expect = 0.25;
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	a -> release();
	return result;
}

bool test_max_m(){
	cout<<"testing max_m --- ";
	Mat *a = getTestMatrix_3();
	vector3f *res = NULL;
	res = max(a);
	vector3f *expect = new vector3f(0.8, 1.7, 2.6);
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_min_m(){
	cout<<"testing min_m --- ";
	Mat *a = getTestMatrix_3();
	vector3f *res = NULL;
	res = min(a);
	vector3f *expect = new vector3f(0.0, 0.9, 1.8);
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_maxLoc(){
	cout<<"testing maxLoc --- ";
	Mat *a = getTestMatrix_3();
	vector3f *resVal = new vector3f();
	vector3f *resLoc = new vector3f();
	max(a, resVal, resLoc);
	vector3f *expectVal = new vector3f(0.8, 1.7, 2.6);
	vector3f *expectLoc = new vector3f(8, 8, 8);
	bool result = areApproximatelyIdentical(resVal, expectVal) && areApproximatelyIdentical(resLoc, expectLoc);
	a -> release();
	resVal -> release();
	expectVal -> release();
	resLoc -> release();
	expectLoc -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_minLoc(){
	cout<<"testing minLoc --- ";
	Mat *a = getTestMatrix_3();
	vector3f *resVal = new vector3f();
	vector3f *resLoc = new vector3f();
	min(a, resVal, resLoc);
	vector3f *expectVal = new vector3f(0.0, 0.9, 1.8);
	vector3f *expectLoc = new vector3f(0, 0, 0);
	bool result = areApproximatelyIdentical(resVal, expectVal) && areApproximatelyIdentical(resLoc, expectLoc);
	a -> release();
	resVal -> release();
	expectVal -> release();
	resLoc -> release();
	expectLoc -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_minMaxLoc(){
	cout<<"testing minLoc --- ";
	Mat *a = getTestMatrix_3();
	vector3f *resMinVal = new vector3f();
	vector3f *resMinLoc = new vector3f();
	vector3f *resMaxVal = new vector3f();
	vector3f *resMaxLoc = new vector3f();
	minMaxLoc(a, resMaxVal, resMaxLoc, resMinVal, resMinLoc);
	vector3f *expectMaxVal = new vector3f(0.8, 1.7, 2.6);
	vector3f *expectMaxLoc = new vector3f(8, 8, 8);
	vector3f *expectMinVal = new vector3f(0.0, 0.9, 1.8);
	vector3f *expectMinLoc = new vector3f(0, 0, 0);
	bool result = areApproximatelyIdentical(resMinVal, expectMinVal) &&
				  areApproximatelyIdentical(resMinLoc, expectMinLoc) &&
				  areApproximatelyIdentical(resMaxVal, expectMaxVal) &&
				  areApproximatelyIdentical(resMaxLoc, expectMaxLoc) ;
	a -> release();
	resMinVal -> release();
	expectMinVal -> release();
	resMinLoc -> release();
	expectMinLoc -> release();
	resMaxVal -> release();
	expectMaxVal -> release();
	resMaxLoc -> release();
	expectMaxLoc -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_greaterThan(){
	const float array_greaterThan[27] = {
			0, 0, 0, 0, 0, 0, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1};
	cout<<"testing greaterThan --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, greaterThan(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_greaterThan, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_lessThan(){
	const float array_lessThan[27] = {
			1, 1, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0};
	cout<<"testing lessThan --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, lessThan(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_lessThan, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_equalTo(){
	const float array_equalTo[27] = {
			0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0};
	cout<<"testing equalTo --- ";
	Mat *a = getTestMatrix_3();
	float b = getTestFloat();
	Mat *res = NULL;
	safeGetPt(res, equalTo(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_equalTo, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_convert_vv(){
	cout<<"testing convert_vv --- ";
	const float array_convert_vv[162] = {
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000};
	std::vector<std::vector<Mat*> > vec;
	getTestVectorVectorMat(vec);
	Mat *res = new Mat();
	convert(vec, res);
	Mat *expect = new Mat(vec[0].size() * vec[0][0] -> getLength(), vec.size(), 1);
	memcpy(expect -> hostData, array_convert_vv, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	releaseVector(vec);
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_convert_m(){
	cout<<"testing convert_m --- ";
	const float array_convert_m[162] = {
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
			0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
			0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
			1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000};
	std::vector<std::vector<Mat*> > expect;
	std::vector<std::vector<Mat*> > res;
	getTestVectorVectorMat(expect);
	Mat *a = new Mat(2 * 27, 3, 1);
	memcpy(a -> hostData, array_convert_m, a -> getLength() * sizeof(float));
	a -> hostToDevice();
	convert(a, res, 3, 3);
	bool result = areApproximatelyIdentical(res, expect);
	releaseVector(res);
	releaseVector(expect);
	a -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_Tanh(){
	const float array_Tanh[27] = {
			 0.2836, -0.8171, -0.9463,
			 0.1953,  0.1045,  0.6860,
			 0.9198,  0.6183, -0.7104,
			-0.6665,  0.9887,  0.0998,
			 0.6022, -0.5829, -0.4964,
			 0.6832,  0.1852,  0.2945,
			-0.2390, -0.0823, -0.5373,
			 0.2124, -0.9590,  0.4542,
			-0.8229, -0.4128,  0.6288};
	cout<<"testing Tanh --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, Tanh(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_Tanh, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_dTanh(){
	const float array_dTanh[27] = {
			 0.9150, -0.3179, -2.2209,
			 0.9609,  0.9890,  0.2938,
			-1.5208,  0.4783,  0.2114,
			 0.3528, -5.6848,  0.9900,
			 0.5147,  0.5553,  0.7035,
			 0.3026,  0.9649,  0.9079,
			 0.9406,  0.9932,  0.6396,
			 0.9535, -2.7365,  0.7599,
			-0.3591,  0.8073,  0.4533};
	cout<<"testing dtanh --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, dTanh(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_dTanh, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_sigmoid(){
	const float array_sigmoid[27] = {
			 0.5724, 0.2409, 0.1425,
			 0.5493, 0.5262, 0.6985,
			 0.8303, 0.6731, 0.2915,
			 0.3091, 0.9299, 0.5250,
			 0.6674, 0.3392, 0.3671,
			 0.6974, 0.5467, 0.5753,
			 0.4394, 0.4794, 0.3543,
			 0.5537, 0.1264, 0.6201,
			 0.2376, 0.3920, 0.6769};
	cout<<"testing sigmoid --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, sigmoid(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_sigmoid, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_dsigmoid(){
	const float array_dsigmoid[27] = {
			 0.2448, 0.1828, 0.1222,
			 0.2476, 0.2493, 0.2106,
			 0.1409, 0.2200, 0.2065,
			 0.2135, 0.0652, 0.2494,
			 0.2220, 0.2241, 0.2323,
			 0.2110, 0.2478, 0.2443,
			 0.2463, 0.2496, 0.2288,
			 0.2471, 0.1104, 0.2356,
			 0.1811, 0.2383, 0.2187};
	cout<<"testing dsigmoid --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, dsigmoid(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_dsigmoid, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_dsigmoid_a(){
	const float array_dsigmoid_a[27] = {
			 0.2066, -2.4659, -5.0156,
			 0.1587,  0.0939,  0.1341,
			-0.9331,  0.2006, -1.6766,
			-1.4517, -4.0993,  0.0901,
			 0.2113, -1.1116, -0.8410,
			 0.1377,  0.1522,  0.2114,
			-0.3031, -0.0893, -0.9607,
			 0.1692, -5.6695,  0.2499,
			-2.5249, -0.6317,  0.1927};
	cout<<"testing dsigmoid_a --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, dsigmoid_a(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_dsigmoid_a, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_ReLU(){
	const float array_ReLU[27] = {
			0.2916, 0.0000, 0.0000,
			0.1978, 0.1049, 0.8404,
			1.5877, 0.7223, 0.0000,
			0.0000, 2.5855, 0.1001,
			0.6966, 0.0000, 0.0000,
			0.8351, 0.1873, 0.3035,
			0.0000, 0.0000, 0.0000,
			0.2157, 0.0000, 0.4900,
			0.0000, 0.0000, 0.7394
};
	cout<<"testing ReLU --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, ReLU(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_ReLU, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_dReLU(){
	const float array_dReLU[27] = {
			 1, 0, 0,
			 1, 1, 1,
			 1, 1, 0,
			 0, 1, 1,
			 1, 0, 0,
			 1, 1, 1,
			 0, 0, 0,
			 1, 0, 1,
			 0, 0, 1};
	cout<<"testing dReLU --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, dReLU(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_dReLU, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_LeakyReLU(){
	const float array_LeakyReLU[27] = {
			 0.2916, -0.0115, -0.0179,
			 0.1978,  0.1049,  0.8404,
			 1.5877,  0.7223, -0.0089,
			-0.0080,  2.5855,  0.1001,
			 0.6966, -0.0067, -0.0054,
			 0.8351,  0.1873,  0.3035,
			-0.0024, -0.0008, -0.0060,
			 0.2157, -0.0193,  0.4900,
			-0.0117, -0.0044,  0.7394};
	cout<<"testing LeakyReLU --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, LeakyReLU(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_LeakyReLU, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_dLeakyReLU(){
	const float array_dLeakyReLU[27] = {
			1.0000, 0.0100, 0.0100,
			1.0000, 1.0000, 1.0000,
			1.0000, 1.0000, 0.0100,
			0.0100, 1.0000, 1.0000,
			1.0000, 0.0100, 0.0100,
			1.0000, 1.0000, 1.0000,
			0.0100, 0.0100, 0.0100,
			1.0000, 0.0100, 1.0000,
			0.0100, 0.0100, 1.0000};
	cout<<"testing dLeakyReLU --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, dLeakyReLU(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_dLeakyReLU, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_fliplr(){
	const float array_fliplr[27] = {
			0.2000, 0.1000, 0.0000,
			0.5000, 0.4000, 0.3000,
			0.8000, 0.7000, 0.6000,
			1.1000, 1.0000, 0.9000,
			1.4000, 1.3000, 1.2000,
			1.7000, 1.6000, 1.5000,
			2.0000, 1.9000, 1.8000,
			2.3000, 2.2000, 2.1000,
			2.6000, 2.5000, 2.4000};
	cout<<"testing fliplr --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, fliplr(a));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_fliplr, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_rot90(){
	const float array_rot90[27] = {
			0.8000, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000, 0.2000, 0.1000, 0.0000,
			1.7000, 1.6000, 1.5000, 1.4000, 1.3000, 1.2000, 1.1000, 1.0000, 0.9000,
			2.6000, 2.5000, 2.4000, 2.3000, 2.2000, 2.1000, 2.0000, 1.9000, 1.8000};
	cout<<"testing rot90 --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, rot90(a, 2));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_rot90, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_dopadding(){
	const float array_dopadding[75] = {
			0,0,0,0,0,0,0,0.1000,0.2000,0,0,0.3000,0.4000,0.5000,0,0,0.6000,0.7000,0.8000,0,0,0,0,0,0,
			0,0,0,0,0,0,0.9000,1.0000,1.1000,0,0,1.2000,1.3000,1.4000,0,0,1.5000,1.6000,1.7000,0,0,0,0,0,0,
			0,0,0,0,0,0,1.8000,1.9000,2.0000,0,0,2.1000,2.2000,2.3000,0,0,2.4000,2.5000,2.6000,0,0,0,0,0,0};
	cout<<"testing dopadding --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, dopadding(a, 1));
	Mat *expect = new Mat(a -> rows + 2, a -> cols + 2, a -> channels);
	memcpy(expect -> hostData, array_dopadding, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_depadding(){
	const float array_depadding[75] = {
			0.6000, 0.7000, 0.8000, 1.1000, 1.2000, 1.3000, 1.6000, 1.7000, 1.8000,
			3.1000, 3.2000, 3.3000, 3.6000, 3.7000, 3.8000, 4.1000, 4.2000, 4.3000,
			5.6000, 5.7000, 5.8000, 6.1000, 6.2000, 6.3000, 6.6000, 6.7000, 6.8000};
	cout<<"testing depadding --- ";
	Mat *a = getTestMatrix_5();
	Mat *res = NULL;
	safeGetPt(res, depadding(a, 1));
	Mat *expect = new Mat(a -> rows - 2, a -> cols - 2, a -> channels);
	memcpy(expect -> hostData, array_depadding, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_reduce(){
	const float array_reduce_row_sum[9] = {0.9, 1.2, 1.5, 3.6, 3.9, 4.2, 6.3, 6.6, 6.9};
	const float array_reduce_col_sum[9] = {0.3, 1.2, 2.1, 3.0, 3.9, 4.8, 5.7, 6.6, 7.5};
	const float array_reduce_row_max[9] = {0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 2.4, 2.5, 2.6};
	const float array_reduce_col_max[9] = {0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6};
	cout<<"testing reduce --- ";
	Mat *a = getTestMatrix_3();
	Mat *res_row_sum = NULL;
	Mat *res_col_sum = NULL;
	Mat *res_row_max = NULL;
	Mat *res_col_max = NULL;
	safeGetPt(res_row_sum, reduce(a, REDUCE_TO_SINGLE_ROW, REDUCE_SUM));
	safeGetPt(res_col_sum, reduce(a, REDUCE_TO_SINGLE_COL, REDUCE_SUM));
	safeGetPt(res_row_max, reduce(a, REDUCE_TO_SINGLE_ROW, REDUCE_MAX));
	safeGetPt(res_col_max, reduce(a, REDUCE_TO_SINGLE_COL, REDUCE_MAX));
	Mat *expect_row_sum = new Mat(1, a -> cols, a -> channels);
	Mat *expect_col_sum = new Mat(a -> rows, 1, a -> channels);
	Mat *expect_row_max = new Mat(1, a -> cols, a -> channels);
	Mat *expect_col_max = new Mat(a -> rows, 1, a -> channels);
	memcpy(expect_row_sum -> hostData, array_reduce_row_sum, expect_row_sum -> getLength() * sizeof(float));
	memcpy(expect_col_sum -> hostData, array_reduce_col_sum, expect_col_sum -> getLength() * sizeof(float));
	memcpy(expect_row_max -> hostData, array_reduce_row_max, expect_row_max -> getLength() * sizeof(float));
	memcpy(expect_col_max -> hostData, array_reduce_col_max, expect_col_max -> getLength() * sizeof(float));
	expect_row_sum -> hostToDevice();
	expect_col_sum -> hostToDevice();
	expect_row_max -> hostToDevice();
	expect_col_max -> hostToDevice();
	bool result = 	areApproximatelyIdentical(res_row_sum, expect_row_sum) &&
					areApproximatelyIdentical(res_col_sum, expect_col_sum) &&
					areApproximatelyIdentical(res_row_max, expect_row_max) &&
					areApproximatelyIdentical(res_col_max, expect_col_max);
	a -> release();
    res_row_sum -> release();
    res_col_sum -> release();
    res_row_max -> release();
    res_col_max -> release();
    expect_row_sum -> release();
    expect_col_sum -> release();
    expect_row_max -> release();
    expect_col_max -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_getRange(){
	const float array_getRange[24] = {
			0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4,
			3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9,
			5.6, 5.7, 5.8, 5.9, 6.1, 6.2, 6.3, 6.4};
	cout<<"testing getRange --- ";
	Mat *a = getTestMatrix_5();
	Mat *res = NULL;
	safeGetPt(res, getRange(a, 1, 4, 1, 2));
	Mat *expect = new Mat(2, 4, a -> channels);
	memcpy(expect -> hostData, array_getRange, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_interpolation(){
	const float array_interpolation[75] = {
			0, 0, 0.1, 0, 0.2, 0, 0, 0, 0, 0, 0.3, 0, 0.4, 0, 0.5, 0, 0, 0, 0, 0, 0.6, 0, 0.7, 0, 0.8,
		  0.9, 0, 1.0, 0, 1.1, 0, 0, 0, 0, 0, 1.2, 0, 1.3, 0, 1.4, 0, 0, 0, 0, 0, 1.5, 0, 1.6, 0, 1.7,
		  1.8, 0, 1.9, 0, 2.0, 0, 0, 0, 0, 0, 2.1, 0, 2.2, 0, 2.3, 0, 0, 0, 0, 0, 2.4, 0, 2.5, 0, 2.6};
	cout<<"testing interpolation --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, interpolation(a, 5));
	Mat *expect = new Mat(5, 5, a -> channels);
	memcpy(expect -> hostData, array_interpolation, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_repmat(){
	const float array_repmat[162] = {
				0.0000, 0.1000, 0.2000, 0.0000, 0.1000, 0.2000, 0.0000, 0.1000, 0.2000,
				0.3000, 0.4000, 0.5000, 0.3000, 0.4000, 0.5000, 0.3000, 0.4000, 0.5000,
				0.6000, 0.7000, 0.8000, 0.6000, 0.7000, 0.8000, 0.6000, 0.7000, 0.8000,
				0.0000, 0.1000, 0.2000, 0.0000, 0.1000, 0.2000, 0.0000, 0.1000, 0.2000,
				0.3000, 0.4000, 0.5000, 0.3000, 0.4000, 0.5000, 0.3000, 0.4000, 0.5000,
				0.6000, 0.7000, 0.8000, 0.6000, 0.7000, 0.8000, 0.6000, 0.7000, 0.8000,
				0.9000, 1.0000, 1.1000, 0.9000, 1.0000, 1.1000, 0.9000, 1.0000, 1.1000,
				1.2000, 1.3000, 1.4000, 1.2000, 1.3000, 1.4000, 1.2000, 1.3000, 1.4000,
				1.5000, 1.6000, 1.7000, 1.5000, 1.6000, 1.7000, 1.5000, 1.6000, 1.7000,
				0.9000, 1.0000, 1.1000, 0.9000, 1.0000, 1.1000, 0.9000, 1.0000, 1.1000,
				1.2000, 1.3000, 1.4000, 1.2000, 1.3000, 1.4000, 1.2000, 1.3000, 1.4000,
				1.5000, 1.6000, 1.7000, 1.5000, 1.6000, 1.7000, 1.5000, 1.6000, 1.7000,
				1.8000, 1.9000, 2.0000, 1.8000, 1.9000, 2.0000, 1.8000, 1.9000, 2.0000,
				2.1000, 2.2000, 2.3000, 2.1000, 2.2000, 2.3000, 2.1000, 2.2000, 2.3000,
				2.4000, 2.5000, 2.6000, 2.4000, 2.5000, 2.6000, 2.4000, 2.5000, 2.6000,
				1.8000, 1.9000, 2.0000, 1.8000, 1.9000, 2.0000, 1.8000, 1.9000, 2.0000,
				2.1000, 2.2000, 2.3000, 2.1000, 2.2000, 2.3000, 2.1000, 2.2000, 2.3000,
				2.4000, 2.5000, 2.6000, 2.4000, 2.5000, 2.6000, 2.4000, 2.5000, 2.6000};
	cout<<"testing repmat --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, repmat(a, 2, 3));
	Mat *expect = new Mat(6, 9, a -> channels);
	memcpy(expect -> hostData, array_repmat, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_downSample(){
	const float array_downsample[27] = {
			0.0, 0.2, 0.4, 1.0, 1.2, 1.4, 2.0, 2.2, 2.4,
			2.5, 2.7, 2.9, 3.5, 3.7, 3.9, 4.5, 4.7, 4.9,
			5.0, 5.2, 5.4, 6.0, 6.2, 6.4, 7.0, 7.2, 7.4};
	cout<<"testing downsample --- ";
	Mat *a = getTestMatrix_5();
	Mat *res = NULL;
	safeGetPt(res, downSample(a, 2, 2));
	Mat *expect = new Mat(3, 3, a -> channels);
	memcpy(expect -> hostData, array_downsample, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_copyMakeBorder(){
	const float array_copyMakeBorder[180] = {
			0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
			0.2500, 0.2500, 0.2500, 0.0000, 0.1000, 0.2000, 0.2500, 0.2500, 0.2500, 0.2500,
			0.2500, 0.2500, 0.2500, 0.3000, 0.4000, 0.5000, 0.2500, 0.2500, 0.2500, 0.2500,
			0.2500, 0.2500, 0.2500, 0.6000, 0.7000, 0.8000, 0.2500, 0.2500, 0.2500, 0.2500,
			0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
			0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
			0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
			0.5000, 0.5000, 0.5000, 0.9000, 1.0000, 1.1000, 0.5000, 0.5000, 0.5000, 0.5000,
			0.5000, 0.5000, 0.5000, 1.2000, 1.3000, 1.4000, 0.5000, 0.5000, 0.5000, 0.5000,
			0.5000, 0.5000, 0.5000, 1.5000, 1.6000, 1.7000, 0.5000, 0.5000, 0.5000, 0.5000,
			0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
			0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
			0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500,
			0.7500, 0.7500, 0.7500, 1.8000, 1.9000, 2.0000, 0.7500, 0.7500, 0.7500, 0.7500,
			0.7500, 0.7500, 0.7500, 2.1000, 2.2000, 2.3000, 0.7500, 0.7500, 0.7500, 0.7500,
			0.7500, 0.7500, 0.7500, 2.4000, 2.5000, 2.6000, 0.7500, 0.7500, 0.7500, 0.7500,
			0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500,
			0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500, 0.7500};
	cout<<"testing copyMakeBorder --- ";
	Mat *a = getTestMatrix_3();
	Mat *res = NULL;
	vector3f *v = getTestVector3f_0();
	safeGetPt(res, copyMakeBorder(a, 1, 2, 3, 4, v));
	Mat *expect = new Mat(6, 10, a -> channels);
	memcpy(expect -> hostData, array_copyMakeBorder, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	v -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_kron(){
	const float array_kron[108] = {
			 0.0000,  0.0000,  0.1000,  0.2000,  0.2000,  0.4000,
			 0.0000,  0.0000,  0.3000,  0.4000,  0.6000,  0.8000,
			 0.3000,  0.6000,  0.4000,  0.8000,  0.5000,  1.0000,
			 0.9000,  1.2000,  1.2000,  1.6000,  1.5000,  2.0000,
			 0.6000,  1.2000,  0.7000,  1.4000,  0.8000,  1.6000,
			 1.8000,  2.4000,  2.1000,  2.8000,  2.4000,  3.2000,
			 0.9000,  1.8000,  1.0000,  2.0000,  1.1000,  2.2000,
			 2.7000,  3.6000,  3.0000,  4.0000,  3.3000,  4.4000,
			 1.2000,  2.4000,  1.3000,  2.6000,  1.4000,  2.8000,
			 3.6000,  4.8000,  3.9000,  5.2000,  4.2000,  5.6000,
			 1.5000,  3.0000,  1.6000,  3.2000,  1.7000,  3.4000,
			 4.5000,  6.0000,  4.8000,  6.4000,  5.1000,  6.8000,
			 1.8000,  3.6000,  1.9000,  3.8000,  2.0000,  4.0000,
			 5.4000,  7.2000,  5.7000,  7.6000,  6.0000,  8.0000,
			 2.1000,  4.2000,  2.2000,  4.4000,  2.3000,  4.6000,
			 6.3000,  8.4000,  6.6000,  8.8000,  6.9000,  9.2000,
			 2.4000,  4.8000,  2.5000,  5.0000,  2.6000,  5.2000,
			 7.2000,  9.6000,  7.5000, 10.0000,  7.8000, 10.4000};
	cout<<"testing kron --- ";
	Mat *a = getTestMatrix_3();
	Mat *b =new Mat(2, 2, 1);
	b -> set(0, 0, 0, 1.0);
	b -> set(0, 1, 0, 2.0);
	b -> set(1, 0, 0, 3.0);
	b -> set(1, 1, 0, 4.0);
	Mat *res = NULL;
	safeGetPt(res, kron(a, b));
	Mat *expect = new Mat(6, 6, a -> channels);
	memcpy(expect -> hostData, array_kron, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_conv2_kernel(){
	const float array_conv2_kernel[75] = {
			0.24, 0.45, 0.81, 1.17, 1.44,
			0.99, 1.2, 1.56, 1.92, 2.19,
			2.79, 3, 3.36, 3.72, 3.99,
			4.59, 4.8, 5.16, 5.52, 5.79,
			6.24, 6.45, 6.81, 7.17, 7.44,
			31.11, 31.86, 33.03, 34.2, 35.01,
			34.56, 35.31, 36.48, 37.65, 38.46,
			40.41, 41.16, 42.33, 43.5, 44.31,
			46.26, 47.01, 48.18, 49.35, 50.16,
			50.61, 51.36, 52.53, 53.7, 54.51,
			102.48, 103.77, 105.75, 107.73, 109.08,
			108.63, 109.92, 111.9, 113.88, 115.23,
			118.53, 119.82, 121.8, 123.78, 125.13,
			128.43, 129.72, 131.7, 133.68, 135.03,
			135.48, 136.77, 138.75, 140.73, 142.08};
	cout<<"testing conv_kernel --- ";
	Mat *a = getTestMatrix_5();
	Mat *b = getTestMatrix_3();
	Mat *res = NULL;
	safeGetPt(res, conv2(a, b));
	Mat *expect = new Mat(5, 5, a -> channels);
	memcpy(expect -> hostData, array_conv2_kernel, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	b -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_conv2(){
	const float array_conv2_valid[27] = {
			0.0800,    0.4100,    0.5600,
		    1.5900,    3.3600,    3.0300,
		    3.4400,    6.1700,    4.8800,
		   12.1600,   20.0900,   14.7200,
		   26.0100,   42.3300,   30.5700,
		   23.9200,   38.4500,   27.4400,
		   42.2400,   66.7700,   46.8800,
		   77.4300,  121.8000,   85.1100,
		   62.4000,   97.7300,   68.0000};
	const float array_conv2_same[48] = {
		    0.0000,    0.0100,    0.0700,    0.0800,
		    0.1500,    1.2000,    1.9200,    1.0500,
		    1.0500,    4.8000,    5.5200,    2.5500,
		    1.2000,    4.3900,    4.8100,    1.9200,
		    2.2500,    7.7800,    8.3800,    3.1900,
		   10.5000,   35.3100,   37.6500,   13.9800,
		   14.1000,   47.0100,   49.3500,   18.1800,
		    6.7500,   22.0600,   23.0200,    8.3300,
		    9.0000,   29.0500,   30.1900,   10.8000,
		   34.3500,  109.9200,  113.8800,   40.4100,
		   40.6500,  129.7200,  133.6800,   47.3100,
		   16.8000,   53.2300,   54.7300,   19.2400};
	const float array_conv2_full[75] = {
			 0,         0,         0,         0,         0,
	         0,    0.0800,    0.4100,    0.5600,         0,
	         0,    1.5900,    3.3600,    3.0300,         0,
	         0,    3.4400,    6.1700,    4.8800,         0,
	         0,         0,         0,         0,         0,
	         0,         0,         0,         0,         0,
	         0,   12.1600,   20.0900,   14.7200,         0,
	         0,   26.0100,   42.3300,   30.5700,         0,
	         0,   23.9200,   38.4500,   27.4400,         0,
	         0,         0,         0,         0,         0,
	         0,         0,         0,         0,         0,
	         0,   42.2400,   66.7700,   46.8800,         0,
	         0,   77.4300,  121.8000,   85.1100,         0,
	         0,   62.4000,   97.7300,   68.0000,         0,
	         0,         0,         0,         0,         0};
	cout<<"testing conv --- ";
	Mat *a = getTestMatrix_5();
	Mat *b = getTestMatrix_3();
	Mat *res_valid = NULL;
	Mat *res_same = NULL;
	Mat *res_full = NULL;
	safeGetPt(res_valid, conv2(a, b, CONV_VALID, 1, 2));
	safeGetPt(res_same, conv2(a, b, CONV_SAME, 1, 2));
	safeGetPt(res_full, conv2(a, b, CONV_FULL, 1, 2));
	Mat *expect_valid = new Mat(3, 3, a -> channels);
	Mat *expect_same = new Mat(4, 4, a -> channels);
	Mat *expect_full = new Mat(5, 5, a -> channels);
	memcpy(expect_valid -> hostData, array_conv2_valid, expect_valid -> getLength() * sizeof(float));
	memcpy(expect_same -> hostData, array_conv2_same, expect_same -> getLength() * sizeof(float));
	memcpy(expect_full -> hostData, array_conv2_full, expect_full -> getLength() * sizeof(float));
	expect_valid -> hostToDevice();
	expect_same -> hostToDevice();
	expect_full -> hostToDevice();
	bool result = 	areApproximatelyIdentical(res_valid, expect_valid) &&
					areApproximatelyIdentical(res_same, expect_same) &&
					areApproximatelyIdentical(res_full, expect_full);
	a -> release();
	b -> release();
	res_valid -> release();
	res_same -> release();
	res_full -> release();
	expect_valid -> release();
	res_same -> release();
	res_full -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_pooling_max(){
	// I did this array by hand lol
	const float array_pooling_max[58] = {
			1.8339, 2.9080, 1.6821, 1.0826,
			1.5442, 2.5855, 1.0391, 0.2571,
			3.5784, 1.3514, 1.6039, 0.9248,
			2.7694,-0.5890, 0.0414,-0.0549,

			3.0349, 0.8404, 1.6035, 0.9111,
			0.8886, 2.5260, 0.8261, 1.2503,
			1.4897, 1.7119, 2.2294, 1.1921,
			1.4172, 0.7914, 0.6252,-1.6118,

			1.4193, 1.3546, 1.0001, 1.0205,
			1.6302, 0.9610, 0.5152, 0.8617,
			0.8351, 1.4367, 0.4716, 0.5812,
			1.1174, 0.9642, 1.2424,-2.3193
	};
	// and this...
	const float array_pooling_max_loc_ch0[16] = {
			10, 4, 17, 19,
			42, 33, 56, 49,
			80, 75, 78, 79,
			90, 95, 98, 99};
	const float array_pooling_max_loc_ch1[16] = {
			10, 3, 17, 9,
			32, 35, 56, 39,
			70, 83, 88, 89,
			90, 95, 96, 99};
	const float array_pooling_max_loc_ch2[16] = {
			2, 23, 8, 29,
			30, 43, 56, 39,
			62, 63, 68, 79,
			91, 94, 97, 99};
	std::vector<vector3f*> expect_loc_max(16);
	for(int i = 0; i < 16; ++i){
		expect_loc_max[i] = new vector3f(array_pooling_max_loc_ch0[i], array_pooling_max_loc_ch1[i], array_pooling_max_loc_ch2[i]);
	}
	cout<<"testing pooling max --- ";
	Mat *a = getTestMatrix_10_rand();
	Mat *res_max = NULL;
	std::vector<vector3f*> res_loc_max;
	safeGetPt(res_max, pooling(a, 3, POOL_MAX, res_loc_max));
	Mat *expect_max = new Mat(4, 4, a -> channels);
	memcpy(expect_max -> hostData, array_pooling_max, expect_max -> getLength() * sizeof(float));
	expect_max -> hostToDevice();
	bool result = areApproximatelyIdentical(res_max, expect_max) && areApproximatelyIdentical(res_loc_max, expect_loc_max);

//	expect_max -> printHost("EXP");
//	res_max -> printHost("RES");
//	Mat *tmp = new Mat();
//	safeGetPt(tmp, subtract(expect_max, res_max));
//	tmp -> printHost("TMP");
//	for(int i = 0; i < 16; ++i){
//		cout<<"i = "<<i<<endl;
//		res_loc_max[i] ->print(" ");
//		expect_loc_max[i] ->print(" ");
//	}

	a -> release();
	res_max -> release();
	expect_max -> release();
	releaseVector(res_loc_max);
	releaseVector(expect_loc_max);
	res_loc_max.clear();
	expect_loc_max.clear();
	std::vector<vector3f*>().swap(res_loc_max);
	std::vector<vector3f*>().swap(expect_loc_max);
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}


bool test_pooling_mean(){
	// I did this array by hand lol
	const float array_pooling_mean[58] = {
		-0.1910, 0.5841, -0.1404, 0.8053, 
		 0.0321, 0.0438, -0.2805, -0.4461, 
		 0.1258, -0.0578, 0.3872, -0.1323, 
		 1.1361, -1.4785, -0.0918, -0.0549, 

		 0.2585, -0.5515, 0.0584, 0.6186, 
		-0.1571, 0.4968, -0.1625, 0.8066, 
		 0.3577, 0.0874, 0.4042, -0.0500, 
		 0.7768, 0.0994, 0.4687, -1.6118, 

		 0.0099, -0.9693, -0.3287, -0.3176, 
		 0.5607, -0.1008, -0.2375, 0.2640, 
		 0.1001, -0.5375, -0.5546, -1.3658, 
		-0.2786, 0.2061, 0.5829, -2.3193
	};
	// and this...
	const float array_pooling_mean_loc_ch0[16] = {
			0, 3, 6, 9,
			30, 33, 36, 39,
			60, 63, 66, 69,
			90, 93, 96, 99};
	const float array_pooling_mean_loc_ch1[16] = {
			0, 3, 6, 9,
			30, 33, 36, 39,
			60, 63, 66, 69,
			90, 93, 96, 99};
	const float array_pooling_mean_loc_ch2[16] = {
			0, 3, 6, 9,
			30, 33, 36, 39,
			60, 63, 66, 69,
			90, 93, 96, 99};
	std::vector<vector3f*> expect_loc_mean(16);
	for(int i = 0; i < 16; ++i){
		expect_loc_mean[i] = new vector3f(array_pooling_mean_loc_ch0[i], array_pooling_mean_loc_ch1[i], array_pooling_mean_loc_ch2[i]);
	}
	cout<<"testing pooling mean --- ";
	Mat *a = getTestMatrix_10_rand();
	Mat *res_mean = NULL;
	std::vector<vector3f*> res_loc_mean;
	safeGetPt(res_mean, pooling(a, 3, POOL_MEAN, res_loc_mean));
	Mat *expect_mean = new Mat(4, 4, a -> channels);
	memcpy(expect_mean -> hostData, array_pooling_mean, expect_mean -> getLength() * sizeof(float));
	expect_mean -> hostToDevice();
	bool result = areApproximatelyIdentical(res_mean, expect_mean) && areApproximatelyIdentical(res_loc_mean, expect_loc_mean);

//	expect_mean -> printHost("EXP");
//	res_mean -> printHost("RES");
//	Mat *tmp = new Mat();
//	safeGetPt(tmp, subtract(expect_mean, res_mean));
//	tmp -> printHost("TMP");
//	for(int i = 0; i < 16; ++i){
//		cout<<"i = "<<i<<endl;
//		res_loc_mean[i] ->print(" ");
//		expect_loc_mean[i] ->print(" ");
//	}
	a -> release();
	res_mean -> release();
	expect_mean -> release();
	releaseVector(res_loc_mean);
	releaseVector(expect_loc_mean);
	res_loc_mean.clear();
	expect_loc_mean.clear();
	std::vector<vector3f*>().swap(res_loc_mean);
	std::vector<vector3f*>().swap(expect_loc_mean);
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_pooling_overlap_max(){
	// I did this array by hand lol
	const float array_pooling_max[18] = {
		0.7394, 2.9080, 2.9080, 
		2.5855, 2.9080, 2.9080, 
		1.4193, 1.3546, 1.3546, 
		1.5877, 1.3546, 1.3546, 
		1.4367, 1.4367, 1.4367, 
		1.4367, 1.4367, 1.4367
	};
	// and this...
	const float array_pooling_max_loc_ch0[6] = {
		17, 8, 8, 
		21, 8, 8
	};
	const float array_pooling_max_loc_ch1[6] = {
		5, 17, 17, 
		20, 17, 17
	};
	const float array_pooling_max_loc_ch2[6] = {
		12, 12, 12, 
		12, 12, 12
	};
	std::vector<vector3f*> expect_loc_max(6);
	for(int i = 0; i < 6; ++i){
		expect_loc_max[i] = new vector3f(array_pooling_max_loc_ch0[i], array_pooling_max_loc_ch1[i], array_pooling_max_loc_ch2[i]);
	}
	cout<<"testing pooling overlap max--- ";
	Mat *a = getTestMatrix_5_rand();
	Mat *res_max = NULL;
	std::vector<vector3f*> res_loc_max;
	vector2i *window_size = new vector2i(3, 4);
	safeGetPt(res_max, pooling_with_overlap(a, window_size, 1, POOL_MAX, res_loc_max));
	Mat *expect_max = new Mat(2, 3, a -> channels);
	memcpy(expect_max -> hostData, array_pooling_max, expect_max -> getLength() * sizeof(float));
	expect_max -> hostToDevice();
	bool result = areApproximatelyIdentical(res_max, expect_max) && areApproximatelyIdentical(res_loc_max, expect_loc_max);
//	expect_max -> printHost("EXP");
//	res_max -> printHost("RES");
//	Mat *tmp = new Mat();
//	safeGetPt(tmp, subtract(expect_max, res_max));
//	tmp -> printHost("TMP");
//	for(int i = 0; i < 16; ++i){
//		cout<<"i = "<<i<<endl;
//		res_loc_max[i] ->print(" ");
//		expect_loc_max[i] ->print(" ");
//	}

	a -> release();
	res_max -> release();
	expect_max -> release();
	releaseVector(res_loc_max);
	releaseVector(expect_loc_max);
	res_loc_max.clear();
	expect_loc_max.clear();
	std::vector<vector3f*>().swap(res_loc_max);
	std::vector<vector3f*>().swap(expect_loc_max);
	window_size -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_findMax(){
	const float array_findMax[3] = {2, 2, 1};
	cout<<"testing findMax --- ";
	Mat *a = getTestMatrix_3_rand();
	Mat *res = NULL;
	safeGetPt(res, findMax(a));
	Mat *expect = new Mat(1, a -> cols, 1);
	memcpy(expect -> hostData, array_findMax, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

// only calculates first channel.
bool test_sameValuesInMat(){
	const float array_sameValuesInMat[27] = {
            0.2916, -1.1480, -1.8888,
            0.3434,  0.2323,  0.8404,
            1.5877,  0.7223, -0.8880,
           -0.8045,  2.5855,  0.1001,
            0.6966, -0.2284, -0.5445,
            0.6565,  0.1873,  0.6666,
           -0.2437, -0.0825, -0.6003,
            0.2157, -1.5656,  0.4900,
           -1.1658, -0.4390,  0.7394};
	cout<<"testing sameValuesInMat --- ";
	Mat *sameMat = new Mat(3, 3, 3);
	memcpy(sameMat -> hostData, array_sameValuesInMat, sameMat -> getLength() * sizeof(float));
	sameMat -> hostToDevice();
	Mat *a = getTestMatrix_3_rand();
	int res = sameValuesInMat(a, sameMat);
	bool result = (res == 6);
	a -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}



/*

bool test_pooling_with_overlap();
bool test_unpooling_with_overlap();
bool test_unpooling();
 */
