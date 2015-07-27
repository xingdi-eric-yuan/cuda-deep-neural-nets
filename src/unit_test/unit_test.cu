#include "unit_test.h"


void runAllTest(){
	int counter = 0;
	int success = 0;
	if(test_add_v_f()){ ++ success;} ++ counter;
	if(test_add_m_v()){ ++ success;} ++ counter;
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

vector3f* getTestVector3f(){
	vector3f *v = new vector3f();
	for(int i = 0; i < 3; ++i){
		v -> set(i, (float)(i + 1) / 4.0);
	}
	return v;
}

float getTestFloat(){
	return 0.5;
}

bool hostEqualToDevice(const Mat* a){
	if(NULL == a -> hostData || NULL == a -> devData) return false;
	float *tmpMemory = (float*)MemoryMonitor::instance()->cpuMalloc(a -> getLength() * sizeof(float));
	cudaMemcpy(tmpMemory, a -> devData, a -> getLength() * sizeof(float), cudaMemcpyDeviceToHost);
	int n = memcmp(tmpMemory, a -> hostData, a -> getLength() * sizeof(float));
	free(tmpMemory);
	return 0 == n ? true : false;
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
		float n = a -> get(i) != b -> get(i);
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
	vector3f *a = getTestVector3f();
	float b = getTestFloat();
	vector3f *res = NULL;
	res = add(a, b);
	vector3f *expect = new vector3f(0.75, 1.0, 1.25);
	bool result = areApproximatelyIdentical(res, expect);
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}

bool test_add_m_v(){
	const float array_add_m_v[27] = {	0.2500, 0.3500, 0.4500, 0.5500, 0.6500, 0.7500, 0.8500, 0.9500, 1.0500,
										1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 2.1000, 2.2000,
										2.5500, 2.6500, 2.7500, 2.8500, 2.9500, 3.0500, 3.1500, 3.2500, 3.3500};
	cout<<"testing add_m_v --- ";
	Mat *a = getTestMatrix_3();
	vector3f *b = getTestVector3f();
	Mat *res = NULL;
	safeGetPt(res, add(a, b));
	Mat *expect = new Mat(a -> rows, a -> cols, a -> channels);
	memcpy(expect -> hostData, array_add_m_v, expect -> getLength() * sizeof(float));
	expect -> hostToDevice();
	bool result = areApproximatelyIdentical(res, expect);
	a -> release();
	res -> release();
	expect -> release();
	cout<<(result ? "success" : "failed")<<endl;
	return result;
}



















