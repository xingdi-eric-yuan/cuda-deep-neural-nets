#include "matrix_maths.h"

void safeGetPt(Mat* &dst, Mat* src){
	if(dst){
		dst -> release();
	}
	dst = src;
}

void safeGetPt(cpuMat* &dst, cpuMat* src){
	if(dst){
		dst -> release();
	}
	dst = src;
}

///
vector3f* add(const vector3f* src, float a){
	if(NULL == src){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, src -> get(i) + a);
	}
	return res;
}

vector3f* add(const vector3f* src1, const vector3f* src2){
	if(NULL == src1 || NULL == src2){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, src1 -> get(i) + src2 -> get(i));
	}
	return res;
}

Mat* add(const Mat* src, float a){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(src -> devData, tmp -> devData, a, len);
	tmp -> deviceToHost();
	return tmp;
}

Mat* add(const Mat* src, const vector3f *val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		cu_plus<<<num_blocks, block_size>>>(src -> devData + len * ch, tmp -> devData + len * ch, val -> get(ch), len);
	}
	tmp -> deviceToHost();
	return tmp;
}

Mat* add(const Mat* a, const Mat* b){
	if(NULL == a -> hostData || NULL == a -> devData ||
	   NULL == b -> hostData || NULL == b -> devData ||
	   a -> getLength() != b -> getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(a -> rows, a -> cols, a -> channels);
	int len = a -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(a -> devData, b -> devData, tmp -> devData, len);
	tmp -> deviceToHost();
	return tmp;
}

vector3f* subtract(const vector3f* src, float a){
	if(NULL == src){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, src -> get(i) - a);
	}
	return res;
}

vector3f* subtract(const vector3f* src1, const vector3f* src2){
	if(NULL == src1 || NULL == src2){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, src1 -> get(i) - src2 -> get(i));
	}
	return res;
}

Mat* subtract(const Mat* src, float a){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(src -> devData, tmp -> devData, a, len);
	tmp -> deviceToHost();
	return tmp;
}

Mat* subtract(const Mat* src, const vector3f *val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		cu_minus<<<num_blocks, block_size>>>(src -> devData + len * ch, tmp -> devData + len * ch, val -> get(ch), len);
	}
	tmp -> deviceToHost();
	return tmp;
}

Mat* subtract(const Mat* a, const Mat* b){
	if(NULL == a -> hostData || NULL == a -> devData ||
	   NULL == b -> hostData || NULL == b -> devData ||
	   a -> getLength() != b -> getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(a -> rows, a -> cols, a -> channels);
	int len = a -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(a -> devData, b -> devData, tmp -> devData, len);
	tmp -> deviceToHost();
	return tmp;
}

vector3f* multiply_elem(const vector3f* src, float a){
	if(NULL == src){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, src -> get(i) * a);
	}
	return res;
}

vector3f* multiply_elem(const vector3f* src1, const vector3f* src2){
	if(NULL == src1 || NULL == src2){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, src1 -> get(i) * src2 -> get(i));
	}
	return res;
}

Mat* multiply_elem(const Mat* src, float a){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	float val = a;
	Mat* tmp = new Mat();
	src -> copyTo(*tmp);
	int len = src -> getLength();
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSscal(handle, len, &val, tmp -> devData, 1);
	tmp -> deviceToHost();
    cublasDestroy(handle);
	return tmp;
}

Mat* multiply_elem(const Mat* src, const vector3f *a){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	src -> copyTo(*tmp);
	int len = src -> rows * src -> cols;
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int ch = 0; ch < src -> channels; ++ch){
		float val = a -> get(ch);
		cublasSscal(handle, len, &val, tmp -> devData + ch * len, 1);
	}
	tmp -> deviceToHost();
    cublasDestroy(handle);
	return tmp;
}

Mat* multiply(const Mat* a, const Mat* b){

	if(NULL == a -> hostData || NULL == a -> devData ||
	   NULL == b -> hostData || NULL == b -> devData||
	   a -> cols != b -> rows || a -> channels != b -> channels){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	tmp -> setSize(a -> rows, b -> cols, a -> channels);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	float alpha = 1.0;
	float beta = 1.0;
	for(int i = 0; i < a -> channels; ++i){
		cublasSetMatrix (a -> rows, a -> cols, sizeof(float), a -> hostData + i * (a -> rows * a -> cols), a -> rows, a -> devData + i * (a -> rows * a -> cols), a -> rows); // cp x- >d_x
		cublasSetMatrix (b -> rows, b -> cols, sizeof(float), b -> hostData + i * (b -> rows * b -> cols), b -> rows, b -> devData + i * (b -> rows * b -> cols), b -> rows); // cp y- >d_y
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a -> rows, b -> cols, a -> cols, &alpha, a -> devData + i * (a -> rows * a -> cols), a -> rows, b -> devData + i * (b -> rows * b -> cols), a -> cols, &beta, tmp -> devData + i * (tmp -> rows * tmp -> cols), a -> rows);
		cublasGetMatrix (a -> rows, b -> cols, sizeof(float), tmp -> devData + i * (tmp -> rows * tmp -> cols), a -> rows, tmp -> hostData + i * (tmp -> rows * tmp -> cols), a -> rows);
	}
	cublasDestroy (handle); // destroy CUBLAS context
	tmp -> deviceToHost();
	return tmp;
}

Mat* multiply_elem(const Mat* a, const Mat* b){
	if(NULL == a -> hostData || NULL == a -> devData ||
	   NULL == b -> hostData || NULL == b -> devData||
	   a -> getLength() != b -> getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(a -> rows, a -> cols, a -> channels);
	int len = res -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(a -> devData, b -> devData, res -> devData, len);
	res -> deviceToHost();
	return res;
}



Mat* t(const Mat* a){
	if(NULL == a -> hostData || NULL == a -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat();
	a -> copyTo(*tmp);
	//tmpmat.zeros();
    float const alpha(1.0);
    float const beta(0.0);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < a -> channels; ++i){
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, a -> cols, a -> rows, &alpha, a -> devData + i * (a -> rows * a -> cols), a -> rows, &beta, a -> devData + i * (a -> rows * a -> cols), a -> cols, tmp -> devData + i * (a -> rows * a -> cols), a -> cols);
	}
	int _swap = tmp -> rows;
	tmp -> rows = tmp -> cols;
	tmp -> cols = _swap;
	cublasDestroy(handle);
	tmp -> deviceToHost();
	return tmp;
}

vector3f* div_rem(vector3f* src, int a){
	if(NULL == src){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, (float)((int)(src -> get(i)) % a));
	}
	return res;
}

vector3f* div_no_rem(vector3f* src, int a){
	if(NULL == src){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	for(int i = 0; i < 3; ++i){
		res -> set(i, (float)((int)(src -> get(i)) / a));
	}
	return res;
}

Mat* exp(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_exp<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

Mat* log(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_log<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

Mat* pow(const Mat* src, float power){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	if(0.0 == power){
		dst -> ones();
		return dst;
	}
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_pow<<<num_blocks, block_size>>>(src -> devData, dst -> devData, power, len);
	dst -> deviceToHost();
	return dst;
}

Mat* square(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_square<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

Mat* sqrt(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_sqrt<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

vector3f* square(const vector3f* src){
	vector3f* dst = new vector3f();
	for(int i = 0; i < 3; ++i){
		dst -> set(i, src -> get(i) * src -> get(i));
	}
	return dst;
}

vector3f* sqrt(const vector3f* src){
	vector3f* dst = new vector3f();
	for(int i = 0; i < 3; ++i){
		dst -> set(i, sqrt(src -> get(i)));
	}
	return dst;
}

Mat* divide(const Mat* numerator, float denominator){
	if(NULL == numerator -> hostData || NULL == numerator -> devData){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(numerator -> rows, numerator -> cols, numerator -> channels);
	if(0.0 == denominator){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	int len = numerator -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator -> devData, dst -> devData, denominator, len);
	dst -> deviceToHost();
	return dst;
}

Mat* divide(float numerator, const Mat* denominator){
	if(NULL == denominator -> hostData || NULL == denominator -> devData){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(denominator -> rows, denominator -> cols, denominator -> channels);
	if(0.0 == numerator){
		return dst;
	}
	int len = denominator -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator, denominator -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

vector3f* divide(const vector3f* numerator, float denominator){
	vector3f* dst = new vector3f();
	if(0.0 == denominator){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < 3; ++i){
		dst -> set(i, (numerator -> get(i) / denominator));
	}
	return dst;
}

vector3f* divide(float numerator, const vector3f* denominator){
	vector3f* dst = new vector3f();
	if(0.0 == numerator){
		return dst;
	}
	for(int i = 0; i < 3; ++i){
		if(denominator -> get(i) == 0.0){
			std::cout<<"invalid denominator..."<<std::endl;
			exit(0);
		}
		dst -> set(i, (numerator / denominator -> get(i)));
	}
	return dst;
}

Mat* divide(const Mat* numerator, const vector3f* denominator){
	if(NULL == numerator -> hostData || NULL == numerator -> devData){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(numerator -> rows, numerator -> cols, numerator -> channels);
	int len = numerator -> rows * numerator -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < numerator -> channels; ++i){
		cu_divide<<<num_blocks, block_size>>>(numerator -> devData + i * len, dst -> devData + i * len, denominator -> get(i), len);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* divide(const vector3f* numerator, const Mat* denominator){
	if(NULL == denominator -> hostData || NULL == denominator -> devData){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(denominator -> rows, denominator -> cols, denominator -> channels);
	int len = denominator -> rows * denominator -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < denominator -> channels; ++i){
		cu_divide<<<num_blocks, block_size>>>(numerator -> get(i), denominator -> devData + i * len, dst -> devData + i * len, len);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* divide(const Mat* numerator, const Mat* denominator){
	if(NULL == denominator -> hostData || NULL == denominator -> devData ||
	   NULL == numerator -> hostData || NULL == numerator -> devData || numerator -> getLength() != denominator -> getLength()){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(denominator -> rows, denominator -> cols, denominator -> channels);
	int len = numerator -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator -> devData, denominator -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

vector3f* divide(const vector3f* numerator, const vector3f* denominator){
	vector3f *dst = new vector3f();
	for(int i = 0; i < 3; ++i){
		if(denominator -> get(i) == 0.0){
			std::cout<<"invalid denominator..."<<std::endl;
			exit(0);
		}
		dst -> set(i, (numerator -> get(i) / denominator -> get(i)));
	}
	return dst;
}

cpuMat* divide(const cpuMat* numerator, const vector3f* denominator){
	if(NULL == numerator -> Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	cpuMat *dst = new cpuMat(numerator -> rows, numerator -> cols, numerator -> channels);
	int len = dst -> rows * dst -> cols;
	for(int ch = 0; ch < numerator -> channels; ++ch){
		for(int i = 0; i < len; ++i){
			if(denominator -> get(ch) == 0.0){
				std::cout<<"invalid denominator..."<<std::endl;
				exit(0);
			}
			dst -> Data[ch * len + i] = numerator -> Data[ch * len + i] / denominator -> get(ch);
		}
	}
	return dst;
}

cpuMat* divide(const cpuMat* numerator, float denominator){
	if(NULL == numerator -> Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	if(denominator == 0.0){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	cpuMat *dst = new cpuMat(numerator -> rows, numerator -> cols, numerator -> channels);
	int len = dst -> getLength();
	for(int i = 0; i < len; ++i){
		dst -> Data[i] = numerator -> Data[i] / denominator;
	}
	return dst;
}

cpuMat* subtract(const cpuMat* a, float b){
	if(NULL == a -> Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	cpuMat *dst = new cpuMat(a -> rows, a -> cols, a -> channels);
	int len = dst -> getLength();
	for(int i = 0; i < len; ++i){
		dst -> Data[i] = a -> Data[i] - b;
	}
	return dst;
}

cpuMat* subtract(const cpuMat* a, const vector3f* b){
	if(NULL == a -> Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	cpuMat *dst = new cpuMat(a -> rows, a -> cols, a -> channels);
	int len = dst -> rows * dst -> cols;
	for(int i = 0; i < len; ++i){
		for(int ch = 0; ch < a -> channels; ++ch){
			dst -> Data[ch * len + i] = a -> Data[ch * len + i] - b -> get(ch);
		}
	}
	return dst;
}

cpuMat* subtract(const cpuMat* a, const cpuMat* b){
	if(NULL == a -> Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	cpuMat *dst = new cpuMat(a -> rows, a -> cols, a -> channels);
	int len = dst -> getLength();
	for(int i = 0; i < len; ++i){
		dst -> Data[i] = a -> Data[i] - b -> Data[i];
	}
	return dst;
}

float sum(const vector3f* src){
	float res = 0.0;
	for(int i = 0; i < 3; ++i){
		res = (res + src -> get(i));
	}
	return res;
}

vector3f* sum(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *devRes = 0;
		cudaMalloc((void**)&devRes, sizeof(float));
		cu_sum<<<num_blocks, block_size, block_size * sizeof(float)>>>(src -> devData + i * len, devRes, len);
		float hostRes = 0;
		cudaMemcpy(&hostRes, devRes, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(devRes);
		res -> set(i, hostRes);
	}
	return res;
}

vector3f* average(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = NULL;
	safeGetPt(tmp, divide(src, src -> rows * src -> cols));
	vector3f *res = NULL;
	res = sum(tmp);
	tmp -> release();
	return res;
}

vector3f* average(const cpuMat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	int len = src -> rows * src ->  cols;
	for(int ch = 0; ch < src -> channels; ++ch){
		for(int i = 0; i < src -> rows * src -> cols; ++i){
			res -> set(ch, res -> get(ch) + src -> Data[ch * len + i] / len);
		}
	}
	return res;
}

vector3f* stddev(const cpuMat* src, const vector3f* avg){
	if(NULL == src -> Data ){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	cpuMat *tmpmat = new cpuMat(src -> rows, src -> cols, src -> channels);
	for(int ch = 0; ch < src -> channels; ++ch){
		for(int i = 0; i < src -> rows * src -> cols; ++i){
			float tmp = src -> Data[ch * src -> rows * src -> cols + i] - avg -> get(ch);
			tmp = tmp * tmp;
			tmpmat -> Data[ch * tmpmat -> rows * tmpmat -> cols + i] = tmp;
		}
	}
	vector3f *res = average(tmpmat);
	res = sqrt(res);
	tmpmat -> release();
	return res;
}

vector3f* stddev(const Mat* src, const vector3f* avg){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = new Mat();
	src -> copyTo(*tmp);
	safeGetPt(tmp, subtract(tmp, avg));
	safeGetPt(tmp, square(tmp));
	vector3f *res = new vector3f();
	res = average(tmp);
	tmp -> release();
	res = sqrt(res);
	return res;
}

float max(const vector3f* src){
	float res = src -> get(0);
	for(int i = 1; i < 3; ++i){
		if(src -> get(i) > res) res = src -> get(i);
	}
	return res;
}

vector3f* max(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src -> devData + i * len, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, len);
		float host_maxVal = 0;
		float host_minVal = 0;
		int host_maxLoc = 0;
		int host_minLoc = 0;
		cudaMemcpy(&host_maxVal, dev_maxVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minVal, dev_minVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_maxLoc, dev_maxLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minLoc, dev_minLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(dev_maxVal);
		cudaFree(dev_minVal);
		cudaFree(dev_maxLoc);
		cudaFree(dev_maxLoc);
		res -> set(i, host_maxVal);
	}
	return res;
}

void max(const Mat* src, vector3f* max_val, vector3f* max_loc){
	if(NULL == src -> hostData || NULL == src -> devData ||
	   NULL == max_val || NULL == max_loc){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src -> devData + i * len, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, len);
		float host_maxVal = 0;
		float host_minVal = 0;
		int host_maxLoc = 0;
		int host_minLoc = 0;
		cudaMemcpy(&host_maxVal, dev_maxVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minVal, dev_minVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_maxLoc, dev_maxLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minLoc, dev_minLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(dev_maxVal);
		cudaFree(dev_minVal);
		cudaFree(dev_maxLoc);
		cudaFree(dev_maxLoc);
		max_val -> set(i, host_maxVal);
		max_loc -> set(i, host_maxLoc);
	}
}

float min(const vector3f* src){
	float res = src -> get(0);
	for(int i = 1; i < 3; ++i){
		if(src -> get(i) < res) res = src -> get(i);
	}
	return res;
}

vector3f* min(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src -> devData + i * len, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, len);
		float host_maxVal = 0;
		float host_minVal = 0;
		int host_maxLoc = 0;
		int host_minLoc = 0;
		cudaMemcpy(&host_maxVal, dev_maxVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minVal, dev_minVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_maxLoc, dev_maxLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minLoc, dev_minLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(dev_maxVal);
		cudaFree(dev_minVal);
		cudaFree(dev_maxLoc);
		cudaFree(dev_maxLoc);
		res -> set(i, host_minVal);
	}
	return res;
}

void min(const Mat* src, vector3f* min_val, vector3f* min_loc){
	if(NULL == src -> hostData || NULL == src -> devData ||
	   NULL == min_val || NULL == min_loc){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src -> devData + i * len, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, len);
		float host_maxVal = 0;
		float host_minVal = 0;
		int host_maxLoc = 0;
		int host_minLoc = 0;
		cudaMemcpy(&host_maxVal, dev_maxVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minVal, dev_minVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_maxLoc, dev_maxLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minLoc, dev_minLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(dev_maxVal);
		cudaFree(dev_minVal);
		cudaFree(dev_maxLoc);
		cudaFree(dev_maxLoc);
		min_val -> set(i, host_minVal);
		min_loc -> set(i, host_minLoc);
	}
}

void minMaxLoc(const Mat* src, vector3f* max_val, vector3f* max_loc, vector3f* min_val, vector3f* min_loc){
	if(NULL == src -> hostData || NULL == src -> devData ||
	   NULL == min_val || NULL == min_loc ||
	   NULL == max_val || NULL == max_loc){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src -> devData + i * len, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, len);
		float host_maxVal = 0;
		float host_minVal = 0;
		int host_maxLoc = 0;
		int host_minLoc = 0;
		cudaMemcpy(&host_maxVal, dev_maxVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minVal, dev_minVal, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_maxLoc, dev_maxLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_minLoc, dev_minLoc, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(dev_maxVal);
		cudaFree(dev_minVal);
		cudaFree(dev_maxLoc);
		cudaFree(dev_maxLoc);
		max_val -> set(i, host_maxVal);
		max_loc -> set(i, host_maxLoc);
		min_val -> set(i, host_minVal);
		min_loc -> set(i, host_minLoc);
	}
}

Mat* greaterThan(const Mat *src, float val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_greaterThan<<<num_blocks, block_size>>>(src -> devData, dst -> devData, val, len);
	dst -> deviceToHost();
	return dst;
}

Mat* lessThan(const Mat *src, float val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_lessThan<<<num_blocks, block_size>>>(src -> devData, dst -> devData, val, len);
	dst -> deviceToHost();
	return dst;
}

// convert from vector of img to matrix
// vec.size() == nsamples
void convert(std::vector<std::vector<Mat*> >& vec, Mat *M){
    int subFeatures = vec[0][0] -> rows * vec[0][0] -> cols;
    Mat *res = new Mat(3 * vec[0].size() * subFeatures, vec.size(), 1);
    for(int i = 0; i < vec.size(); i++){
        for(int m = 0; m < vec[i].size(); m++){
			memcpy(res -> hostData + vec[i][m] -> getLength() * (m + i * vec[i].size()), vec[i][m] -> hostData, vec[i][m] -> getLength() * sizeof(float));
        }
    }
    res -> hostToDevice();
    res -> moveTo(*M);
}

// convert from matrix to vector of img
// vec.size() == nsamples
void convert(Mat *M, std::vector<std::vector<Mat*> >& vec, int nsamples, int imagesize){
    std::vector<Mat*> tmpvec;
    for(int i = 0; i < nsamples; i++){
        tmpvec.clear();
        int dim = imagesize * imagesize;
        for(int j = 0; j < M -> rows; j += dim * 3){
        	Mat *tmp =  new Mat(imagesize, imagesize, 3);
        	memcpy(tmp -> hostData, M -> hostData + i * M -> rows + j, dim * 3 * sizeof(float));
        	tmp -> hostToDevice();
        	tmpvec.push_back(tmp);
        }
        vec.push_back(tmpvec);
    }
    tmpvec.clear();
    std::vector<Mat*>().swap(tmpvec);
}

// non-linearity

Mat* sigmoid(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = new Mat();
	safeGetPt(tmp, multiply_elem(src, -1.0));
	safeGetPt(tmp, exp(tmp));
	safeGetPt(tmp, add(tmp, 1.0));
	safeGetPt(tmp, divide(1.0, tmp));
	return tmp;
}

Mat* dsigmoid(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = new Mat();
	Mat *tmp2 = new Mat();
	safeGetPt(tmp, exp(src));
	safeGetPt(tmp2, add(tmp, 1.0));
	safeGetPt(tmp2, square(tmp2));
	safeGetPt(tmp, divide(tmp, tmp2));
	tmp2 -> release();
    return tmp;
}

Mat* dsigmoid_a(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = new Mat(src -> rows, src -> cols, src -> channels);
	res -> ones();
	safeGetPt(res, subtract(res, src));
	safeGetPt(res, multiply_elem(res, src));
	return res;
}

Mat* ReLU(const Mat *M){
	if(NULL == M -> hostData || NULL == M -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat* res = NULL;
	res = greaterThan(M, 0.0);
	safeGetPt(res, multiply_elem(res, M));
    return res;
}

Mat* dReLU(const Mat* M){
	if(NULL == M -> hostData || NULL == M -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = NULL;
	res = greaterThan(M, 0.0);
    return res;
}

Mat* LeakyReLU(const Mat* M){
	if(NULL == M -> hostData || NULL == M -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *p = NULL;
	Mat *n = NULL;

	p = greaterThan(M, 0.0);
	n = lessThan(M, 0.0);
	safeGetPt(p, multiply_elem(p, M));
	safeGetPt(n, multiply_elem(n, M));
	safeGetPt(n, divide(n, leaky_relu_alpha));
	safeGetPt(n, add(n, p));
	p -> release();
	return n;
}

Mat* dLeakyReLU(const Mat* M){
	if(NULL == M -> hostData || NULL == M -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *p = NULL;
	Mat *n = NULL;

	p = greaterThan(M, 0.0);
	n = lessThan(M, 0.0);
	safeGetPt(n, divide(n, leaky_relu_alpha));
	safeGetPt(n, add(n, p));
	p -> release();
	return n;
}

Mat* Tanh(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_tanh<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
	dst -> deviceToHost();
	return dst;
}

Mat* dTanh(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = new Mat(src -> rows, src -> cols, src -> channels);
	res -> ones();
	Mat *tmp = NULL;
	safeGetPt(tmp, square(src));
	safeGetPt(res, subtract(res, tmp));
    tmp -> release();
    return res;
}

Mat* nonLinearity(const Mat *M, int method){
	if(NULL == M -> hostData || NULL == M -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(method == NL_RELU){
        return ReLU(M);
    }elif(method == NL_TANH){
        return Tanh(M);
    }elif(method == NL_LEAKY_RELU){
        return LeakyReLU(M);
    }else{
        return sigmoid(M);
    }
}

Mat* dnonLinearity(const Mat *M, int method){
	if(NULL == M -> hostData || NULL == M -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(method == NL_RELU){
        return dReLU(M);
    }elif(method == NL_TANH){
        return dTanh(M);
    }elif(method == NL_LEAKY_RELU){
        return dLeakyReLU(M);
    }else{
        return dsigmoid(M);
    }
}

// convolution and pooling
Mat* fliplr(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_fliplr<<<num_blocks, block_size>>>(src -> devData + i * len, dst -> devData + i * len, src -> rows, src -> cols, len);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* rot90(const Mat *src, int k){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = new Mat();
	src -> copyTo(*res);
    if(0 == k) return res;
    else{
    	if(k > 1) safeGetPt(res, rot90(res, k - 1));
    	safeGetPt(res, t(res));
    	safeGetPt(res, fliplr(res));
    }
    return res;
}

Mat* dopadding(const Mat *src, int pad){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(pad <= 0){
		Mat *dst = new Mat();
		src -> copyTo(*dst);
		return dst;
	}
	Mat *dst = new Mat(src -> rows + pad * 2, src -> cols + pad * 2, src -> channels);
	int lensrc = src -> rows * src -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lensrc / block_size) + ((lensrc % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_padding<<<num_blocks, block_size>>>(src -> devData + i * lensrc, dst -> devData + i * lendst, src -> rows, src -> cols, dst -> cols, lensrc);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* depadding(const Mat *src, int pad){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(pad <= 0){
		Mat *dst = new Mat();
		src -> copyTo(*dst);
		return dst;
	}
	Mat *dst = new Mat(src -> rows - pad * 2, src -> cols - pad * 2, src -> channels);
	int lensrc = src -> rows * src -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lendst / block_size) + ((lendst % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_depadding<<<num_blocks, block_size>>>(src -> devData + i * lensrc, dst -> devData + i * lendst, src -> rows, src -> cols, dst -> cols, lendst);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* reduce(const Mat* src, int direction, int mode){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = new Mat();
	Mat *dst = new Mat();
	if(REDUCE_TO_SINGLE_ROW == direction){
		dst -> setSize(1, src -> cols, src -> channels);
		for(int i = 0; i < src -> cols; ++i){
			safeGetPt(tmp, getRange(src, i, i, 0, src -> rows - 1));
			if(REDUCE_SUM == mode){
				dst -> set(i, *(sum(tmp)));
			}elif(REDUCE_MAX == mode){
				dst -> set(i, *(max(tmp)));
			}
		}
	}else{ // REDUCE_TO_SINGLE_COL == direction
		dst -> setSize(src -> rows, 1, src -> channels);
		for(int i = 0; i < src -> rows; ++i){
			safeGetPt(tmp, getRange(src, 0, src -> cols - 1, i, i));
			if(REDUCE_SUM == mode){
				dst -> set(i, *(sum(tmp)));
			}elif(REDUCE_MAX == mode){
				dst -> set(i, *(max(tmp)));
			}
		}
	}
	tmp -> release();
	return dst;
}

Mat* interpolation(const Mat* src, int _size){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    int stride = _size / src -> cols;
    if(_size % src -> cols > 0) ++ stride;
    if(stride == 0 || stride == 1) {
    	Mat *dst = new Mat();
    	src -> copyTo(*dst);
    	return dst;
    }
    Mat *dst = new Mat(_size, _size, src -> channels);
	int lensrc = src -> rows * src -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lensrc / block_size) + ((lensrc % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_interpolation<<<num_blocks, block_size>>>(src -> devData + i * lensrc, dst -> devData + i * lendst, src -> cols, dst -> cols, stride, lensrc);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* repmat(const Mat *src, int vert, int hori){
	if(NULL == src -> hostData || NULL == src -> devData || 0 == vert || 0 == hori){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(1 == vert && 1 == hori) {
    	Mat *dst = new Mat();
    	src -> copyTo(*dst);
    	return dst;
    }
	Mat *dst = new Mat(src -> rows * vert, src -> cols * hori, src -> channels);
	int lensrc = src -> rows * src -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lendst / block_size) + ((lendst % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_repmat<<<num_blocks, block_size>>>(src -> devData + i * lensrc, dst -> devData + i * lendst, src -> rows, src -> cols, dst -> rows, dst -> cols, lendst);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* kron(const Mat *a, const Mat *b){
	if(NULL == a -> hostData || NULL == a -> devData || NULL == b -> hostData || NULL == b -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(a -> rows * b -> rows, a -> cols * b -> cols, a -> channels);
	int lensrc = a -> rows * a -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lendst / block_size) + ((lendst % block_size) ? 1 : 0);
	for(int i = 0; i < a -> channels; ++i){
		cu_kron<<<num_blocks, block_size>>>(a -> devData + i * lensrc, b -> devData, dst -> devData + i * lendst, a -> rows, a -> cols, dst -> rows, dst -> cols, lendst);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* conv2(const Mat *m, const Mat *kernel){
	if(NULL == m -> hostData || NULL == m -> devData || NULL == kernel -> hostData || NULL == kernel -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = new Mat(m -> rows, m -> cols, m -> channels);
	float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;
    fComplex *d_DataSpectrum0, *d_KernelSpectrum0;
    cufftHandle fftPlan;
    float *host_result_tmp, *host_result;
    const int kernelH = kernel -> rows;
    const int kernelW = kernel -> cols;
    const int kernelY = kernel -> rows / 2;
    const int kernelX = kernel -> cols / 2;
    const int dataH = m -> rows;
    const int dataW = m -> cols;
    const int fftH = snapTransformSize(dataH + kernelH - 1);
    const int fftW = snapTransformSize(dataW + kernelW - 1);
    host_result = (float *)malloc(dataH * dataW * sizeof(float));
    host_result_tmp = (float *)malloc(fftH * fftW * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));
    // std::cout<<"...creating C2C FFT plan for "<<fftH<<" x "<<fftW/2<<std::endl;
    checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));
    for(int i = 0; i < m -> channels; ++i){
    	checkCudaErrors(cudaMemcpy(d_Data, m -> devData + dataH * dataW * i, dataH * dataW * sizeof(float), cudaMemcpyDeviceToDevice));
    	checkCudaErrors(cudaMemcpy(d_Kernel, kernel -> devData + kernelH * kernelW * i, kernelH * kernelW * sizeof(float), cudaMemcpyDeviceToDevice));
    	checkCudaErrors(cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float)));
    	checkCudaErrors(cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float)));
        padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW,
        					dataH, dataW, kernelH, kernelW, kernelY, kernelX);
        padKernel(d_PaddedKernel, d_Kernel, fftH, fftW, kernelH, kernelW, kernelY, kernelX);
        //CUFFT_INVERSE works just as well...
        const int FFT_DIR = CUFFT_FORWARD;
        //Not including kernel transformation into time measurement,
        //since convolution kernel is not changed very frequently
        // std::cout<<"...transforming convolution kernel"<<std::endl;
        checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum0, FFT_DIR));
        // std::cout<<"...running GPU FFT convolution: "<<std::endl;
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData, (cufftComplex *)d_DataSpectrum0, FFT_DIR));
        spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH, fftW / 2, FFT_DIR);
        checkCudaErrors(cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0, (cufftComplex *)d_PaddedData, -FFT_DIR));
        checkCudaErrors(cudaDeviceSynchronize());
        // std::cout<<"...reading back GPU FFT results"<<std::endl;
        checkCudaErrors(cudaMemcpy(host_result_tmp, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost));
        for(int y = 0; y < dataH; y++){
            for(int x = 0; x < dataW; x++){
                host_result[y * dataW + x] = host_result_tmp[y * fftW  + x];
            }
        }
        memcpy(res -> hostData + i * res -> rows * res -> cols, host_result, res -> rows * res -> cols * sizeof(float));
    }
    res -> hostToDevice();
    checkCudaErrors(cufftDestroy(fftPlan));
    checkCudaErrors(cudaFree(d_KernelSpectrum0));
    checkCudaErrors(cudaFree(d_DataSpectrum0));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Data));
    free(host_result);
    free(host_result_tmp);
    return res;
}

Mat* conv2(const Mat *m, const Mat *kernel, int convtype, int pad, int stride){
	if(NULL == m -> hostData || NULL == m -> devData || NULL == kernel -> hostData || NULL == kernel -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *src = NULL;
	Mat *res = NULL;
	safeGetPt(src, dopadding(m, kernel -> cols / 2 + 1));
	safeGetPt(src, dopadding(src, pad));
	safeGetPt(res, conv2(src, kernel));
	safeGetPt(res, getRange(res, (res -> cols - (m -> cols + kernel -> cols - 1 + pad * 2)) / 2,
								 (res -> cols - (m -> cols + kernel -> cols - 1 + pad * 2)) / 2 + m -> cols + kernel -> cols - 1 + pad * 2 - 1,
								 (res -> rows - (m -> rows + kernel -> rows - 1 + pad * 2)) / 2,
								 (res -> rows - (m -> rows + kernel -> rows - 1 + pad * 2)) / 2 + m -> rows + kernel -> rows - 1 + pad * 2 - 1));
	if(CONV_SAME == convtype){
		safeGetPt(res, getRange(res, kernel -> cols / 2, 
									 res -> cols - 1 - kernel -> cols / 2, 
									 kernel -> rows / 2, 
									 res -> rows - 1 - kernel -> rows / 2));
	}
	if(CONV_VALID == convtype){
        int tmpx = m -> cols + pad * 2 - kernel -> cols + 1;
        int tmpy = m -> rows + pad * 2 - kernel -> rows + 1;
        safeGetPt(res, getRange(res, (res -> cols - tmpx) / 2, 
        							 res -> cols - 1 - (res -> cols - tmpx) / 2,
                		 	 	     (res -> rows - tmpy) / 2, 
                		 	 	     res -> rows - 1 - (res -> rows - tmpy) / 2));
	}
	safeGetPt(res, downSample(res, stride, stride));
	src -> release();
	return res;
}

Mat* getRange(const Mat* src, int xstart, int xend, int ystart, int yend){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(xstart < 0 || xstart > xend || xend >= src -> cols ||
	   ystart < 0 || ystart > yend || yend >= src -> rows){
		std::cout<<"invalid range..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(yend - ystart + 1, xend - xstart + 1, src -> channels);
	int len = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_getRange<<<num_blocks, block_size>>>(src -> devData + i * src -> rows * src -> cols, dst -> devData + i * len, xstart, xend, ystart, yend, src -> cols, len);
	}
	dst -> deviceToHost();
	return dst;
}

Mat* downSample(const Mat* src, int y_stride, int x_stride){
	if(NULL == src -> hostData || NULL == src -> devData || y_stride < 1 || x_stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(1 == y_stride && 1 == x_stride){
		Mat *res = new Mat();
		src ->copyTo(*res);
		return res;
	}
	int dst_rows = src -> rows / y_stride;
	if(src -> rows % y_stride > 0) ++dst_rows;
	int dst_cols = src -> cols / x_stride;
	if(src -> cols % x_stride > 0) ++dst_cols;
	Mat *res = new Mat(dst_rows, dst_cols, src -> channels);
	int len = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_downSample<<<num_blocks, block_size>>>(src -> devData + i * src -> rows * src -> cols, res -> devData + i * len, y_stride, x_stride, src -> cols, len);
	}
	res -> deviceToHost();
	return res;
}

Mat* copyMakeBorder(const Mat* src, int up, int down, int left, int right, const vector3f* val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(0 == up && 0 == down && 0 == left && 0 == right){
		Mat *dst = new Mat();
		src -> copyTo(*dst);
		return dst;
	}
	Mat *dst = new Mat(src -> rows + up + down, src -> cols + left + right, src -> channels);
	dst -> setAll(*val);
	int lensrc = src -> rows * src -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lensrc / block_size) + ((lensrc % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_copyMakeBorder<<<num_blocks, block_size>>>(src -> devData + i * lensrc, dst -> devData + i * lendst, src -> rows, src -> cols, up, down, left, right, lensrc);
	}
	dst -> deviceToHost();
	return dst;
}


// THE FOLLOWING POOLING METHOD SUCKS, NEED TO SPEED UP!!!!

// Pooling with overlap
// Max pooling and stochastic pooling supported
// output size = (input size - window size) / stride + 1
Mat* pooling_with_overlap(const Mat *src, vector2i *window_size, int stride, int poolingMethod, std::vector<vector3f*> &locat){
	if(NULL == src -> hostData || NULL == src -> devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmpres = new Mat(src -> rows - window_size -> get(1) + 1, src -> cols - window_size -> get(0) + 1, src -> channels);
	std::vector<vector3f*> tmplocat;
	Mat *tmp = NULL;
    for(int i = 0; i < tmpres -> rows; ++i){
        for(int j = 0; j < tmpres -> cols; ++j){
        	int xstart = j;
        	int xend = j + window_size -> get(0) - 1 < src -> cols - 1 ? j + window_size -> get(0) - 1 : src -> cols - 1;
        	int ystart = i;
        	int yend = i + window_size -> get(1) - 1 < src -> rows - 1 ? i + window_size -> get(1) - 1 : src -> rows - 1;
        	safeGetPt(tmp, getRange(src, xstart, xend, ystart, yend));
        	vector3f *val = new vector3f();
        	vector3f *loc = new vector3f();
        	if(POOL_MAX == poolingMethod){
        		max(tmp, val, loc);
        	}
        	vector3f *tmpr = new vector3f();
        	vector3f *tmpc = new vector3f();
        	tmpc = div_rem(loc, window_size -> get(0));
        	tmpr = div_no_rem(loc, window_size -> get(0));
        	tmpr = add(tmpr, i);
        	tmpc = add(tmpc, j);
        	loc = multiply_elem(tmpr, src -> cols);
        	loc = add(loc, tmpc);
            tmplocat.push_back(loc);
            tmpres -> set(i, j, *val);
        }
    }
    Mat *dst = new Mat();
    safeGetPt(dst, downSample(tmpres, stride, stride));
    for(int i = 0; i < dst -> cols; ++i){
    	for(int j = 0; j < dst -> rows; ++j){
    		locat.push_back(tmplocat[i * stride * tmpres -> rows + j * stride]);
    	}
    }
    tmpres -> release();
    tmp -> release();
    tmplocat.clear();
    std::vector<vector3f*>().swap(tmplocat);
    return dst;
}

// Max pooling and stochastic pooling supported
Mat* unpooling_with_overlap(const Mat* src, vector2i* window_size, int stride, int poolingMethod, std::vector<vector3f*> &locat, vector2i* up_size){
	if(NULL == src -> hostData || NULL == src -> devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(1 == window_size -> get(0) && 1 == window_size -> get(1) && 1 == stride){
    	Mat *res = new Mat();
    	src -> copyTo(*res);
        return res;
    }
    Mat *res = new Mat(up_size -> get(1), up_size -> get(0), src -> channels);
    for(int i = 0; i < src -> rows; i++){
        for(int j = 0; j < src -> cols; j++){
        	for(int ch = 0; ch < src -> channels; ++ch){
            	res -> set(locat[i * src -> cols + j] -> get(ch), ch, src -> get(i, j, ch));
        	}
        }
    }
    return res;
}

Mat* pooling(const Mat* src, int stride, int poolingMethod, std::vector<vector3f*> &locat){
	if(NULL == src -> hostData || NULL == src -> devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(stride == 1){
    	for(int i = 0; i < src -> rows * src -> cols; ++i){
    		vector3f* tmp = new vector3f(i, i, i);
    		locat.push_back(tmp);
    	}
    	Mat *res = new Mat();
    	src -> copyTo(*res);
        return res;
    }
	int dst_rows = src -> rows / stride;
	if(src -> rows % stride > 0) ++dst_rows;
	int dst_cols = src -> cols / stride;
	if(src -> cols % stride > 0) ++dst_cols;
	Mat *res = new Mat(dst_rows, dst_cols, src -> channels);
	Mat *tmp = NULL;

    for(int i = 0; i < res -> rows; ++i){
        for(int j = 0; j < res -> cols; ++j){
        	int xstart = j * stride;
        	int xend = (j * stride + stride  - 1) < (src -> cols - 1) ? (j * stride + stride  - 1) : (src -> cols - 1);
        	int ystart = i * stride;
        	int yend = (i * stride + stride  - 1) < (src -> rows - 1) ? (i * stride + stride  - 1) : (src -> rows - 1);
        	safeGetPt(tmp, getRange(src, xstart, xend, ystart, yend));
        	vector3f *val = new vector3f();
        	vector3f *loc = new vector3f();
        	if(POOL_MAX == poolingMethod){
        		// max poling
        		max(tmp, val, loc);
        	}elif(POOL_MEAN == poolingMethod){
        		// Mean Pooling
        		val = average(tmp);
        		loc -> setAll(0.0);
            }
        	vector3f *tmpr = new vector3f();
        	vector3f *tmpc = new vector3f();
        	tmpc = div_rem(loc, tmp -> cols);
        	tmpr = div_no_rem(loc, tmp -> cols);
        	tmpr = add(tmpr, i * stride);
        	tmpc = add(tmpc, j * stride);
        	loc = multiply_elem(tmpr, src -> cols);
        	loc = add(loc, tmpc);
            locat.push_back(loc);
            res -> set(i, j, *val);
        }
    }
    tmp -> release();
    return res;
}

Mat* unpooling(const Mat* src, int stride, int poolingMethod, std::vector<vector3f*>& locat, vector2i* up_size){
	if(NULL == src -> hostData || NULL == src -> devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(stride == 1){
    	Mat *res = new Mat();
    	src -> copyTo(*res);
        return res;
    }
    if(POOL_MEAN == poolingMethod){
    	Mat *one = new Mat(stride, stride, src -> channels);
    	one -> ones();
    	Mat *res = NULL;
    	safeGetPt(res, kron(src, one));
    	safeGetPt(res, divide(res, stride * stride));
    	vector3f *tmp = new vector3f();
    	safeGetPt(res, copyMakeBorder(res, 0, up_size -> get(1) - res -> rows, 0, up_size -> get(0) - res -> cols, tmp));
        one -> release();
        return res;
    }else{ //(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod)
        Mat *res = new Mat(up_size -> get(1), up_size -> get(0), src -> channels);
        for(int i = 0; i < src -> rows; i++){
            for(int j = 0; j < src -> cols; j++){
            	for(int ch = 0; ch < src -> channels; ++ch){
                	res -> set(locat[i * src -> cols + j] -> get(ch), ch, src -> get(i, j, ch));
            	}
            }
        }
        return res;
    }
}






