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

void safeGetPt(vector3f* &dst, vector3f* src){
	if(dst){
		dst -> release();
	}
	dst = src;
}

void safeGetPt(vector2i* &dst, vector2i* src){
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
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(src -> devData, tmp -> devData, a, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	tmp -> deviceToHost();
	return tmp;
}

Mat* add(const Mat* src, const vector3f *val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		cu_plus<<<num_blocks, block_size>>>(src -> devData + len * ch, tmp -> devData + len * ch, val -> get(ch), len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	Mat* tmp = new Mat(a -> rows, a -> cols, a -> channels);
	int len = a -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(a -> devData, b -> devData, tmp -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(src -> devData, tmp -> devData, a, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	tmp -> deviceToHost();
	return tmp;
}

Mat* subtract(const Mat* src, const vector3f *val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		cu_minus<<<num_blocks, block_size>>>(src -> devData + len * ch, tmp -> devData + len * ch, val -> get(ch), len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	Mat* tmp = new Mat(a -> rows, a -> cols, a -> channels);
	int len = a -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(a -> devData, b -> devData, tmp -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
	Mat* res = new Mat(src -> rows, src -> cols, src -> channels);
	int len = res -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(src -> devData, a, res -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	res -> deviceToHost();
	return res;
}

Mat* multiply_elem(const Mat* src, const vector3f *a){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(src -> rows, src -> cols, src -> channels);
	int len = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		float val = a -> get(ch);
		cu_elementWiseMultiply<<<num_blocks, block_size>>>(src -> devData + ch * len, val, res -> devData + ch * len, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	res -> deviceToHost();
	return res;
}

Mat* multiply(const Mat* a, const Mat* b){
	if(NULL == a -> hostData || NULL == a -> devData ||
	   NULL == b -> hostData || NULL == b -> devData||
	   a -> cols != b -> rows || a -> channels != b -> channels){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(a -> rows, b -> cols, a -> channels);
	int lena = a -> rows * a -> cols;
	int lenb = b -> rows * b -> cols;
	int lenres = res -> rows * res -> cols;
	int TILE_WIDTH = 32;
    dim3 dimGrid((res -> cols - 1) / TILE_WIDTH + 1, (res -> rows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	for(int ch = 0; ch < a -> channels; ++ch){
		cu_multiply<<<dimGrid, dimBlock>>>(a -> devData + ch * lena , b -> devData + ch * lenb, res -> devData + ch * lenres,
													a -> rows, a -> cols, b -> rows, b -> cols, res -> rows, res -> cols);
	    getLastCudaError("kernel execution failed\n");
	    checkCudaErrors(cudaDeviceSynchronize());
	}
	res -> deviceToHost();
	return res;
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	res -> deviceToHost();
	return res;
}

Mat* t(const Mat* a){
	if(NULL == a -> hostData || NULL == a -> devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(a -> cols, a -> rows, a -> channels);
	int len = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < a -> channels; ++ch){
		cu_transpose<<<num_blocks, block_size>>>(a -> devData + ch * len, res -> devData + ch * len, a -> cols, res -> cols, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	res -> deviceToHost();
	return res;
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		float *data = NULL;
		int data_len = len;
		checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * data_len));
		checkCudaErrors(cudaMemcpy(data, src -> devData + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		float *d_partial_sums = NULL;
		float *global_mem = NULL;
		while(true){
			checkCudaErrors(cudaMalloc((void**)&global_mem, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(global_mem, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMalloc((void**)&d_partial_sums, sizeof(float) * tmp_num_blocks));
			cu_sum<<<tmp_num_blocks, tmp_block_size>>>(data, d_partial_sums, global_mem, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaFree(global_mem));
			checkCudaErrors(cudaFree(data));
			data_len = tmp_num_blocks;
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res = 0;
				checkCudaErrors(cudaMemcpy(&host_res, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaFree(d_partial_sums));
				res -> set(ch, host_res);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
				checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * data_len));
				checkCudaErrors(cudaMemcpy(data, d_partial_sums, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaFree(d_partial_sums));
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
				checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * data_len));
				checkCudaErrors(cudaMemcpy(data, d_partial_sums, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaFree(d_partial_sums));
			}
		}
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
		float *dev_maxVal_partial = 0;
		float *dev_minVal_partial = 0;
		int *dev_maxLoc_partial = 0;
		int *dev_minLoc_partial = 0;
		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks)));

		float *glob_mem_maxVal = 0;
		float *glob_mem_minVal = 0;
		int *glob_mem_maxLoc = 0;
		int *glob_mem_minLoc = 0;
		checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * (block_size)));
		checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * (block_size)));
		checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * (block_size)));
		checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * (block_size)));
		checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * block_size));
		checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * block_size));
		checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * block_size));
		checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * block_size));


		// launch one kernel to compute, per-block, a partial max
		cu_minMaxLoc<<<num_blocks, block_size>>>(src -> devData + i * len,
															dev_minVal_partial, dev_maxVal_partial,
															dev_minLoc_partial, dev_maxLoc_partial,
															glob_mem_maxVal,
															glob_mem_minVal,
															glob_mem_maxLoc,
															glob_mem_minLoc, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());

		float *data_maxVal = 0;
		float *data_minVal = 0;
		int *data_maxLoc = 0;
		int *data_minLoc = 0;
		checkCudaErrors(cudaMalloc((void**)&data_maxVal, sizeof(float) * num_blocks));
		checkCudaErrors(cudaMalloc((void**)&data_minVal, sizeof(float) * num_blocks));
		checkCudaErrors(cudaMalloc((void**)&data_maxLoc, sizeof(int) * num_blocks));
		checkCudaErrors(cudaMalloc((void**)&data_minLoc, sizeof(int) * num_blocks));
		checkCudaErrors(cudaMemcpy(data_maxVal, dev_maxVal_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(data_minVal, dev_minVal_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(data_maxLoc, dev_maxLoc_partial, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(data_minLoc, dev_minLoc_partial, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));



		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
		checkCudaErrors(cudaFree(glob_mem_maxVal));
		checkCudaErrors(cudaFree(glob_mem_minVal));
		checkCudaErrors(cudaFree(glob_mem_maxLoc));
		checkCudaErrors(cudaFree(glob_mem_minLoc));

		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int)));

		checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * num_blocks));
		checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * num_blocks));
		checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * num_blocks));
		checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * num_blocks));
		checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * num_blocks));
		checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * num_blocks));
		checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * num_blocks));
		checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * num_blocks));




		// launch a single block to compute the max
		cu_minMaxLoc<<<1, num_blocks>>>(dev_maxVal_partial,
										data_maxVal, data_minVal, data_maxLoc, data_minLoc,
										glob_mem_maxVal,
										glob_mem_minVal,
										glob_mem_maxLoc,
										glob_mem_minLoc, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_maxVal = 0;
		checkCudaErrors(cudaMemcpy(&host_maxVal, dev_maxVal_partial, sizeof(float), cudaMemcpyDeviceToHost));
		res -> set(i, host_maxVal);
		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
		checkCudaErrors(cudaFree(glob_mem_maxVal));
		checkCudaErrors(cudaFree(glob_mem_minVal));
		checkCudaErrors(cudaFree(glob_mem_maxLoc));
		checkCudaErrors(cudaFree(glob_mem_minLoc));
		checkCudaErrors(cudaFree(data_maxVal));
		checkCudaErrors(cudaFree(data_minVal));
		checkCudaErrors(cudaFree(data_maxLoc));
		checkCudaErrors(cudaFree(data_minLoc));
	}
	return res;
}



vector3f* max1(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		float *dev_maxVal_partial = 0;
		float *dev_minVal_partial = 0;
		int *dev_maxLoc_partial = 0;
		int *dev_minLoc_partial = 0;
		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks + 1)));
		// launch one kernel to compute, per-block, a partial max
		cu_minMaxLoc1<<<num_blocks, block_size, block_size * sizeof(float)>>>(src -> devData + i * len,
															dev_minVal_partial, dev_maxVal_partial,
															dev_minLoc_partial, dev_maxLoc_partial, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		// launch a single block to compute the max
		cu_minMaxLoc1<<<1, num_blocks, num_blocks * sizeof(float)>>>(dev_maxVal_partial,
															dev_minVal_partial + num_blocks,
															dev_maxVal_partial + num_blocks,
															dev_minLoc_partial + num_blocks,
															dev_maxLoc_partial + num_blocks, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_maxVal = 0;
		checkCudaErrors(cudaMemcpy(&host_maxVal, dev_maxVal_partial + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));
		res -> set(i, host_maxVal);
		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
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
		float *dev_maxVal_partial = 0;
		float *dev_minVal_partial = 0;
		int *dev_maxLoc_partial = 0;
		int *dev_minLoc_partial = 0;
		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks + 1)));
		// launch one kernel to compute, per-block, a partial max
		cu_minMaxLoc1<<<num_blocks, block_size, block_size * sizeof(float)>>>(src -> devData + i * len,
															dev_minVal_partial, dev_maxVal_partial,
															dev_minLoc_partial, dev_maxLoc_partial, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		int* host_maxLoc_partial = (int *)malloc(sizeof(int) * (num_blocks + 1));
		checkCudaErrors(cudaMemcpy(host_maxLoc_partial, dev_maxLoc_partial, (num_blocks + 1) * sizeof(int), cudaMemcpyDeviceToHost));
		// launch a single block to compute the max
		cu_minMaxLoc1<<<1, num_blocks, num_blocks * sizeof(float)>>>(dev_maxVal_partial,
															dev_minVal_partial + num_blocks,
															dev_maxVal_partial + num_blocks,
															dev_minLoc_partial + num_blocks,
															dev_maxLoc_partial + num_blocks, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_maxVal = 0;
		int host_maxLoc = 0;
		checkCudaErrors(cudaMemcpy(&host_maxVal, dev_maxVal_partial + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&host_maxLoc, dev_maxLoc_partial + num_blocks, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
		max_val -> set(i, host_maxVal);
		max_loc -> set(i, host_maxLoc_partial[host_maxLoc]);
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
		float *dev_maxVal_partial = 0;
		float *dev_minVal_partial = 0;
		int *dev_maxLoc_partial = 0;
		int *dev_minLoc_partial = 0;
		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks + 1)));
		// launch one kernel to compute, per-block, a partial min
		cu_minMaxLoc1<<<num_blocks, block_size, block_size * sizeof(float)>>>(src -> devData + i * len,
															dev_minVal_partial, dev_maxVal_partial,
															dev_minLoc_partial, dev_maxLoc_partial, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		// launch a single block to compute the min
		cu_minMaxLoc1<<<1, num_blocks, num_blocks * sizeof(float)>>>(dev_minVal_partial,
															dev_minVal_partial + num_blocks,
															dev_maxVal_partial + num_blocks,
															dev_minLoc_partial + num_blocks,
															dev_maxLoc_partial + num_blocks, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_minVal = 0;
		checkCudaErrors(cudaMemcpy(&host_minVal, dev_minVal_partial + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));
		res -> set(i, host_minVal);
		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
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
		float *dev_maxVal_partial = 0;
		float *dev_minVal_partial = 0;
		int *dev_maxLoc_partial = 0;
		int *dev_minLoc_partial = 0;
		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks + 1)));
		// launch one kernel to compute, per-block, a partial min
		cu_minMaxLoc1<<<num_blocks, block_size, block_size * sizeof(float)>>>(src -> devData + i * len,
															dev_minVal_partial, dev_maxVal_partial,
															dev_minLoc_partial, dev_maxLoc_partial, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		int* host_minLoc_partial = (int *)malloc(sizeof(int) * (num_blocks + 1));
		checkCudaErrors(cudaMemcpy(host_minLoc_partial, dev_minLoc_partial, (num_blocks + 1) * sizeof(int), cudaMemcpyDeviceToHost));
		// launch a single block to compute the min
		cu_minMaxLoc1<<<1, num_blocks, num_blocks * sizeof(float)>>>(dev_minVal_partial,
															dev_minVal_partial + num_blocks,
															dev_maxVal_partial + num_blocks,
															dev_minLoc_partial + num_blocks,
															dev_maxLoc_partial + num_blocks, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_minVal = 0;
		int host_minLoc = 0;
		checkCudaErrors(cudaMemcpy(&host_minVal, dev_minVal_partial + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&host_minLoc, dev_minLoc_partial + num_blocks, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
		min_val -> set(i, host_minVal);
		min_loc -> set(i, host_minLoc_partial[host_minLoc]);
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
		float *dev_maxVal_partial = 0;
		float *dev_minVal_partial = 0;
		int *dev_maxLoc_partial = 0;
		int *dev_minLoc_partial = 0;
		checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks + 1)));
		checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks + 1)));
		// launch one kernel to compute, per-block, a partial result
		cu_minMaxLoc1<<<num_blocks, block_size, block_size * sizeof(float)>>>(src -> devData + i * len,
															dev_minVal_partial, dev_maxVal_partial,
															dev_minLoc_partial, dev_maxLoc_partial, len);
	    getLastCudaError("kernel execution failed\n");
		// store partial results
        checkCudaErrors(cudaDeviceSynchronize());
		int* host_minLoc_partial = (int *)malloc(sizeof(int) * (num_blocks + 1));
		checkCudaErrors(cudaMemcpy(host_minLoc_partial, dev_minLoc_partial, (num_blocks + 1) * sizeof(int), cudaMemcpyDeviceToHost));
		int* host_maxLoc_partial = (int *)malloc(sizeof(int) * (num_blocks + 1));
		checkCudaErrors(cudaMemcpy(host_maxLoc_partial, dev_maxLoc_partial, (num_blocks + 1) * sizeof(int), cudaMemcpyDeviceToHost));
		// launch a single block to compute min
		cu_minMaxLoc1<<<1, num_blocks, num_blocks * sizeof(float)>>>(dev_minVal_partial,
															dev_minVal_partial + num_blocks,
															dev_maxVal_partial + num_blocks,
															dev_minLoc_partial + num_blocks,
															dev_maxLoc_partial + num_blocks, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_minVal = 0;
		int host_minLoc = 0;
		checkCudaErrors(cudaMemcpy(&host_minVal, dev_minVal_partial + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&host_minLoc, dev_minLoc_partial + num_blocks, sizeof(int), cudaMemcpyDeviceToHost));
		// launch a single block to compute max
		cu_minMaxLoc1<<<1, num_blocks, num_blocks * sizeof(float)>>>(dev_maxVal_partial,
															dev_minVal_partial + num_blocks,
															dev_maxVal_partial + num_blocks,
															dev_minLoc_partial + num_blocks,
															dev_maxLoc_partial + num_blocks, num_blocks);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
		  // copy the result back to the host
		float host_maxVal = 0;
		int host_maxLoc = 0;
		checkCudaErrors(cudaMemcpy(&host_maxVal, dev_maxVal_partial + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&host_maxLoc, dev_maxLoc_partial + num_blocks, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(dev_maxVal_partial));
		checkCudaErrors(cudaFree(dev_minVal_partial));
		checkCudaErrors(cudaFree(dev_maxLoc_partial));
		checkCudaErrors(cudaFree(dev_minLoc_partial));
		min_val -> set(i, host_minVal);
		min_loc -> set(i, host_minLoc_partial[host_minLoc]);
		max_val -> set(i, host_maxVal);
		max_loc -> set(i, host_maxLoc_partial[host_maxLoc]);
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* equalTo(const Mat *src, float val){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_equalTo<<<num_blocks, block_size>>>(src -> devData, dst -> devData, val, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* getBernoulliMatrix(int height, int width, int nchannels, float prob){
	Mat *ran = new Mat(height, width, nchannels);
	ran -> randu();
	Mat *res = NULL;
	safeGetPt(res, greaterThan(ran, prob));
	ran -> release();
	return res;
}


// convert from vector of img to matrix
// vec.size() == nsamples
void convert(std::vector<std::vector<Mat*> >& vec, Mat *M){
    Mat *res = new Mat(vec.size(), vec[0].size() * vec[0][0] -> getLength(), 1);
    for(int i = 0; i < vec.size(); i++){
        for(int m = 0; m < vec[i].size(); m++){
			memcpy(res -> hostData + vec[i][m] -> getLength() * (m + i * vec[i].size()), vec[i][m] -> hostData, vec[i][m] -> getLength() * sizeof(float));
        }
    }
    res -> hostToDevice();
    Mat *tmp = NULL;
    safeGetPt(tmp, t(res));
    tmp -> moveTo(*M);
    res -> release();
}

// convert from matrix to vector of img
// vec.size() == nsamples
void convert(Mat *M, std::vector<std::vector<Mat*> >& vec, int nsamples, int imagesize){
    std::vector<Mat*> tmpvec;
    Mat *Mt = NULL;
    safeGetPt(Mt, t(M));

    int dim = imagesize * imagesize * 3;
    releaseVector(vec);
    vec.resize(nsamples);
    for(int i = 0; i < vec.size(); ++i){
    	vec[i].clear();
    	vec[i].resize(Mt -> cols / dim);
    	for(int j = 0; j < vec[i].size(); ++j){
    		vec[i][j] = new Mat(imagesize, imagesize, 3);
        	memcpy(vec[i][j] -> hostData, Mt -> hostData + i * Mt -> cols + j * dim, dim * sizeof(float));
        	vec[i][j] -> hostToDevice();
    	}
    }
    Mt -> release();
}

// non-linearity
Mat *sigmoid(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();

	dim3 ThreadsPerBlock(16, 16);  // 256 threads
	dim3 numBlocks(	iDivUp(src -> cols, ThreadsPerBlock.x),  // for instance 512/8 = 64
					iDivUp(src -> rows, ThreadsPerBlock.y));

	cu_sigmoid<<<numBlocks, ThreadsPerBlock>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* dsigmoid(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dsigmoid<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* dsigmoid_a(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dsigmoid_a<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* ReLU(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_relu<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* dReLU(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_drelu<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* LeakyReLU(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_leaky_relu<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* dLeakyReLU(const Mat* src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dleaky_relu<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
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
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
}

Mat* dTanh(const Mat *src){
	if(NULL == src -> hostData || NULL == src -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dtanh<<<num_blocks, block_size>>>(src -> devData, dst -> devData, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	dst -> deviceToHost();
	return dst;
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
	    getLastCudaError("kernel execution failed\n");
		checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
        checkCudaErrors(cudaDeviceSynchronize());
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
    checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
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
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	dst -> deviceToHost();
	return dst;
}

// THE FOLLOWING POOLING METHOD SUCKS, NEED TO SPEED UP!!!!

// Pooling with overlap
// Max pooling and stochastic pooling supported
// output size = (input size - window size) / stride + 1
Mat* pooling_with_overlap(const Mat *src, vector2i *window_size, int stride, int poolingMethod, Mat*& locat){
	if(NULL == src -> hostData || NULL == src -> devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int dst_rows = src -> rows - window_size -> get(1) + 1;
	int dst_cols = src -> cols - window_size -> get(0) + 1;
	Mat *res = new Mat(dst_rows, dst_cols, src -> channels);
	Mat *loc = new Mat(dst_rows, dst_cols, src -> channels);
	safeGetPt(res, getRange(src, 0, src -> cols - window_size -> get(0), 0, src -> rows - window_size -> get(1)));
	int lensrc = src -> rows * src -> cols;
	int lenres = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lenres / block_size) + ((lenres % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_pooling_overlap_max<<<num_blocks, block_size>>>(src -> devData + i * lensrc, res -> devData + i * lenres,  loc -> devData + i * lenres, src -> rows, src -> cols, res -> rows, res -> cols, window_size -> get(0), window_size -> get(1), lenres);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	res -> deviceToHost();
	loc -> deviceToHost();
    Mat *dst = new Mat();
    safeGetPt(dst, downSample(res, stride, stride));
    safeGetPt(locat, downSample(loc, stride, stride));
    res -> release();
	loc -> release();
    return dst;
}

// Max pooling and stochastic pooling supported
Mat* unpooling_with_overlap(const Mat* src, vector2i* window_size, int stride, int poolingMethod, const Mat* locat, vector2i* up_size){
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
	int lenres = res -> rows * res -> cols;
	int lensrc = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lensrc / block_size) + ((lensrc % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_unpooling<<<num_blocks, block_size>>>(src -> devData + i * lensrc, locat -> devData + i * lensrc, res -> devData + i * lenres, res -> cols, lensrc);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	res -> deviceToHost();
	return res;
}

Mat* pooling(const Mat* src, int stride, int poolingMethod, Mat*& locat){
	if(NULL == src -> hostData || NULL == src -> devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(stride == 1){
    	locat = new Mat(src -> rows, src -> cols, src -> channels);
    	for(int i = 0; i < src -> rows * src -> cols; ++i){
    		vector3f* tmp = new vector3f(i, i, i);
    		locat -> set(i, *tmp);
    		tmp -> release();
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
	Mat *loc = new Mat(dst_rows, dst_cols, src -> channels);
	if(POOL_MAX == poolingMethod) safeGetPt(res, downSample(src, stride, stride));
	int lensrc = src -> rows * src -> cols;
	int lenres = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lenres / block_size) + ((lenres % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		if(POOL_MAX == poolingMethod){
			cu_pooling_max<<<num_blocks, block_size>>>(src -> devData + i * lensrc, res -> devData + i * lenres,  loc -> devData + i * lenres, src -> rows, src -> cols, res -> rows, res -> cols, stride, stride, lenres);
		    getLastCudaError("kernel execution failed\n");
		}elif(POOL_MEAN == poolingMethod){
			cu_pooling_mean<<<num_blocks, block_size>>>(src -> devData + i * lensrc, res -> devData + i * lenres,  loc -> devData + i * lenres, src -> rows, src -> cols, res -> rows, res -> cols, stride, stride, lenres);
		    getLastCudaError("kernel execution failed\n");
		}
        checkCudaErrors(cudaDeviceSynchronize());
	}
	res -> deviceToHost();
	loc -> deviceToHost();
	loc -> copyTo(*locat);
	loc -> release();
    return res;
}

Mat* unpooling(const Mat* src, int stride, int poolingMethod, const Mat* locat, vector2i* up_size){
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
    	int lenres = res -> rows * res -> cols;
    	int lensrc = src -> rows * src -> cols;
    	const size_t block_size = threadsPerBlock;
    	const size_t num_blocks = (lensrc / block_size) + ((lensrc % block_size) ? 1 : 0);
    	for(int i = 0; i < src -> channels; ++i){
    		cu_unpooling<<<num_blocks, block_size>>>(src -> devData + i * lensrc, locat -> devData + i * lensrc, res -> devData + i * lenres, res -> cols, lensrc);
    	    getLastCudaError("kernel execution failed\n");
            checkCudaErrors(cudaDeviceSynchronize());
    	}
    	res -> deviceToHost();
    	return res;
    }
}

Mat* findMax(const Mat* m){
	if(NULL == m -> hostData || NULL == m -> devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = new Mat(1, m -> cols, 1);
	Mat *tmp = NULL;
	vector3f *val = new vector3f();
	vector3f *loc = new vector3f();
	for(int i = 0; i < m -> cols; ++i){
		safeGetPt(tmp, getRange(m, i, i, 0, m -> rows - 1));
		max(tmp, val, loc);
		res -> set(0, i, 0, loc -> get(0));
	}
	tmp -> release();
	val -> release();
	loc -> release();
	return res;
}

// only calculates first channel.
int sameValuesInMat(const Mat* a, const Mat* b){
	if(NULL == a -> hostData || NULL == a -> devData || NULL == b -> hostData || NULL == b -> devData ||
			a -> getLength() != b -> getLength()){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = NULL;
	safeGetPt(tmp, subtract(a, b));
	safeGetPt(tmp, equalTo(tmp, 0.0));
	int res = (int)(sum(tmp) -> get(0));
	tmp -> release();
	return res;
}

