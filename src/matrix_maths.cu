#include "matrix_maths.h"

// SAFE GET POINTER
// assign value to pointer after free that pointer
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

// ADD
// returns src + a
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

// ADD
// returns src1 + src2
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

// ADD
// returns src + a
Mat* add(const Mat* src, float a){
	if(NULL == src -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(src -> Data, tmp -> Data, a, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return tmp;
}

// ADD
// returns src + val
Mat* add(const Mat* src, const vector3f *val){
	if(NULL == src -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		cu_plus<<<num_blocks, block_size>>>(src -> Data + len * ch, tmp -> Data + len * ch, val -> get(ch), len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return tmp;
}

// ADD
// returns a + b
Mat* add(const Mat* a, const Mat* b){
	if(NULL == a -> Data ||
	   NULL == b -> Data ||
	   a -> getLength() != b -> getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(a -> rows, a -> cols, a -> channels);
	int len = a -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(a -> Data, b -> Data, tmp -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return tmp;
}

// SUBTRACT
// returns src - a
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

// SUBTRACT
// returns src1 - src2
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

// SUBTRACT
// returns src - a
Mat* subtract(const Mat* src, float a){
	if(NULL == src -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(src -> Data, tmp -> Data, a, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return tmp;
}

// SUBTRACT
// returns src - val
Mat* subtract(const Mat* src, const vector3f *val){
	if(NULL == src -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		cu_minus<<<num_blocks, block_size>>>(src -> Data + len * ch, tmp -> Data + len * ch, val -> get(ch), len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return tmp;
}

// SUBTRACT
// returns a - b
Mat* subtract(const Mat* a, const Mat* b){
	if(NULL == a -> Data ||
	   NULL == b -> Data ||
	   a -> getLength() != b -> getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* tmp = new Mat(a -> rows, a -> cols, a -> channels);
	int len = a -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(a -> Data, b -> Data, tmp -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return tmp;
}

// MULTIPLY ELEMENT WISE
// return src(i) * a
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

// MULTIPLY ELEMENT WISE
// return src1(i) * src2(i)
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

// MULTIPLY ELEMENT WISE
// return src(i) * a
Mat* multiply_elem(const Mat* src, float a){
	if(NULL == src -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(src -> rows, src -> cols, src -> channels);
	int len = res -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(src -> Data, a, res -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return res;
}

// MULTIPLY ELEMENT WISE
// return src(i) * a
Mat* multiply_elem(const Mat* src, const vector3f *a){
	if(NULL == src -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(src -> rows, src -> cols, src -> channels);
	int len = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < src -> channels; ++ch){
		float val = a -> get(ch);
		cu_elementWiseMultiply<<<num_blocks, block_size>>>(src -> Data + ch * len, val, res -> Data + ch * len, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return res;
}

// MULTIPLY
// matrix multiplication
// result size should be (rowsa, colsb)
Mat* multiply(const Mat* a, const Mat* b){
	if(NULL == a -> Data ||
	   NULL == b -> Data||
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
		cu_multiply<<<dimGrid, dimBlock>>>(a -> Data + ch * lena , b -> Data + ch * lenb, res -> Data + ch * lenres,
													a -> rows, a -> cols, b -> rows, b -> cols, res -> rows, res -> cols);
	    getLastCudaError("kernel execution failed\n");
	    checkCudaErrors(cudaDeviceSynchronize());
	}
	return res;
}

// MULTIPLY ELEMENT WISE
// return a(i) * b(i)
Mat* multiply_elem(const Mat* a, const Mat* b){
	if(NULL == a -> Data ||
	   NULL == b -> Data||
	   a -> getLength() != b -> getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(a -> rows, a -> cols, a -> channels);
	int len = res -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(a -> Data, b -> Data, res -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return res;
}

// TRANSPOSE
// return matrix transpose
Mat* t(const Mat* a){
	if(NULL == a -> Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat* res = new Mat(a -> cols, a -> rows, a -> channels);
	int len = res -> rows * res -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < a -> channels; ++ch){
		cu_transpose<<<num_blocks, block_size>>>(a -> Data + ch * len, res -> Data + ch * len, a -> cols, res -> cols, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return res;
}

// DIVIDE REMAINDER
// returns src % a
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

// DIVIDE NO REMAINDER
// returns src / a
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

// EXP
// returns exp(src)
Mat* exp(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_exp<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// LOG
// returns log(src)
Mat* log(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_log<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// POW
// returns pow(src, power)
Mat* pow(const Mat* src, float power){
	if(NULL == src -> Data){
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
	cu_pow<<<num_blocks, block_size>>>(src -> Data, dst -> Data, power, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// SQUARE
// returns square
Mat* square(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_square<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// SQRT
// returns square root
Mat* sqrt(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_sqrt<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// SQUARE
// returns square
vector3f* square(const vector3f* src){
	vector3f* dst = new vector3f();
	for(int i = 0; i < 3; ++i){
		dst -> set(i, src -> get(i) * src -> get(i));
	}
	return dst;
}

// SQRT
// returns square root
vector3f* sqrt(const vector3f* src){
	vector3f* dst = new vector3f();
	for(int i = 0; i < 3; ++i){
		dst -> set(i, sqrt(src -> get(i)));
	}
	return dst;
}

// DIVIDE
// returns numerator / denominator
Mat* divide(const Mat* numerator, float denominator){
	if(NULL == numerator -> Data){
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
	cu_divide<<<num_blocks, block_size>>>(numerator -> Data, dst -> Data, denominator, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DIVIDE
// returns numerator / denominator
Mat* divide(float numerator, const Mat* denominator){
	if(NULL == denominator -> Data){
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
	cu_divide<<<num_blocks, block_size>>>(numerator, denominator -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DIVIDE
// returns numerator / denominator
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

// DIVIDE
// returns numerator / denominator
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

// DIVIDE
// returns numerator / denominator
Mat* divide(const Mat* numerator, const vector3f* denominator){
	if(NULL == numerator -> Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(numerator -> rows, numerator -> cols, numerator -> channels);
	int len = numerator -> rows * numerator -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < numerator -> channels; ++i){
		cu_divide<<<num_blocks, block_size>>>(numerator -> Data + i * len, dst -> Data + i * len, denominator -> get(i), len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// DIVIDE
// returns numerator / denominator
Mat* divide(const vector3f* numerator, const Mat* denominator){
	if(NULL == denominator -> Data){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(denominator -> rows, denominator -> cols, denominator -> channels);
	int len = denominator -> rows * denominator -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < denominator -> channels; ++i){
		cu_divide<<<num_blocks, block_size>>>(numerator -> get(i), denominator -> Data + i * len, dst -> Data + i * len, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// DIVIDE
// returns numerator / denominator
Mat* divide(const Mat* numerator, const Mat* denominator){
	if(NULL == denominator -> Data ||
	   NULL == numerator -> Data || numerator -> getLength() != denominator -> getLength()){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(denominator -> rows, denominator -> cols, denominator -> channels);
	int len = numerator -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator -> Data, denominator -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DIVIDE
// returns numerator / denominator
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

// DIVIDE
// returns numerator / denominator
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

// DIVIDE
// returns numerator / denominator
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

// SUBTRACT
// returns a - b
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

// SUBTRACT
// returns a - b
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

// SUBTRACT
// returns a - b
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

// SUM
// returns the sum of a vector3f
float sum(const vector3f* src){
	float res = 0.0;
	for(int i = 0; i < 3; ++i){
		res = (res + src -> get(i));
	}
	return res;
}

// SUM
// get sum of a matrix.
// because sometimes the size of matrix is larger than maxThreadsPerBlock, so each time
// it calculates the partial sum for each block, and re-calling the kernel function to the partial result,
// until data size is less than threadsPerBlock.
vector3f* sum(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	float *data = NULL;
	float *d_partial_sums = NULL;
	float *global_mem = NULL;
	checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * len));
	checkCudaErrors(cudaMalloc((void**)&global_mem, sizeof(float) * block_size));
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums, sizeof(float) * num_blocks));

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		int data_len = len;
		checkCudaErrors(cudaMemcpy(data, src -> Data + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		while(true){
			checkCudaErrors(cudaMemset(global_mem, 0, sizeof(float) * tmp_block_size));
			cu_sum<<<tmp_num_blocks, tmp_block_size>>>(data, d_partial_sums, global_mem, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			data_len = tmp_num_blocks;
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res = 0;
				checkCudaErrors(cudaMemcpy(&host_res, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost));
				res -> set(ch, host_res);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
				checkCudaErrors(cudaMemcpy(data, d_partial_sums, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
				checkCudaErrors(cudaMemcpy(data, d_partial_sums, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
			}
		}
	}
	checkCudaErrors(cudaFree(global_mem));
	checkCudaErrors(cudaFree(data));
	checkCudaErrors(cudaFree(d_partial_sums));
	return res;
}

// AVERAGE
// returns mean of a matrix
vector3f* average(const Mat* src){
	if(NULL == src -> Data){
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

// AVERAGE
// returns mean of a matrix
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

// STDDEV
// returns standard deviation of a matrix
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

// STDDEV
// returns standard deviation of a matrix
vector3f* stddev(const Mat* src, const vector3f* avg){
	if(NULL == src -> Data){
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

// MAX
// get max value of a vector3f
float max(const vector3f* src){
	float res = src -> get(0);
	for(int i = 1; i < 3; ++i){
		if(src -> get(i) > res) res = src -> get(i);
	}
	return res;
}


// MAX
// get min value, max value, min location, max location of a matrix.
// because sometimes the size of matrix is larger than maxThreadsPerBlock, so each time
// it calculates the minMaxLoc for each block, and re-calling the kernel function to the partial result,
// until data size is less than threadsPerBlock.
vector3f* max(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}

	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	float *dev_maxVal_partial = NULL;
	float *dev_minVal_partial = NULL;
	int *dev_maxLoc_partial = NULL;
	int *dev_minLoc_partial = NULL;
	float *glob_mem_maxVal = NULL;
	float *glob_mem_minVal = NULL;
	int *glob_mem_maxLoc = NULL;
	int *glob_mem_minLoc = NULL;
	float *data = NULL;
	checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * len));

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		int data_len = len;
		checkCudaErrors(cudaMemcpy(data, src -> Data + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		while(true){
			checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * tmp_block_size));
			cu_minMaxLoc<<<num_blocks, block_size>>>(data,
													dev_minVal_partial, dev_maxVal_partial,
													dev_minLoc_partial, dev_maxLoc_partial,
													glob_mem_maxVal,
													glob_mem_minVal,
													glob_mem_maxLoc,
													glob_mem_minLoc, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(data, dev_maxVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			data_len = tmp_num_blocks;
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res = 0;
				checkCudaErrors(cudaMemcpy(&host_res, dev_maxVal_partial, sizeof(float), cudaMemcpyDeviceToHost));
				res -> set(ch, host_res);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
			}
		}
	}
	checkCudaErrors(cudaFree(glob_mem_maxVal));
	checkCudaErrors(cudaFree(glob_mem_minVal));
	checkCudaErrors(cudaFree(glob_mem_maxLoc));
	checkCudaErrors(cudaFree(glob_mem_minLoc));
	checkCudaErrors(cudaFree(data));
	checkCudaErrors(cudaFree(dev_maxVal_partial));
	checkCudaErrors(cudaFree(dev_minVal_partial));
	checkCudaErrors(cudaFree(dev_maxLoc_partial));
	checkCudaErrors(cudaFree(dev_minLoc_partial));
	return res;
}

// MAX
// get min value, max value, min location, max location of a matrix.
// because sometimes the size of matrix is larger than maxThreadsPerBlock, so each time
// it calculates the minMaxLoc for each block, and re-calling the kernel function to the partial result,
// until data size is less than threadsPerBlock.
void max(const Mat* src, vector3f* max_val, vector3f* max_loc){
	if(NULL == src -> Data ||
	   NULL == max_val || NULL == max_loc){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	float *dev_maxVal_partial = NULL;
	float *dev_minVal_partial = NULL;
	int *dev_maxLoc_partial = NULL;
	int *dev_minLoc_partial = NULL;
	float *glob_mem_maxVal = NULL;
	float *glob_mem_minVal = NULL;
	int *glob_mem_maxLoc = NULL;
	int *glob_mem_minLoc = NULL;
	float *data = NULL;
	int *loc_tmp1 = (int*)malloc(sizeof(int) * num_blocks);
	int *loc_tmp2 = (int*)malloc(sizeof(int) * num_blocks);
	memset(loc_tmp1, 0, sizeof(int) * num_blocks);
	memset(loc_tmp2, 0, sizeof(int) * num_blocks);
	checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * len));

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		int data_len = len;
		int counter = 0;
		checkCudaErrors(cudaMemcpy(data, src -> Data + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		while(true){
			checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * tmp_block_size));
			cu_minMaxLoc<<<num_blocks, block_size>>>(data,
													dev_minVal_partial, dev_maxVal_partial,
													dev_minLoc_partial, dev_maxLoc_partial,
													glob_mem_maxVal,
													glob_mem_minVal,
													glob_mem_maxLoc,
													glob_mem_minLoc, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(data, dev_maxVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			data_len = tmp_num_blocks;
			checkCudaErrors(cudaMemcpy(loc_tmp2, dev_maxLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(0 < counter){
				for(int i = 0; i < tmp_num_blocks; ++i){
					loc_tmp2[i] = loc_tmp1[loc_tmp2[i]];
				}
			}
			++counter;
			checkCudaErrors(cudaMemcpy(loc_tmp1, dev_maxLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res = 0;
				checkCudaErrors(cudaMemcpy(&host_res, dev_maxVal_partial, sizeof(float), cudaMemcpyDeviceToHost));
				max_val -> set(ch, host_res);
				max_loc -> set(ch, loc_tmp2[0]);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
			}
		}
	}
	checkCudaErrors(cudaFree(glob_mem_maxVal));
	checkCudaErrors(cudaFree(glob_mem_minVal));
	checkCudaErrors(cudaFree(glob_mem_maxLoc));
	checkCudaErrors(cudaFree(glob_mem_minLoc));
	checkCudaErrors(cudaFree(data));
	checkCudaErrors(cudaFree(dev_maxVal_partial));
	checkCudaErrors(cudaFree(dev_minVal_partial));
	checkCudaErrors(cudaFree(dev_maxLoc_partial));
	checkCudaErrors(cudaFree(dev_minLoc_partial));
	free(loc_tmp1);
	free(loc_tmp2);
}

// MIN
// get min value of a vector3f
float min(const vector3f* src){
	float res = src -> get(0);
	for(int i = 1; i < 3; ++i){
		if(src -> get(i) < res) res = src -> get(i);
	}
	return res;
}

// MIN
// get min value, max value, min location, max location of a matrix.
// because sometimes the size of matrix is larger than maxThreadsPerBlock, so each time
// it calculates the minMaxLoc for each block, and re-calling the kernel function to the partial result,
// until data size is less than threadsPerBlock.
vector3f* min(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}

	vector3f *res = new vector3f();
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	float *dev_maxVal_partial = NULL;
	float *dev_minVal_partial = NULL;
	int *dev_maxLoc_partial = NULL;
	int *dev_minLoc_partial = NULL;
	float *glob_mem_maxVal = NULL;
	float *glob_mem_minVal = NULL;
	int *glob_mem_maxLoc = NULL;
	int *glob_mem_minLoc = NULL;
	float *data = NULL;
	checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * len));

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		int data_len = len;
		checkCudaErrors(cudaMemcpy(data, src -> Data + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		while(true){
			checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * tmp_block_size));
			cu_minMaxLoc<<<num_blocks, block_size>>>(data,
													dev_minVal_partial, dev_maxVal_partial,
													dev_minLoc_partial, dev_maxLoc_partial,
													glob_mem_maxVal,
													glob_mem_minVal,
													glob_mem_maxLoc,
													glob_mem_minLoc, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(data, dev_minVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			data_len = tmp_num_blocks;
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res = 0;
				checkCudaErrors(cudaMemcpy(&host_res, dev_minVal_partial, sizeof(float), cudaMemcpyDeviceToHost));
				res -> set(ch, host_res);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
			}
		}
	}
	checkCudaErrors(cudaFree(glob_mem_maxVal));
	checkCudaErrors(cudaFree(glob_mem_minVal));
	checkCudaErrors(cudaFree(glob_mem_maxLoc));
	checkCudaErrors(cudaFree(glob_mem_minLoc));
	checkCudaErrors(cudaFree(data));
	checkCudaErrors(cudaFree(dev_maxVal_partial));
	checkCudaErrors(cudaFree(dev_minVal_partial));
	checkCudaErrors(cudaFree(dev_maxLoc_partial));
	checkCudaErrors(cudaFree(dev_minLoc_partial));
	return res;
}

// MIN
// get min value, max value, min location, max location of a matrix.
// because sometimes the size of matrix is larger than maxThreadsPerBlock, so each time
// it calculates the minMaxLoc for each block, and re-calling the kernel function to the partial result,
// until data size is less than threadsPerBlock.
void min(const Mat* src, vector3f* min_val, vector3f* min_loc){
	if(NULL == src -> Data ||
	   NULL == min_val || NULL == min_loc){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	float *dev_maxVal_partial = NULL;
	float *dev_minVal_partial = NULL;
	int *dev_maxLoc_partial = NULL;
	int *dev_minLoc_partial = NULL;
	float *glob_mem_maxVal = NULL;
	float *glob_mem_minVal = NULL;
	int *glob_mem_maxLoc = NULL;
	int *glob_mem_minLoc = NULL;
	float *data = NULL;
	int *loc_tmp1 = (int*)malloc(sizeof(int) * num_blocks);
	int *loc_tmp2 = (int*)malloc(sizeof(int) * num_blocks);
	memset(loc_tmp1, 0, sizeof(int) * num_blocks);
	memset(loc_tmp2, 0, sizeof(int) * num_blocks);
	checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&data, sizeof(float) * len));

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		int data_len = len;
		int counter = 0;
		checkCudaErrors(cudaMemcpy(data, src -> Data + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		while(true){
			checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * tmp_block_size));
			cu_minMaxLoc<<<num_blocks, block_size>>>(data,
													dev_minVal_partial, dev_maxVal_partial,
													dev_minLoc_partial, dev_maxLoc_partial,
													glob_mem_maxVal,
													glob_mem_minVal,
													glob_mem_maxLoc,
													glob_mem_minLoc, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(data, dev_minVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			data_len = tmp_num_blocks;
			checkCudaErrors(cudaMemcpy(loc_tmp2, dev_minLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(0 < counter){
				for(int i = 0; i < tmp_num_blocks; ++i){
					loc_tmp2[i] = loc_tmp1[loc_tmp2[i]];
				}
			}
			++counter;
			checkCudaErrors(cudaMemcpy(loc_tmp1, dev_minLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res = 0;
				checkCudaErrors(cudaMemcpy(&host_res, dev_minVal_partial, sizeof(float), cudaMemcpyDeviceToHost));
				min_val -> set(ch, host_res);
				min_loc -> set(ch, loc_tmp2[0]);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
			}
		}
	}
	checkCudaErrors(cudaFree(glob_mem_maxVal));
	checkCudaErrors(cudaFree(glob_mem_minVal));
	checkCudaErrors(cudaFree(glob_mem_maxLoc));
	checkCudaErrors(cudaFree(glob_mem_minLoc));
	checkCudaErrors(cudaFree(data));
	checkCudaErrors(cudaFree(dev_maxVal_partial));
	checkCudaErrors(cudaFree(dev_minVal_partial));
	checkCudaErrors(cudaFree(dev_maxLoc_partial));
	checkCudaErrors(cudaFree(dev_minLoc_partial));
	free(loc_tmp1);
	free(loc_tmp2);
}

// MIN MAX LOCATION
// get min value, max value, min location, max location of a matrix.
// because sometimes the size of matrix is larger than maxThreadsPerBlock, so each time
// it calculates the minMaxLoc for each block, and re-calling the kernel function to the partial result,
// until data size is less than threadsPerBlock.
void minMaxLoc(const Mat* src, vector3f* max_val, vector3f* max_loc, vector3f* min_val, vector3f* min_loc){
	if(NULL == src -> Data ||
	   NULL == min_val || NULL == min_loc ||
	   NULL == max_val || NULL == max_loc){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);

	float *dev_maxVal_partial = NULL;
	float *dev_minVal_partial = NULL;
	int *dev_maxLoc_partial = NULL;
	int *dev_minLoc_partial = NULL;
	float *glob_mem_maxVal = NULL;
	float *glob_mem_minVal = NULL;
	int *glob_mem_maxLoc = NULL;
	int *glob_mem_minLoc = NULL;
	float *data_max = NULL;
	float *data_min = NULL;
	int *loc_tmp1 = (int*)malloc(sizeof(int) * num_blocks);
	int *loc_tmp2 = (int*)malloc(sizeof(int) * num_blocks);
	int *loc_tmp3 = (int*)malloc(sizeof(int) * num_blocks);
	int *loc_tmp4 = (int*)malloc(sizeof(int) * num_blocks);
	memset(loc_tmp1, 0, sizeof(int) * num_blocks);
	memset(loc_tmp2, 0, sizeof(int) * num_blocks);
	memset(loc_tmp3, 0, sizeof(int) * num_blocks);
	memset(loc_tmp4, 0, sizeof(int) * num_blocks);
	checkCudaErrors(cudaMalloc((void**)&dev_maxVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minVal_partial, sizeof(float) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_maxLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&dev_minLoc_partial, sizeof(int) * (num_blocks)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minVal, sizeof(float) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_maxLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&glob_mem_minLoc, sizeof(int) * (block_size)));
	checkCudaErrors(cudaMalloc((void**)&data_max, sizeof(float) * len));
	checkCudaErrors(cudaMalloc((void**)&data_min, sizeof(float) * len));

	for(int ch = 0; ch < src -> channels; ++ch){
		int tmp_block_size = block_size;
		int tmp_num_blocks = num_blocks;
		int data_len = len;
		int counter = 0;
		checkCudaErrors(cudaMemcpy(data_max, src -> Data + ch * len, data_len * sizeof(float), cudaMemcpyDeviceToDevice));
		while(true){
			checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * tmp_block_size));
			checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * tmp_block_size));
			cu_minMaxLoc<<<num_blocks, block_size>>>(data_max,
													dev_minVal_partial, dev_maxVal_partial,
													dev_minLoc_partial, dev_maxLoc_partial,
													glob_mem_maxVal,
													glob_mem_minVal,
													glob_mem_maxLoc,
													glob_mem_minLoc, data_len);
		    getLastCudaError("kernel execution failed\n");
	        checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(data_max, dev_maxVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(loc_tmp2, dev_maxLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(0 < counter){
				for(int i = 0; i < tmp_num_blocks; ++i){
					loc_tmp2[i] = loc_tmp1[loc_tmp2[i]];
				}
			}
			checkCudaErrors(cudaMemcpy(loc_tmp1, dev_maxLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(0 < counter){
				checkCudaErrors(cudaMemset(glob_mem_maxVal, 0, sizeof(float) * tmp_block_size));
				checkCudaErrors(cudaMemset(glob_mem_minVal, 0, sizeof(float) * tmp_block_size));
				checkCudaErrors(cudaMemset(glob_mem_maxLoc, 0, sizeof(int) * tmp_block_size));
				checkCudaErrors(cudaMemset(glob_mem_minLoc, 0, sizeof(int) * tmp_block_size));
				cu_minMaxLoc<<<num_blocks, block_size>>>(data_min,
														dev_minVal_partial, dev_maxVal_partial,
														dev_minLoc_partial, dev_maxLoc_partial,
														glob_mem_maxVal,
														glob_mem_minVal,
														glob_mem_maxLoc,
														glob_mem_minLoc, data_len);
			    getLastCudaError("kernel execution failed\n");
		        checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaMemcpy(data_min, dev_minVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			}else{
				checkCudaErrors(cudaMemcpy(data_min, dev_minVal_partial, tmp_num_blocks * sizeof(float), cudaMemcpyDeviceToDevice));
			}

			checkCudaErrors(cudaMemcpy(loc_tmp4, dev_minLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			if(0 < counter){
				for(int i = 0; i < tmp_num_blocks; ++i){
					loc_tmp4[i] = loc_tmp3[loc_tmp4[i]];
				}
			}
			checkCudaErrors(cudaMemcpy(loc_tmp3, dev_minLoc_partial, tmp_num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
			data_len = tmp_num_blocks;
			++counter;
			if(tmp_num_blocks == 1){
				// copy the result back to the host
				float host_res_min = 0;
				float host_res_max = 0;
				checkCudaErrors(cudaMemcpy(&host_res_min, data_min, sizeof(float), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(&host_res_max, data_max, sizeof(float), cudaMemcpyDeviceToHost));
				max_val -> set(ch, host_res_max);
				max_loc -> set(ch, loc_tmp2[0]);
				min_val -> set(ch, host_res_min);
				min_loc -> set(ch, loc_tmp4[0]);
				break;
			}elif(tmp_num_blocks <= block_size){
				tmp_block_size = data_len;
				tmp_num_blocks = 1;
			}else{
				tmp_block_size = threadsPerBlock;
				tmp_num_blocks = (data_len / tmp_block_size) + ((data_len % tmp_block_size) ? 1 : 0);
			}
		}
	}
	checkCudaErrors(cudaFree(glob_mem_maxVal));
	checkCudaErrors(cudaFree(glob_mem_minVal));
	checkCudaErrors(cudaFree(glob_mem_maxLoc));
	checkCudaErrors(cudaFree(glob_mem_minLoc));
	checkCudaErrors(cudaFree(data_min));
	checkCudaErrors(cudaFree(data_max));
	checkCudaErrors(cudaFree(dev_maxVal_partial));
	checkCudaErrors(cudaFree(dev_minVal_partial));
	checkCudaErrors(cudaFree(dev_maxLoc_partial));
	checkCudaErrors(cudaFree(dev_minLoc_partial));
	free(loc_tmp1);
	free(loc_tmp2);
	free(loc_tmp3);
	free(loc_tmp4);
}

// EQUAL TO
// res = src > val
Mat* greaterThan(const Mat *src, float val){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_greaterThan<<<num_blocks, block_size>>>(src -> Data, dst -> Data, val, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// EQUAL TO
// res = src < val
Mat* lessThan(const Mat *src, float val){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_lessThan<<<num_blocks, block_size>>>(src -> Data, dst -> Data, val, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// EQUAL TO
// res = src == val
Mat* equalTo(const Mat *src, float val){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_equalTo<<<num_blocks, block_size>>>(src -> Data, dst -> Data, val, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// GET BERNOULLI MATRIX
// get random bernoulli matrix, with given probability
Mat* getBernoulliMatrix(int height, int width, int nchannels, float prob){
	Mat *ran = new Mat(height, width, nchannels);
	ran -> randu();
	Mat *res = NULL;
	safeGetPt(res, greaterThan(ran, prob));
	ran -> release();
	return res;
}

// CONVERT
// convert from vector of img to matrix
// vec.size() == nsamples
// for example:
// vec[0][0]:  	[1, 2]	vec[0][1]: 	[a][b]
// 				[3, 4]				[c][d]
// vec[1][0]:  	[5, 6]	vec[1][1]: 	[e][f]
// 				[7, 8]				[g][h]
// result should be:   M[1, 2, 3, 4, a, b, c, d]T
//						[5, 6, 7, 8, e, f, g, h]

void convert(std::vector<std::vector<Mat*> >& vec, Mat *M){
    Mat *res = new Mat(vec.size(), vec[0].size() * vec[0][0] -> getLength(), 1);
    for(int i = 0; i < vec.size(); i++){
        for(int m = 0; m < vec[i].size(); m++){
			checkCudaErrors(cudaMemcpy(res -> Data + vec[i][m] -> getLength() * (m + i * vec[i].size()), vec[i][m] -> Data, vec[i][m] -> getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    Mat *tmp = NULL;
    safeGetPt(tmp, t(res));
    tmp -> copyTo(*M);
    tmp -> release();
    res -> release();
}

// CONVERT
// convert from matrix to vector of img
// vec.size() == nsamples
// details see convert(vec<vec<Mat*>>, Mat*) comments
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
			checkCudaErrors(cudaMemcpy(vec[i][j] -> Data, Mt -> Data + i * Mt -> cols + j * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice));
    	}
    }
    Mt -> release();
}

// SIGMOID
// sigmoid non-linearity
Mat *sigmoid(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_sigmoid<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DSIGMOID
// get derivatives of sigmoid non-linearity function
Mat* dsigmoid(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dsigmoid<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DSIGMOID A
// get derivatives of sigmoid non-linearity function using cache of forward passing matrix
Mat* dsigmoid_a(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dsigmoid_a<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// RELU
// relu non-linearity
Mat* ReLU(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_relu<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DRELU
// get derivatives of relu non-linearity function
Mat* dReLU(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_drelu<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// LEAKY RELU
// leaky-relu non-linearity
Mat* LeakyReLU(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_leaky_relu<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DLEAKYRELU
// get derivatives of leaky-relu non-linearity function
Mat* dLeakyReLU(const Mat* src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dleaky_relu<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// TANH
// tanh non-linearity
Mat* Tanh(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_tanh<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// DTANH
// get derivatives of tanh non-linearity function
Mat* dTanh(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_dtanh<<<num_blocks, block_size>>>(src -> Data, dst -> Data, len);
    getLastCudaError("kernel execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
	return dst;
}

// NONLINEARITY
// non-linearity
Mat* nonLinearity(const Mat *M, int method){
	if(NULL == M -> Data){
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

// DNONLINEARITY
// get derivatives of non-linearity function
Mat* dnonLinearity(const Mat *M, int method){
	if(NULL == M -> Data){
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

// FLIPLR
// flip left and right, for example:
// [a, b]				 [b, a]
// [c, d]   turns into   [d, c]
Mat* fliplr(const Mat *src){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(src -> rows, src -> cols, src -> channels);
	int len = src -> rows * src -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src -> channels; ++i){
		cu_fliplr<<<num_blocks, block_size>>>(src -> Data + i * len, dst -> Data + i * len, src -> rows, src -> cols, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// ROT90
// rotate 90
Mat* rot90(const Mat *src, int k){
	if(NULL == src -> Data){
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

// DOPADDING
// do zero padding around matrix
Mat* dopadding(const Mat *src, int pad){
	if(NULL == src -> Data){
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
		cu_padding<<<num_blocks, block_size>>>(src -> Data + i * lensrc, dst -> Data + i * lendst, src -> rows, src -> cols, dst -> cols, lensrc);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// DEPADDING
// delete padding around matrix
Mat* depadding(const Mat *src, int pad){
	if(NULL == src -> Data){
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
		cu_depadding<<<num_blocks, block_size>>>(src -> Data + i * lensrc, dst -> Data + i * lendst, src -> rows, src -> cols, dst -> cols, lendst);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// REDUCE
// similar with OpenCV reduce function, only support max reduce and sum reduce
Mat* reduce(const Mat* src, int direction, int mode){
	if(NULL == src -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *tmp = new Mat();
	Mat *dst = NULL;
	vector3f *tmpvec = new vector3f();
	if(REDUCE_TO_SINGLE_ROW == direction){
		dst = new Mat(1, src -> cols, src -> channels);
		for(int i = 0; i < src -> cols; ++i){
			safeGetPt(tmp, getRange(src, i, i, 0, src -> rows - 1));
			if(REDUCE_SUM == mode){
				safeGetPt(tmpvec, sum(tmp));
			}elif(REDUCE_MAX == mode){
				safeGetPt(tmpvec, max(tmp));
			}
			dst -> set(i, *(tmpvec));
		}
	}else{ // REDUCE_TO_SINGLE_COL == direction
		dst = new Mat(src -> rows, 1, src -> channels);
		for(int i = 0; i < src -> rows; ++i){
			safeGetPt(tmp, getRange(src, 0, src -> cols - 1, i, i));
			if(REDUCE_SUM == mode){
				safeGetPt(tmpvec, sum(tmp));
			}elif(REDUCE_MAX == mode){
				safeGetPt(tmpvec, max(tmp));
			}
			dst -> set(i, *(tmpvec));
		}
	}
	tmpvec -> release();
	tmp -> release();
	return dst;
}

// INTERPOLATION
// interpolation with zeros
Mat* interpolation(const Mat* src, int _size){
	if(NULL == src -> Data){
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
		cu_interpolation<<<num_blocks, block_size>>>(src -> Data + i * lensrc, dst -> Data + i * lendst, src -> cols, dst -> cols, stride, lensrc);
	    getLastCudaError("kernel execution failed\n");
	    getLastCudaError("kernel execution failed\n");
		checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// REPMAT
// repeat matrix, for example, repmat(a, 2, 3) turns into:	[a, a, a]
// 															[a, a, a]
Mat* repmat(const Mat *src, int vert, int hori){
	if(NULL == src -> Data || 0 == vert || 0 == hori){
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
		cu_repmat<<<num_blocks, block_size>>>(src -> Data + i * lensrc, dst -> Data + i * lendst, src -> rows, src -> cols, dst -> rows, dst -> cols, lendst);
	    getLastCudaError("kernel execution failed\n");
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// KRON
// calculates kronecker product, can be used in unpooling mean pooling
Mat* kron(const Mat *a, const Mat *b){
	if(NULL == a -> Data || NULL == b -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *dst = new Mat(a -> rows * b -> rows, a -> cols * b -> cols, a -> channels);
	int lensrc = a -> rows * a -> cols;
	int lendst = dst -> rows * dst -> cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (lendst / block_size) + ((lendst % block_size) ? 1 : 0);
	for(int i = 0; i < a -> channels; ++i){
		cu_kron<<<num_blocks, block_size>>>(a -> Data + i * lensrc, b -> Data, dst -> Data + i * lendst, a -> rows, a -> cols, dst -> rows, dst -> cols, lendst);
	    getLastCudaError("kernel execution failed\n");
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}

// CONVOLUTION 2D
// using nVidia convolution FFT2D,
// this is actually conv2 with 'same' type
Mat* conv2(const Mat *m, const Mat *kernel){
	if(NULL == m -> Data || NULL == kernel -> Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat *res = new Mat(m -> rows, m -> cols, m -> channels);
	float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;
    fComplex *d_DataSpectrum0, *d_KernelSpectrum0;
    cufftHandle fftPlan;
    float *result_tmp;
    const int kernelH = kernel -> rows;
    const int kernelW = kernel -> cols;
    const int kernelY = kernel -> rows / 2;
    const int kernelX = kernel -> cols / 2;
    const int dataH = m -> rows;
    const int dataW = m -> cols;
    const int fftH = snapTransformSize(dataH + kernelH - 1);
    const int fftW = snapTransformSize(dataW + kernelW - 1);
    checkCudaErrors(cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&result_tmp,   dataH   * dataW   * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex)));
    Mat *res_tmp = new Mat(fftH, fftW, 3);
    // std::cout<<"...creating C2C FFT plan for "<<fftH<<" x "<<fftW/2<<std::endl;
    checkCudaErrors(cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C));
    for(int i = 0; i < m -> channels; ++i){
    	checkCudaErrors(cudaMemcpy(d_Data, m -> Data + dataH * dataW * i, dataH * dataW * sizeof(float), cudaMemcpyDeviceToDevice));
    	checkCudaErrors(cudaMemcpy(d_Kernel, kernel -> Data + kernelH * kernelW * i, kernelH * kernelW * sizeof(float), cudaMemcpyDeviceToDevice));
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
		checkCudaErrors(cudaMemcpy(res_tmp -> Data + res_tmp -> rows * res_tmp -> cols * i, d_PaddedData, res_tmp -> rows * res_tmp -> cols * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    safeGetPt(res, getRange(res_tmp, 0, dataW - 1, 0, dataH - 1));
	res_tmp -> release();
    checkCudaErrors(cufftDestroy(fftPlan));
    checkCudaErrors(cudaFree(d_KernelSpectrum0));
    checkCudaErrors(cudaFree(d_DataSpectrum0));
    checkCudaErrors(cudaFree(d_PaddedKernel));
    checkCudaErrors(cudaFree(d_PaddedData));
    checkCudaErrors(cudaFree(d_Kernel));
    checkCudaErrors(cudaFree(d_Data));
    return res;
}

// CONVOLUTION 2D
// supports full/same/valid type convolution, similar with matlab conv2 type.
// also supports zero padding and stride.
// if using valid type, result size = (m_size + 2 * pad - kernel_size) / stride + 1
Mat* conv2(const Mat *m, const Mat *kernel, int convtype, int pad, int stride){
	if(NULL == m -> Data || NULL == kernel -> Data){
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

// GET RANGE
// similar with OpenCV getRange function.
// BE CAREFUL, xend/yend are last element in the range, but not first element outside range.
Mat* getRange(const Mat* src, int xstart, int xend, int ystart, int yend){
    //cout<<" --- using gpu memory "<<MemoryMonitor::instance() -> getGpuMemory() <<"    inside 1"<<endl;
	if(NULL == src -> Data){
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
		cu_getRange<<<num_blocks, block_size>>>(src -> Data + i * src -> rows * src -> cols, dst -> Data + i * len, xstart, xend, ystart, yend, src -> cols, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
    //cout<<" --- using gpu memory "<<MemoryMonitor::instance() -> getGpuMemory() <<"    inside 2"<<endl;
	return dst;
}

// DOWN SAMPLE
// simply down sample
Mat* downSample(const Mat* src, int y_stride, int x_stride){
	if(NULL == src -> Data || y_stride < 1 || x_stride < 1){
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
		cu_downSample<<<num_blocks, block_size>>>(src -> Data + i * src -> rows * src -> cols, res -> Data + i * len, y_stride, x_stride, src -> cols, len);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return res;
}

// COPY MAKE BORDER
// similar with OpenCV copyMakeBorder function, for example
// a =  [b, c]		, copyMakeBorder(a, 1, 0, 1, 0, val=x) is 	[x, x, x]
//		[d, e]													[x, b, c]
//																[x, d, e]
Mat* copyMakeBorder(const Mat* src, int up, int down, int left, int right, const vector3f* val){
	if(NULL == src -> Data){
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
		cu_copyMakeBorder<<<num_blocks, block_size>>>(src -> Data + i * lensrc, dst -> Data + i * lendst, src -> rows, src -> cols, up, down, left, right, lensrc);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return dst;
}


// POOLING WITH OVERLAP
// Max pooling supported
// output size = (input size - window size) / stride + 1
// TODO: stochastic pooling
Mat* pooling_with_overlap(const Mat *src, vector2i *window_size, int stride, int poolingMethod, Mat*& locat){
	if(NULL == src -> Data || stride < 1){
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
		cu_pooling_overlap_max<<<num_blocks, block_size>>>(src -> Data + i * lensrc, res -> Data + i * lenres,  loc -> Data + i * lenres, src -> rows, src -> cols, res -> rows, res -> cols, window_size -> get(0), window_size -> get(1), lenres);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
    Mat *dst = new Mat();
    safeGetPt(dst, downSample(res, stride, stride));
    safeGetPt(locat, downSample(loc, stride, stride));
    res -> release();
	loc -> release();
    return dst;
}

// UNPOOLING WITH OVERLAP
// unpooling with overlap
Mat* unpooling_with_overlap(const Mat* src, vector2i* window_size, int stride, int poolingMethod, const Mat* locat, vector2i* up_size){
	if(NULL == src -> Data || stride < 1){
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
		cu_unpooling<<<num_blocks, block_size>>>(src -> Data + i * lensrc, locat -> Data + i * lensrc, res -> Data + i * lenres, res -> cols, lensrc);
	    getLastCudaError("kernel execution failed\n");
        checkCudaErrors(cudaDeviceSynchronize());
	}
	return res;
}

// POOLING
// pooling without overlap
// TODO: stochastic pooling
Mat* pooling(const Mat* src, int stride, int poolingMethod, Mat*& locat){
	if(NULL == src -> Data || stride < 1){
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
			cu_pooling_max<<<num_blocks, block_size>>>(src -> Data + i * lensrc, res -> Data + i * lenres,  loc -> Data + i * lenres, src -> rows, src -> cols, res -> rows, res -> cols, stride, stride, lenres);
		    getLastCudaError("kernel execution failed\n");
		}elif(POOL_MEAN == poolingMethod){
			cu_pooling_mean<<<num_blocks, block_size>>>(src -> Data + i * lensrc, res -> Data + i * lenres,  loc -> Data + i * lenres, src -> rows, src -> cols, res -> rows, res -> cols, stride, stride, lenres);
		    getLastCudaError("kernel execution failed\n");
		}
        checkCudaErrors(cudaDeviceSynchronize());
	}
	loc -> copyTo(*locat);
	loc -> release();
    return res;
}

// UNPOOLING
// unpooling
Mat* unpooling(const Mat* src, int stride, int poolingMethod, const Mat* locat, vector2i* up_size){
	if(NULL == src -> Data || stride < 1){
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
    		cu_unpooling<<<num_blocks, block_size>>>(src -> Data + i * lensrc, locat -> Data + i * lensrc, res -> Data + i * lenres, res -> cols, lensrc);
    	    getLastCudaError("kernel execution failed\n");
            checkCudaErrors(cudaDeviceSynchronize());
    	}
    	return res;
    }
}

// FIND MAX
// use during testing, find which has max val in each column
Mat* findMax(const Mat* m){
	if(NULL == m -> Data){
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

// SAME VALUES IN MATRIX
// use during testing, only calculates first channel,
// and find how many values are same between two matrices.
int sameValuesInMat(const Mat* a, const Mat* b){
	if(NULL == a -> Data || NULL == b -> Data ||
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

