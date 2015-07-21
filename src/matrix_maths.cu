#include "matrix_maths.h"

Mat exp(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	int tmp = src.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_exp<<<num_blocks, block_size>>>(src.devData, dst.devData, tmp);
	dst.deviceToHost();
	return dst;
}

Mat log(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	int tmp = src.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_log<<<num_blocks, block_size>>>(src.devData, dst.devData, tmp);
	dst.deviceToHost();
	return dst;
}

Mat pow(const Mat &src, int power){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid src..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	if(0 == power){
		dst.ones();
		return dst;
	}
	int tmp = src.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_pow<<<num_blocks, block_size>>>(src.devData, dst.devData, power, tmp);
	dst.deviceToHost();
	return dst;
}

Mat divide(const Mat &numerator, float denominator){
	if(NULL == numerator.hostData || NULL == numerator.devData){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	Mat dst(numerator);
	if(0.0 == denominator){
		std::cout<<"invalid denominator..."<<std::endl;
		dst.zeros();
		return dst;
	}
	int tmp = numerator.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator.devData, dst.devData, denominator, tmp);
	dst.deviceToHost();
	return dst;
}

Mat divide(float numerator, const Mat& denominator){
	if(NULL == denominator.hostData || NULL == denominator.devData){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	Mat dst(denominator);
	if(0.0 == numerator){
		dst.zeros();
		return dst;
	}
	int tmp = denominator.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator, denominator.devData, dst.devData, tmp);
	dst.deviceToHost();
	return dst;
}

vector3f divide(const vector3f& numerator, float denominator){
	vector3f dst(numerator);
	if(0.0 == denominator){
		dst.zeros();
		return dst;
	}
	for(int i = 0; i < 3; ++i){
		dst.set(i, (dst.get(i) / denominator));
	}
	return dst;
}

vector3f divide(float numerator, const vector3f& denominator){
	vector3f dst(denominator);
	if(0.0 == numerator){
		dst.zeros();
		return dst;
	}
	for(int i = 0; i < 3; ++i){
		if(dst.get(i) == 0.0) continue;
		dst.set(i, (numerator / dst.get(i)));
	}
	return dst;
}

Mat divide(const Mat& numerator, const vector3f& denominator){
	if(NULL == numerator.hostData || NULL == numerator.devData){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	Mat dst(numerator);
	int tmp = numerator.rows * numerator.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < numerator.channels; ++i){
		cu_divide<<<num_blocks, block_size>>>(numerator.devData + i * tmp, dst.devData + i * tmp, denominator.get(i), tmp);
	}
	dst.deviceToHost();
	return dst;
}

Mat divide(const vector3f& numerator, const Mat& denominator){
	if(NULL == denominator.hostData || NULL == denominator.devData){
		std::cout<<"invalid denominator..."<<std::endl;
		exit(0);
	}
	Mat dst(denominator);
	int tmp = denominator.rows * denominator.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < denominator.channels; ++i){
		cu_divide<<<num_blocks, block_size>>>(numerator.get(i), denominator.devData + i * tmp, dst.devData + i * tmp, tmp);
	}
	dst.deviceToHost();
	return dst;
}

Mat divide(const Mat& numerator, const Mat& denominator){
	if(NULL == denominator.hostData || NULL == denominator.devData ||
	   NULL == numerator.hostData || NULL == numerator.devData || numerator.getLength() != denominator.getLength()){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(numerator);
	int tmp = numerator.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_divide<<<num_blocks, block_size>>>(numerator.devData, denominator.devData, dst.devData, tmp);
	dst.deviceToHost();
	return dst;
}

vector3f divide(const vector3f& numerator, const vector3f& denominator){
	vector3f dst(denominator);
	for(int i = 0; i < 3; ++i){
		if(dst.get(i) == 0.0) continue;
		dst.set(i, (numerator.get(i) / dst.get(i)));
	}
	return dst;
}

cpuMat divide(const cpuMat& numerator, const vector3f& denominator){
	if(NULL == numerator.Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	cpuMat dst(numerator);
	int len = dst.rows * dst.cols;
	for(int ch = 0; ch < numerator.channels; ++ch){
		for(int i = 0; i < len; ++i){
			dst.Data[ch * len + i] = dst.Data[ch * len + i] / denominator.get(ch);
		}
	}
	return dst;
}

cpuMat divide(const cpuMat& numerator, float denominator){
	if(NULL == numerator.Data){
		std::cout<<"invalid numerator..."<<std::endl;
		exit(0);
	}
	cpuMat dst(numerator);
	int len = dst.getLength();
	for(int i = 0; i < len; ++i){
		dst.Data[i] = dst.Data[i] / denominator;
	}
	return dst;
}

float sum(const vector3f& src){
	float res = 0.0;
	for(int i = 0; i < 3; ++i){
		res = (res + src.get(i));
	}
	return res;
}

vector3f sum(const Mat& src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f res;
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		float *devRes = 0;
		cudaMalloc((void**)&devRes, sizeof(float));
		cu_sum<<<num_blocks, block_size, block_size * sizeof(float)>>>(src.devData + i * tmp, devRes, tmp);
		float hostRes = 0;
		cudaMemcpy(&hostRes, devRes, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(devRes);
		res.set(i, hostRes);
	}
	return res;
}

vector3f average(const Mat& src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat tmp = divide(src, src.rows * src.cols);
	vector3f res = sum(tmp);
	return res;
}

vector3f average(const cpuMat& src){
	if(NULL == src.Data){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f res;
	for(int ch = 0; ch < src.channels; ++ch){
		for(int i = 0; i < src.rows * src.cols; ++i){
			res.set(ch, res.get(ch) + src.Data[ch * src.rows * src.cols + i] / src.rows / src.cols);
		}
	}
	return res;
}

vector3f stddev(const cpuMat& src, const vector3f& avg){
	if(NULL == src.Data ){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	cpuMat tmpmat(src);
	for(int ch = 0; ch < src.channels; ++ch){
		for(int i = 0; i < tmpmat.rows * tmpmat.cols; ++i){
			float tmp = tmpmat.Data[ch * tmpmat.rows * src.cols + i] - avg.get(ch);
			tmp = tmp * tmp;
			tmpmat.Data[ch * tmpmat.rows * tmpmat.cols + i] = tmp;
		}
	}
	vector3f res = average(tmpmat);
	return res;
}

vector3f stddev(const Mat& src, const vector3f& avg){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat tmp = src - avg;
	tmp = pow(tmp, 2);
	vector3f res = average(tmp);
	return res;
}

float max(const vector3f &src){
	float res = src.get(0);
	for(int i = 1; i < 3; ++i){
		if(src.get(i) > res) res = src.get(i);
	}
	return res;
}

vector3f max(const Mat& src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f res;
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src.devData + i * tmp, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, tmp);
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
		res.set(i, host_maxVal);
	}
	return res;
}

void max(const Mat& src, vector3f& max_val, vector3f& max_loc){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src.devData + i * tmp, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, tmp);
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
		max_val.set(i, host_maxVal);
		max_loc.set(i, host_maxLoc);
	}
}

float min(const vector3f &src){
	float res = src.get(0);
	for(int i = 1; i < 3; ++i){
		if(src.get(i) < res) res = src.get(i);
	}
	return res;
}

vector3f min(const Mat& src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	vector3f res;
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src.devData + i * tmp, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, tmp);
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
		res.set(i, host_minVal);
	}
	return res;
}

void min(const Mat& src, vector3f& min_val, vector3f& min_loc){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src.devData + i * tmp, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, tmp);
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
		min_val.set(i, host_minVal);
		min_loc.set(i, host_minLoc);
	}
}

void minMaxLoc(const Mat& src, vector3f& max_val, vector3f& max_loc, vector3f& min_val, vector3f& min_loc){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		float *dev_maxVal = 0;
		float *dev_minVal = 0;
		int *dev_maxLoc = 0;
		int *dev_minLoc = 0;
		cudaMalloc((void**)&dev_maxVal, sizeof(float));
		cudaMalloc((void**)&dev_minVal, sizeof(float));
		cudaMalloc((void**)&dev_maxLoc, sizeof(int));
		cudaMalloc((void**)&dev_minLoc, sizeof(int));
		cu_minMaxLoc<<<num_blocks, block_size>>>(src.devData + i * tmp, dev_minVal, dev_maxVal, dev_minLoc, dev_maxLoc, tmp);
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
		max_val.set(i, host_maxVal);
		max_loc.set(i, host_maxLoc);
		min_val.set(i, host_minVal);
		min_loc.set(i, host_minLoc);
	}
}

Mat greaterThan(const Mat &src, float val){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	int tmp = src.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_greaterThan<<<num_blocks, block_size>>>(src.devData, dst.devData, val, tmp);
	dst.deviceToHost();
	return dst;
}

Mat lessThan(const Mat &src, float val){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	int tmp = src.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_lessThan<<<num_blocks, block_size>>>(src.devData, dst.devData, val, tmp);
	dst.deviceToHost();
	return dst;
}

// convert from vector of img to matrix
// vec.size() == nsamples
void convert(std::vector<std::vector<Mat> >& vec, Mat &M){
    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    Mat res(3 * vec[0].size() * subFeatures, vec.size(), 1);
    for(int i = 0; i < vec.size(); i++){
        for(int m = 0; m < vec[i].size(); m++){
			memcpy(res.hostData + vec[i][m].getLength() * (m + i * vec[i].size()), vec[i][m].hostData, vec[i][m].getLength() * sizeof(float));
        }
    }
    res.hostToDevice();
    res.copyTo(M);
}

// convert from matrix to vector of img
// vec.size() == nsamples
void convert(Mat &M, std::vector<std::vector<Mat> >& vec, int nsamples, int imagesize){
    std::vector<Mat> tmpvec;
    for(int i = 0; i < nsamples; i++){
        tmpvec.clear();
        int dim = imagesize * imagesize;
        for(int j = 0; j < M.rows; j += dim * 3){
        	Mat tmp(imagesize, imagesize, 3);
        	memcpy(tmp.hostData, M.hostData + i * M.rows + j, dim * 3 * sizeof(float));
        	tmp.hostToDevice();
        	tmpvec.push_back(tmp);
        }
        vec.push_back(tmpvec);
    }
    tmpvec.clear();
    std::vector<Mat>().swap(tmpvec);
}

// non-linearity
Mat sigmoid(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat tmp(src);
	tmp = tmp.mul(-1.0);
	tmp = exp(tmp) + 1.0;
	tmp = divide(1.0, tmp);
	return tmp;
}

Mat dsigmoid(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat tmp = exp(src);
    Mat tmp2 = tmp + 1.0;
    tmp2 = pow(tmp2, 2);
    return divide(tmp, tmp2);
}

Mat dsigmoid_a(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat res(src);
    res.ones();
    res = res - src;
    res = res.mul(src);
    return res;
}

Mat ReLU(const Mat& M){
	if(NULL == M.hostData || NULL == M.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat res = greaterThan(M, 0.0);
    res = res.mul(M);
    return res;
}

Mat dReLU(const Mat& M){
	if(NULL == M.hostData || NULL == M.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat res = greaterThan(M, 0.0);
    return res;
}

Mat LeakyReLU(const Mat& M){
	if(NULL == M.hostData || NULL == M.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat p = greaterThan(M, 0.0);
    p = p.mul(M);
    Mat n = lessThan(M, 0.0);
    n = n.mul(M);
    n = divide(n, leaky_relu_alpha);
    return p + n;
}

Mat dLeakyReLU(const Mat& M){
	if(NULL == M.hostData || NULL == M.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat p = greaterThan(M, 0.0);
    Mat n = lessThan(M, 0.0);
    n = divide(n, leaky_relu_alpha);
    return p + n;
}

Mat Tanh(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	int tmp = src.getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_tanh<<<num_blocks, block_size>>>(src.devData, dst.devData, tmp);
	dst.deviceToHost();
	return dst;
}

Mat dTanh(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat res(src);
    res.ones();
    res = res - pow(src, 2);
    return res;
}

Mat nonLinearity(const Mat &M, int method){
	if(NULL == M.hostData || NULL == M.devData){
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

Mat dnonLinearity(const Mat &M, int method){
	if(NULL == M.hostData || NULL == M.devData){
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
Mat fliplr(const Mat &src){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(src);
	int tmp = src.rows * src.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_fliplr<<<num_blocks, block_size>>>(src.devData + i * tmp, dst.devData + i * tmp, src.rows, src.cols, tmp);
	}
	dst.deviceToHost();
	return dst;
}

Mat rot90(const Mat &src, int k){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    Mat res(src);
    if(0 == k) return res;
    elif(1 == k) {
    	res = res.t();
    	res = fliplr(res);
    }else{
    	res = rot90(res, k - 1);
    	res = res.t();
    	res = fliplr(res);
    }
    return res;
}

Mat dopadding(const Mat &src, int pad){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(pad <= 0) return src;
	Mat dst(src.rows + pad * 2, src.cols + pad * 2, src.channels);
	int tmp1 = src.rows * src.cols;
	int tmp2 = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp1 / block_size) + ((tmp1 % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_padding<<<num_blocks, block_size>>>(src.devData + i * tmp1, dst.devData + i * tmp2, src.rows, src.cols, dst.rows, tmp1);
	}
	dst.deviceToHost();
	return dst;
}

Mat depadding(const Mat &src, int pad){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(pad <= 0) return src;
	Mat dst(src.rows - pad * 2, src.cols - pad * 2, src.channels);
	int tmp1 = src.rows * src.cols;
	int tmp2 = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp2 / block_size) + ((tmp2 % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_depadding<<<num_blocks, block_size>>>(src.devData + i * tmp1, dst.devData + i * tmp2, src.rows, src.cols, dst.rows, tmp2);
	}
	dst.deviceToHost();
	return dst;
}

Mat reduce(const Mat& src, int direction, int mode){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst;
	if(REDUCE_TO_SINGLE_ROW == direction){
		dst.setSize(1, src.cols, src.channels);
		for(int i = 0; i < src.cols; ++i){
			Mat tmp = getRange(src, i, i, 0, src.rows - 1);
			if(REDUCE_SUM == mode){
				dst.set(i, sum(tmp));
			}elif(REDUCE_MAX == mode){
				dst.set(i, max(tmp));
			}
		}
	}else{ // REDUCE_TO_SINGLE_COL == direction
		dst.setSize(src.rows, 1, src.channels);
		for(int i = 0; i < src.rows; ++i){
			Mat tmp = getRange(src, 0, src.cols - 1, i, i);
			if(REDUCE_SUM == mode){
				dst.set(i, sum(tmp));
			}elif(REDUCE_MAX == mode){
				dst.set(i, max(tmp));
			}
		}
	}
	return dst;
}

Mat interpolation(const Mat& src, int _size){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    int stride = _size / src.rows;
    if(_size % src.rows > 0) ++ stride;
    if(stride == 0 || stride == 1) {Mat dst(src); return dst;}
    Mat dst(_size, _size, src.channels);
	int tmp1 = src.rows * src.cols;
	int tmp2 = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp1 / block_size) + ((tmp1 % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_interpolation<<<num_blocks, block_size>>>(src.devData + i * tmp1, dst.devData + i * tmp2, src.rows, dst.rows, stride, tmp1);
	}
	dst.deviceToHost();
	return dst;
}

Mat repmat(const Mat &src, int vert, int hori){
	if(NULL == src.hostData || NULL == src.devData || 0 == vert || 0 == hori){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(src.rows * vert, src.cols * hori, src.channels);
	if(1 == vert && 1 == hori) {dst = src; return dst;}
	int tmp1 = src.rows * src.cols;
	int tmp2 = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp2 / block_size) + ((tmp2 % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_repmat<<<num_blocks, block_size>>>(src.devData + i * tmp1, dst.devData + i * tmp2, src.rows, src.cols, dst.rows, dst.cols, tmp2);
	}
	dst.deviceToHost();
	return dst;
}

Mat kron(const Mat &a, const Mat &b){
	if(NULL == a.hostData || NULL == a.devData || NULL == b.hostData || NULL == b.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat dst(a.rows * b.rows, a.cols * b.cols, a.channels);
	int tmp1 = a.rows * a.cols;
	int tmp2 = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp2 / block_size) + ((tmp2 % block_size) ? 1 : 0);
	for(int i = 0; i < a.channels; ++i){
		cu_kron<<<num_blocks, block_size>>>(a.devData + i * tmp1, b.devData, dst.devData + i * tmp2, a.rows, a.cols, dst.rows, dst.cols, tmp2);
	}
	dst.deviceToHost();
	return dst;
}

Mat conv2(const Mat &m, const Mat &kernel){
	if(NULL == m.hostData || NULL == m.devData || NULL == kernel.hostData || NULL == kernel.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat res(m.rows, m.cols, m.channels);
    float *d_Data, *d_Kernel, *d_PaddedData, *d_PaddedKernel;
    fComplex *d_DataSpectrum0, *d_KernelSpectrum0;
    cufftHandle fftPlan;
    float *host_result_tmp, *host_result;

    const int kernelH = kernel.rows;
    const int kernelW = kernel.cols;
    const int kernelY = kernel.rows / 2;
    const int kernelX = kernel.cols / 2;
    const int dataH = m.rows;
    const int dataW = m.cols;
    const int fftH = snapTransformSize(dataH + kernelH - 1);
    const int fftW = snapTransformSize(dataW + kernelW - 1);
    host_result = (float *)malloc(dataH * dataW * sizeof(float));
    host_result_tmp = (float *)malloc(fftH * fftW * sizeof(float));
    cudaMalloc((void **)&d_Data,   dataH   * dataW   * sizeof(float));
    cudaMalloc((void **)&d_Kernel, kernelH * kernelW * sizeof(float));
    cudaMalloc((void **)&d_PaddedData,   fftH * fftW * sizeof(float));
    cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float));
    cudaMalloc((void **)&d_DataSpectrum0,   fftH * (fftW / 2) * sizeof(fComplex));
    cudaMalloc((void **)&d_KernelSpectrum0, fftH * (fftW / 2) * sizeof(fComplex));
    // std::cout<<"...creating C2C FFT plan for "<<fftH<<" x "<<fftW/2<<std::endl;
    cufftPlan2d(&fftPlan, fftH, fftW / 2, CUFFT_C2C);
    for(int i = 0; i < m.channels; ++i){
        cudaMemcpy(d_Data, m.devData + m.rows * m.cols * i, dataH * dataW * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_Kernel, kernel.devData + kernel.rows * kernel.cols * i, kernelH * kernelW * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(d_PaddedData,   0, fftH * fftW * sizeof(float));
        cudaMemset(d_PaddedKernel, 0, fftH * fftW * sizeof(float));
        padDataClampToBorder(d_PaddedData, d_Data, fftH, fftW,
        					dataH, dataW, kernelH, kernelW, kernelY, kernelX);
        padKernel(d_PaddedKernel, d_Kernel, fftH, fftW,
            kernelH, kernelW, kernelY, kernelX);
        //CUFFT_INVERSE works just as well...
        const int FFT_DIR = CUFFT_FORWARD;
        //Not including kernel transformation into time measurement,
        //since convolution kernel is not changed very frequently
        // std::cout<<"...transforming convolution kernel"<<std::endl;
        cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum0, FFT_DIR);
        // std::cout<<"...running GPU FFT convolution: "<<std::endl;
        cudaDeviceSynchronize();
        cufftExecC2C(fftPlan, (cufftComplex *)d_PaddedData, (cufftComplex *)d_DataSpectrum0, FFT_DIR);
        spProcess2D(d_DataSpectrum0, d_DataSpectrum0, d_KernelSpectrum0, fftH, fftW / 2, FFT_DIR);
        cufftExecC2C(fftPlan, (cufftComplex *)d_DataSpectrum0, (cufftComplex *)d_PaddedData, -FFT_DIR);
        cudaDeviceSynchronize();
        // std::cout<<"...reading back GPU FFT results"<<std::endl;
        cudaMemcpy(host_result_tmp, d_PaddedData, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost);
        for(int y = 0; y < dataH; y++){
            for(int x = 0; x < dataW; x++){
                host_result[y * dataW + x] = host_result_tmp[y * fftW  + x];
            }
        }
        memcpy(res.hostData + i * res.rows * res.cols, host_result, res.rows * res.cols * sizeof(float));
    }
    res.hostToDevice();
    cudaFree(d_KernelSpectrum0);
    cudaFree(d_DataSpectrum0);
    cudaFree(d_PaddedKernel);
    cudaFree(d_PaddedData);
    cudaFree(d_Kernel);
    cudaFree(d_Data);
    free(host_result);
    free(host_result_tmp);
    return res;
}

Mat conv2(const Mat &m, const Mat &kernel, int convtype, int pad, int stride){
	if(NULL == m.hostData || NULL == m.devData || NULL == kernel.hostData || NULL == kernel.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat src = dopadding(m, kernel.cols - 1);
	src = dopadding(src, pad);
	Mat res = conv2(src, kernel);
	res = getRange(res, (res.cols - (m.cols + kernel.cols - 1)) / 2, (res.cols - (m.cols + kernel.cols - 1)) / 2 + m.cols + kernel.cols - 1 - 1,
						(res.rows - (m.rows + kernel.rows - 1)) / 2, (res.rows - (m.rows + kernel.rows - 1)) / 2 + m.rows + kernel.rows - 1 - 1);
	if(CONV_SAME == convtype){
		res = getRange(res, kernel.cols / 2, res.cols - 1 - kernel.cols / 2, kernel.rows / 2, res.rows - 1 - kernel.rows / 2);
	}
	if(CONV_VALID == convtype){
        int tmpx = m.cols + pad * 2 - kernel.cols + 1;
        int tmpy = m.rows + pad * 2 - kernel.rows + 1;
        res = getRange(res, (res.cols - tmpx) / 2, res.cols - 1 - (res.cols - tmpx) / 2,
        		 	 	    (res.rows - tmpy) / 2, res.rows - 1 - (res.rows - tmpy) / 2);
	}
	res = downSample(res, stride, stride);
	return res;
}

Mat getRange(const Mat& src, int xstart, int xend, int ystart, int yend){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(xstart < 0 || xstart > xend || xend >= src.cols ||
	   ystart < 0 || ystart > yend || yend >= src.rows){
		std::cout<<"invalid range..."<<std::endl;
		exit(0);
	}
	Mat dst(yend - ystart + 1, xend - xstart + 1, src.channels);
	int len = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_getRange<<<num_blocks, block_size>>>(src.devData + i * src.rows * src.cols, dst.devData + i * len, xstart, xend, ystart, yend, src.rows, len);
	}
	dst.deviceToHost();
	return dst;
}

Mat downSample(const Mat& src, int y_stride, int x_stride){
	if(NULL == src.hostData || NULL == src.devData || y_stride < 1 || x_stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(1 == y_stride && 1 == x_stride){
		Mat res(src);
		return res;
	}
	int dst_rows = src.rows / y_stride;
	if(src.rows % y_stride > 0) ++dst_rows;
	int dst_cols = src.cols / x_stride;
	if(src.cols % x_stride > 0) ++dst_cols;
	Mat res(dst_rows, dst_cols, src.channels);
	int len = res.rows * res.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_downSample<<<num_blocks, block_size>>>(src.devData + i * src.rows * src.cols, res.devData + i * len, y_stride, x_stride, src.rows, len);
	}
	res.deviceToHost();
	return res;
}

Mat copyMakeBorder(const Mat& src, int up, int down, int left, int right, const vector3f& val){
	if(NULL == src.hostData || NULL == src.devData){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	if(0 == up && 0 == down && 0 == left && 0 == right){
		Mat dst(src);
		return dst;
	}
	Mat dst(src.rows + up + down, src.cols + left + right, src.channels);
	dst.setAll(val);
	int tmp1 = src.rows * src.cols;
	int tmp2 = dst.rows * dst.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp1 / block_size) + ((tmp1 % block_size) ? 1 : 0);
	for(int i = 0; i < src.channels; ++i){
		cu_copyMakeBorder<<<num_blocks, block_size>>>(src.devData + i * tmp1, dst.devData + i * tmp2, src.rows, src.cols, up, down, left, right, tmp1);
	}
	dst.deviceToHost();
	return dst;
}

// Pooling with overlap
// Max pooling and stochastic pooling supported
// output size = (input size - window size) / stride + 1
Mat pooling_with_overlap(const Mat &src, vector2i window_size, int stride, int poolingMethod, std::vector<vector3f> &locat){
	if(NULL == src.hostData || NULL == src.devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
	Mat tmpres(src.rows - window_size.get(1) + 1, src.cols - window_size.get(0) + 1, src.channels);
	std::vector<vector3f> tmplocat;
    for(int i = 0; i < tmpres.rows; ++i){
        for(int j = 0; j < tmpres.cols; ++j){
        	Mat tmp = getRange(src, j, j + window_size.get(0) - 1, i, i + window_size.get(1) - 1);
        	vector3f val;
        	vector3f loc;
        	if(POOL_MAX == poolingMethod){
        		max(tmp, val, loc);
        	}
        	vector3f tmpr = loc % window_size.get(1);
        	vector3f tmpc = loc.divNoRem(window_size.get(1));
        	tmpr = tmpr + i;
        	tmpc = tmpc + j;
        	loc = tmpc * src.rows + tmpr;
            tmplocat.push_back(loc);
            tmpres.set(i, j, val);
        }
    }
    Mat dst = downSample(tmpres, stride, stride);
    for(int i = 0; i < tmpres.cols; i++){
        for(int j = 0; j < tmpres.rows; j++){
            if(i % stride > 0 || j % stride > 0) continue;
            locat.push_back(tmplocat[i * tmpres.rows + j]);
        }
    }
    tmplocat.clear();
    std::vector<vector3f>().swap(tmplocat);
    return dst;
}

// Max pooling and stochastic pooling supported
Mat unpooling_with_overlap(const Mat &src, vector2i window_size, int stride, int poolingMethod, std::vector<vector3f> &locat, vector2i& up_size){
	if(NULL == src.hostData || NULL == src.devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(1 == window_size.get(0) && 1 == window_size.get(1) && 1 == stride){
        Mat res(src);
        return res;
    }
    Mat res(up_size.get(1), up_size.get(0), src.channels);
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
        	for(int ch = 0; ch < src.channels; ++ch){
            	res.set(locat[i * src.cols + j].get(ch), ch, src.get(i, j, ch));
        	}
        }
    }
    return res;
}

Mat pooling(const Mat& src, int stride, int poolingMethod, std::vector<vector3f> &locat){
	if(NULL == src.hostData || NULL == src.devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(stride == 1){
    	for(int i = 0; i < src.rows * src.cols; ++i){
    		vector3f tmp(i, i, i);
    		locat.push_back(tmp);
    	}
        Mat res(src);
        return res;
    }
	int dst_rows = src.rows / stride;
	if(src.rows % stride > 0) ++dst_rows;
	int dst_cols = src.cols / stride;
	if(src.cols % stride > 0) ++dst_cols;
	Mat res(dst_rows, dst_cols, src.channels);
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
        	Mat tmp = getRange(src, j * stride, j * stride + stride  - 1, i * stride, i * stride + stride - 1);
        	vector3f val;
        	vector3f loc;
        	if(POOL_MAX == poolingMethod){
        		// max poling
        		max(tmp, val, loc);
        	}elif(POOL_MEAN == poolingMethod){
        		// Mean Pooling
        		val = average(src);
        		loc.setAll(0.0);
            }
        	vector3f tmpr = loc % stride;
        	vector3f tmpc = loc.divNoRem(stride);
        	tmpr = tmpr + i * stride;
        	tmpc = tmpc + j * stride;
        	loc = tmpc * src.rows + tmpr;
            locat.push_back(loc);
            res.set(i, j, val);
        }
    }
    return res;
}

Mat unpooling(const Mat& src, int stride, int poolingMethod, std::vector<vector3f>& locat, vector2i& up_size){
	if(NULL == src.hostData || NULL == src.devData || stride < 1){
		std::cout<<"invalid input..."<<std::endl;
		exit(0);
	}
    if(stride == 1){
        Mat res(src);
        return res;
    }
    if(POOL_MEAN == poolingMethod){
    	Mat one(stride, stride, src.channels);
    	one.ones();
        Mat res = kron(src, one);
        res = divide(res, stride * stride);
        vector3f tmp(0.0, 0.0, 0.0);
        res = copyMakeBorder(res, 0, up_size.get(1) - res.rows, 0, up_size.get(0) - res.cols, tmp);
        return res;
    }else{ //(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod)
        Mat res(up_size.get(1), up_size.get(0), src.channels);
        for(int i = 0; i < src.rows; i++){
            for(int j = 0; j < src.cols; j++){
            	for(int ch = 0; ch < src.channels; ++ch){
                	res.set(locat[i * src.cols + j].get(ch), ch, src.get(i, j, ch));
            	}
            }
        }
        return res;
    }
}







