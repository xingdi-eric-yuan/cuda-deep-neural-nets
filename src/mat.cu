#include "data_structure.h"

using namespace std;

Mat::Mat(){
	rows = 0;
	cols = 0;
	channels = 0;
	hostData = NULL;
	devData = NULL;
}
Mat::Mat(const Mat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	hostData = NULL;
	devData = NULL;
	mallocHost();
	mallocDevice();
	memcpy(hostData, m.hostData, getLength() * sizeof(float));
	cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
}
Mat::Mat(int height, int width, int nchannels){
	cols = width;
	rows = height;
	channels = nchannels;
	hostData = NULL;
	devData = NULL;
	mallocHost();
	mallocDevice();
	zeros();
}
Mat::~Mat(){
	if(NULL != hostData)
		MemoryMonitor::instance()->freeCpuMemory(hostData);
	if(NULL != devData)
		MemoryMonitor::instance()->freeGpuMemory(devData);
}

Mat& Mat::operator=(const Mat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	hostData = NULL;
	devData = NULL;
	mallocHost();
	mallocDevice();
	memcpy(hostData, m.hostData, getLength() * sizeof(float));
	cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
    return *this;
}

void Mat::zeros(){
	setAll(0.0);
}

void Mat::ones(){
	setAll(1.0);
}

void Mat::randn(){
	if(NULL == hostData) mallocHost();
	if(NULL == devData) mallocDevice();
	curandGenerator_t gen;
	// Create pseudo-random number generator
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	// Set seed
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	// Generate n floats on device
	curandGenerateUniform(gen, devData, getLength());
	// Cleanup generator
	curandDestroyGenerator(gen);
	deviceToHost();
}

void Mat::set(int pos_y, int pos_x, int pos_channel, float val){
	if(NULL == hostData || NULL == devData) {zeros();}
	if(pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	hostData[IDX2C(pos_y, pos_x, cols) + pos_channel * (rows * cols)] = val;
	hostToDevice();
}

void Mat::setAll(float val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_setAll<<<num_blocks, block_size>>>(devData, val, len);
	deviceToHost();
}

void Mat::setAll(const vector3f &v){
	if(channels != 3){
		std::cout<<"this is not a 3 channel mat..."<<std::endl;
		exit(0);
	}
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_setAll<<<num_blocks, block_size>>>(devData + i * len, v.get(i), len);
	}
	deviceToHost();
}

float Mat::get(int pos_y, int pos_x, int pos_channel) const{
	if(NULL == hostData || NULL == devData||
	   pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	return hostData[IDX2C(pos_y, pos_x, cols) + pos_channel * (rows * cols)];
}

int Mat::getLength() const{
	return rows * cols * channels;
}

void Mat::deviceToHost(){
	if(NULL == hostData) mallocHost();
	if(NULL == devData) mallocDevice();
	// Copy device memory to host
	cudaMemcpy(hostData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToHost);
}

void Mat::hostToDevice(){
	if(NULL == hostData) mallocHost();
	if(NULL == devData) mallocDevice();
	// Copy host memory to device
	cudaMemcpy(devData, hostData, getLength() * sizeof(float), cudaMemcpyHostToDevice);
}

void Mat::copyTo(Mat &m){
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	m.hostData = NULL;
	m.devData = NULL;
	m.mallocHost();
	m.mallocDevice();
	memcpy(m.hostData, hostData, getLength() * sizeof(float));
	cudaMemcpy(m.devData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
}

// only changes devData (on GPU)
Mat Mat::operator+(const Mat &m){
	if(NULL == hostData || NULL == devData ||
	   NULL == m.hostData || NULL == m.devData||
	   getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	Mat tmpmat(m);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSetVector (n, sizeof (float), hostData, 1, devData, 1); // cp x- >d_x
	cublasSetVector (n, sizeof (float), tmpmat.hostData, 1, tmpmat.devData, 1); // cp y- >d_y
	float alpha = 1.0;
	// multiply the vector d_x by the scalar alpha and add to d_y
	cublasSaxpy(handle, n, &alpha, devData, 1, tmpmat.devData, 1);
	cublasGetVector (n, sizeof (float), tmpmat.devData, 1, tmpmat.hostData, 1); // cp d_y - >y
	cublasDestroy ( handle ); // destroy CUBLAS context
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator+(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(tmpmat.devData, val, len);
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator+(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	if(channels != 3){
		std::cout<<"this is not a 3 channel mat..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		float tmp = v.get(i);
		cu_plus<<<num_blocks, block_size>>>(tmpmat.devData + i * len, tmp, len);
	}
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator-(const Mat &m){

	if(NULL == hostData || NULL == devData ||
	   NULL == m.hostData || NULL == m.devData||
	   getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSetVector (n, sizeof (float), m.hostData, 1, m.devData, 1); // cp x- >d_x
	cublasSetVector (n, sizeof (float), tmpmat.hostData, 1, tmpmat.devData, 1); // cp y- >d_y
	float alpha = -1.0;
	// multiply the vector d_x by the scalar alpha and add to d_y
	cublasSaxpy(handle, n, &alpha, m.devData, 1, tmpmat.devData, 1);
	cublasGetVector (n, sizeof (float) ,tmpmat.devData, 1, tmpmat.hostData, 1); // cp d_y - >y
	cublasDestroy ( handle ); // destroy CUBLAS context
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator-(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(tmpmat.devData, val, tmp);
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator-(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	if(channels != 3){
		std::cout<<"this is not a 3 channel mat..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_minus<<<num_blocks, block_size>>>(tmpmat.devData + i * len, v.get(i), len);
	}
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator*(const Mat &m){
	if(NULL == hostData || NULL == devData ||
	   NULL == m.hostData || NULL == m.devData||
	   cols != m.rows || channels != m.channels){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat(rows, m.cols, channels);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	float alpha = 1.0;
	float beta = 1.0;
	for(int i = 0; i < channels; ++i){
		cublasSetMatrix (rows, cols, sizeof(float), hostData + i * (rows * cols), rows, devData + i * (rows * cols), rows); // cp x- >d_x
		cublasSetMatrix (m.rows, m.cols, sizeof(float), m.hostData + i * (m.rows * m.cols), m.rows, m.devData + i * (m.rows * m.cols), m.rows); // cp y- >d_y
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, m.cols, cols, &alpha, devData + i * (rows * cols), rows, m.devData + i * (m.rows * m.cols), cols, &beta, tmpmat.devData + i * (tmpmat.rows * tmpmat.cols), rows);
		cublasGetMatrix (rows, m.cols, sizeof(float), tmpmat.devData + i * (tmpmat.rows * tmpmat.cols), rows, tmpmat.hostData + i * (tmpmat.rows * tmpmat.cols), rows);
	}
	cublasDestroy (handle); // destroy CUBLAS context
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::operator*(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSscal(handle, n, &val, tmpmat.devData, 1);
	tmpmat.deviceToHost();
    cublasDestroy(handle);
	return tmpmat;
}

Mat Mat::operator*(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	if(channels != 3){
		std::cout<<"this is not a 3 channel mat..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < channels; ++i){
		float tmp = v.get(i);
		cublasSscal(handle, rows * cols, &tmp, tmpmat.devData + i * rows * cols, 1);
	}
	tmpmat.deviceToHost();
    cublasDestroy(handle);
	return tmpmat;
}

Mat Mat::mul(const Mat &m){
	if(NULL == hostData || NULL == devData ||
	   NULL == m.hostData || NULL == m.devData||
	   getLength()!= m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData, m.devData, tmpmat.devData, tmp);
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::mul(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSscal(handle, n, &val, tmpmat.devData, 1);
	tmpmat.deviceToHost();
    cublasDestroy(handle);
	return tmpmat;
}

Mat Mat::mul(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	if(channels != 3){
		std::cout<<"this is not a 3 channel mat..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < channels; ++i){
		float tmp = v.get(i);
		cublasSscal(handle, rows * cols, &tmp, tmpmat.devData + i * rows * cols, 1);
	}
	tmpmat.deviceToHost();
    cublasDestroy(handle);
	return tmpmat;
}

Mat Mat::t(){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	//tmpmat.zeros();
    float const alpha(1.0);
    float const beta(0.0);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < channels; ++i){
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &alpha, devData + i * (rows * cols), rows, &beta, devData + i * (rows * cols), cols, tmpmat.devData + i * (rows * cols), cols);
	}
	int tmp = tmpmat.rows;
	tmpmat.rows = tmpmat.cols;
	tmpmat.cols = tmp;
	cublasDestroy(handle);
	tmpmat.deviceToHost();
	return tmpmat;
}

// memory
void Mat::mallocHost(){
	if(NULL == hostData){
		// malloc host data
		hostData = (float*)MemoryMonitor::instance()->cpuMalloc(cols * rows * channels * sizeof(float));
		if(NULL == hostData) {
			std::cout<<"host memory allocation failed..."<<std::endl;
			exit(0);
		}
		memset(hostData, 0, cols * rows * channels * sizeof(float));
	}
}

void Mat::mallocDevice(){
	if(NULL == devData){
		cudaError_t cudaStat;
		// malloc device data
		cudaStat = MemoryMonitor::instance()->gpuMalloc((void**)&devData, cols * rows * channels * sizeof(float));
		if(cudaStat != cudaSuccess) {
			std::cout<<"device memory allocation failed..."<<std::endl;
			exit(0);
		}
		cudaStat = cudaMemset(devData, 0, sizeof(float) * cols * rows * channels);
		if(cudaStat != cudaSuccess) {
			std::cout<<"device memory cudaMemset failed..."<<std::endl;
			exit(0);
		}
	}
}

void Mat::printHost(const std::string &str){
	std::cout<<str<<std::endl;
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid matrix..."<<std::endl;
		exit(0);
	}
	Mat show = t();
	int counter = 0;
	std::cout<<"Matrix with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<"columns."<<std::endl;
	for(int i = 0; i < channels; ++i){
		std::cout<<"Channel "<<i<<" : "<<std::endl;
		for(int j = 0; j < rows; ++j){
			for(int k = 0; k < cols; ++k){
				std::cout<<show.hostData[counter]<<" ";
				++ counter;
			}
			std::cout<<std::endl;
		}
	}
}

void Mat::printDevice(const std::string &str){
	std::cout<<str<<std::endl;
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid matrix..."<<std::endl;
		exit(0);
	}
	Mat show = t();
	float *host_data = 0;
	host_data = (float*)MemoryMonitor::instance()->cpuMalloc(show.cols * show.rows * show.channels * sizeof(float));
	cudaMemcpy(host_data, show.devData, show.cols * show.rows * show.channels * sizeof(float), cudaMemcpyDeviceToHost);
	int counter = 0;
	std::cout<<"Matrix with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<"columns."<<std::endl;
	for(int i = 0; i < channels; ++i){
		std::cout<<"Channel "<<i<<" : "<<std::endl;
		for(int j = 0; j < rows; ++j){
			for(int k = 0; k < cols; ++k){
				std::cout<<host_data[counter]<<" ";
				++ counter;
			}
			std::cout<<std::endl;
		}
	}
	if(NULL != host_data)
		MemoryMonitor::instance()->freeCpuMemory(host_data);
}
