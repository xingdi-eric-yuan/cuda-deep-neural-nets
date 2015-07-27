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

Mat::Mat(const cpuMat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	hostData = NULL;
	devData = NULL;
	mallocHost();
	mallocDevice();
	memcpy(hostData, m.Data, getLength() * sizeof(float));
	cudaMemcpy(devData, m.Data, getLength() * sizeof(float), cudaMemcpyHostToDevice);
}

Mat::Mat(int height, int width, int nchannels){
	cols = width;
	rows = height;
	channels = nchannels;
	hostData = NULL;
	devData = NULL;
	mallocHost();
	mallocDevice();
	//zeros();
}

Mat::~Mat(){
	if(NULL != hostData)
		MemoryMonitor::instance()->freeCpuMemory(hostData);
	if(NULL != devData)
		MemoryMonitor::instance()->freeGpuMemory(devData);
	rows = 0;
	cols = 0;
	channels = 0;
	hostData = NULL;
	devData = NULL;
}

void Mat::release(){
	if(NULL != hostData)
		MemoryMonitor::instance()->freeCpuMemory(hostData);
	if(NULL != devData)
		MemoryMonitor::instance()->freeGpuMemory(devData);
	rows = 0;
	cols = 0;
	channels = 0;
	hostData = NULL;
	devData = NULL;
}

Mat& Mat::operator=(const Mat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	if(NULL != hostData){
		MemoryMonitor::instance()->freeCpuMemory(hostData);
		hostData = NULL;
	}
	if(NULL != devData){
		MemoryMonitor::instance()->freeGpuMemory(devData);
		devData = NULL;
	}
	mallocHost();
	mallocDevice();
	memcpy(hostData, m.hostData, getLength() * sizeof(float));
	cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
    return *this;
}

Mat& Mat::operator<<=(Mat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	if(NULL != hostData){
		MemoryMonitor::instance()->freeCpuMemory(hostData);
		hostData = NULL;
	}
	if(NULL != devData){
		MemoryMonitor::instance()->freeGpuMemory(devData);
		devData = NULL;
	}
	mallocHost();
	mallocDevice();
	memcpy(hostData, m.hostData, getLength() * sizeof(float));
	cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
	m.release();
    return *this;
}

void Mat::setSize(int r, int c, int ch){
	rows = r;
	cols = c;
	channels = ch;
	if(NULL != hostData){
		MemoryMonitor::instance()->freeCpuMemory(hostData);
		hostData = NULL;
	}
	if(NULL != devData){
		MemoryMonitor::instance()->freeGpuMemory(devData);
		devData = NULL;
	}
	mallocHost();
	mallocDevice();
	//zeros();
}

void Mat::zeros(){
	setAll(0.0);
}

void Mat::ones(){
	setAll(1.0);
}

void Mat::randu(){
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
	for(int i = 0; i < getLength(); ++i){
		hostData[i] = hostData[i] * 2.0 - 1.0;
	}
	hostToDevice();
}

void Mat::set(int pos_y, int pos_x, int pos_channel, float val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	if(pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	hostData[IDX2C(pos_y, pos_x, rows) + pos_channel * (rows * cols)] = val;
	hostToDevice();
}

void Mat::set(int pos_y, int pos_x, const vector3f& val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	if(pos_x >= cols || pos_y >= rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < channels; ++i){
		set(pos_y, pos_x, i, val.get(i));
	}
	hostToDevice();
}

void Mat::set(int pos, const vector3f& val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	if(pos >= cols * rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < channels; ++i){
		hostData[pos + i * (rows * cols)] = val.get(i);
	}
	hostToDevice();
}

void Mat::set(int pos, int pos_channel, float val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	if(pos >= cols * rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	hostData[pos + pos_channel * (rows * cols)] = val;
	hostToDevice();
}

void Mat::set(int pos, float val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	if(pos >= getLength()){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	hostData[pos] = val;
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
	return hostData[IDX2C(pos_y, pos_x, rows) + pos_channel * (rows * cols)];
}

vector3f Mat::get(int pos_y, int pos_x) const{
	if(NULL == hostData || NULL == devData||
	   pos_x >= cols || pos_y >= rows || channels < 3){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	vector3f res;
	for(int i = 0; i < 3; ++i){
		res.set(i, hostData[IDX2C(pos_y, pos_x, rows) + i * (rows * cols)]);
	}
	return res;
}

int Mat::getLength() const{
	return rows * cols * channels;
}

void Mat::deviceToHost(){
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"can't do that, host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"can't do that, device data is NULL..."<<std::endl;
		exit(0);
	}
	// Copy device memory to host
	cudaMemcpy(hostData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToHost);
}

void Mat::hostToDevice(){
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"can't do that, host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"can't do that, device data is NULL..."<<std::endl;
		exit(0);
	}
	// Copy host memory to device
	cudaMemcpy(devData, hostData, getLength() * sizeof(float), cudaMemcpyHostToDevice);
}

void Mat::copyTo(Mat &m) const{
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.hostData){
		MemoryMonitor::instance()->freeCpuMemory(m.hostData);
		m.hostData = NULL;
	}
	if(NULL != m.devData){
		MemoryMonitor::instance()->freeGpuMemory(m.devData);
		m.devData = NULL;
	}
	m.mallocHost();
	m.mallocDevice();
	memcpy(m.hostData, hostData, getLength() * sizeof(float));
	cudaMemcpy(m.devData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
}

void Mat::copyTo(cpuMat &m) const{
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeCpuMemory(m.Data);
		m.Data = NULL;
	}
	m.mallocMat();
	memcpy(m.Data, hostData, getLength() * sizeof(float));
}

void Mat::moveTo(Mat &m){
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.hostData){
		MemoryMonitor::instance()->freeCpuMemory(m.hostData);
		m.hostData = NULL;
	}
	if(NULL != m.devData){
		MemoryMonitor::instance()->freeGpuMemory(m.devData);
		m.devData = NULL;
	}
	m.mallocHost();
	m.mallocDevice();
	memcpy(m.hostData, hostData, getLength() * sizeof(float));
	cudaMemcpy(m.devData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice);
	release();
}

void Mat::moveTo(cpuMat &m){
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeCpuMemory(m.Data);
		m.Data = NULL;
	}
	m.mallocMat();
	memcpy(m.Data, hostData, getLength() * sizeof(float));
	release();
}

Mat Mat::operator+(const Mat &m) const{
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

Mat Mat::operator+(float val) const{
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

Mat Mat::operator+(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
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

Mat& Mat::operator+=(const Mat &m){
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
	cublasSaxpy(handle, n, &alpha, tmpmat.devData, 1, devData, 1);
	cublasGetVector (n, sizeof (float), devData, 1, hostData, 1); // cp d_y - >y
	cublasDestroy ( handle ); // destroy CUBLAS context
	deviceToHost();
	tmpmat.release();
    return *this;
}

Mat& Mat::operator+=(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(devData, val, len);
	deviceToHost();
	return *this;
}

Mat& Mat::operator+=(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_plus<<<num_blocks, block_size>>>(devData + i * len, v.get(i), len);
	}
	deviceToHost();
	return *this;
}

Mat Mat::operator-(const Mat &m) const{

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

Mat Mat::operator-(float val) const{
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

Mat Mat::operator-(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
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

Mat& Mat::operator-=(const Mat &m){
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
	float alpha = -1.0;
	// multiply the vector d_x by the scalar alpha and add to d_y
	cublasSaxpy(handle, n, &alpha, tmpmat.devData, 1, devData, 1);
	cublasGetVector (n, sizeof (float), devData, 1, hostData, 1); // cp d_y - >y
	cublasDestroy ( handle ); // destroy CUBLAS context
	deviceToHost();
	tmpmat.release();
    return *this;
}

Mat& Mat::operator-=(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(devData, val, len);
	deviceToHost();
	return *this;
}

Mat& Mat::operator-=(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_minus<<<num_blocks, block_size>>>(devData + i * len, v.get(i), len);
	}
	deviceToHost();
	return *this;
}

Mat Mat::operator*(const Mat &m) const{
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

Mat Mat::operator*(float val) const{
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

Mat Mat::operator*(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
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

Mat& Mat::operator*=(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSscal(handle, n, &val, devData, 1);
	deviceToHost();
    cublasDestroy(handle);
	return *this;
}

Mat& Mat::operator*=(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < channels; ++i){
		float tmp = v.get(i);
		cublasSscal(handle, rows * cols, &tmp, devData + i * rows * cols, 1);
	}
	deviceToHost();
    cublasDestroy(handle);
	return *this;
}

Mat Mat::operator/(float val) const{
	if(NULL == hostData || NULL == devData || 0 == val){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	float tmp = 1 / val;
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSscal(handle, n, &tmp, tmpmat.devData, 1);
	tmpmat.deviceToHost();
    cublasDestroy(handle);
	return tmpmat;
}

Mat Mat::operator/(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < channels; ++i){
		float tmp = 1 / v.get(i);
		cublasSscal(handle, rows * cols, &tmp, tmpmat.devData + i * rows * cols, 1);
	}
	tmpmat.deviceToHost();
    cublasDestroy(handle);
	return tmpmat;
}

Mat& Mat::operator/=(float val){
	if(NULL == hostData || NULL == devData || 0 == val){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int n = getLength();
	float tmp = 1 / val;
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	cublasSscal(handle, n, &tmp, devData, 1);
	deviceToHost();
    cublasDestroy(handle);
	return *this;
}

Mat& Mat::operator/=(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	cublasHandle_t handle; // CUBLAS context
	cublasCreate (&handle); // initialize CUBLAS context
	for(int i = 0; i < channels; ++i){
		float tmp = 1 / v.get(i);
		cublasSscal(handle, rows * cols, &tmp, devData + i * rows * cols, 1);
	}
	deviceToHost();
    cublasDestroy(handle);
	return *this;
}

Mat Mat::mul(const Mat &m) const{
	if(NULL == hostData || NULL == devData ||
	   NULL == m.hostData || NULL == m.devData||
	   getLength()!= m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat(m);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(tmpmat.devData, devData, tmp);
	tmpmat.deviceToHost();
	return tmpmat;
}

Mat Mat::mul(float val) const{
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

Mat Mat::mul(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
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

Mat Mat::t() const{
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
			std::cout<<"device memory allocation failed... because of error number : "<<(int)cudaStat<<std::endl;
			exit(0);
		}
		cudaStat = cudaMemset(devData, 0, sizeof(float) * cols * rows * channels);
		if(cudaStat != cudaSuccess) {
			std::cout<<"device memory cudaMemset failed... because of error number : "<<(int)cudaStat<<std::endl;
			exit(0);
		}
	}
}

void Mat::printHost(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"device data is NULL..."<<std::endl;
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
	show.release();
}

void Mat::printDevice(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"device data is NULL..."<<std::endl;
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
	if(NULL != host_data) MemoryMonitor::instance()->freeCpuMemory(host_data);
	show.release();
}
