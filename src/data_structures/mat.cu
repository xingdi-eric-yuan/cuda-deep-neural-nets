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
	checkCudaErrors(cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
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
	checkCudaErrors(cudaMemcpy(devData, m.Data, getLength() * sizeof(float), cudaMemcpyHostToDevice));
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
	checkCudaErrors(cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
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
	checkCudaErrors(cudaMemcpy(devData, m.devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
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
	checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	// Set seed
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	// Generate n floats on device
	checkCudaErrors(curandGenerateUniform(gen, devData, getLength()));
	// Cleanup generator
	checkCudaErrors(curandDestroyGenerator(gen));
	deviceToHost();
}

void Mat::randn(){

	if(NULL == hostData) mallocHost();
	if(NULL == devData) mallocDevice();
	int len = cols * rows;
	curandGenerator_t gen;
	// Create pseudo-random number generator
	checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	for(int ch = 0; ch < channels; ++ch){
		// Set seed
		checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, ch * 1000ULL));
		// Generate n floats on device
		if(len % 2){
			// In general, the normal generating functions (e.g. curandGenerateNormal, curandGenerateLogNormal, etc.)
			// require the number of requested points to be a multiple of 2, for a pseudorandom RNG.
			float *tmp = NULL;
			checkCudaErrors(MemoryMonitor::instance()->gpuMalloc((void**)&tmp, (len + 1) * sizeof(float)));
			checkCudaErrors(curandGenerateNormal(gen, tmp, len + 1, 0.0, 1.0));
			checkCudaErrors(cudaMemcpy(devData + len * ch, tmp, len * sizeof(float), cudaMemcpyDeviceToDevice));
			MemoryMonitor::instance()->freeGpuMemory(tmp);
		}else{
			checkCudaErrors(curandGenerateNormal(gen, devData + ch * len, len, 0.0, 1.0));
		}
	}
	// Cleanup generator
	checkCudaErrors(curandDestroyGenerator(gen));
	deviceToHost();
}

void Mat::set(int pos_y, int pos_x, int pos_channel, float val){
	if(NULL == hostData) {mallocHost();}
	if(NULL == devData) {mallocDevice();}
	if(pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	hostData[RC2IDX(pos_y, pos_x, cols) + pos_channel * (rows * cols)] = val;
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
	return hostData[RC2IDX(pos_y, pos_x, cols) + pos_channel * (rows * cols)];
}

vector3f Mat::get(int pos_y, int pos_x) const{
	if(NULL == hostData || NULL == devData||
	   pos_x >= cols || pos_y >= rows || channels < 3){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	vector3f res;
	for(int i = 0; i < 3; ++i){
		res.set(i, hostData[RC2IDX(pos_y, pos_x, cols) + i * (rows * cols)]);
	}
	return res;
}

vector3f Mat::get(int pos) const{
	if(NULL == hostData || NULL == devData||
	   pos >= getLength() || channels < 3){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	vector3f res;
	for(int i = 0; i < 3; ++i){
		res.set(i, hostData[pos + i * (rows * cols)]);
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
	checkCudaErrors(cudaMemcpy(hostData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToHost));
}

void Mat::hostToDevice(){
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"can't do that, host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"can't do that, device data is NULL..."<<std::endl;
		exit(0);
	}
	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(devData, hostData, getLength() * sizeof(float), cudaMemcpyHostToDevice));
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
	checkCudaErrors(cudaMemcpy(m.devData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
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
	checkCudaErrors(cudaMemcpy(m.devData, devData, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
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
	Mat tmpmat(m);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(tmpmat.devData, devData, tmp);
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
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_plus<<<num_blocks, block_size>>>(devData, m.devData, len);
    deviceToHost();
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
	Mat tmpmat(m);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(tmpmat.devData, devData, tmp);
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
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_minus<<<num_blocks, block_size>>>(devData, m.devData, len);
    deviceToHost();
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
	Mat res(rows, m.cols, channels);
	int lena = rows * cols;
	int lenb = m.rows * m.cols;
	int lenres = res.rows * res.cols;
	int TILE_WIDTH = 32;
    dim3 dimGrid((res.cols - 1) / TILE_WIDTH + 1, (res.rows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	for(int ch = 0; ch < channels; ++ch){
		cu_multiply<<<dimGrid, dimBlock>>>(devData + ch * lena , m.devData + ch * lenb, res.devData + ch * lenres,
													rows, cols, m.rows, m.cols, res.rows, res.cols);
		checkCudaErrors(cudaThreadSynchronize());
	}
	res.deviceToHost();
	return res;
}

Mat Mat::operator*(float val) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData, val, res.devData, len);
    res.deviceToHost();
    return res;
}

Mat Mat::operator*(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData + ch * len, val, res.devData + ch * len, len);
    }
    res.deviceToHost();
    return res;
}

Mat& Mat::operator*=(float val){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData, val, len);
    deviceToHost();
	return *this;
}

Mat& Mat::operator*=(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData + ch * len, val, len);
    }
	return *this;
}

Mat Mat::operator/(float val) const{
	if(NULL == hostData || NULL == devData || 0 == val){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_divide<<<num_blocks, block_size>>>(devData, res.devData, val, len);
    res.deviceToHost();
    return res;
}

Mat Mat::operator/(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_divide<<<num_blocks, block_size>>>(devData + ch * len, res.devData + ch * len, val, len);
    }
    res.deviceToHost();
    return res;
}

Mat& Mat::operator/=(float val){
	if(NULL == hostData || NULL == devData || 0 == val){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_divide<<<num_blocks, block_size>>>(devData, val, len);
    deviceToHost();
	return *this;
}

Mat& Mat::operator/=(const vector3f &v){
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_divide<<<num_blocks, block_size>>>(devData + ch * len, val, len);
    }
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
    Mat res(rows, cols, channels);
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData, val, res.devData, len);
    res.deviceToHost();
    return res;
}

Mat Mat::mul(const vector3f &v) const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(devData + ch * len, val, res.devData + ch * len, len);
    }
    res.deviceToHost();
    return res;
}

Mat Mat::t() const{
	if(NULL == hostData || NULL == devData){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat res(cols, rows, channels);
	int len = res.rows * res.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < channels; ++ch){
		cu_transpose<<<num_blocks, block_size>>>(devData + ch * len, res.devData + ch * len, cols, res.cols, len);
	}
	res.deviceToHost();
	return res;
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
		checkCudaErrors(cudaMemset(devData, 0, sizeof(float) * cols * rows * channels));
	}
}

void Mat::printHost(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"device data is NULL..."<<std::endl;
		exit(0);
	}
	int counter = 0;
	std::cout<<"Matrix with "<<channels<<" channels, "<<rows<<" rows, "<<cols<<"columns."<<std::endl;
	for(int i = 0; i < channels; ++i){
		std::cout<<"Channel "<<i<<" : "<<std::endl;
		for(int j = 0; j < rows; ++j){
			for(int k = 0; k < cols; ++k){
				std::cout<<hostData[counter]<<" ";
				++ counter;
			}
			std::cout<<std::endl;
		}
	}
}

void Mat::printDevice(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == hostData || NULL == devData){
		if(NULL == hostData) std::cout<<"host data is NULL..."<<std::endl;
		if(NULL == devData) std::cout<<"device data is NULL..."<<std::endl;
		exit(0);
	}
	float *host_data = 0;
	host_data = (float*)MemoryMonitor::instance()->cpuMalloc(getLength() * sizeof(float));
	checkCudaErrors(cudaMemcpy(host_data, devData, getLength() * sizeof(float), cudaMemcpyDeviceToHost));
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
}
