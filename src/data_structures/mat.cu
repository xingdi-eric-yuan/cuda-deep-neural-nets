#include "data_structure.h"

using namespace std;

Mat::Mat(){
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

Mat::Mat(const Mat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	Data = NULL;
	mallocDevice();
	checkCudaErrors(cudaMemcpy(Data, m.Data, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
}

Mat::Mat(const cpuMat &m){
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	Data = NULL;
	mallocDevice();
	checkCudaErrors(cudaMemcpy(Data, m.Data, getLength() * sizeof(float), cudaMemcpyHostToDevice));
}

Mat::Mat(int height, int width, int nchannels){
	cols = width;
	rows = height;
	channels = nchannels;
	Data = NULL;
	mallocDevice();
	//zeros();
}

Mat::~Mat(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeGpuMemory(Data);
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

void Mat::release(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeGpuMemory(Data);
	rows = 0;
	cols = 0;
	channels = 0;
	Data = NULL;
}

Mat& Mat::operator=(const Mat &m){
	if(NULL != Data)
		MemoryMonitor::instance()->freeGpuMemory(Data);
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	mallocDevice();
	checkCudaErrors(cudaMemcpy(Data, m.Data, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
    return *this;
}

Mat& Mat::operator<<=(Mat &m){
	if(NULL != Data)
		MemoryMonitor::instance()->freeGpuMemory(Data);
	cols = m.cols;
	rows = m.rows;
	channels = m.channels;
	mallocDevice();
	checkCudaErrors(cudaMemcpy(Data, m.Data, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
	m.release();
    return *this;
}

void Mat::setSize(int r, int c, int ch){
	if(NULL != Data)
		MemoryMonitor::instance()->freeGpuMemory(Data);
	rows = r;
	cols = c;
	channels = ch;
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
	if(NULL == Data) mallocDevice();
	curandGenerator_t gen;
	// Create pseudo-random number generator
	checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	// Set seed
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, rand() % 3456));
	// Generate n floats on device
	checkCudaErrors(curandGenerateUniform(gen, Data, getLength()));
	// Cleanup generator
	checkCudaErrors(curandDestroyGenerator(gen));
}

void Mat::randn(){

	if(NULL == Data) mallocDevice();
	int len = cols * rows;
	curandGenerator_t gen;
	// Create pseudo-random number generator
	checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	for(int ch = 0; ch < channels; ++ch){
		// Set seed
		checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, rand() % 1234));
		// Generate n floats on device
		if(len % 2){
			// In general, the normal generating functions (e.g. curandGenerateNormal, curandGenerateLogNormal, etc.)
			// require the number of requested points to be a multiple of 2, for a pseudorandom RNG.
			float *tmp = NULL;
			checkCudaErrors(MemoryMonitor::instance()->gpuMalloc((void**)&tmp, (len + 1) * sizeof(float)));
			checkCudaErrors(curandGenerateNormal(gen, tmp, len + 1, 0.0, 1.0));
			checkCudaErrors(cudaMemcpy(Data + len * ch, tmp, len * sizeof(float), cudaMemcpyDeviceToDevice));
			MemoryMonitor::instance()->freeGpuMemory(tmp);
		}else{
			checkCudaErrors(curandGenerateNormal(gen, Data + ch * len, len, 0.0, 1.0));
		}
	}
	// Cleanup generator
	checkCudaErrors(curandDestroyGenerator(gen));
}

void Mat::set(int pos_y, int pos_x, int pos_channel, float val){
	if(NULL == Data) {mallocDevice();}
	if(pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
    float *host_data = (float *)malloc(sizeof(float));
    *host_data = val;
    checkCudaErrors(cudaMemcpy(Data + RC2IDX(pos_y, pos_x, cols) + pos_channel * (rows * cols), host_data, sizeof(float), cudaMemcpyHostToDevice));
    free(host_data);
}

void Mat::set(int pos_y, int pos_x, const vector3f& val){
	if(NULL == Data) {mallocDevice();}
	if(pos_x >= cols || pos_y >= rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	float host_data = 0;
	for(int i = 0; i < channels; ++i){
	    host_data = val.get(i);
	    checkCudaErrors(cudaMemcpy(Data + RC2IDX(pos_y, pos_x, cols) + i * (rows * cols), &host_data, sizeof(float), cudaMemcpyHostToDevice));
	}
}

void Mat::set(int pos, const vector3f& val){
	if(NULL == Data) {mallocDevice();}
	if(pos >= cols * rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	float host_data = 0;
	for(int i = 0; i < channels; ++i){
	    host_data = val.get(i);
	    checkCudaErrors(cudaMemcpy(Data + pos + i * (rows * cols), &host_data, sizeof(float), cudaMemcpyHostToDevice));
	}
}

void Mat::set(int pos, int pos_channel, float val){
	if(NULL == Data) {mallocDevice();}
	if(pos >= cols * rows){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	float *host_data = (float *)malloc(sizeof(float));
	*host_data = val;
	checkCudaErrors(cudaMemcpy(Data + pos + pos_channel * (rows * cols), host_data, sizeof(float), cudaMemcpyHostToDevice));
	free(host_data);
}

void Mat::set(int pos, float val){
	if(NULL == Data) {mallocDevice();}
	if(pos >= getLength()){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	float *host_data = (float *)malloc(sizeof(float));
	*host_data = val;
	checkCudaErrors(cudaMemcpy(Data + pos, host_data, sizeof(float), cudaMemcpyHostToDevice));
	free(host_data);
}

void Mat::setAll(float val){
	if(NULL == Data) {mallocDevice();}
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_setAll<<<num_blocks, block_size>>>(Data, val, len);
}

void Mat::setAll(const vector3f &v){
	if(NULL == Data) {mallocDevice();}
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_setAll<<<num_blocks, block_size>>>(Data + i * len, v.get(i), len);
	}
}

float Mat::get(int pos_y, int pos_x, int pos_channel) const{
	if(NULL == Data||
	   pos_x >= cols || pos_y >= rows || pos_channel >= channels){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	float host_data = 0;
	checkCudaErrors(cudaMemcpy(&host_data, Data + RC2IDX(pos_y, pos_x, cols) + pos_channel * (rows * cols), sizeof(float), cudaMemcpyDeviceToHost));
	return host_data;
}

vector3f Mat::get(int pos_y, int pos_x) const{
	if(NULL == Data||
	   pos_x >= cols || pos_y >= rows || channels < 3){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	vector3f res;
	float host_data = 0;
	for(int i = 0; i < 3; ++i){
		checkCudaErrors(cudaMemcpy(&host_data, Data + RC2IDX(pos_y, pos_x, cols) + i * (rows * cols), sizeof(float), cudaMemcpyDeviceToHost));
		res.set(i, host_data);
	}
	return res;
}

vector3f Mat::get(int pos) const{
	if(NULL == Data||
	   pos >= getLength() || channels < 3){
		std::cout<<"invalid position..."<<std::endl;
		exit(0);
	}
	vector3f res;
	float host_data = 0;
	for(int i = 0; i < 3; ++i){
		checkCudaErrors(cudaMemcpy(&host_data, Data + pos + i * (rows * cols), sizeof(float), cudaMemcpyDeviceToHost));
		res.set(i, host_data);
	}
	return res;
}

int Mat::getLength() const{
	return rows * cols * channels;
}

void Mat::copyTo(Mat &m) const{
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeGpuMemory(m.Data);
		m.Data = NULL;
	}
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	m.mallocDevice();
	checkCudaErrors(cudaMemcpy(m.Data, Data, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Mat::copyTo(cpuMat &m) const{
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeCpuMemory(m.Data);
		m.Data = NULL;
	}
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	m.mallocMat();
	checkCudaErrors(cudaMemcpy(m.Data, Data, getLength() * sizeof(float), cudaMemcpyDeviceToHost));
}

void Mat::moveTo(Mat &m){
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeGpuMemory(m.Data);
		m.Data = NULL;
	}
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	m.mallocDevice();
	checkCudaErrors(cudaMemcpy(m.Data, Data, getLength() * sizeof(float), cudaMemcpyDeviceToDevice));
	release();
}

void Mat::moveTo(cpuMat &m){
	if(NULL != m.Data){
		MemoryMonitor::instance()->freeCpuMemory(m.Data);
		m.Data = NULL;
	}
	m.rows = rows;
	m.cols = cols;
	m.channels = channels;
	m.mallocMat();
	checkCudaErrors(cudaMemcpy(m.Data, Data, getLength() * sizeof(float), cudaMemcpyDeviceToHost));
	release();
}

Mat Mat::operator+(const Mat &m) const{
	if(NULL == Data || NULL == m.Data||
	   getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat(m);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(tmpmat.Data, Data, tmp);
	return tmpmat;
}

Mat Mat::operator+(float val) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(tmpmat.Data, val, len);
	return tmpmat;
}

Mat Mat::operator+(const vector3f &v) const{
	if(NULL == Data){
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
		cu_plus<<<num_blocks, block_size>>>(tmpmat.Data + i * len, tmp, len);
	}
	return tmpmat;
}

Mat& Mat::operator+=(const Mat &m){
	if(NULL == Data || NULL == m.Data||
	   getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_plus<<<num_blocks, block_size>>>(Data, m.Data, len);
	return *this;
}

Mat& Mat::operator+=(float val){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_plus<<<num_blocks, block_size>>>(Data, val, len);
	return *this;
}

Mat& Mat::operator+=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_plus<<<num_blocks, block_size>>>(Data + i * len, v.get(i), len);
	}
	return *this;
}

Mat Mat::operator-(const Mat &m) const{

	if(NULL == Data ||
	   NULL == m.Data||
	   getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat(m);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(tmpmat.Data, Data, tmp);
	return tmpmat;
}

Mat Mat::operator-(float val) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(tmpmat.Data, val, tmp);
	return tmpmat;
}

Mat Mat::operator-(const vector3f &v) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat;
	copyTo(tmpmat);
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_minus<<<num_blocks, block_size>>>(tmpmat.Data + i * len, v.get(i), len);
	}
	return tmpmat;
}

Mat& Mat::operator-=(const Mat &m){
	if(NULL == Data ||
	   NULL == m.Data||
	   getLength() != m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_minus<<<num_blocks, block_size>>>(Data, m.Data, len);
	return *this;
}

Mat& Mat::operator-=(float val){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	cu_minus<<<num_blocks, block_size>>>(Data, val, len);
	return *this;
}

Mat& Mat::operator-=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	int len = rows * cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int i = 0; i < channels; ++i){
		cu_minus<<<num_blocks, block_size>>>(Data + i * len, v.get(i), len);
	}
	return *this;
}

Mat Mat::operator*(const Mat &m) const{
	if(NULL == Data ||
	   NULL == m.Data||
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
		cu_multiply<<<dimGrid, dimBlock>>>(Data + ch * lena , m.Data + ch * lenb, res.Data + ch * lenres,
													rows, cols, m.rows, m.cols, res.rows, res.cols);
		checkCudaErrors(cudaThreadSynchronize());
	}
	return res;
}

Mat Mat::operator*(float val) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_elementWiseMultiply<<<num_blocks, block_size>>>(Data, val, res.Data, len);
    return res;
}

Mat Mat::operator*(const vector3f &v) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(Data + ch * len, val, res.Data + ch * len, len);
    }
    return res;
}

Mat& Mat::operator*=(float val){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_elementWiseMultiply<<<num_blocks, block_size>>>(Data, val, len);
	return *this;
}

Mat& Mat::operator*=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(Data + ch * len, val, len);
    }
	return *this;
}

Mat Mat::operator/(float val) const{
	if(NULL == Data || 0 == val){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_divide<<<num_blocks, block_size>>>(Data, res.Data, val, len);
    return res;
}

Mat Mat::operator/(const vector3f &v) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_divide<<<num_blocks, block_size>>>(Data + ch * len, res.Data + ch * len, val, len);
    }
    return res;
}

Mat& Mat::operator/=(float val){
	if(NULL == Data || 0 == val){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_divide<<<num_blocks, block_size>>>(Data, val, len);
	return *this;
}

Mat& Mat::operator/=(const vector3f &v){
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_divide<<<num_blocks, block_size>>>(Data + ch * len, val, len);
    }
	return *this;
}

Mat Mat::mul(const Mat &m) const{
	if(NULL == Data ||
	   NULL == m.Data||
	   getLength()!= m.getLength()){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat tmpmat(m);
	int tmp = getLength();
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (tmp / block_size) + ((tmp % block_size) ? 1 : 0);
	cu_elementWiseMultiply<<<num_blocks, block_size>>>(tmpmat.Data, Data, tmp);
	return tmpmat;
}

Mat Mat::mul(float val) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = getLength();
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    cu_elementWiseMultiply<<<num_blocks, block_size>>>(Data, val, res.Data, len);
    return res;
}

Mat Mat::mul(const vector3f &v) const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
    Mat res(rows, cols, channels);
    int len = rows * cols;
    const size_t block_size = threadsPerBlock;
    const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    for(int ch = 0; ch < channels; ++ch){
        float val = v.get(ch);
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(Data + ch * len, val, res.Data + ch * len, len);
    }
    return res;
}

Mat Mat::t() const{
	if(NULL == Data){
		std::cout<<"invalid vectors..."<<std::endl;
		exit(0);
	}
	Mat res(cols, rows, channels);
	int len = res.rows * res.cols;
	const size_t block_size = threadsPerBlock;
	const size_t num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
	for(int ch = 0; ch < channels; ++ch){
		cu_transpose<<<num_blocks, block_size>>>(Data + ch * len, res.Data + ch * len, cols, res.cols, len);
	}
	return res;
}

// memory
void Mat::mallocDevice(){
	if(NULL == Data){
		cudaError_t cudaStat;
		// malloc device data
		cudaStat = MemoryMonitor::instance()->gpuMalloc((void**)&Data, cols * rows * channels * sizeof(float));
		if(cudaStat != cudaSuccess) {
			std::cout<<"device memory allocation failed... because of error number : "<<(int)cudaStat<<std::endl;
			exit(0);
		}
		checkCudaErrors(cudaMemset(Data, 0, sizeof(float) * cols * rows * channels));
	}
}

void Mat::print(const std::string &str) const{
	std::cout<<str<<std::endl;
	if(NULL == Data){
		if(NULL == Data) std::cout<<"device data is NULL..."<<std::endl;
		exit(0);
	}
	float *host_data = 0;
	host_data = (float*)MemoryMonitor::instance()->cpuMalloc(getLength() * sizeof(float));
	checkCudaErrors(cudaMemcpy(host_data, Data, getLength() * sizeof(float), cudaMemcpyDeviceToHost));
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

void Mat::printDim(const std::string &str) const{

	std::cout<<str<<std::endl;
	if(NULL == Data){
		if(NULL == Data) std::cout<<"device data is NULL..."<<std::endl;
		exit(0);
	}
	cout<<"Matrix Dimension = ["<<rows<<", "<<cols<<", "<<channels<<"]"<<endl;
}
