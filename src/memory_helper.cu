#include "memory_helper.h"
using namespace std;

// CPU MALLOC
// cpu memory allocation
void* MemoryMonitor::cpuMalloc(int size){
	cpuMemory += size;
	void* p = malloc(size);
	cpuPoint[p] = 1.0f * size;
	return p;
}

// FREE CPU MEMORY
// cpu memory free
void MemoryMonitor::freeCpuMemory(void* ptr)
{
	if(cpuPoint.find(ptr) != cpuPoint.end()){
		cpuMemory -= cpuPoint[ptr];
		free(ptr);
		cpuPoint.erase(ptr);
	}
}

// GPU MALLOC
// gpu memory allocation
cudaError_t MemoryMonitor::gpuMalloc(void** devPtr, int size){
	cudaError_t error = cudaMalloc(devPtr, size);
	checkCudaErrors(error);
	gpuMemory += size;
	gpuPoint[*devPtr] = (double)size;
	return error;
}

// FREE GPU MEMORY
// gpu memory free
void MemoryMonitor::freeGpuMemory(void* ptr){
	if(gpuPoint.find(ptr) != gpuPoint.end()){
		gpuMemory -= gpuPoint[ptr];
		checkCudaErrors(cudaFree(ptr));
		gpuPoint.erase(ptr);
	}
}

// GET CPU MEMORY
// returns size of cpu memory which currently using
double MemoryMonitor::getCpuMemory() const{
	return cpuMemory;
}

// GET GPU MEMORY
// returns size of gpu memory which currently using
double MemoryMonitor::getGpuMemory() const{
	return gpuMemory;
}
