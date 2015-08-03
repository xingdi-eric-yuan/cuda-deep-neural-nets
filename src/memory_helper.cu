#include "memory_helper.h"
using namespace std;

void* MemoryMonitor::cpuMalloc(int size){
	cpuMemory += size;
	void* p = malloc(size);
	cpuPoint[p] = 1.0f * size;
 	//if(size >= Mb){
 	//	printf("******************************* cpu malloc memory %fMb\n", 1.0 * size / Mb);
 	//}
	return p;
}

void MemoryMonitor::freeCpuMemory(void* ptr)
{
	if(cpuPoint.find(ptr) != cpuPoint.end()){
 		//if(cpuPoint[ptr] >= Mb){
 		//	printf("+++++++++++++++++++++++++++++++ free cpu memory %fMb\n", cpuPoint[ptr] / Mb);
 		//}
		cpuMemory -= cpuPoint[ptr];
		free(ptr);
		cpuPoint.erase(ptr);
	}
}

cudaError_t MemoryMonitor::gpuMalloc(void** devPtr, int size){
	cudaError_t error = cudaMalloc(devPtr, size);
	checkCudaErrors(error);
	gpuMemory += size;
	gpuPoint[*devPtr] = (float)size;
	return error;
 	//if(size >= Mb){
 	//	printf("******************************* gpu malloc memory %fMb\n", 1.0 * size / Mb);
 	//}
}

void MemoryMonitor::freeGpuMemory(void* ptr){
	if(gpuPoint.find(ptr) != gpuPoint.end()){
 		//if(gpuPoint[ptr] >= Mb){
 		//printf("+++++++++++++++++++++++++++++++ free gpu memory %fMb\n", gpuPoint[ptr] / Mb);
 		//}
		gpuMemory -= gpuPoint[ptr];
		checkCudaErrors(cudaFree(ptr));
		gpuPoint.erase(ptr);
	}
}


float MemoryMonitor::getCpuMemory() const{
	return cpuMemory;
}

float MemoryMonitor::getGpuMemory() const{
	return gpuMemory;
}
