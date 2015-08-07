#pragma once
#include "general_settings.h"

using namespace std;

class MemoryMonitor{
public:
	static MemoryMonitor* instance(){
		static MemoryMonitor* monitor = new MemoryMonitor();
		return monitor;
	}
	void* cpuMalloc(int size);
	cudaError_t gpuMalloc(void** devPtr, int size);
	MemoryMonitor(): gpuMemory(0), cpuMemory(0){}
	void printCpuMemory(){printf("total malloc cpu memory %fMb\n", cpuMemory / 1024 / 1024);}
	void printGpuMemory(){printf("total malloc gpu memory %fMb\n", gpuMemory / 1024 / 1024);}
	void freeGpuMemory(void* ptr);
	void freeCpuMemory(void* ptr);
	double getCpuMemory() const;
	double getGpuMemory() const;
private:
	double cpuMemory;
	double gpuMemory;
	std::unordered_map<void*, double>cpuPoint;
	std::unordered_map<void*, double>gpuPoint;
};
