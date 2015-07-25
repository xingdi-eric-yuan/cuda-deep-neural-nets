#include "helper.h"

int snapTransformSize(int dataSize){
    int hiBit;
    unsigned int lowPOT, hiPOT;
    dataSize = iAlignUp(dataSize, 16);
    for (hiBit = 31; hiBit >= 0; hiBit--){
        if (dataSize & (1U << hiBit)){
            break;
        }
    }
    lowPOT = 1U << hiBit;
    if (lowPOT == (unsigned int)dataSize){
        return dataSize;
    }
    hiPOT = 1U << (hiBit + 1);
    if (hiPOT <= 1024){
        return hiPOT;
    }else{
        return iAlignUp(dataSize, 512);
    }
}

void copyVector(const std::vector<vector<Mat*> >& _from, std::vector<vector<Mat*> >& _to){
	_to.clear();
	_to.resize(_from.size());
	for(int i = 0; i < _to.size(); ++i){
		_to[i].clear();
		_to[i].resize(_from[i].size());
	}
	for(int i = 0; i < _to.size(); ++i){
		for(int j = 0; j < _to[i].size(); ++j){
			_to[i][j] = new Mat();
			_from[i][j] -> copyTo(*(_to[i][j]));
		}
	}
}

void copyVector(const std::vector<Mat*>& _from, std::vector<Mat*>& _to){
	_to.clear();
	_to.resize(_from.size());
	for(int i = 0; i < _to.size(); ++i){
		_to[i] = new Mat();
		_from[i] -> copyTo(*(_to[i]));
	}
}

void releaseVector(std::vector<std::vector<Mat*> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			vec[i][j] -> release();
		}
	}
}

void releaseVector(std::vector<std::vector<Mat> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			vec[i][j].release();
		}
	}
}
void releaseVector(std::vector<Mat*>& vec){
	for(int i = 0; i < vec.size(); ++i){
		vec[i] -> release();
	}
}
void releaseVector(std::vector<Mat>& vec){
	for(int i = 0; i < vec.size(); ++i){
		vec[i].release();
	}
}

void releaseVector(std::vector<std::vector<cpuMat*> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			vec[i][j] -> release();
		}
	}
}

void releaseVector(std::vector<std::vector<cpuMat> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			vec[i][j].release();
		}
	}
}
void releaseVector(std::vector<cpuMat*>& vec){
	for(int i = 0; i < vec.size(); ++i){
		vec[i] -> release();
	}
}
void releaseVector(std::vector<cpuMat>& vec){
	for(int i = 0; i < vec.size(); ++i){
		vec[i].release();
	}
}

void showGpuProperty(){

	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	for(int i = 0; i < count; ++i){
		cudaGetDeviceProperties(&prop, i);
		cout<<"--- general information for device "<<i<<endl;
		cout<<"name: "<<prop.name<<endl;
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap: " );
		if (prop.deviceOverlap) printf( "Enabled\n" );
		else printf( "Disabled\n" );
		printf( "Kernel execition timeout : " );
		if (prop.kernelExecTimeoutEnabled) printf( "Enabled\n" );
		else printf( "Disabled\n" );
		printf( "   --- Memory Information for device %d ---\n", i );
		printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
		printf( "Max mem pitch:  %ld\n", prop.memPitch );
		printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
		printf( "   --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count:  %d\n", prop.multiProcessorCount );
		printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp:  %d\n", prop.regsPerBlock );
	    printf( "Threads in warp:  %d\n", prop.warpSize );
		printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
	    printf( "Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
		printf( "Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
		printf( "\n" );
	}
}
