#include "helper.h"

// SNAP TRANSFORM SIZE
// transforms data size, use in convolution FFT 2D
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

// COPY VECTOR
// DEEP copy between vectors of pointers
void copyVector(const std::vector<std::vector<Mat*> >& _from, std::vector<std::vector<Mat*> >& _to){
	releaseVector(_to);
	_to.clear();
	_to.resize(_from.size());
	for(int i = 0; i < _to.size(); ++i){
		_to[i].clear();
		_to[i].resize(_from[i].size());
		for(int j = 0; j < _to[i].size(); ++j){
			_to[i][j] = new Mat();
			_from[i][j] -> copyTo(*(_to[i][j]));
		}
	}
}

void copyVector(const std::vector<Mat*>& _from, std::vector<Mat*>& _to){
	releaseVector(_to);
	_to.clear();
	_to.resize(_from.size());
	for(int i = 0; i < _to.size(); ++i){
		_to[i] = new Mat();
		_from[i] -> copyTo(*(_to[i]));
	}
}

void copyVector(const std::vector<std::vector<vector3f*> >& _from, std::vector<std::vector<vector3f*> >& _to){
	releaseVector(_to);
	_to.clear();
	_to.resize(_from.size());
	for(int i = 0; i < _to.size(); ++i){
		_to[i].clear();
		_to[i].resize(_from[i].size());
		for(int j = 0; j < _to[i].size(); ++j){
			_to[i][j] = new vector3f();
			_from[i][j] -> copyTo(*(_to[i][j]));
		}
	}
}

void copyVector(const std::vector<vector3f*>& _from, std::vector<vector3f*>& _to){
	releaseVector(_to);
	_to.clear();
	_to.resize(_from.size());
	for(int i = 0; i < _to.size(); ++i){
		_to[i] = new vector3f();
		_from[i] -> copyTo(*(_to[i]));
	}
}

// RELEASE VECTOR
// release vector of Mat, vector3f, cpuMat, or Mat*, vector3f*, cpuMat*. (first make sure the pointer is not NULL before you free it)
void releaseVector(std::vector<std::vector<Mat*> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			if(vec[i][j]) vec[i][j] -> release();
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
		if(vec[i]) vec[i] -> release();
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
			if(vec[i][j]) vec[i][j] -> release();
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
		if(vec[i]) vec[i] -> release();
	}
}
void releaseVector(std::vector<cpuMat>& vec){
	for(int i = 0; i < vec.size(); ++i){
		vec[i].release();
	}
}
void releaseVector(std::vector<std::vector<std::vector<vector3f*> > >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			for(int k = 0; k < vec[i][j].size(); ++k){
				if(vec[i][j][k]) vec[i][j][k] -> release();
			}
		}
	}
}
void releaseVector(std::vector<std::vector<std::vector<vector3f> > >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			for(int k = 0; k < vec[i][j].size(); ++k){
				vec[i][j][k].release();
			}
		}
	}
}
void releaseVector(std::vector<std::vector<vector3f*> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			if(vec[i][j]) vec[i][j] -> release();
		}
	}
}
void releaseVector(std::vector<std::vector<vector3f> >& vec){
	for(int i = 0; i < vec.size(); ++i){
		for(int j = 0; j < vec[i].size(); ++j){
			vec[i][j].release();
		}
	}
}
void releaseVector(std::vector<vector3f*>& vec){
	for(int i = 0; i < vec.size(); ++i){
		if(vec[i]) vec[i] -> release();
	}
}
void releaseVector(std::vector<vector3f>& vec){
	for(int i = 0; i < vec.size(); ++i){
		vec[i].release();
	}
}

// SHOW GPU PROPERTY
// cuda function, shows gpu properties
void showGpuProperty(){
	cudaDeviceProp prop;
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
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
