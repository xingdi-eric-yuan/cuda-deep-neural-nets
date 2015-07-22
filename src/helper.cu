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
