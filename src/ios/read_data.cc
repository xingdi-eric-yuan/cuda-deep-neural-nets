#include "read_data.h"

void read_batch(std::string filename, std::vector<cpuMat*>& vec, cpuMat* label){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            cpuMat *fin_img = new cpuMat(n_rows, n_cols, 3);
            for(int ch = 0; ch < 3; ++ch){
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        fin_img -> set(r, c, ch, (float)temp);
                    }
                }
            }
            vec.push_back(fin_img);
            label -> set(0, i, 0, (float)tplabel);
        }
    }
	file.close();
}

void read_CIFAR10_data(std::vector<cpuMat*> &trainX, std::vector<cpuMat*> &testX, cpuMat *&trainY, cpuMat *&testY){
	std::string filename;
    filename = "cifar-10-batches-bin/data_batch_";
    std::vector<cpuMat*> labels;
    std::vector<std::vector<cpuMat*> > batches;
	int number_batch = 1;
    for(int i = 1; i <= number_batch; i++){
    	std::vector<cpuMat*> tpbatch;
        cpuMat *tplabel = new cpuMat(1, 10000, 1);
        std::string name = filename + std::to_string((long long)i) + ".bin";
        read_batch(name, tpbatch, tplabel);
        labels.push_back(tplabel);
        batches.push_back(tpbatch);
        tpbatch.clear();
    }
    // trainX
    trainX.reserve(batches[0].size() * number_batch);
    for(int i = 0; i < number_batch; i++){
        trainX.insert(trainX.end(), batches[i].begin(), batches[i].end());
    }
    // trainY
    trainY = new cpuMat(1, 10000 * number_batch, 1);
    for(int i = 0; i < number_batch; ++i){
    	for(int j = 0; j < 10000; ++j){
    		trainY -> set(0, i * 10000 + j, 0, labels[i] -> get(0, j, 0));
    	}
    }
    // testX, testY
    filename = "cifar-10-batches-bin/test_batch.bin";
    testY = new cpuMat(1, 10000, 1);
    std::cout<<"Reading data..."<<std::endl;
    read_batch(filename, testX, testY);
    std::cout<<"Doing preprocessing"<<std::endl;
    preProcessing(trainX, testX);
    // dataEnlarge(trainX, trainY);

    cout<<"****************************************************************************"<<endl
        <<"**                        READ DATASET COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;
    cout<<"The training data has "<<trainX.size()<<" images, each images has "<<trainX[0] -> cols<<" columns and "<<trainX[0] -> rows<<" rows."<<endl;
    cout<<"The testing data has "<<testX.size()<<" images, each images has "<<testX[0] -> cols<<" columns and "<<testX[0] -> rows<<" rows."<<endl;
    cout<<"There are "<<trainY -> cols<<" training labels and "<<testY -> cols<<" testing labels."<<endl<<endl;
}

cpuMat* concat(const std::vector<cpuMat*> &vec){
    int height = vec[0] -> rows * vec[0] -> cols;
    int width = vec.size();
    cpuMat *res = new cpuMat(height, width, vec[0] -> channels);
    for(int i = 0; i < vec.size(); i++){
    	for(int ch = 0; ch < res -> channels; ++ch){
    		memcpy(res -> Data + ch * res -> cols * res -> rows + i * vec[0] -> rows * vec[0] -> cols, vec[i] -> Data + ch * vec[0] -> rows * vec[0] -> cols, vec[0] -> rows * vec[0] -> cols * sizeof(float));
    	}
    }
    return res;
}

void
preProcessing(std::vector<cpuMat*> &trainX, std::vector<cpuMat*> &testX){
    for(int i = 0; i < trainX.size(); i++){
    	safeGetPt(trainX[i], divide(trainX[i], 255.0));
    }
    for(int i = 0; i < testX.size(); i++){
    	safeGetPt(testX[i], divide(testX[i], 255.0));
    }
    // first convert vec of mat into a single mat
    cpuMat *tmp = new cpuMat();
    safeGetPt(tmp, concat(trainX));

    vector3f *mean = new vector3f();
    mean = average(tmp);
    vector3f *sdev = new vector3f();
    sdev = stddev(tmp, mean);

    mean -> print("mean");
    sdev -> print("sdev");

    for(int i = 0; i < trainX.size(); i++){
    	safeGetPt(trainX[i], subtract(trainX[i], mean));
    	safeGetPt(trainX[i], divide(trainX[i], sdev));
    }
    for(int i = 0; i < testX.size(); i++){
    	safeGetPt(testX[i], subtract(testX[i], mean));
    	safeGetPt(testX[i], divide(testX[i], sdev));
    }

    mean -> release();
    sdev -> release();
    tmp -> release();
}



