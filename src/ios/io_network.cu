#include "io_network.h"

void save(const std::string &path, const Mat* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid matrix..."<<std::endl;
		exit(0);
	}
	ofstream outfile(path.c_str(), ios::out);
	outfile.precision(16);
	if (outfile.is_open()){
		outfile<<"path = "<<path<<"\n";
		outfile<<"rows = "<<m -> rows<<"\n";
		outfile<<"cols = "<<m -> cols<<"\n";
		outfile<<"channels = "<<m -> channels<<"\n";
		outfile<<"data =\n";
		float *data = (float*)malloc(m -> getLength() * sizeof(float));
		checkCudaErrors(cudaMemcpy(data, m -> Data, m -> getLength() * sizeof(float), cudaMemcpyDeviceToHost));
		for(int i = 0; i < m -> getLength(); ++i){
			outfile<<data[i]<<" ";
		}
		outfile<<"\n";
		outfile.close();
		free(data);
	}else {
		std::cout<<"unable to open file..."<<std::endl;
		exit(0);
	}
}

void save(const std::string &path, const cpuMat* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid matrix..."<<std::endl;
		exit(0);
	}
	ofstream outfile(path.c_str(), ios::out);
	outfile.precision(16);
	if (outfile.is_open()){
		outfile<<"path = "<<path<<"\n";
		outfile<<"rows = "<<m -> rows<<"\n";
		outfile<<"cols = "<<m -> cols<<"\n";
		outfile<<"channels = "<<m -> channels<<"\n";
		outfile<<"data =\n";
		float *data = (float*)malloc(m -> getLength() * sizeof(float));
		memcpy(data, m -> Data, m -> getLength() * sizeof(float));
		for(int i = 0; i < m -> getLength(); ++i){
			outfile<<data[i]<<" ";
		}
		outfile<<"\n";
		outfile.close();
		free(data);
	}else {
		std::cout<<"unable to open file..."<<std::endl;
		exit(0);
	}
}

void save(const std::string &path, const vector3f* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid vector3f..."<<std::endl;
		exit(0);
	}
	ofstream outfile(path.c_str(), ios::out);
	outfile.precision(16);
	if (outfile.is_open()){
		outfile<<"path = "<<path<<"\n";
		outfile<<"data =\n";
		float *data = (float*)malloc(3 * sizeof(float));
		memcpy(data, m -> Data, 3 * sizeof(float));
		for(int i = 0; i < 3; ++i){
			outfile<<data[i]<<" ";
		}
		outfile<<"\n";
		outfile.close();
		free(data);
	}else {
		std::cout<<"unable to open file..."<<std::endl;
		exit(0);
	}
}

void save(const std::string &path, const vector2i* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid vector3f..."<<std::endl;
		exit(0);
	}
	ofstream outfile(path.c_str(), ios::out);
	outfile.precision(16);
	if (outfile.is_open()){
		outfile<<"path = "<<path<<"\n";
		outfile<<"data =\n";
		int *data = (int*)malloc(2 * sizeof(int));
		memcpy(data, m -> Data, 2 * sizeof(int));
		for(int i = 0; i < 2; ++i){
			outfile<<data[i]<<" ";
		}
		outfile<<"\n";
		outfile.close();
		free(data);
	}else {
		std::cout<<"unable to open file..."<<std::endl;
		exit(0);
	}
}

void read(const std::string &path, Mat* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid matrix..."<<std::endl;
		exit(0);
	}
	std::ifstream infile(path.c_str());
	std::string line, tmpstr;
	getline(infile, line); // path = XX
	getline(infile, line); // rows = XX
	std::istringstream iss_row(line);
	if(!(iss_row >> tmpstr >> tmpstr >> tmpstr)){
		std::cout<<"invalid file...1"<<std::endl;
		exit(0);
	}
	if(std::stoi(tmpstr) != m -> rows){
		std::cout<<"invalid file...2"<<std::endl;
		exit(0);
	}
	getline(infile, line); // cols = XX
	std::istringstream iss_col(line);
	if(!(iss_col >> tmpstr >> tmpstr >> tmpstr)){
		cout<<tmpstr<<"---#########"<<endl;
		std::cout<<"invalid file...3"<<std::endl;
		exit(0);
	}
	if(std::stoi(tmpstr) != m -> cols){
		std::cout<<"invalid file...4"<<std::endl;
		exit(0);
	}
	getline(infile, line); // channels = XX
	std::istringstream iss_channels(line);
	if(!(iss_channels >> tmpstr >> tmpstr >> tmpstr)){
		std::cout<<"invalid file...5"<<std::endl;
		exit(0);
	}
	if(std::stoi(tmpstr) != m -> channels){
		std::cout<<"invalid file...6"<<std::endl;
		exit(0);
	}
	getline(infile, line); // data = 
	std::istringstream iss_data_title(line);
	if(!(iss_data_title >> tmpstr >> tmpstr)){
		std::cout<<"invalid file..."<<std::endl;
		exit(0);
	}
	getline(infile, line); // [data]
	std::istringstream iss_data(line);
	float *hostData = (float*)malloc(m -> getLength() * sizeof(float));
	for(int i = 0; i < m -> getLength(); ++i){
		if(!(iss_data >> tmpstr)){
			std::cout<<"invalid file..."<<std::endl;
			exit(0);
		}
		hostData[i] = std::stof(tmpstr);
	}
	checkCudaErrors(cudaMemcpy(m -> Data, hostData, m -> getLength() * sizeof(float), cudaMemcpyHostToDevice));
	free(hostData);
}

void read(const std::string &path, cpuMat* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid matrix..."<<std::endl;
		exit(0);
	}
	std::ifstream infile(path.c_str());
	std::string line, tmpstr;
	getline(infile, line); // path = XX
	getline(infile, line); // rows = XX
	std::istringstream iss_row(line);
	if(!(iss_row >> tmpstr >> tmpstr >> tmpstr)){
		std::cout<<"invalid file...1"<<std::endl;
		exit(0);
	}
	if(std::stoi(tmpstr) != m -> rows){
		std::cout<<"invalid file...2"<<std::endl;
		exit(0);
	}
	getline(infile, line); // cols = XX
	std::istringstream iss_col(line);
	if(!(iss_col >> tmpstr >> tmpstr >> tmpstr)){
		cout<<tmpstr<<"---#########"<<endl;
		std::cout<<"invalid file...3"<<std::endl;
		exit(0);
	}
	if(std::stoi(tmpstr) != m -> cols){
		std::cout<<"invalid file...4"<<std::endl;
		exit(0);
	}
	getline(infile, line); // channels = XX
	std::istringstream iss_channels(line);
	if(!(iss_channels >> tmpstr >> tmpstr >> tmpstr)){
		std::cout<<"invalid file...5"<<std::endl;
		exit(0);
	}
	if(std::stoi(tmpstr) != m -> channels){
		std::cout<<"invalid file...6"<<std::endl;
		exit(0);
	}
	getline(infile, line); // data = 
	std::istringstream iss_data_title(line);
	if(!(iss_data_title >> tmpstr >> tmpstr)){
		std::cout<<"invalid file..."<<std::endl;
		exit(0);
	}
	getline(infile, line); // [data]
	std::istringstream iss_data(line);
	float *hostData = (float*)malloc(m -> getLength() * sizeof(float));
	for(int i = 0; i < m -> getLength(); ++i){
		if(!(iss_data >> tmpstr)){
			std::cout<<"invalid file..."<<std::endl;
			exit(0);
		}
		hostData[i] = std::stof(tmpstr);
	}
	memcpy(m -> Data, hostData, m -> getLength() * sizeof(float));
	free(hostData);
}

void read(const std::string &path, vector3f* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid vector3f..."<<std::endl;
		exit(0);
	}
	std::ifstream infile(path.c_str());
	std::string line, tmpstr;
	getline(infile, line); // path = XX
	getline(infile, line); // data = 
	std::istringstream iss_data_title(line);
	if(!(iss_data_title >> tmpstr >> tmpstr)){
		std::cout<<"invalid file..."<<std::endl;
		exit(0);
	}
	getline(infile, line); // [data]
	std::istringstream iss_data(line);
	float *hostData = (float*)malloc(3 * sizeof(float));
	for(int i = 0; i < 3; ++i){
		if(!(iss_data >> tmpstr)){
			std::cout<<"invalid file..."<<std::endl;
			exit(0);
		}
		hostData[i] = std::stof(tmpstr);
	}
	memcpy(m -> Data, hostData, 3 * sizeof(float));
	free(hostData);
}

void read(const std::string &path, vector2i* m){
	if(NULL == m || NULL == m -> Data){
		std::cout<<"invalid vector3f..."<<std::endl;
		exit(0);
	}
	std::ifstream infile(path.c_str());
	std::string line, tmpstr;
	getline(infile, line); // path = XX
	getline(infile, line); // data = 
	std::istringstream iss_data_title(line);
	if(!(iss_data_title >> tmpstr >> tmpstr)){
		std::cout<<"invalid file..."<<std::endl;
		exit(0);
	}
	getline(infile, line); // [data]
	std::istringstream iss_data(line);
	int *hostData = (int*)malloc(2 * sizeof(int));
	for(int i = 0; i < 2; ++i){
		if(!(iss_data >> tmpstr)){
			std::cout<<"invalid file..."<<std::endl;
			exit(0);
		}
		hostData[i] = std::stoi(tmpstr);
	}
	memcpy(m -> Data, hostData, 2 * sizeof(int));
	free(hostData);
}

void saveNetwork(const std::string &path, const std::vector<network_layer*> &flow){
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string pathw = "";
    string pathb = "";
    for(int i = 0; i < flow.size(); ++i){
    	string layerpath = path + "/layer_" + std::to_string(i);
        if(flow[i] -> layer_type == "convolutional"){
        	for(int k = 0; k < ((convolutional_layer*)flow[i]) -> kernels.size(); ++k){
        		pathw = layerpath + "_kernel_" + std::to_string(k) + "_w.txt";
        		pathb = layerpath + "_kernel_" + std::to_string(k) + "_b.txt";
        		save(pathw, ((convolutional_layer*)flow[i]) -> kernels[k] -> w);
        		save(pathb, ((convolutional_layer*)flow[i]) -> kernels[k] -> b);
        	}
        }elif(flow[i] -> layer_type == "fully_connected"){
    		pathw = layerpath + "_w.txt";
    		pathb = layerpath + "_b.txt";
    		save(pathw, ((fully_connected_layer*)flow[i]) -> w);
    		save(pathb, ((fully_connected_layer*)flow[i]) -> b);
        }elif(flow[i] -> layer_type == "softmax"){
    		pathw = layerpath + "_w.txt";
    		pathb = layerpath + "_b.txt";
    		save(pathw, ((softmax_layer*)flow[i]) -> w);
    		save(pathb, ((softmax_layer*)flow[i]) -> b);
        }
    }
}

// READ NETWORK
void readNetwork(const string &path, std::vector<network_layer*> &flow){
    string pathw = "";
    string pathb = "";
    for(int i = 0; i < flow.size(); ++i){
        string layerpath = path + "/layer_" + std::to_string(i);
        if(flow[i] -> layer_type == "convolutional"){
            for(int k = 0; k < ((convolutional_layer*)flow[i]) -> kernels.size(); ++k){
                pathw = layerpath + "_kernel_" + std::to_string(k) + "_w.txt";
                pathb = layerpath + "_kernel_" + std::to_string(k) + "_b.txt";
                read(pathw, ((convolutional_layer*)flow[i]) -> kernels[k] -> w);
                read(pathb, ((convolutional_layer*)flow[i]) -> kernels[k] -> b);
            }
        }elif(flow[i] -> layer_type == "fully_connected"){
            pathw = layerpath + "_w.txt";
            pathb = layerpath + "_b.txt";
            read(pathw, ((fully_connected_layer*)flow[i]) -> w);
            read(pathb, ((fully_connected_layer*)flow[i]) -> b);
        }elif(flow[i] -> layer_type == "softmax"){
            pathw = layerpath + "_w.txt";
            pathb = layerpath + "_b.txt";
            read(pathw, ((softmax_layer*)flow[i]) -> w);
            read(pathb, ((softmax_layer*)flow[i]) -> b);
        }
    }
}


