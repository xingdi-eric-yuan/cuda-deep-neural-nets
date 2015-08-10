#include "save_network.h"


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

void saveNetworkConfig(const std::string &path, const std::vector<network_layer*> &flow){

	ofstream outfile(path.c_str(), ios::out);
	outfile.precision(16);
	if (outfile.is_open()){

		outfile<<"/*\npath = "<<path<<"\nSaved Network Config File\n*/\n";
		outfile<<"/*******************************************************\n";
		outfile<<"*\n";
		outfile<<"*	General Parameters Config\n";
		outfile<<"*\n";
		outfile<<"*******************************************************/\n\n";
		outfile<<"IS_GRADIENT_CHECKING = "<<(is_gradient_checking == true ? "true" : "false")<<";\n";
		outfile<<"USE_LOG = "<<(use_log == true ? "true" : "false")<<";\n";
		outfile<<"LEARNING_RATE_W = "<<std::to_string(lrate_w)<<";\n";
		outfile<<"LEARNING_RATE_B = "<<std::to_string(lrate_b)<<";\n";
		outfile<<"TRAINING_EPOCHS = "<<std::to_string(training_epochs)<<";\n";
		outfile<<"ITER_PER_EPO = "<<std::to_string(iter_per_epo)<<";\n";
		outfile<<"MOMENTUM_W_INIT = "<<std::to_string(momentum_w_init)<<";\n";
		outfile<<"MOMENTUM_D2_INIT = "<<std::to_string(momentum_d2_init)<<";\n";
		outfile<<"MOMENTUM_W_ADJUST = "<<std::to_string(momentum_w_adjust)<<";\n";
		outfile<<"MOMENTUM_D2_ADJUST = "<<std::to_string(momentum_d2_adjust)<<";\n\n";

		outfile<<"/*******************************************************\n";
		outfile<<"*\n";
		outfile<<"*	Layers Config\n";
		outfile<<"*\n";
		outfile<<"*******************************************************/\n\n";

	    for(int i = 0; i < flow.size(); ++i){
	    	outfile<<"$\n";

	    	outfile<<"LAYER = "<<flow[i] ->layer_type<<";\n";
        	outfile<<"NAME = "<<flow[i] ->layer_name<<";\n";
        	outfile<<"OUTPUT_TYPE = "<<flow[i] -> output_format<<";\n";
	        if(flow[i] -> layer_type == "input"){
	        	outfile<<"BATCH_SIZE = "<<std::to_string(((input_layer*)flow[i]) -> batch_size)<<";\n";
	        }elif(flow[i] -> layer_type == "convolutional"){
	        	outfile<<"KERNEL_SIZE = "<<std::to_string(((convolutional_layer*)flow[i]) -> kernels[0] -> w -> rows)<<";\n";
	        	outfile<<"KERNEL_AMOUNT = "<<std::to_string(((convolutional_layer*)flow[i]) -> kernels.size())<<";\n";
	        	outfile<<"PADDING = "<<std::to_string(((convolutional_layer*)flow[i]) -> padding)<<";\n";
	        	outfile<<"STRIDE = "<<std::to_string(((convolutional_layer*)flow[i]) -> stride)<<";\n";
	        	outfile<<"COMBINE_MAP = "<<std::to_string(((convolutional_layer*)flow[i]) -> combine_feature_map)<<";\n";
	        	outfile<<"WEIGHT_DECAY = "<<std::to_string(((convolutional_layer*)flow[i]) -> kernels[0] -> weight_decay)<<";\n";
	        }elif(flow[i] -> layer_type == "fully_connected"){
	        	outfile<<"NUM_HIDDEN_NEURONS = "<<std::to_string(((fully_connected_layer*)flow[i]) -> size)<<";\n";
	        	outfile<<"WEIGHT_DECAY = "<<std::to_string(((fully_connected_layer*)flow[i]) -> weight_decay)<<";\n";
	        }elif(flow[i] -> layer_type == "softmax"){
	        	outfile<<"NUM_CLASSES = "<<std::to_string(((softmax_layer*)flow[i]) -> output_size)<<";\n";
	        	outfile<<"WEIGHT_DECAY = "<<std::to_string(((softmax_layer*)flow[i]) -> weight_decay)<<";\n";
	        }elif(flow[i] -> layer_type == "combine"){
	            ;
	        }elif(flow[i] -> layer_type == "branch"){
	            ;
	        }elif(flow[i] -> layer_type == "non_linearity"){
	        	outfile<<"METHOD = ";
	        	string tmp ="";
	        	if(NL_SIGMOID == ((non_linearity_layer*)flow[i]) -> method){
	        		tmp += "sigmoid";
	        	}elif(NL_TANH == ((non_linearity_layer*)flow[i]) -> method){
	        		tmp += "tanh";
	        	}elif(NL_RELU == ((non_linearity_layer*)flow[i]) -> method){
	        		tmp += "relu";
	        	}elif(NL_LEAKY_RELU == ((non_linearity_layer*)flow[i]) -> method){
	        		tmp += "leaky_relu";
	        	}
	        	outfile<<tmp<<";\n";
	        }elif(flow[i] -> layer_type == "pooling"){
	        	outfile<<"METHOD = ";
	        	string tmp ="";
	        	if(POOL_MAX == ((pooling_layer*)flow[i]) -> method){
	        		tmp += "max";
	        	}elif(POOL_MEAN == ((pooling_layer*)flow[i]) -> method){
	        		tmp += "mean";
	        	}elif(POOL_STOCHASTIC == ((pooling_layer*)flow[i]) -> method){
	        		tmp += "stochastic";
	        	}
	        	outfile<<tmp<<";\n";
	        	outfile<<"OVERLAP = "<<(((pooling_layer*)flow[i]) -> overlap ? "true" : "false")<<";\n";
	        	outfile<<"STRIDE = "<<std::to_string(((pooling_layer*)flow[i]) -> stride)<<";\n";
	        	if(((pooling_layer*)flow[i]) -> overlap){
		        	outfile<<"WINDOW_SIZE = "<<std::to_string(((pooling_layer*)flow[i]) -> window_size)<<";\n";
	        	}
	        }elif(flow[i] -> layer_type == "local_response_normalization"){
	        	outfile<<"ALPHA = "<<std::to_string(((local_response_normalization_layer*)flow[i]) -> alpha)<<";\n";
	        	outfile<<"BETA = "<<std::to_string(((local_response_normalization_layer*)flow[i]) -> beta)<<";\n";
	        	outfile<<"K = "<<std::to_string(((local_response_normalization_layer*)flow[i]) -> k)<<";\n";
	        	outfile<<"N = "<<std::to_string(((local_response_normalization_layer*)flow[i]) -> n)<<";\n";
	        }elif(flow[i] -> layer_type == "dropout"){
	        	outfile<<"DROPOUT_RATE = "<<std::to_string(((dropout_layer*)flow[i]) -> dropout_rate)<<";\n";
	        }
	    	outfile<<"&\n\n";
	    }
		outfile.close();

	}else {
		std::cout<<"unable to open file..."<<std::endl;
		exit(0);
	}
}
