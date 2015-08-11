#include "io_config.h"

// READ TO STRING
// read file into string
string read_2_string(string File_name){
    string res = "";
    ifstream infile(File_name);
    string line;
    while(getline(infile, line)){
        res += line;
    }
    return res;
}

// GET WORD BOOL
// get boolean value
bool get_word_bool(string &str, string name){

    size_t pos = str.find(name);
    int i = pos + 1;
    bool res = true;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        if(!content.compare("true")) res = true;
        else res = false;
    }
    str.erase(pos, i - pos + 1);
    return res;
}

// GET WORD INT
// get int value
int get_word_int(string &str, string name){

    size_t pos = str.find(name);
    int i = pos + 1;
    int res = 1;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = stoi(content);
    }
    str.erase(pos, i - pos + 1);
    return res;
}

// GET WORD DOUBLE
// get double value
double get_word_double(string &str, string name){

    size_t pos = str.find(name);
    int i = pos + 1;
    double res = 0.0;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = atof(content.c_str());
    }
    str.erase(pos, i - pos + 1);
    return res;
}

// GET WORD STRING
// get string value
string get_word_string(string &str, string name){

    size_t pos = str.find(name);
    int i = pos + 1;
    string res = "";
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = content;
    }
    str.erase(pos, i - pos + 1);
    return res;
}

// GET WORD TYPE
// get network layer type
int get_word_type(string &str, string name){

    size_t pos = str.find(name);
    int i = pos + 1;
    int res = 0;
    while(1){
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';'){
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        // layer type
        if(!content.compare("input")) res = 0;
        elif(!content.compare("convolutional")) res = 1;
        elif(!content.compare("fully_connected")) res = 2;
        elif(!content.compare("softmax")) res = 3;
        elif(!content.compare("combine")) res = 4;
        elif(!content.compare("branch")) res = 5;
        elif(!content.compare("non_linearity")) res = 6;
        elif(!content.compare("pooling")) res = 7;
        elif(!content.compare("local_response_normalization")) res = 8;
        elif(!content.compare("dropout")) res = 9;
    }
    str.erase(pos, i - pos + 1);
    return res;
}

// DELETE COMMENT
// delete /**/ and comments inside
void delete_comment(string &str){
    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1){
        if(head == str.length()) break;
        if(str[head] == '/'){
            tail = head + 1;
            while(1){
                if(tail == str.length()) break;
                if(str[tail] == '/') break;
                ++ tail;
            }
            str.erase(head, tail - head + 1);
        }else ++ head;
    }
}

// DELETE SPACE
// delete spaces
void delete_space(string &str){
    if(str.empty()) return;
    int i = 0;
    while(1){
        if(i == str.length()) break;
        if(str[i] == '\t' || str[i] == '\n' || str[i] == ' '){
            str.erase(str.begin() + i);
        }else ++ i;
    }
}

// GET LAYERS CONFIG
// get layer config and build network use layers
void get_layers_config(string &str, std::vector<network_layer*> &flow){
    vector<string> layers;
    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1){
        if(head == str.length()) break;
        if(str[head] == '$'){
            tail = head + 1;
            while(1){
                if(tail == str.length()) break;
                if(str[tail] == '&') break;
                ++ tail;
            }
            string sub = str.substr(head, tail - head + 1);
            if(sub[sub.length() - 1] == '&'){
                sub.erase(sub.begin() + sub.length() - 1);
                sub.erase(sub.begin());
                layers.push_back(sub);
            }
            str.erase(head, tail - head + 1);
        }else ++ head;
    }
    for(int i = 0; i < layers.size(); i++){
        int type = get_word_type(layers[i], "LAYER");
        switch(type){
            case 0:{ // input layer
                input_layer *tmp = new input_layer();
                string namestr = get_word_string(layers[i], "NAME");
                int batchsize = get_word_int(layers[i], "BATCH_SIZE");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                tmp -> init_config(namestr, batchsize, of);
                flow.push_back(tmp);
                break;
            }case 1:{ // convolutional layer
                convolutional_layer *tmp = new convolutional_layer();
                string namestr = get_word_string(layers[i], "NAME");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                int ks = get_word_int(layers[i], "KERNEL_SIZE");
                int ka = get_word_int(layers[i], "KERNEL_AMOUNT");
                int cm = get_word_int(layers[i], "COMBINE_MAP");
                double wd = get_word_double(layers[i], "WEIGHT_DECAY");
                int pa = get_word_int(layers[i], "PADDING");
                int st = get_word_int(layers[i], "STRIDE");
                tmp -> init_config(namestr, ka, ks, cm, pa, st, wd, of);
                flow.push_back(tmp);
                break;
            }case 2:{ // full_connectedlayer
                fully_connected_layer *tmp = new fully_connected_layer();
                string namestr = get_word_string(layers[i], "NAME");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                int hs = get_word_int(layers[i], "NUM_HIDDEN_NEURONS");
                double wd = get_word_double(layers[i], "WEIGHT_DECAY");
                tmp -> init_config(namestr, hs, wd, of);
                flow.push_back(tmp);
                break;
            }
            case 3:{ // softmax layer
                softmax_layer *tmp = new softmax_layer();
                string namestr = get_word_string(layers[i], "NAME");
                int classes = get_word_int(layers[i], "NUM_CLASSES");
                double wd = get_word_double(layers[i], "WEIGHT_DECAY");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                tmp -> init_config(namestr, classes, wd, of);
                flow.push_back(tmp);
                break;
            }case 4:{ // combine layer
                break;
            }case 5:{ // branch layer
                break;
            }
            case 6:{ // non linearity layer
                non_linearity_layer *tmp = new non_linearity_layer();
                string namestr = get_word_string(layers[i], "NAME");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                string strmethod = get_word_string(layers[i], "METHOD");
                int method = 0;
                // non-linearity
                if(strmethod == "sigmoid") method = 0;
                elif(strmethod == "tanh") method = 1;
                elif(strmethod == "relu") method = 2;
                elif(strmethod == "leaky_relu") method = 3;
                tmp -> init_config(namestr, method, of);
                flow.push_back(tmp);
                break;
            }case 7:{ // pooling layer
                pooling_layer *tmp = new pooling_layer();
                string namestr = get_word_string(layers[i], "NAME");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                string strmethod = get_word_string(layers[i], "METHOD");
                int method = 0;
                if(strmethod == "max") method = 0;
                elif(strmethod == "mean") method = 1;
                elif(strmethod == "stochastic") method = 2;
                bool if_overlap = get_word_bool(layers[i], "OVERLAP");
                int stride = get_word_int(layers[i], "STRIDE");
                if(true == if_overlap){
                    int windowsize = get_word_int(layers[i], "WINDOW_SIZE");
                    tmp -> init_config(namestr, method, of, stride, windowsize);
                }else{
                    tmp -> init_config(namestr, method, of, stride);
                }
                flow.push_back(tmp);
                break;
            }case 8:{ // local response normalization layer
                local_response_normalization_layer *tmp = new local_response_normalization_layer();
                string namestr = get_word_string(layers[i], "NAME");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                double alpha = get_word_double(layers[i], "ALPHA");
                double beta = get_word_double(layers[i], "BETA");
                double k = get_word_double(layers[i], "K");
                int n = get_word_int(layers[i], "N");
                tmp -> init_config(namestr, of, alpha, beta, k, n);
                flow.push_back(tmp);

                break;
            }case 9:{ // dropout layer
                dropout_layer *tmp = new dropout_layer();
                string namestr = get_word_string(layers[i], "NAME");
                string of = get_word_string(layers[i], "OUTPUT_TYPE");
                double dor = get_word_double(layers[i], "DROPOUT_RATE");
                tmp -> init_config(namestr, of, dor);
                flow.push_back(tmp);
                break;
            }
        }
    }
}

// BUILD NETWORK FROM CONFIG FILE
// build network from config file
void buildNetworkFromConfigFile(const string &filepath, std::vector<network_layer*> &flow, const std::vector<cpuMat*> &x, const cpuMat *y){

    string str = read_2_string(filepath);
    delete_space(str);
    delete_comment(str);
    get_layers_config(str, flow);
    is_gradient_checking = get_word_bool(str, "IS_GRADIENT_CHECKING");
    use_log = get_word_bool(str, "USE_LOG");
    training_epochs = get_word_int(str, "TRAINING_EPOCHS");
    iter_per_epo = get_word_int(str, "ITER_PER_EPO");
    lrate_w = get_word_double(str, "LEARNING_RATE_W");
    lrate_b = get_word_double(str, "LEARNING_RATE_B");
    momentum_w_init = get_word_double(str, "MOMENTUM_W_INIT");
    momentum_d2_init = get_word_double(str, "MOMENTUM_D2_INIT");
    momentum_w_adjust = get_word_double(str, "MOMENTUM_W_ADJUST");
    momentum_d2_adjust = get_word_double(str, "MOMENTUM_D2_ADJUST");

    if(is_gradient_checking){
        ((input_layer*)flow[0]) -> batch_size = 2;
    }

    cout<<"****************************************************************************"<<endl
        <<"**                    READ CONFIG FILE COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;

    cout<<"********** general config **********"<<endl;
    cout<<"is_gradient_checking = "<<is_gradient_checking<<endl;
    cout<<"use_log = "<<use_log<<endl;
    cout<<"training epochs = "<<training_epochs<<endl;
    cout<<"iteration per epoch = "<<iter_per_epo<<endl;
    cout<<"learning rate for weight matrices = "<<lrate_w<<endl;
    cout<<"learning rate for bias = "<<lrate_b<<endl;
    cout<<"********** layers config **********"<<endl;

    forwardPassInit(x, y, flow);
    printNetwork(flow);
}

// BUILD NETWORK FROM SAVED DATA
// build network from saved data
void buildNetworkFromSavedData(const string &folder, std::vector<network_layer*> &flow, const std::vector<cpuMat*> &x, const cpuMat *y){
    string path = folder + "/saved_config.txt";
    buildNetworkFromConfigFile(path, flow, x, y);
    readNetwork(folder, flow);
}

void saveNetworkConfig(const std::string &path, const std::vector<network_layer*> &flow){

    ofstream outfile(path.c_str(), ios::out);
    outfile.precision(16);
    if (outfile.is_open()){

        outfile<<"/*\nSaved Network Config File\n*/\n";
        outfile<<"/*******************************************************\n";
        outfile<<"*\n";
        outfile<<"* General Parameters Config\n";
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
        outfile<<"* Layers Config\n";
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




