#include "train_network.h"

using namespace std;

void forwardPassInit(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){
     //cout<<"---------------- forward init"<<endl;
    // forward pass
    int batch_size = 0;
    for(int i = 0; i < flow.size(); i++){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            batch_size = ((input_layer*)flow[i]) -> batch_size;
            ((input_layer*)flow[i]) -> forwardPass(batch_size, x, y);
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> init_weight(flow[i - 1]);
            ((convolutional_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> init_weight(flow[i - 1]);
            ((fully_connected_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> init_weight(flow[i - 1]);
            ((softmax_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }/*elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }*/
    }
}

void forwardPass(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){

    //cout<<"---------------- forward "<<endl;
    // forward pass
    int batch_size = 0;
    Mat *tmp = new Mat();
	vector3f *tmpvec3 = new vector3f();
    float J1 = 0, J2 = 0, J3 = 0, J4 = 0;
    for(int i = 0; i < flow.size(); i++){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            batch_size = ((input_layer*)flow[i]) -> batch_size;
            ((input_layer*)flow[i]) -> forwardPass(batch_size, x, y);
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
            // get cost
            for(int k = 0; k < ((convolutional_layer*)flow[i]) -> kernels.size(); ++k){
            	safeGetPt(tmp, square(((convolutional_layer*)flow[i]) -> kernels[k] -> w));
            	tmpvec3 = sum(tmp);
                J4 += sum(tmpvec3) * ((convolutional_layer*)flow[i]) -> kernels[k] -> weight_decay / 2.0;
            }
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
            // get cost
            safeGetPt(tmp, square(((fully_connected_layer*)flow[i]) -> w));
        	tmpvec3 = sum(tmp);
            J3 += sum(tmpvec3) * ((fully_connected_layer*)flow[i]) -> weight_decay / 2.0;
        }elif(flow[i] -> layer_type == "softmax"){
        	((softmax_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
            // get cost
            Mat *groundTruth = new Mat(((softmax_layer*)flow[i]) -> output_size, batch_size, 1);
            for(int i = 0; i < batch_size; i++){
            	groundTruth -> set(((input_layer*)flow[0]) -> label -> get(0, i, 0), i, 0, 1.0);
            }
            safeGetPt(tmp, log(flow[i] -> output_matrix));
            safeGetPt(tmp, multiply_elem(tmp, groundTruth));
        	tmpvec3 = sum(tmp);
            J1 += -sum(tmpvec3) / batch_size;
            safeGetPt(tmp, square(((softmax_layer*)flow[i]) -> w));
            tmpvec3 = sum(tmp);
            J2 += sum(tmpvec3) * ((softmax_layer*)flow[i]) -> weight_decay / 2.0;
            groundTruth -> release();
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }/*elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> forwardPass(batch_size, flow[i - 1]);
        }*/
    }
    ((softmax_layer*)flow[flow.size() - 1]) -> network_cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking)
    	cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<((softmax_layer*)flow[flow.size() - 1]) -> network_cost;//endl;
    tmp -> release();
}

void forwardPassTest(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){

    // forward pass
    int batch_size = x.size();
    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "input"){
            ((input_layer*)flow[i]) -> forwardPassTest(batch_size, x, y);
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }/*elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> forwardPassTest(batch_size, flow[i - 1]);
        }*/
    }
}

void backwardPass(std::vector<network_layer*> &flow){
    //cout<<"---------------- backward"<<endl;
    // backward pass
    int batch_size = ((input_layer*)flow[0]) -> batch_size;
    Mat *groundTruth = new Mat(((softmax_layer*)flow[flow.size() - 1]) -> output_size, batch_size, 1);
    for(int i = 0; i < batch_size; i++){
    	groundTruth -> set(((input_layer*)flow[0]) -> label -> get(0, i, 0), i, 0, 1.0);
    }
    for(int i = flow.size() - 1; i >= 0; --i){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            ((input_layer*)flow[i]) -> backwardPass();
        }elif(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], groundTruth);
        }elif(flow[i] -> layer_type == "combine"){
            ((combine_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "branch"){
            ((branch_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "non_linearity"){
            ((non_linearity_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "pooling"){
            ((pooling_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            ((local_response_normalization_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }/*elif(flow[i] -> layer_type == "dropout"){
            ((dropout_layer*)flow[i]) -> backwardPass(batch_size, flow[i - 1], flow[i + 1]);
        }*/
    }
    groundTruth -> release();
}

void updateNetwork(std::vector<network_layer*> &flow, int iter){
    //cout<<"---------------- update"<<endl;
    for(int i = 0; i < flow.size(); ++i){
        //cout<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "convolutional"){
            ((convolutional_layer*)flow[i]) -> update(iter);
        }elif(flow[i] -> layer_type == "fully_connected"){
            ((fully_connected_layer*)flow[i]) -> update(iter);
        }elif(flow[i] -> layer_type == "softmax"){
            ((softmax_layer*)flow[i]) -> update(iter);
        }
    }
}

void printNetwork(std::vector<network_layer*> &flow){
    cout<<"****************************************************************************"<<endl
        <<"**                       NETWORK LAYERS                                     "<<endl
        <<"****************************************************************************"<<endl<<endl;
    for(int i = 0; i < flow.size(); ++i){
        cout<<"##-------------------layer number "<<i<<", layer name is "<<flow[i] -> layer_name<<endl;
        if(flow[i] -> layer_type == "input"){
            cout<<"batch size = "<<((input_layer*)flow[i]) -> batch_size<<endl;
        }elif(flow[i] -> layer_type == "convolutional"){
            cout<<"kernel amount = "<<((convolutional_layer*)flow[i]) -> kernels.size()<<endl;
            cout<<"kernel size = ["<<((convolutional_layer*)flow[i]) -> kernels[0] -> w -> rows<<", "<<((convolutional_layer*)flow[i]) -> kernels[0] -> w -> cols<<"]"<<endl;
            cout<<"padding = "<<((convolutional_layer*)flow[i]) -> padding<<endl;
            cout<<"stride = "<<((convolutional_layer*)flow[i]) -> stride<<endl;
            cout<<"combine feature map = "<<((convolutional_layer*)flow[i]) -> combine_feature_map<<endl;
            cout<<"weight decay = "<<((convolutional_layer*)flow[i]) -> kernels[0] -> weight_decay<<endl;
        }elif(flow[i] -> layer_type == "fully_connected"){
            cout<<"hidden size = "<<((fully_connected_layer*)flow[i]) -> size<<endl;
            cout<<"weight decay = "<<((fully_connected_layer*)flow[i]) -> weight_decay<<endl;
        }elif(flow[i] -> layer_type == "softmax"){
            cout<<"output size = "<<((softmax_layer*)flow[i]) -> output_size<<endl;
            cout<<"weight decay size = "<<((softmax_layer*)flow[i]) -> weight_decay<<endl;
        }elif(flow[i] -> layer_type == "combine"){
            ;
        }elif(flow[i] -> layer_type == "branch"){
            ;
        }elif(flow[i] -> layer_type == "non_linearity"){
            cout<<"non-lin method = "<<((non_linearity_layer*)flow[i]) -> method<<endl;
        }elif(flow[i] -> layer_type == "pooling"){
            cout<<"pooling method = "<<((pooling_layer*)flow[i]) -> method<<endl;
            cout<<"overlap = "<<((pooling_layer*)flow[i]) -> overlap<<endl;
            cout<<"stride = "<<((pooling_layer*)flow[i]) -> stride<<endl;
            cout<<"window size = "<<((pooling_layer*)flow[i]) -> window_size<<endl;
        }elif(flow[i] -> layer_type == "local_response_normalization"){
            cout<<"alpha = "<<((local_response_normalization_layer*)flow[i]) -> alpha<<endl;
            cout<<"beta = "<<((local_response_normalization_layer*)flow[i]) -> beta<<endl;
            cout<<"k = "<<((local_response_normalization_layer*)flow[i]) -> k<<endl;
            cout<<"n = "<<((local_response_normalization_layer*)flow[i]) -> n<<endl;
        }/*elif(flow[i] -> layer_type == "dropout"){
            cout<<"dropout rate = "<<((dropout_layer*)flow[i]) -> dropout_rate<<endl;
        }*/
        if(flow[i] -> output_format == "matrix"){
            cout<<"output matrix size is ["<<flow[i] -> output_matrix -> rows<<", "<<flow[i] -> output_matrix -> cols<<"]"<<endl;
        }else{
            cout<<"output vector size is "<<flow[i] -> output_vector.size()<<" * "<<flow[i] -> output_vector[0].size()<<" * ["<<flow[i] -> output_vector[0][0] -> rows<<", "<<flow[i] -> output_vector[0][0] -> cols<<"]"<<endl;
        }
        cout<<"---------------------"<<endl;
    }
}

void testNetwork(const std::vector<cpuMat*> &x, const cpuMat *y, std::vector<network_layer*> &flow){
/*
    int batch_size = 100;

    int batch_amount = x.size() / batch_size;
    int correct = 0;
    for(int i = 0; i < batch_amount; i++){
        std::vector<Mat> batchX(batch_size);
        Mat batchY = Mat::zeros(1, batch_size, CV_64FC1);
        for(int j = 0; j < batch_size; j++){
            x[i * batch_size + j].copyTo(batchX[j]);
        }
        y(Rect(i * batch_size, 0, batch_size, 1)).copyTo(batchY);
        forwardPassTest(batchX, batchY, flow);
        Mat res = findMax(flow[flow.size() - 1] -> output_matrix);

        //if(i < 3)
        //cout<<" "<<flow[flow.size() - 1] -> output_matrix<<endl<<endl<<endl;
        //cout<<" "<<res<<endl;
        correct += compareMatrix(res, batchY);
        batchX.clear();
        std::vector<Mat>().swap(batchX);
    }
    if(x.size() % batch_size){
        std::vector<Mat> batchX(x.size() % batch_size);
        Mat batchY = Mat::zeros(1, x.size() % batch_size, CV_64FC1);
        for(int j = 0; j < batchX.size(); j++){
            x[batch_amount * batch_size + j].copyTo(batchX[j]);
        }
        y(Rect(batch_amount * batch_size, 0, batchX.size(), 1)).copyTo(batchY);
        forwardPassTest(batchX, batchY, flow);
        Mat res = findMax(flow[flow.size() - 1] -> output_matrix);
        correct += compareMatrix(res, batchY);
        batchX.clear();
        std::vector<Mat>().swap(batchX);
    }
    cout<<"correct: "<<correct<<", total: "<<x.size()<<", accuracy: "<<float(correct) / (float)(x.size())<<endl;
    */
}

void trainNetwork(const std::vector<cpuMat*> &x, const cpuMat *y, const std::vector<cpuMat*> &tx, const cpuMat *ty, std::vector<network_layer*> &flow){

    forwardPassInit(x, y, flow);
    printNetwork(flow);

    if (is_gradient_checking){
        gradient_checking_network_layers(flow, x, y);
    }else{
    cout<<"****************************************************************************"<<endl
        <<"**                       TRAINING NETWORK......                             "<<endl
        <<"****************************************************************************"<<endl<<endl;
        int k = 0;
        for(int epo = 1; epo <= training_epochs; epo++){
            for(; k <= iter_per_epo * epo; k++){
                cout<<"epoch: "<<epo<<", iter: "<<k;//<<endl;
                forwardPass(x, y, flow);
                backwardPass(flow);
                updateNetwork(flow, k);
                cout<<" --- using gpu memory "<<MemoryMonitor::instance() -> getGpuMemory() / Mb<<" Mb"<<endl;
            }

            //cout<<"Test training data: "<<endl;
            //testNetwork(x, y, flow);
            //cout<<"Test testing data: "<<endl;
            //testNetwork(tx, ty, flow);

            //if(use_log){
            //    save2XML("log", i2str(k), flow);
            //}

        }

    }
    //*/

}


