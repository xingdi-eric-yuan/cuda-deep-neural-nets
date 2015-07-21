#include "gradient_checking.h"

using namespace std;

void gradient_checking(const std::vector<cpuMat> &sampleX, const cpuMat &sampleY, std::vector<network_layer*> &flow, Mat* gradient, Mat* alt){
    float epsilon = 1e-3;
    if(alt -> channels == 3){
        for(int i = 0; i < alt -> rows; i++){
            for(int j = 0; j < alt -> cols; j++){
                for(int ch = 0; ch < 3; ch++){
                    float memo = alt -> get(i, j, ch);
                    alt -> set(i, j, ch, memo + epsilon);
                    forwardPass(sampleX, sampleY, flow);
                    float value1 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                    alt -> set(i, j, ch, memo - epsilon);
                    forwardPass(sampleX, sampleY, flow);
                    float value2 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                    float tp = (value1 - value2) / (2 * epsilon);
                    float tpgrad = gradient -> get(i, j, ch);
                    if(tp == 0.0 && tpgrad == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<tpgrad<<", "<<1<<endl;
                    else cout<<i<<", "<<j<<", "<<ch<<", "<<tp<<", "<<tpgrad<<", "<<tp / tpgrad<<endl;
                    alt -> set(i, j, ch, memo);
                }
            }
        }
    }else{
    	int ch = 0;
        for(int i = 0; i < alt -> rows; i++){
            for(int j = 0; j < alt -> cols; j++){
                float memo = alt -> get(i, j, ch);
                alt -> set(i, j, ch, memo + epsilon);
                forwardPass(sampleX, sampleY, flow);
                float value1 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                alt -> set(i, j, ch, memo - epsilon);
                forwardPass(sampleX, sampleY, flow);
                float value2 = ((softmax_layer*)flow[flow.size() - 1]) -> network_cost;
                float tp = (value1 - value2) / (2 * epsilon);
                float tpgrad = gradient -> get(i, j, ch);
                if(tp == 0.0 && tpgrad == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<tpgrad<<", "<<1<<endl;
                else cout<<i<<", "<<j<<", "<<tp<<", "<<tpgrad<<", "<<tp / tpgrad<<endl;
                alt -> set(i, j, ch, memo);
            }
        }
    }
}

void gradientChecking_SoftmaxLayer(std::vector<network_layer*> &flow, const std::vector<cpuMat> &sampleX, const cpuMat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the
    //cost function and dJ function are correct)

    // forwardPassInit(sampleX, sampleY, flow);
    forwardPass(sampleX, sampleY, flow);
    backwardPass(flow);

    Mat *p;// = new Mat();
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer !!!!"<<endl;
    cout<<"################################################"<<endl;

    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "softmax"){
            cout<<"---------------- checking layer number "<<i<<" ..."<<endl;
            p = ((softmax_layer*)flow[i]) -> w;
            gradient_checking(sampleX, sampleY, flow, ((softmax_layer*)flow[i]) -> wgrad, p);
        }
    }
}

void gradientChecking_FullyConnectedLayer(std::vector<network_layer*> &flow, const std::vector<cpuMat> &sampleX, const cpuMat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the
    //cost function and dJ function are correct)

    // forwardPassInit(sampleX, sampleY, flow);
    forwardPass(sampleX, sampleY, flow);
    backwardPass(flow);

    Mat *p;
    cout<<"################################################"<<endl;
    cout<<"## test fully connected layer !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "fully_connected"){
            cout<<"---------------- checking layer number "<<i<<" ..."<<endl;
            p = ((fully_connected_layer*)flow[i]) -> w;
            gradient_checking(sampleX, sampleY, flow, ((fully_connected_layer*)flow[i]) -> wgrad, p);
        }
    }

}

void gradientChecking_ConvolutionalLayer(std::vector<network_layer*> &flow, const std::vector<cpuMat> &sampleX, const cpuMat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the
    //cost function and dJ function are correct)

    // forwardPassInit(sampleX, sampleY, flow);
    forwardPass(sampleX, sampleY, flow);
    backwardPass(flow);

    Mat *p;
    cout<<"################################################"<<endl;
    cout<<"## test convolutional layer !!!!"<<endl;
    cout<<"################################################"<<endl;

    for(int i = 0; i < flow.size(); i++){
        if(flow[i] -> layer_type == "convolutional"){
            cout<<"---------------- checking layer number "<<i<<" ..."<<endl;

            for(int j = 0; j < ((convolutional_layer*)flow[i]) -> kernels.size(); j++){
                cout<<"------ checking kernel number "<<j<<" ..."<<endl;
                p = ((convolutional_layer*)flow[i]) -> kernels[j] -> w;
                gradient_checking(sampleX, sampleY, flow, ((convolutional_layer*)flow[i]) -> kernels[j] -> wgrad, p);
            }

            cout<<"-------------------------------------- checking combine feature map weight"<<endl;
            p = ((convolutional_layer*)flow[i]) -> combine_weight;
            gradient_checking(sampleX, sampleY, flow, ((convolutional_layer*)flow[i]) -> combine_weight_grad, p);
        }
    }
}

void gradient_checking_network_layers(std::vector<network_layer*> &flow, const std::vector<cpuMat> &sampleX, const cpuMat &sampleY){

    // delete dropout layer when doing gradient checking
    std::vector<network_layer*> tmpflow(flow);
    int i = 0;
    while(true){
        if(i >= tmpflow.size()) break;
        if(tmpflow[i] -> layer_type == "dropout"){
            tmpflow.erase(tmpflow.begin() + i);
        }else ++i;
    }
    //gradientChecking_ConvolutionalLayer(tmpflow, sampleX, sampleY);
    //gradientChecking_FullyConnectedLayer(tmpflow, sampleX, sampleY);
    gradientChecking_SoftmaxLayer(tmpflow, sampleX, sampleY);

}






