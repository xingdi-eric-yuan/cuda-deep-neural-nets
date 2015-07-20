#include "layer_bank.h"

// kernel
convolutional_kernel::convolutional_kernel(){
	weight_decay = 0.0;
	kernel_size = 0;
}
convolutional_kernel::~convolutional_kernel(){
}

void convolutional_kernel::init_config(int width, float weightDecay){
    kernel_size = width;
    weight_decay = weightDecay;
    w.setSize(kernel_size, kernel_size, 3);
    w.randu();
    b.setAll(0.0);
    wgrad.setSize(kernel_size, kernel_size, 3);
    wd2.setSize(kernel_size, kernel_size, 3);
    bgrad.setAll(0.0);
    bd2.setAll(0.0);
    float epsilon = 0.12;
    w = w * epsilon;
}

// layer
convolutional_layer::convolutional_layer(){
	stride = 1;
	padding = 0;
	combine_feature_map = 0;
    momentum_derivative = 0.0;
    momentum_second_derivative = 0.0;
    iter = 0;
    mu = 0.0;
}
convolutional_layer::~convolutional_layer(){
    kernels.clear();
    vector<convolutional_kernel*>().swap(kernels);
}

void convolutional_layer::init_config(string namestr, int kernel_amount, int kernel_size, int output_amount, int _padding, int _stride, float weight_decay, string outputformat){
    layer_type = "convolutional";
    layer_name = namestr;
    output_format = outputformat;
    padding = _padding;
    stride = _stride;
    combine_feature_map = output_amount;

    kernels.clear();
    for(int i = 0; i < kernel_amount; ++i){
        convolutional_kernel *tmp_kernel = new convolutional_kernel();
        tmp_kernel -> init_config(kernel_size, weight_decay);
        kernels.push_back(tmp_kernel);
    }
}

void convolutional_layer::init_weight(network_layer* previous_layer){

    if(combine_feature_map > 0){
    	combine_weight.setSize(kernels.size(), combine_feature_map, 1);
    	combine_weight.randu();
    	combine_weight = combine_weight.mul(0.12);
    	combine_weight_grad.setSize(kernels.size(), combine_feature_map, 1);
    	combine_weight_d2.setSize(kernels.size(), combine_feature_map, 1);
    }
    // updater
    Mat tmpw(kernels[0] -> w.rows, kernels[0] -> w.cols, 3);
    velocity_combine_weight.setSize(combine_weight.rows, combine_weight.cols, 1);
    second_derivative_combine_weight.setSize(combine_weight.rows, combine_weight.cols, 1);

    velocity_w.resize(kernels.size());
    velocity_b.resize(kernels.size());
    second_derivative_w.resize(kernels.size());
    second_derivative_b.resize(kernels.size());
    for(int i = 0; i < kernels.size(); ++i){
        tmpw.copyTo(velocity_w[i]);
        tmpw.copyTo(second_derivative_w[i]);
        velocity_b[i].zeros();
        second_derivative_b[i].zeros();
    }
    iter = 0;
    mu = 1e-2;
    convolutional_layer::setMomentum();
}

void convolutional_layer::setMomentum(){
    if(iter < 30){
        momentum_derivative = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_derivative = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void convolutional_layer::update(int iter_num){
    iter = iter_num;
    if(iter == 30) convolutional_layer::setMomentum();
    vector3f allmu(mu, mu, mu);
    for(int i = 0; i < kernels.size(); ++i){
        second_derivative_w[i] = second_derivative_w[i] * momentum_second_derivative + kernels[i] -> wd2 * (1.0 - momentum_second_derivative);
        learning_rate_w = divide(lrate_w, (second_derivative_w[i] + allmu));
        velocity_w[i] = velocity_w[i] * momentum_derivative + kernels[i] -> wgrad.mul(learning_rate_w) * (1.0 - momentum_derivative);
        kernels[i] -> w = kernels[i] -> w - velocity_w[i];

        second_derivative_b[i] = second_derivative_b[i] * momentum_second_derivative + kernels[i] -> bd2 * (1.0 - momentum_second_derivative);
        learning_rate_b = divide(lrate_b, (second_derivative_b[i] + allmu));
        velocity_b[i] = velocity_b[i] * momentum_derivative + kernels[i] -> bgrad.mul(learning_rate_b) * (1.0 - momentum_derivative);
        kernels[i] -> b = kernels[i] -> b - velocity_b[i];
    }
    if(combine_feature_map > 0){
        second_derivative_combine_weight = second_derivative_combine_weight * momentum_second_derivative + combine_weight_d2 * (1.0 - momentum_second_derivative);
        learning_rate_w = divide(lrate_w, (second_derivative_combine_weight + mu));
        velocity_combine_weight = velocity_combine_weight * momentum_derivative + combine_weight_grad.mul(learning_rate_w) * (1.0 - momentum_derivative);
        combine_weight = combine_weight - velocity_combine_weight;
    }
}

void convolutional_layer::forwardPass(int nsamples, network_layer* previous_layer){

    std::vector<std::vector<Mat> > input;
    if(previous_layer -> output_format == "image"){
        input = previous_layer -> output_vector;
    }else{
        // no!!!!
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    Mat c_weight;
    if(combine_feature_map > 0){
        c_weight = exp(combine_weight);
        c_weight = divide(c_weight, repmat(reduce(c_weight, REDUCE_TO_SINGLE_ROW, REDUCE_SUM), c_weight.rows, 1));
    }
    output_vector.clear();
    for(int i = 0; i < input.size(); ++i){
        std::vector<Mat> eachsample;
        for(int j = 0; j < input[i].size(); ++j){
            std::vector<Mat> tmpvec;
            for(int k = 0; k < kernels.size(); ++k){
                Mat temp = rot90(kernels[k] -> w, 2);
                Mat tmpconv = conv2(input[i][j], temp, CONV_VALID, padding, stride);
                tmpconv = tmpconv + kernels[k] -> b;
                tmpvec.push_back(tmpconv);
            }
            if(combine_feature_map > 0){
                std::vector<Mat> outputvec(combine_feature_map);
                Mat zero(tmpvec[0].rows, tmpvec[0].cols, 3);
                for(int k = 0; k < outputvec.size(); k++) {zero.copyTo(outputvec[k]);}
                for(int m = 0; m < kernels.size(); m++){
                    for(int n = 0; n < combine_feature_map; n++){
                    	vector3f tmpvec3;
                    	tmpvec3.setAll(c_weight.get(m, n, 0));
                        outputvec[n] = outputvec[n] + tmpvec[m].mul(tmpvec3);
                    }
                }
                for(int k = 0; k < outputvec.size(); k++) {eachsample.push_back(outputvec[k]);}
                outputvec.clear();
            }
            else{
                for(int k = 0; k < tmpvec.size(); k++) {eachsample.push_back(tmpvec[k]);}
            }
            tmpvec.clear();
        }
        output_vector.push_back(eachsample);
    }
    input.clear();
    std::vector<std::vector<Mat> >().swap(input);
}

void convolutional_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    convolutional_layer::forwardPass(nsamples, previous_layer);
}

void convolutional_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    std::vector<std::vector<Mat> > derivative;
    std::vector<std::vector<Mat> > deriv2;
    if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0].rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0].rows);
    }else{
        derivative = next_layer -> delta_vector;
        deriv2 = next_layer -> d2_vector;
    }
    if(previous_layer -> output_format != "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(previous_layer -> output_vector.size());
    d2_vector.resize(previous_layer -> output_vector.size());
    for(int i = 0; i < delta_vector.size(); i++){
        delta_vector[i].resize(previous_layer -> output_vector[i].size());
        d2_vector[i].resize(previous_layer -> output_vector[i].size());
    }
    Mat tmp, tmp2, tmp3;
    std::vector<Mat> tmp_wgrad(kernels.size());
    std::vector<Mat> tmp_wd2(kernels.size());
    std::vector<vector3f> tmpgradb;
    std::vector<vector3f> tmpbd2;
    tmp.setSize(kernels[0] -> w.rows, kernels[0] -> w.cols, 3);
    vector3f tmpscalar(0.0, 0.0, 0.0);
    for(int m = 0; m < kernels.size(); m++) {
        tmp.copyTo(tmp_wgrad[m]);
        tmp.copyTo(tmp_wd2[m]);
        tmpgradb.push_back(tmpscalar);
        tmpbd2.push_back(tmpscalar);
    }
    Mat c_weight, c_weightgrad, c_weightd2;
    if(combine_feature_map > 0){
        c_weight = exp(combine_weight);
        c_weight = divide(c_weight, repmat(reduce(c_weight, REDUCE_TO_SINGLE_ROW, REDUCE_SUM), c_weight.rows, 1));
        c_weightgrad.setSize(c_weight.rows, c_weight.cols, 1);
        c_weightd2.setSize(c_weight.rows, c_weight.cols, 1);
    }

    for(int i = 0; i < nsamples; i++){
        for(int j = 0; j < previous_layer -> output_vector[i].size(); j++){
            std::vector<Mat> sensi(kernels.size());
            std::vector<Mat> sensid2(kernels.size());
            Mat tmp_delta;
            Mat tmp_d2;
            tmp.setSize(output_vector[0][0].rows, output_vector[0][0].cols, 3);
            for(int m = 0; m < kernels.size(); m++) {
                tmp.copyTo(sensi[m]);
                tmp.copyTo(sensid2[m]);
                if(combine_feature_map > 0){
                    for(int n = 0; n < combine_feature_map; n++){
                    	vector3f tmpvec3_1, tmpvec3_2;
                    	tmpvec3_1.setAll(c_weight.get(m, n, 0));
                    	float tmpfloat = c_weight.get(m, n, 0) * c_weight.get(m, n, 0);
                    	tmpvec3_2.setAll(tmpfloat);
                        sensi[m] = sensi[m] + derivative[i][j * combine_feature_map + n].mul(tmpvec3_1);
                        sensid2[m] = sensid2[m] + deriv2[i][j * combine_feature_map + n].mul(tmpvec3_2);
                    }
                }else{
                    sensi[m] = sensi[m] + derivative[i][j * kernels.size() + m];
                    sensid2[m] = sensid2[m] + deriv2[i][j * kernels.size() + m];
                }

                if(stride > 1){
                    int len = previous_layer -> output_vector[0][0].rows + padding * 2 - kernels[0] -> w.rows + 1;
                    sensi[m] = interpolation(sensi[m], len);
                    sensid2[m] = interpolation(sensid2[m], len);
                }
                if(m == 0){
                    tmp_delta = conv2(sensi[m], kernels[m] -> w, CONV_FULL, 0, 1);
                    tmp_d2 = conv2(sensid2[m], pow(kernels[m] -> w, 2), CONV_FULL, 0, 1);
                }else{
                    tmp_delta = tmp_delta + conv2(sensi[m], kernels[m] -> w, CONV_FULL, 0, 1);
                    tmp_d2 = tmp_d2 + conv2(sensid2[m], pow(kernels[m] -> w, 2), CONV_FULL, 0, 1);
                }
                Mat input;
                if(padding > 0){
                    input = dopadding(previous_layer -> output_vector[i][j], padding);
                }else{
                    previous_layer -> output_vector[i][j].copyTo(input);
                }
                tmp2 = rot90(sensi[m], 2);
                tmp3 = rot90(sensid2[m], 2);
                tmp_wgrad[m] = tmp_wgrad[m] + conv2(input, tmp2, CONV_VALID, 0, 1);
                tmp_wd2[m] = tmp_wd2[m] + conv2(pow(input, 2), tmp3, CONV_VALID, 0, 1);
                tmpgradb[m] = tmpgradb[m] + sum(tmp2);
                tmpbd2[m] = tmpbd2[m] + sum(tmp3);

                if(combine_feature_map > 0){
                    // combine feature map weight matrix (after softmax)
                    previous_layer -> output_vector[i][j].copyTo(input);
                    tmp2 = rot90(kernels[m] -> w, 2);
                    tmp2.copyTo(tmp3);
                    tmp2 = conv2(input, tmp2, CONV_VALID, padding, stride);
                    tmp3 = conv2(pow(input, 2), pow(tmp3, 2), CONV_VALID, padding, stride);
                    for(int n = 0; n < combine_feature_map; n++){
                        Mat tmpd;
                        tmpd = tmp2.mul(derivative[i][j * combine_feature_map + n]);
                        c_weightgrad.set(m, n, 0, c_weightgrad.get(m, n, 0) + sum(tmpd).get(0));
                        tmpd = tmp3.mul(deriv2[i][j * combine_feature_map + n]);
                        c_weightd2.set(m, n, 0, c_weightd2.get(m, n, 0) + sum(tmpd).get(0));
                    }
                }
            }
            if(padding > 0){
                tmp_delta = depadding(tmp_delta, padding);
                tmp_d2 = depadding(tmp_d2, padding);
            }
            tmp_delta.copyTo(delta_vector[i][j]);
            tmp_d2.copyTo(d2_vector[i][j]);
            sensi.clear();
            std::vector<Mat>().swap(sensi);
            sensid2.clear();
            std::vector<Mat>().swap(sensid2);
        }
    }
    for(int i = 0; i < kernels.size(); i++){
    	vector3f tmpvec3;
    	tmpvec3.setAll(kernels[i] -> weight_decay);
        kernels[i] -> wgrad = divide(tmp_wgrad[i], nsamples) + kernels[i] -> w * kernels[i] -> weight_decay;
        kernels[i] -> wd2 = divide(tmp_wd2[i], nsamples) + tmpvec3;
        kernels[i] -> bgrad = divide(tmpgradb[i], nsamples);
        kernels[i] -> bd2 = divide(tmpbd2[i], nsamples);
    }

    if(combine_feature_map > 0){
        tmp2 = c_weightgrad.mul(c_weight);
        tmp2 = repmat(reduce(tmp2, REDUCE_TO_SINGLE_ROW, REDUCE_SUM), c_weightgrad.rows, 1);
        tmp = c_weightgrad - tmp2;
        tmp = c_weight.mul(tmp);
        tmp = divide(tmp, nsamples);
        tmp.copyTo(combine_weight_grad);

        tmp2 = c_weightd2.mul(c_weight);
        tmp2 = repmat(reduce(tmp2, REDUCE_TO_SINGLE_ROW, REDUCE_SUM), c_weightd2.rows, 1);
        tmp = c_weightd2 - tmp2;
        tmp = c_weight.mul(tmp);
        tmp = divide(tmp, nsamples);
        tmp.copyTo(combine_weight_d2);
    }
    tmp_wgrad.clear();
    std::vector<Mat>().swap(tmp_wgrad);
    tmp_wd2.clear();
    std::vector<Mat>().swap(tmp_wd2);
    derivative.clear();
    std::vector<std::vector<Mat> >().swap(derivative);
    deriv2.clear();
    std::vector<std::vector<Mat> >().swap(deriv2);
    tmpgradb.clear();
    std::vector<vector3f>().swap(tmpgradb);
    tmpbd2.clear();
    std::vector<vector3f>().swap(tmpbd2);
}



