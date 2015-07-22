#include "layer_bank.h"

// kernel
convolutional_kernel::convolutional_kernel(){
	weight_decay = 0.0;
	kernel_size = 0;
    w = new Mat();
    b = new vector3f();
    wgrad = new Mat();
    bgrad = new vector3f();
    wd2 = new Mat();
    bd2 = new vector3f();
}
convolutional_kernel::~convolutional_kernel(){
	w -> release();
	wgrad -> release();
    wd2 -> release();
}

void convolutional_kernel::init_config(int width, float weightDecay){
    kernel_size = width;
    weight_decay = weightDecay;
    w -> setSize(kernel_size, kernel_size, 3);
    w -> randu();
    b -> setAll(0.0);
    wgrad -> setSize(kernel_size, kernel_size, 3);
    wd2 -> setSize(kernel_size, kernel_size, 3);
    bgrad -> setAll(0.0);
    bd2 -> setAll(0.0);
    float epsilon = 0.12;
    (*w) *= epsilon;
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

    combine_weight = new Mat();
    combine_weight_grad = new Mat();
    combine_weight_d2 = new Mat();
    velocity_combine_weight = new Mat();
    second_derivative_combine_weight = new Mat();
    learning_rate_w = new Mat();
    learning_rate_b = new vector3f();
}
convolutional_layer::~convolutional_layer(){
    kernels.clear();
    vector<convolutional_kernel*>().swap(kernels);
    combine_weight -> release();
    combine_weight_grad -> release();
    combine_weight_d2 -> release();
    velocity_combine_weight -> release();
    second_derivative_combine_weight -> release();
    learning_rate_w -> release();
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
    	combine_weight -> setSize(kernels.size(), combine_feature_map, 1);
    	combine_weight -> randu();
    	//(*combine_weight) *= 0.12;
    	combine_weight_grad -> setSize(kernels.size(), combine_feature_map, 1);
    	combine_weight_d2 -> setSize(kernels.size(), combine_feature_map, 1);
    }
    // updater
    velocity_combine_weight -> setSize(combine_weight -> rows, combine_weight -> cols, 1);
    second_derivative_combine_weight -> setSize(combine_weight -> rows, combine_weight -> cols, 1);

    velocity_w.resize(kernels.size());
    velocity_b.resize(kernels.size());
    second_derivative_w.resize(kernels.size());
    second_derivative_b.resize(kernels.size());
    for(int i = 0; i < kernels.size(); ++i){
    	velocity_w[i] = new Mat();
    	second_derivative_w[i] = new Mat();
    	velocity_b[i] = new vector3f();
    	second_derivative_b[i] = new vector3f();
    	velocity_w[i] -> setSize(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    	second_derivative_w[i] -> setSize(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
        velocity_b[i] -> zeros();
        second_derivative_b[i] -> zeros();
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

    Mat tmp;

    for(int i = 0; i < kernels.size(); ++i){
        ((*(second_derivative_w[i])) * momentum_second_derivative).moveTo(*(second_derivative_w[i]));
        kernels[i] -> wd2 -> mul(1.0 - momentum_second_derivative).moveTo(tmp);
        (*(second_derivative_w[i])) += tmp;
        (*(second_derivative_w[i]) + mu).moveTo(tmp);
        divide(lrate_w, tmp).moveTo(*learning_rate_w);
        ((*(velocity_w[i])) * momentum_derivative).moveTo(*(velocity_w[i]));
        kernels[i] -> wgrad -> mul(*learning_rate_w).moveTo(tmp);
        tmp.mul(1.0 - momentum_derivative).moveTo(tmp);
        (*(velocity_w[i])) += tmp;
        (*(kernels[i] -> w)) -= (*(velocity_w[i]));

        *(second_derivative_b[i]) = (*(second_derivative_b[i])) * momentum_second_derivative + (*(kernels[i] -> bd2)) * (1.0 - momentum_second_derivative);
        *learning_rate_b = divide(lrate_b, (*(second_derivative_b[i]) + allmu));
        *(velocity_b[i]) = (*(velocity_b[i])) * momentum_derivative + kernels[i] -> bgrad -> mul(*learning_rate_b) * (1.0 - momentum_derivative);
        *(kernels[i] -> b) -= *(velocity_b[i]);
    }

    if(combine_feature_map > 0){
        ((*second_derivative_combine_weight) * momentum_second_derivative).moveTo(*second_derivative_combine_weight);
        combine_weight_d2 -> mul(1.0 - momentum_second_derivative).moveTo(tmp);
        (*second_derivative_combine_weight) += tmp;
        (*second_derivative_combine_weight + mu).moveTo(tmp);
        divide(lrate_b, tmp).moveTo(*learning_rate_w);
        ((*velocity_combine_weight) * momentum_derivative).moveTo(*velocity_combine_weight);
        combine_weight_grad -> mul(*learning_rate_w).moveTo(tmp);
        tmp.mul(1.0 - momentum_derivative).moveTo(tmp);
        (*velocity_combine_weight) += tmp;
        (*combine_weight) -= (*velocity_combine_weight);
    }
    tmp.release();
}

void convolutional_layer::forwardPass(int nsamples, network_layer* previous_layer){

    releaseVector(output_vector);
    output_vector.clear();
    std::vector<std::vector<Mat*> > input;
    if(previous_layer -> output_format == "image"){
    	copyVector(previous_layer -> output_vector, input);
    }else{
        // no!!!!
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    Mat c_weight;
    Mat tmp;
    if(combine_feature_map > 0){
        exp(*combine_weight).moveTo(c_weight);
        reduce(c_weight, REDUCE_TO_SINGLE_ROW, REDUCE_SUM).moveTo(tmp);
        repmat(tmp, c_weight.rows, 1).moveTo(tmp);
        divide(c_weight, tmp).moveTo(c_weight);
    }
    for(int i = 0; i < input.size(); ++i){
        std::vector<Mat*> eachsample;
        for(int j = 0; j < input[i].size(); ++j){
            std::vector<Mat*> tmpvec(kernels.size());
            for(int k = 0; k < kernels.size(); ++k){
                Mat *temp = new Mat();
                rot90(kernels[k] -> w, 2) -> moveTo(*temp);
                Mat *tmpconv = new Mat();
                conv2(input[i][j], temp, CONV_VALID, padding, stride) -> moveTo(*tmpconv);
                *tmpconv += *(kernels[k] -> b);
                tmpvec[k] = new Mat();
                tmpconv -> moveTo(*(tmpvec[k]));
                temp -> release();
            }
            if(combine_feature_map > 0){
                std::vector<Mat*> outputvec(combine_feature_map);
                Mat zero(tmpvec[0] -> rows, tmpvec[0] -> cols, 3);
                for(int k = 0; k < outputvec.size(); k++) {
                	outputvec[k] = new Mat();
                	zero.copyTo(*(outputvec[k]));
                }
                for(int m = 0; m < kernels.size(); m++){
                    for(int n = 0; n < combine_feature_map; n++){
                    	vector3f tmpvec3;
                    	tmpvec3.setAll(c_weight.get(m, n, 0));
                    	(tmpvec[m] -> mul(tmpvec3)).moveTo(tmp);
                        *(outputvec[n]) += tmp;
                    }
                }
                for(int k = 0; k < outputvec.size(); k++) {eachsample.push_back(outputvec[k]);}
                zero.release();
                outputvec.clear();
                std::vector<Mat*>().swap(outputvec);
            }
            else{
                for(int k = 0; k < tmpvec.size(); k++) {eachsample.push_back(tmpvec[k]);}
            }
            tmpvec.clear();
            std::vector<Mat*>().swap(tmpvec);
        }
        output_vector.push_back(eachsample);
        eachsample.clear();
        std::vector<Mat*>().swap(eachsample);
    }
    c_weight.release();
    tmp.release();
    releaseVector(input);
    input.clear();
    std::vector<std::vector<Mat*> >().swap(input);
}

void convolutional_layer::forwardPassTest(int nsamples, network_layer* previous_layer){
    convolutional_layer::forwardPass(nsamples, previous_layer);
}

void convolutional_layer::backwardPass(int nsamples, network_layer* previous_layer, network_layer* next_layer){

    std::vector<std::vector<Mat*> > derivative;
    std::vector<std::vector<Mat*> > deriv2;
    if(next_layer -> output_format == "matrix"){
        convert(next_layer -> delta_matrix, derivative, nsamples, output_vector[0][0] -> rows);
        convert(next_layer -> d2_matrix, deriv2, nsamples, output_vector[0][0] -> rows);
    }else{
    	copyVector(next_layer -> delta_vector, derivative);
    	copyVector(next_layer -> d2_vector, deriv2);
    }
    if(previous_layer -> output_format != "image"){
        cout<<"??? image after matrix??? I can't do that for now..."<<endl;
        return;
    }
    releaseVector(delta_vector);
    releaseVector(d2_vector);
    delta_vector.clear();
    d2_vector.clear();
    delta_vector.resize(previous_layer -> output_vector.size());
    d2_vector.resize(previous_layer -> output_vector.size());
    for(int i = 0; i < delta_vector.size(); i++){
    	delta_vector[i].clear();
    	d2_vector[i].clear();
        delta_vector[i].resize(previous_layer -> output_vector[i].size());
        d2_vector[i].resize(previous_layer -> output_vector[i].size());
        for(int j = 0; j < delta_vector[i].size(); ++j){
        	delta_vector[i][j] = new Mat();
        	d2_vector[i][j] = new Mat();
        }
    }
    Mat tmp, tmp2, tmp3, tmp4;
    std::vector<Mat> tmp_wgrad(kernels.size());
    std::vector<Mat> tmp_wd2(kernels.size());
    std::vector<vector3f> tmpgradb(kernels.size());
    std::vector<vector3f> tmpbd2(kernels.size());
    for(int m = 0; m < kernels.size(); m++) {
    	tmp_wgrad[m].setSize(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
    	tmp_wd2[m].setSize(kernels[0] -> w -> rows, kernels[0] -> w -> cols, 3);
        tmpgradb[m].zeros();
        tmpbd2[m].zeros();
    }
    Mat c_weight, c_weightgrad, c_weightd2;
    if(combine_feature_map > 0){
        exp(*combine_weight).moveTo(c_weight);
        reduce(c_weight, REDUCE_TO_SINGLE_ROW, REDUCE_SUM).moveTo(tmp);
        repmat(tmp, c_weight.rows, 1).moveTo(tmp);
        divide(c_weight, tmp).moveTo(c_weight);
        c_weightgrad.setSize(c_weight.rows, c_weight.cols, 1);
        c_weightd2.setSize(c_weight.rows, c_weight.cols, 1);
    }
    for(int i = 0; i < nsamples; i++){
        for(int j = 0; j < previous_layer -> output_vector[i].size(); j++){
            std::vector<Mat> sensi(kernels.size());
            std::vector<Mat> sensid2(kernels.size());
            Mat tmp_delta;
            Mat tmp_d2;
            for(int m = 0; m < kernels.size(); m++) {
            	sensi[m].setSize(output_vector[0][0] -> rows, output_vector[0][0] -> cols, 3);
            	sensid2[m].setSize(output_vector[0][0] -> rows, output_vector[0][0] -> cols, 3);
                if(combine_feature_map > 0){
                    for(int n = 0; n < combine_feature_map; n++){
                    	vector3f tmpvec3_1, tmpvec3_2;
                    	tmpvec3_1.setAll(c_weight.get(m, n, 0));
                    	float tmpfloat = c_weight.get(m, n, 0) * c_weight.get(m, n, 0);
                    	tmpvec3_2.setAll(tmpfloat);
                    	derivative[i][j * combine_feature_map + n] -> mul(tmpvec3_1).moveTo(tmp);
                    	deriv2[i][j * combine_feature_map + n] -> mul(tmpvec3_2).moveTo(tmp2);
                        sensi[m] += tmp;
                        sensid2[m] += tmp2;
                    }
                }else{
                    sensi[m] += *(derivative[i][j * kernels.size() + m]);
                    sensid2[m] += *(deriv2[i][j * kernels.size() + m]);
                }

                if(stride > 1){
                    int len = previous_layer -> output_vector[0][0] -> rows + padding * 2 - kernels[0] -> w -> rows + 1;
                    interpolation(sensi[m], len).moveTo(sensi[m]);
                    interpolation(sensid2[m], len).moveTo(sensid2[m]);
                }
            	square(kernels[m] -> w) -> moveTo(tmp3);
                if(m == 0){
                	conv2(sensi[m], *(kernels[m] -> w), CONV_FULL, 0, 1).moveTo(tmp_delta);
                	conv2(sensid2[m], tmp3, CONV_FULL, 0, 1).moveTo(tmp_d2);
                }else{
                	conv2(sensi[m], *(kernels[m] -> w), CONV_FULL, 0, 1).moveTo(tmp);
                	conv2(sensid2[m], tmp3, CONV_FULL, 0, 1).moveTo(tmp2);
                    tmp_delta += tmp;
                    tmp_d2 += tmp2;
                }
                Mat input;
                if(padding > 0){
                	dopadding(previous_layer -> output_vector[i][j], padding) -> moveTo(input);
                }else{
                    previous_layer -> output_vector[i][j] -> copyTo(input);
                }
                square(input).moveTo(tmp);
                rot90(sensi[m], 2).moveTo(tmp2);
                rot90(sensid2[m], 2).moveTo(tmp3);
                tmpgradb[m] += sum(tmp2);
                tmpbd2[m] += sum(tmp3);
                conv2(input, tmp2, CONV_VALID, 0, 1).moveTo(tmp2);
                conv2(tmp, tmp3, CONV_VALID, 0, 1).moveTo(tmp3);
                tmp_wgrad[m] += tmp2;
                tmp_wd2[m] += tmp3;

                if(combine_feature_map > 0){
                    // combine feature map weight matrix (after softmax)
                    previous_layer -> output_vector[i][j] -> copyTo(input);
                    square(input).moveTo(tmp);
                    square(tmp3).moveTo(tmp4);
                    rot90(kernels[m] -> w, 2) -> moveTo(tmp2);
                    tmp2.copyTo(tmp3);
                    conv2(input, tmp2, CONV_VALID, padding, stride).moveTo(tmp2);
                    conv2(tmp, tmp4, CONV_VALID, padding, stride).moveTo(tmp3);
                    for(int n = 0; n < combine_feature_map; n++){
                        Mat tmpd;
                        derivative[i][j * combine_feature_map + n] -> mul(tmp2).moveTo(tmpd);
                        c_weightgrad.set(m, n, 0, c_weightgrad.get(m, n, 0) + sum(tmpd).get(0));
                        deriv2[i][j * combine_feature_map + n] -> mul(tmp3).moveTo(tmpd);
                        c_weightd2.set(m, n, 0, c_weightd2.get(m, n, 0) + sum(tmpd).get(0));
                        tmpd.release();
                    }
                }
                input.release();
            }
            if(padding > 0){
            	depadding(tmp_delta, padding).moveTo(tmp_delta);
            	depadding(tmp_d2, padding).moveTo(tmp_d2);
            }
            tmp_delta.moveTo(*(delta_vector[i][j]));
            tmp_d2.moveTo(*(d2_vector[i][j]));
            releaseVector(sensi);
            sensi.clear();
            std::vector<Mat>().swap(sensi);
            releaseVector(sensid2);
            sensid2.clear();
            std::vector<Mat>().swap(sensid2);
        }
    }
    for(int i = 0; i < kernels.size(); i++){
    	vector3f tmpvec3;
    	tmpvec3.setAll(kernels[i] -> weight_decay);
    	divide(tmp_wgrad[i], nsamples).moveTo(*(kernels[i] -> wgrad) );
    	kernels[i] -> w -> mul(kernels[i] -> weight_decay).moveTo(tmp);
    	*(kernels[i] -> wgrad) += tmp;
    	divide(tmp_wd2[i], nsamples).moveTo(*(kernels[i] -> wd2));
    	*(kernels[i] -> wd2) += tmpvec3;
    	(*(kernels[i] -> bgrad)) = divide(tmpgradb[i], nsamples);
    	(*(kernels[i] -> bd2)) = divide(tmpbd2[i], nsamples);
    }
    if(combine_feature_map > 0){
    	c_weightgrad.mul(c_weight).moveTo(tmp2);
    	reduce(tmp2, REDUCE_TO_SINGLE_ROW, REDUCE_SUM).moveTo(tmp2);
    	repmat(tmp2, c_weightgrad.rows, 1).moveTo(tmp2);
    	(c_weightgrad - tmp2).moveTo(tmp);
    	c_weight.mul(tmp).moveTo(tmp);
    	divide(tmp, nsamples).moveTo(*combine_weight_grad);

    	c_weightd2.mul(c_weight).moveTo(tmp2);
    	reduce(tmp2, REDUCE_TO_SINGLE_ROW, REDUCE_SUM).moveTo(tmp2);
    	repmat(tmp2, c_weightd2.rows, 1).moveTo(tmp2);
    	(c_weightd2 - tmp2).moveTo(tmp);
    	c_weight.mul(tmp).moveTo(tmp);
    	divide(tmp, nsamples).moveTo(*combine_weight_d2);
    }
    tmp.release();
    tmp2.release();
    tmp3.release();
    tmp4.release();
    c_weight.release();
    c_weightgrad.release();
    c_weightd2.release();
    releaseVector(tmp_wgrad);
    tmp_wgrad.clear();
    std::vector<Mat>().swap(tmp_wgrad);
    releaseVector(tmp_wd2);
    tmp_wd2.clear();
    std::vector<Mat>().swap(tmp_wd2);
    releaseVector(derivative);
    derivative.clear();
    std::vector<std::vector<Mat*> >().swap(derivative);
    releaseVector(deriv2);
    deriv2.clear();
    std::vector<std::vector<Mat*> >().swap(deriv2);
    tmpgradb.clear();
    std::vector<vector3f>().swap(tmpgradb);
    tmpbd2.clear();
    std::vector<vector3f>().swap(tmpbd2);
}


