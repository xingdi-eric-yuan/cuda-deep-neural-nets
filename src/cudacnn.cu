/*
 ============================================================================
 Name        : cudacnn.cu
 Author      : eric_yuan
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include "general_settings.h"

int main(void){


    std::vector<cpuMat> trainX;
    std::vector<cpuMat> testX;
    cpuMat trainY, testY;
    read_CIFAR10_data(trainX, testX, trainY, testY);



	/*
	cpuMat a(3, 2, 3);
	//a.setAll(2.2);
	a.randn();
	a.set(1, 1, 0, 0.7);
	a.set(2, 1, 0, 0.8);
	a.set(1, 0, 1, 0.55);
	a.set(2, 0, 1, 0.45);
	a.print("1st print");

	cpuMat b = a.t();
	b.print("2nd print");
//*/

/*
	Mat a(3, 2, 3);
	a.randn();
	a.set(1, 1, 0, 0.7);
	a.set(2, 1, 0, 0.8);
	a.set(1, 0, 1, 0.55);
	a.set(2, 0, 1, 0.45);
	a.printHost("1st print");

	Mat b = a.t();
	b.printHost("2nd print");

	Mat c = a * b;
	c.printHost("3nd print");

	vector3f v(1.0, 2.0, 3.0);
	Mat d = c * v;
	d.printHost("4th print");

	Mat e = pow(d, 2.0);
	e.printHost("5th print");

	Mat f = padding(e, 2);
	f.printHost("6th print");

	Mat g = depadding(f, 2);
	g.printHost("7th print");

	Mat h = Tanh(g);
	h.printHost("8th print");

	Mat i = rot90(h, 2);
	i.printHost("9th print");

	Mat j = repmat(i, 2, 2);
	j.printHost("10th print");

	Mat k = conv2(j, h, CONV_SAME, 0, 1);
	k.printHost("11th print");

	vector2i _size1(2, 3);
	std::vector<vector3f> locat;

	Mat l = pooling_with_overlap(k, _size1, 1, POOL_MAX, locat);
	l.printHost("12th print");
	for(int counter = 0; counter < locat.size(); counter++){
		string str = "locat_" + to_string(counter);
		locat[counter].print(str);
	}
	vector2i _size2(6, 6);
	Mat m = unpooling_with_overlap(l, _size1, 1, POOL_MAX, locat, _size2);
	m.printHost("13th print");
//*/
	return 0;
}
