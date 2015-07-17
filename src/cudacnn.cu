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

	Mat a(3, 2, 3);
	//a.setAll(2.2);
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
/*
	Mat j(3, 2, 1);
	j.set(0, 0, 0, 0.5);
	j.set(0, 1, 0, 1.0);
	j.set(1, 0, 0, 2.0);
	j.set(1, 1, 0, 3.0);
	j.printHost("10th print");

	Mat k = kron(i, j);
	k.printHost("11th print");
*/
	Mat j = repmat(i, 2, 3);
	j.printHost("10th print");


	return 0;
}
