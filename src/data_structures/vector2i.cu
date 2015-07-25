#include "data_structure.h"

using namespace std;

vector2i::vector2i(){
	for(int i = 0; i < 2; ++i){
		set(i, 0);
	}
}

vector2i::vector2i(int a, int b){
	set(0, a);
	set(1, b);
}

vector2i::vector2i(const vector2i &v){
	for(int i = 0; i < 2; ++i){
		set(i, v.get(i));
	}
}

vector2i::~vector2i(){}

vector2i& vector2i::operator=(const vector2i &v){
	for(int i = 0; i < 2; ++i){
		set(i, v.get(i));
	}
    return *this;
}

void vector2i::zeros(){
	for(int i = 0; i < 2; ++i){
		set(i, 0);
	}
}

void vector2i::ones(){
	for(int i = 0; i < 2; ++i){
		set(i, 1);
	}
}

void vector2i::set(int pos, int val){
	if(0 == pos) val0 = val;
	elif(1 == pos) val1 = val;
	else ;
}

void vector2i::setAll(int val){
	for(int i = 0; i < 2; ++i){
		set(i, val);
	}
}

int vector2i::get(int pos) const{
	if(0 == pos) return val0;
	elif(1 == pos) return val1;
	else return 0;
}
void vector2i::copyTo(vector2i &v) const{
	for(int i = 0; i < 2; ++i){
		v.set(i, get(i));
	}
}

vector2i vector2i::operator+(const vector2i &v) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) + v.get(i));
	}
	return tmp;
}

vector2i vector2i::operator+(int a) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) + a);
	}
	return tmp;
}

vector2i& vector2i::operator+=(const vector2i &v){
	for(int i = 0; i < 2; ++i){
		set(i, get(i) + v.get(i));
	}
	return *this;
}

vector2i& vector2i::operator+=(int a){
	for(int i = 0; i < 2; ++i){
		set(i, get(i) + a);
	}
	return *this;
}

vector2i vector2i::operator-(const vector2i &v) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) - v.get(i));
	}
	return tmp;
}

vector2i vector2i::operator-(int a) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) - a);
	}
	return tmp;
}

vector2i& vector2i::operator-=(const vector2i &v){
	for(int i = 0; i < 2; ++i){
		set(i, get(i) - v.get(i));
	}
	return *this;
}

vector2i& vector2i::operator-=(int a){
	for(int i = 0; i < 2; ++i){
		set(i, get(i) - a);
	}
	return *this;
}

vector2i vector2i::operator*(const vector2i &v) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) * v.get(i));
	}
	return tmp;
}

vector2i vector2i::operator*(int a) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) * a);
	}
	return tmp;
}

vector2i& vector2i::operator*=(const vector2i &v){
	for(int i = 0; i < 2; ++i){
		set(i, get(i) * v.get(i));
	}
	return *this;
}

vector2i& vector2i::operator*=(int a){
	for(int i = 0; i < 2; ++i){
		set(i, get(i) * a);
	}
	return *this;
}

vector2i vector2i::mul(const vector2i &v) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) * v.get(i));
	}
	return tmp;
}

vector2i vector2i::mul(int a) const{
	vector2i tmp;
	copyTo(tmp);
	for(int i = 0; i < 2; ++i){
		tmp.set(i, tmp.get(i) * a);
	}
	return tmp;
}

void vector2i::print(const std::string& str) const{
	std::cout<<str<<std::endl;
	std::cout<<"vector2i: [ ";
	for(int i = 0; i < 2; ++i){
		std::cout<<get(i)<<" ";
	}
	std::cout<<"]"<<std::endl;
}
