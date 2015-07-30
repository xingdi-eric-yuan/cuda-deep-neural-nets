#include "data_structure.h"

using namespace std;

vector3f::vector3f(){
	Data = NULL;
	mallocVector3f();
}

vector3f::vector3f(float a, float b, float c){
	Data = NULL;
	mallocVector3f();
	set(0, a);
	set(1, b);
	set(2, c);
}

vector3f::vector3f(const vector3f &v){
	Data = NULL;
	mallocVector3f();
	memcpy(Data, v.Data, 3 * sizeof(float));
}

vector3f::~vector3f(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeCpuMemory(Data);
	Data = NULL;
}

void vector3f::release(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeCpuMemory(Data);
	Data = NULL;
}

vector3f& vector3f::operator=(const vector3f &v){
	if(NULL != Data){
		MemoryMonitor::instance()->freeCpuMemory(Data);
		Data = NULL;
	}
	mallocVector3f();
	memcpy(Data, v.Data, 3 * sizeof(float));
    return *this;
}

void vector3f::zeros(){
	for(int i = 0; i < 3; ++i){
		set(i, 0.0);
	}
}

void vector3f::ones(){
	for(int i = 0; i < 3; ++i){
		set(i, 1.0);
	}
}

void vector3f::set(int pos, float val){
	Data[pos] = val;
}

void vector3f::setAll(float val){
	for(int i = 0; i < 3; ++i){
		set(i, val);
	}
}

float vector3f::get(int pos) const{
	return Data[pos];
}

void vector3f::copyTo(vector3f &v) const{
	if(NULL != v.Data){
		MemoryMonitor::instance()->freeCpuMemory(v.Data);
		v.Data = NULL;
	}
	v.mallocVector3f();
	memcpy(v.Data, Data, 3 * sizeof(float));
}

vector3f vector3f::operator+(const vector3f &v) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) + v.get(i));
	}
	return tmp;
}

vector3f vector3f::operator+(float a) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) + a);
	}
	return tmp;
}

vector3f& vector3f::operator+=(const vector3f &v){
	for(int i = 0; i < 3; ++i){
		set(i, get(i) + v.get(i));
	}
	return *this;
}

vector3f& vector3f::operator+=(float a){
	for(int i = 0; i < 3; ++i){
		set(i, get(i) + a);
	}
	return *this;
}

vector3f vector3f::operator-(const vector3f &v) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) - v.get(i));
	}
	return tmp;
}

vector3f vector3f::operator-(float a) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) - a);
	}
	return tmp;
}

vector3f& vector3f::operator-=(const vector3f &v){
	for(int i = 0; i < 3; ++i){
		set(i, get(i) - v.get(i));
	}
	return *this;
}

vector3f& vector3f::operator-=(float a){
	for(int i = 0; i < 3; ++i){
		set(i, get(i) - a);
	}
	return *this;
}

vector3f vector3f::operator*(const vector3f &v) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) * v.get(i));
	}
	return tmp;
}

vector3f vector3f::operator*(float a) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) * a);
	}
	return tmp;
}

vector3f& vector3f::operator*=(const vector3f &v){
	for(int i = 0; i < 3; ++i){
		set(i, get(i) * v.get(i));
	}
	return *this;
}

vector3f& vector3f::operator*=(float a){
	for(int i = 0; i < 3; ++i){
		set(i, get(i) * a);
	}
	return *this;
}

vector3f vector3f::operator/(float a) const{
	if(0 == a){
		std::cout<<"denominator is zero..."<<std::endl;
		exit(0);
	}
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) / a);
	}
	return tmp;
}

vector3f& vector3f::operator/=(float a){
	if(0 == a){
		std::cout<<"denominator is zero..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < 3; ++i){
		set(i, get(i) / a);
	}
	return *this;
}

vector3f vector3f::divNoRem(float a) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, (int)(tmp.get(i) / a));
	}
	return tmp;
}

vector3f vector3f::operator%(float a) const{
	if(0 == a){
		std::cout<<"denominator is zero..."<<std::endl;
		exit(0);
	}
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, (float)((int)(tmp.get(i)) % (int)a));
	}
	return tmp;
}

vector3f& vector3f::operator%=(float a){
	if(0 == a){
		std::cout<<"denominator is zero..."<<std::endl;
		exit(0);
	}
	for(int i = 0; i < 3; ++i){
		set(i, (float)((int)(get(i)) % (int)a));
	}
	return *this;
}

vector3f vector3f::mul(const vector3f &v) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) * v.get(i));
	}
	return tmp;
}

vector3f vector3f::mul(float a) const{
	vector3f tmp;
	copyTo(tmp);
	for(int i = 0; i < 3; ++i){
		tmp.set(i, tmp.get(i) * a);
	}
	return tmp;
}

void vector3f::mallocVector3f(){
	if(NULL == Data){
		// malloc data
		Data = (float*)MemoryMonitor::instance()->cpuMalloc(3 * sizeof(float));
		if(NULL == Data) {
			std::cout<<"host memory allocation failed..."<<std::endl;
			exit(0);
		}
		memset(Data, 0, 3 * sizeof(float));
	}
}

void vector3f::print(const std::string& str) const{
	std::cout<<str<<std::endl;
	std::cout<<"vector3f: [ ";
	for(int i = 0; i < 3; ++i){
		std::cout<<get(i)<<" ";
	}
	std::cout<<"]"<<std::endl;
}

