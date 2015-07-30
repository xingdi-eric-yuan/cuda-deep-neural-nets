#include "data_structure.h"

using namespace std;

vector2i::vector2i(){
	Data = NULL;
	mallocVector2i();
}

vector2i::vector2i(int a, int b){
	Data = NULL;
	mallocVector2i();
	set(0, a);
	set(1, b);
}

vector2i::vector2i(const vector2i &v){
	Data = NULL;
	mallocVector2i();
	memcpy(Data, v.Data, 2 * sizeof(int));
}

vector2i::~vector2i(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeCpuMemory(Data);
	Data = NULL;
}

void vector2i::release(){
	if(NULL != Data)
		MemoryMonitor::instance()->freeCpuMemory(Data);
	Data = NULL;
}

vector2i& vector2i::operator=(const vector2i &v){
	if(NULL != Data){
		MemoryMonitor::instance()->freeCpuMemory(Data);
		Data = NULL;
	}
	mallocVector2i();
	memcpy(Data, v.Data, 2 * sizeof(int));
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
	Data[pos] = val;
}

void vector2i::setAll(int val){
	for(int i = 0; i < 2; ++i){
		set(i, val);
	}
}

int vector2i::get(int pos) const{
	return Data[pos];
}
void vector2i::copyTo(vector2i &v) const{
	if(NULL != v.Data){
		MemoryMonitor::instance()->freeCpuMemory(v.Data);
		v.Data = NULL;
	}
	v.mallocVector2i();
	memcpy(v.Data, Data, 2 * sizeof(int));
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

void vector2i::mallocVector2i(){
	if(NULL == Data){
		// malloc data
		Data = (int*)MemoryMonitor::instance()->cpuMalloc(2 * sizeof(int));
		if(NULL == Data) {
			std::cout<<"host memory allocation failed..."<<std::endl;
			exit(0);
		}
		memset(Data, 0, 2 * sizeof(int));
	}
}

void vector2i::print(const std::string& str) const{
	std::cout<<str<<std::endl;
	std::cout<<"vector2i: [ ";
	for(int i = 0; i < 2; ++i){
		std::cout<<get(i)<<" ";
	}
	std::cout<<"]"<<std::endl;
}
