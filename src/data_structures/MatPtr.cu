#include "data_structure.h"

using namespace std;

MatPtr::MatPtr(){
	mptr = NULL;
}
MatPtr::~MatPtr(){
	if(mptr) {
		mptr -> release();
	}
}

MatPtr& MatPtr::operator<=(const MatPtr& pt){
	if(mptr) {
		mptr -> release();
	}
	mptr = pt.mptr;
	return *this;
}

MatPtr& MatPtr::operator=(const MatPtr& pt){
	mptr = pt.mptr;
	return *this;
}

