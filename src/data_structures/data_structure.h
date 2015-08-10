#pragma once
#include "../general_settings.h"

using namespace std;

class Mat;
class cpuMat;
class vector2i;
class vector3f;
// size/position
class vector2i{
public:
	vector2i();
	vector2i(int, int);
	vector2i(const vector2i&);
	~vector2i();
	void release();
	vector2i& operator=(const vector2i&);

	void zeros();
	void ones();
	void set(int, int);
	void setAll(int);
	int get(int) const;
	void copyTo(vector2i&) const;

	vector2i operator+(const vector2i&) const;
	vector2i operator-(const vector2i&) const;
	vector2i operator*(const vector2i&) const;
	vector2i operator+(int) const;
	vector2i operator-(int) const;
	vector2i operator*(int) const;

	vector2i& operator+=(const vector2i&);
	vector2i& operator-=(const vector2i&);
	vector2i& operator*=(const vector2i&);
	vector2i& operator+=(int);
	vector2i& operator-=(int);
	vector2i& operator*=(int);

	vector2i mul(const vector2i&) const;
	vector2i mul(const int) const;
	void mallocVector2i();
	void print(const std::string&) const;

	int *Data;
};
// Scalar3
class vector3f{
public:
	vector3f();
	vector3f(float, float, float);
	vector3f(const vector3f&);
	~vector3f();
	void release();
	vector3f& operator=(const vector3f&);

	void zeros();
	void ones();
	void set(int, float);
	void setAll(float);
	float get(int) const;
	void copyTo(vector3f&) const;

	vector3f operator+(const vector3f&) const;
	vector3f operator-(const vector3f&) const;
	vector3f operator*(const vector3f&) const;
	vector3f operator+(float) const;
	vector3f operator-(float) const;
	vector3f operator*(float) const;
	vector3f operator/(float) const;
	vector3f operator%(float) const;

	vector3f& operator+=(const vector3f&);
	vector3f& operator-=(const vector3f&);
	vector3f& operator*=(const vector3f&);
	vector3f& operator+=(const float);
	vector3f& operator-=(const float);
	vector3f& operator*=(const float);
	vector3f& operator/=(const float);
	vector3f& operator%=(const float);

	vector3f divNoRem(float) const;
	vector3f mul(const vector3f&) const;
	vector3f mul(const float) const;

	void mallocVector3f();
	void print(const std::string&) const;

	float *Data;
};

class MatPtr{
public:
	MatPtr();
	MatPtr(int, int, int);
	~MatPtr();
	MatPtr& operator=(const MatPtr&);
	MatPtr& operator<=(const MatPtr&);
	Mat* mptr;
};

// matrix
class Mat{
public:
	Mat();
	Mat(const Mat&);
	Mat(const cpuMat&);
	Mat(int, int, int);
	~Mat();
	void release();
	Mat& operator=(const Mat&);
	Mat& operator<<=(Mat&);

	int cols;
	int rows;
	int channels;

	float *Data;

	void setSize(int, int, int);
	void zeros();
	void ones();
	void randu();
	void randn();
	void set(int, int, int, float);
	void set(int, int, float);
	void set(int, float);
	void set(int, int, const vector3f&);
	void set(int, const vector3f&);
	void setAll(float);
	void setAll(const vector3f&);
	float get(int, int, int) const;
	vector3f get(int, int) const;
	vector3f get(int) const;
	int getLength() const;

	void copyTo(Mat&) const;
	void copyTo(cpuMat&) const;
	void moveTo(Mat&);
	void moveTo(cpuMat&);

	// only changes devData (on GPU)
	Mat operator+(const Mat&) const;
	Mat operator-(const Mat&) const;
	Mat operator*(const Mat&) const;
	Mat operator+(float) const;
	Mat operator-(float) const;
	Mat operator*(float) const;
	Mat operator/(float) const;
	Mat operator+(const vector3f&) const;
	Mat operator-(const vector3f&) const;
	Mat operator*(const vector3f&) const;
	Mat operator/(const vector3f&) const;

	Mat& operator+=(const Mat&);
	Mat& operator-=(const Mat&);
	Mat& operator+=(float);
	Mat& operator-=(float);
	Mat& operator*=(float);
	Mat& operator/=(float);
	Mat& operator+=(const vector3f&);
	Mat& operator-=(const vector3f&);
	Mat& operator*=(const vector3f&);
	Mat& operator/=(const vector3f&);

	Mat mul(const Mat&) const;
	Mat mul(float) const;
	Mat mul(const vector3f&) const;
	Mat t() const;
	// memory
	void mallocDevice();
	//
	void print(const std::string&) const;
	void printDim(const std::string&) const;
};

// matrix
class cpuMat{
public:
	cpuMat();
	cpuMat(const Mat&);
	cpuMat(const cpuMat&);
	cpuMat(int, int, int);
	~cpuMat();
	void release();
	cpuMat& operator=(const cpuMat&);
	cpuMat& operator<<=(cpuMat&);

	int cols;
	int rows;
	int channels;
	float *Data;

	void setSize(int, int, int);
	void zeros();
	void ones();
	void randn();
	void set(int, int, int, float);
	void set(int, int, float);
	void set(int, int, const vector3f&);
	void set(int, const vector3f&);
	void setAll(float);
	void setAll(const vector3f&);
	float get(int, int, int) const;
	vector3f get(int, int) const;
	int getLength() const;

	void copyTo(Mat&) const;
	void copyTo(cpuMat&) const;
	void moveTo(Mat&);
	void moveTo(cpuMat&);

	// only changes devData (on GPU)
	cpuMat operator+(const cpuMat&) const;
	cpuMat operator-(const cpuMat&) const;
	cpuMat operator*(const cpuMat&) const;
	cpuMat operator+(float) const;
	cpuMat operator-(float) const;
	cpuMat operator*(float) const;
	cpuMat operator+(const vector3f&) const;
	cpuMat operator-(const vector3f&) const;
	cpuMat operator*(const vector3f&) const;

	cpuMat& operator+=(const cpuMat&);
	cpuMat& operator-=(const cpuMat&);
	cpuMat& operator+=(float);
	cpuMat& operator-=(float);
	cpuMat& operator*=(float);
	cpuMat& operator+=(const vector3f&);
	cpuMat& operator-=(const vector3f&);
	cpuMat& operator*=(const vector3f&);

	cpuMat mul(const cpuMat&) const;
	cpuMat mul(float) const;
	cpuMat mul(const vector3f&) const;
	cpuMat t() const;
	// memory
	void mallocMat();
	void print(const std::string&) const;
	void printDim(const std::string&) const;
};
