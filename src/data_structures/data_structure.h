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
	vector2i& operator=(const vector2i&);

	void zeros();
	void ones();
	void set(int, int);
	void setAll(int);
	int get(int) const;
	void copyTo(vector2i&);

	vector2i operator+(const vector2i&);
	vector2i operator-(const vector2i&);
	vector2i operator*(const vector2i&);
	vector2i operator+(int);
	vector2i operator-(int);
	vector2i operator*(int);
	vector2i mul(const vector2i&);
	vector2i mul(const int);
	void print(const std::string&) const;

	int val0;
	int val1;
};
// Scalar3
class vector3f{
public:
	vector3f();
	vector3f(float, float, float);
	vector3f(const vector3f&);
	~vector3f();
	vector3f& operator=(const vector3f&);

	void zeros();
	void ones();
	void set(int, float);
	void setAll(float);
	float get(int) const;
	void copyTo(vector3f&);

	vector3f operator+(const vector3f&);
	vector3f operator-(const vector3f&);
	vector3f operator*(const vector3f&);
	vector3f operator+(float);
	vector3f operator-(float);
	vector3f operator*(float);
	vector3f operator/(float);
	vector3f operator%(float);
	vector3f divNoRem(float);
	vector3f mul(const vector3f&);
	vector3f mul(const float);
	void print(const std::string&) const;

	float val0;
	float val1;
	float val2;
};

// matrix
class Mat{
public:
	Mat();
	Mat(const Mat&);
	Mat(const cpuMat&);
	Mat(int, int, int);
	~Mat();
	Mat& operator=(const Mat&);

	int cols;
	int rows;
	int channels;

	float *hostData;
	float *devData;

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

	void deviceToHost();
	void hostToDevice();
	void copyTo(Mat&) const;
	void copyTo(cpuMat&) const;

	// only changes devData (on GPU)
	Mat operator+(const Mat&) const;
	Mat operator-(const Mat&) const;
	Mat operator*(const Mat&) const;
	Mat operator+(float) const;
	Mat operator-(float) const;
	Mat operator*(float) const;
	Mat operator+(const vector3f&) const;
	Mat operator-(const vector3f&) const;
	Mat operator*(const vector3f&) const;
	Mat mul(const Mat&) const;
	Mat mul(float) const;
	Mat mul(const vector3f&) const;
	Mat t() const;
	// memory
	void mallocHost();
	void mallocDevice();
	//
	void printHost(const std::string&) const;
	void printDevice(const std::string&) const;
};

// matrix
class cpuMat{
public:
	cpuMat();
	cpuMat(const Mat&);
	cpuMat(const cpuMat&);
	cpuMat(int, int, int);
	~cpuMat();
	cpuMat& operator=(const cpuMat&);

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
	cpuMat mul(const cpuMat&) const;
	cpuMat mul(float) const;
	cpuMat mul(const vector3f&) const;
	cpuMat t() const;
	// memory
	void mallocMat();
	void print(const std::string&) const;
};
