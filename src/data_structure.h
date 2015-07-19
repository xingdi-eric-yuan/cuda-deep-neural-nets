#pragma once
#include "general_settings.h"

using namespace std;

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
	Mat(int, int, int);
	~Mat();
	Mat& operator=(const Mat&);

	int cols;
	int rows;
	int channels;

	float *hostData;
	float *devData;

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
	void copyTo(Mat&);

	// only changes devData (on GPU)
	Mat operator+(const Mat&);
	Mat operator-(const Mat&);
	Mat operator*(const Mat&);
	Mat operator+(float);
	Mat operator-(float);
	Mat operator*(float);
	Mat operator+(const vector3f&);
	Mat operator-(const vector3f&);
	Mat operator*(const vector3f&);
	Mat mul(const Mat&);
	Mat mul(float);
	Mat mul(const vector3f&);
	Mat t();
	// memory
	void mallocHost();
	void mallocDevice();
	//
	void printHost(const std::string&);
	void printDevice(const std::string&);
};
