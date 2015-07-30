#pragma once
#include "general_settings.h"

int snapTransformSize(int);
void copyVector(const std::vector<std::vector<Mat*> >&, std::vector<std::vector<Mat*> >&);
void copyVector(const std::vector<Mat*>&, std::vector<Mat*>&);
void copyVector(const std::vector<std::vector<vector3f*> >&, std::vector<std::vector<vector3f*> >&);
void copyVector(const std::vector<Mat*>&, std::vector<vector3f*>&);

void releaseVector(std::vector<std::vector<Mat*> >&);
void releaseVector(std::vector<std::vector<Mat> >&);
void releaseVector(std::vector<Mat*>&);
void releaseVector(std::vector<Mat>&);

void releaseVector(std::vector<std::vector<cpuMat*> >&);
void releaseVector(std::vector<std::vector<cpuMat> >&);
void releaseVector(std::vector<cpuMat*>&);
void releaseVector(std::vector<cpuMat>&);

void releaseVector(std::vector<std::vector<std::vector<vector3f*> > >&);
void releaseVector(std::vector<std::vector<std::vector<vector3f> > >&);
void releaseVector(std::vector<std::vector<vector3f*> >&);
void releaseVector(std::vector<std::vector<vector3f> >&);
void releaseVector(std::vector<vector3f*>&);
void releaseVector(std::vector<vector3f>&);
void showGpuProperty();
