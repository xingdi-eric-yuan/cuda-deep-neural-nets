#cuda-deep-neural-net-0.1.1
=====================
##UNFINISHED
Deep neural network framework (C/C++/CUDA).

To run this code, you should have 
* a cifar-10 dataset( put "cifar-10-batches-bin" where this .md file is, you can get it from [HERE](http://www.cs.toronto.edu/~kriz/cifar.html), make sure to download the binary version which suitable for C programs);
* nVidia graphic card which supports nVidia CUDA

##Compile & Run
add this project into nVidia nsight, add **curand** and **cufft** into path. 

##Updates 
* 0.1.0: Aug.5, the first version released.
* 0.1.1: Aug.10, remove hostData in Mat, only use device memory, for speed up.

##Data Structures
####Mat
* Similar with Mat in OpenCV, has memory in both CPU and GPU, use it to do most of the calculations.

####cpuMat
* Has memory only in CPU, use it to read dataset, and do pre-processing (unless your GPU memory is huge...).

####vector3f
* Similar with Scalar in OpenCV, has 3 float space, which corresponses 3 channels. For example, the sum of a 3-channal Mat is a vector3f.

####vector2i
* Similar with vector3f, but has 2 int space, use it to represent 2-d position, or size. 

##Layer Config Description 
* For each layer, there is a **layer_name**, a **layer_type**, and a **output_format**.
* There are currently 2 output formats: **matrix** single channel Mat, and **image** (vector of 3-channel Mat).

####Input Layer
* **batch size**: the training process is using mini-batch stochastic gradient descent.

####Convolutional Layer
* **kernel size**: size of kernels for convolution calculation.
* **kernel amount**: amount of kernels for convolution calculation.
* **combine map**: amount of combine feature map, details can be found in [Notes on Convolutional Neural Networks](http://cogprints.org/5869/1/cnn_tutorial.pdf).
* **weight decay**: weight decay for convolutional kernels.
* **padding**: padding before doing convolution.
* **stride**: stride when doing convolution (For "VALID" type of convolution, **result size = (image_size + 2 * padding - kernel_size) / stride + 1)**.

####Fully Connected Layer
* **num hidden neurons**: size of fully connected layer.
* **weight decay**: weight decay for fully connected layer.

####Softmax Layer
* **num classes**: output size of softmax layer.
* **weight decay**: weight decay for softmax layer.

####Non-linearity Layer
* **method**: sigmoid/tanh/relu/leaky_relu.

####Pooling Layer
* **method**: max/mean/stochastic.
* **overlap**: if use overlap pooling.
* **window size**: window size when using overlap pooling.
* **stride**: pooling stride.

####Local Response Normalization Layer
* **alpha**, **beta**, **k**, **n**: see [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

####Dropout Layer
* **dropout rate**: percentage of zeros when generating Bernoulli matrix.

####Combine Layer
* for implementing GoogLeNet, TODO...

####Branch Layer
* for implementing GoogLeNet, TODO...

##Structure and Algorithm
See my several posts about CNNs at [my tech-blog](http://eric-yuan.me).

##CPU Version
[There's also a CPU version of this code which I used OpenCV as the linear algebra library.](https://github.com/xingdi-eric-yuan/conv-net-version-3).

##TODO
* combine layer
* branch layer
* stochastic pooling

The MIT License (MIT)
------------------

Copyright (c) 2015 Xingdi (Eric) Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.