#include "cu_matrix_maths.h"

__global__ void cu_plus(float *A, const float *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fadd_rd(A[tid], B[tid]);
		tid += stride;
	}
}

__global__ void cu_plus(const float *A, const float *B, float *C, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		C[tid] = __fadd_rd(A[tid], B[tid]);
		tid += stride;
	}
}

__global__ void cu_plus(float *A, const float b, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fadd_rd(A[tid], b);
		tid += stride;
	}
}

__global__ void cu_plus(const float *A, float *B, const float c, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		B[tid] = __fadd_rd(A[tid], c);
		tid += stride;
	}
}

__global__ void cu_minus(float *A, const float *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fsub_rd(A[tid], B[tid]);
		tid += stride;
	}
}

__global__ void cu_minus(const float *A, const float *B, float *C, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		C[tid] = __fsub_rd(A[tid], B[tid]);
		tid += stride;
	}
}

__global__ void cu_minus(float *A, const float b, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fsub_rd(A[tid], b);
		tid += stride;
	}
}

__global__ void cu_minus(const float *A, float *B, const float c, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		B[tid] = __fsub_rd(A[tid], c);
		tid += stride;
	}
}

__global__ void cu_square(const float *A, float *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		B[tid] = __fmul_rd(A[tid], A[tid]);
		tid += stride;
	}
}

__global__ void cu_sqrt(const float *A, float *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		B[tid] = sqrtf(A[tid]);
		tid += stride;
	}
}

__global__ void cu_elementWiseMultiply(float *A, const float *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fmul_rd(A[tid], B[tid]);
		tid += stride;
	}
}

__global__ void cu_elementWiseMultiply(float *A, float B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fmul_rd(A[tid], B);
		tid += stride;
	}
}

__global__ void cu_elementWiseMultiply(const float *A, const float *B, float *C, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		C[tid] = __fmul_rd(A[tid], B[tid]);
		tid += stride;
	}
}

__global__ void cu_elementWiseMultiply(const float *A, const float B, float *C, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		C[tid] = __fmul_rd(A[tid], B);
		tid += stride;
	}
}

__global__ void cu_setAll(float* A, const float val, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = val;
		tid += stride;
	}
}

__global__ void cu_exp(const float* src, float* dst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		dst[tid] = __expf(src[tid]);
		tid += stride;
	}
}

__global__ void cu_log(const float* src, float* dst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		dst[tid] = __logf(src[tid]);
		tid += stride;
	}
}

__global__ void cu_pow(const float* src, float* dst, const float power, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		dst[tid] = powf(src[tid], power);
		tid += stride;
	}
}

__global__ void cu_divide(float *A, float B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fdividef(A[tid], B);
		tid += stride;
	}
}

__global__ void cu_divide(const float* src, float* dst, const float denominator, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(0 == denominator) dst[tid] = 0.0;
		else dst[tid] = __fdividef(src[tid], denominator);
		tid += stride;
	}
}

__global__ void cu_divide(const float numerator, const float* src, float* dst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(0 == src[tid]) dst[tid] = 0.0;
		else dst[tid] = __fdividef(numerator, src[tid]);
		tid += stride;
	}
}

__global__ void cu_divide(const float* numerator, const float* denominator, float* dst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(0 == denominator[tid]) dst[tid] = 0.0;
		else dst[tid] = __fdividef(numerator[tid], denominator[tid]);
		tid += stride;
	}
}

__global__ void cu_sum(const float* src, float* sum, const int n){
	extern __shared__ float sdata[];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// load input into __shared__ memory
	float x = 0;
	if(tid < n){
		x = src[tid];
	}
	sdata[threadIdx.x] = x;
	__syncthreads();
	// contiguous range pattern
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
		if(threadIdx.x < offset){
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
	    // wait until all threads in the block have
	    // updated their partial sums
		__syncthreads();
	}
	// thread 0 writes the final result
	if(threadIdx.x == 0){
		sum[blockIdx.x] = sdata[0];
	}
	__syncthreads();
}

__global__ void cu_minMaxLoc(const float* src, float* minValue, float* maxValue, int* minLoc, int* maxLoc, const int n){
	__shared__ float minValCache[threadsPerBlock];
	__shared__ float maxValCache[threadsPerBlock];
	__shared__ int minLocCache[threadsPerBlock];
	__shared__ int maxLocCache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//int stride = blockDim.x * gridDim.x;
	float val = src[0];
	int loc = 0;
	if(tid < n){
		val = src[tid];
		loc = tid;
	}
	maxValCache[threadIdx.x] = val;
	minValCache[threadIdx.x] = val;
	maxLocCache[threadIdx.x] = loc;
	minLocCache[threadIdx.x] = loc;
	__syncthreads();
	// contiguous range pattern
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
		if(threadIdx.x < offset){
			// add a partial sum upstream to our own
			if(maxValCache[threadIdx.x] >= maxValCache[threadIdx.x + offset]){
				;
			}else{
				maxValCache[threadIdx.x] = maxValCache[threadIdx.x + offset];
				maxLocCache[threadIdx.x] = maxLocCache[threadIdx.x + offset];
			}
			if(minValCache[threadIdx.x] <= minValCache[threadIdx.x + offset]){
				;
			}else{
				minValCache[threadIdx.x] = minValCache[threadIdx.x + offset];
				minLocCache[threadIdx.x] = minLocCache[threadIdx.x + offset];
			}
		}
	    // wait until all threads in the block have
	    // updated their partial sums
		__syncthreads();
	}
	// thread 0 writes the final result
	if(threadIdx.x == 0){
		minValue[blockIdx.x] = minValCache[0];
		maxValue[blockIdx.x] = maxValCache[0];
		minLoc[blockIdx.x] = minLocCache[0];
		maxLoc[blockIdx.x] = maxLocCache[0];
	}
}


__global__ void cu_greaterThan(const float* src, float* dst, const float val, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] > val) dst[tid] = 1.0;
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_greaterThanOrEqualTo(const float* src, float* dst, const float val, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] >= val) dst[tid] = 1.0;
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_lessThan(const float* src, float* dst, const float val, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] < val) dst[tid] = 1.0;
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_lessThanOrEqualTo(const float* src, float* dst, const float val, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] <= val) dst[tid] = 1.0;
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_equalTo(const float* src, float* dst, const float val, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] == val) dst[tid] = 1.0;
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_fliplr(const float* src, float* dst, const int rows, const int cols, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int c = tid % cols;
		int r = tid / cols;
		dst[tid] = src[(cols - c - 1) + r * cols];
		tid += stride;
	}
}

__global__ void cu_padding(const float* src, float* dst, const int rows1, const int cols1, const int cols2, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int pad = (cols2 - cols1) / 2;
		int c1 = tid % cols1;
		int r1 = tid / cols1;
		int r2 = r1 + pad;
		int c2 = c1 + pad;
		dst[r2 * cols2 + c2] = src[tid];
		tid += stride;
	}
}

__global__ void cu_depadding(const float* src, float* dst, const int rows1, const int cols1, const int cols2, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int pad = (cols1 - cols2) / 2;
		int c2 = tid % cols2;
		int r2 = tid / cols2;
		int r1 = r2 + pad;
		int c1 = c2 + pad;
		dst[tid] = src[r1 * cols1 + c1];
		tid += stride;
	}
}

__global__ void cu_repmat(const float *a, float* dst, const int rowsa, const int colsa, const int rowsdst, const int colsdst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int c2 = tid % colsdst;
		int r2 = tid / colsdst;
		int ra = r2 % rowsa;
		int ca = c2 % colsa;
		dst[tid] = a[ra * colsa + ca];
		tid += stride;
	}
}

__global__ void cu_kron(const float *a, const float* b, float* dst, const int rowsa, const int colsa, const int rowsdst, const int colsdst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int colsb = colsdst / colsa;
	int rowsb = rowsdst / rowsa;
	while(tid < n){
		int c2 = tid % colsdst;
		int r2 = tid / colsdst;
		int rb = r2 % rowsb;
		int cb = c2 % colsb;
		int ra = r2 / rowsb;
		int ca = c2 / colsb;
		dst[tid] = a[ra * colsa + ca] * b[rb * colsb + cb];
		tid += stride;
	}
}

__global__ void cu_downSample(const float *src, float* dst, const int y_stride, const int x_stride, const int colssrc, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int colsdst = colssrc / x_stride;
	if(colssrc % x_stride > 0) ++colsdst;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = rdst * y_stride;
		int csrc = cdst * x_stride;
		dst[tid] = src[rsrc * colssrc + csrc];
		tid += stride;
	}
}

__global__ void cu_interpolation(const float* src, float* dst, const int colssrc, const int colsdst, const int _stride, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int csrc = tid % colssrc;
		int rsrc = tid / colssrc;
		int rdst = rsrc * _stride;
		int cdst = csrc * _stride;
		dst[rdst * colsdst + cdst] = src[tid];
		tid += stride;
	}
}

__global__ void cu_getRange(const float *src, float* dst, const int xstart, const int xend, const int ystart, const int yend, const int colssrc, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int colsdst = xend - xstart + 1;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = rdst + ystart;
		int csrc = cdst + xstart;
		dst[tid] = src[rsrc * colssrc + csrc];
		tid += stride;
	}
}

__global__ void cu_copyMakeBorder(const float *src, float* dst, const int rowssrc, const int colssrc, const int up, const int down, const int left, const int right, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int colsdst = colssrc + left + right;
	while(tid < n){
		int csrc = tid % colssrc;
		int rsrc = tid / colssrc;
		int rdst = up + rsrc;
		int cdst = left + csrc;
		dst[rdst * colsdst + cdst] = src[tid];
		tid += stride;
	}
}

__global__ void cu_pooling_max(const float* src, float* dst, float *loc, const int rowssrc, const int colssrc, const int rowsdst, const int colsdst, const int stridex, const int stridey, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = rdst * stridey;
		int csrc = cdst * stridex;
		int xend = (csrc + stridex - 1) > (colssrc - 1) ? (colssrc - 1) : (csrc + stridex - 1);
		int yend = (rsrc + stridey - 1) > (rowssrc - 1) ? (rowssrc - 1) : (rsrc + stridey - 1);
		loc[tid] = (float)(rsrc * colssrc + csrc);
		for(int i = rsrc; i <= yend; ++i){
			for(int j = csrc; j <= xend; ++j){
				if(src[i * colssrc + j] > dst[tid]){
					dst[tid] = src[i * colssrc + j];
					loc[tid] = (float)(i * colssrc + j);
				}
			}
		}
		tid += stride;
	}
}

__global__ void cu_pooling_mean(const float* src, float* dst, float *loc, const int rowssrc, const int colssrc, const int rowsdst, const int colsdst, const int stridex, const int stridey, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = rdst * stridey;
		int csrc = cdst * stridex;
		int xend = (csrc + stridex - 1) > (colssrc - 1) ? (colssrc - 1) : (csrc + stridex - 1);
		int yend = (rsrc + stridey - 1) > (rowssrc - 1) ? (rowssrc - 1) : (rsrc + stridey - 1);
		loc[tid] = (float)(rsrc * colssrc + csrc);
		for(int i = rsrc; i <= yend; ++i){
			for(int j = csrc; j <= xend; ++j){
				dst[tid] += __fdividef(src[i * colssrc + j], __fmul_rd(yend - rsrc + 1, xend - csrc + 1));	
			}
		}
		tid += stride;
	}
}

__global__ void cu_pooling_overlap_max(const float* src, float* dst, float *loc, const int rowssrc, const int colssrc, const int rowsdst, const int colsdst, const int sizex, const int sizey, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = rdst;
		int csrc = cdst;
		int xend = (csrc + sizex - 1);
		int yend = (rsrc + sizey - 1);
		loc[tid] = (float)(rsrc * colssrc + csrc);
		for(int i = rsrc; i <= yend; ++i){
			for(int j = csrc; j <= xend; ++j){
				if(src[i * colssrc + j] > dst[tid]){
					dst[tid] = src[i * colssrc + j];
					loc[tid] = (float)(i * colssrc + j);
				}
			}
		}
		tid += stride;
	}
}

__global__ void cu_unpooling(const float* src, const float* loc, float* dst, const int colsdst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int cdst = (int)(loc[tid]) % colsdst;
		int rdst = (int)(loc[tid]) / colsdst;
		dst[rdst * colsdst + cdst] = src[tid];
		tid += stride;
	}
}

__global__ void cu_multiply(const float* A, const float* B, float * C,
                                    int rowsa, int colsa,
                                    int rowsb, int colsb,
                                    int rowsc, int colsc){
    __shared__ float sA[32][32];   // Tile size of 32x32
    __shared__ float sB[32][32];
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;
    for (int k = 0; k < (((colsa - 1)/ 32) + 1); k++){
        if ( (Row < rowsa) && (threadIdx.x + (k*32)) < colsa){
            sA[threadIdx.y][threadIdx.x] = A[(Row*colsa) + threadIdx.x + (k*32)];
        }
        else{
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
        if ( Col < colsb && (threadIdx.y + k*32) < rowsb){
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*32)*colsb + Col];
        }
        else{
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < 32; ++j){
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (Row < rowsc && Col < colsc){
        C[Row*colsc + Col] = Cvalue;
    }
}

__global__ void cu_transpose(const float* src, float* dst, int colssrc, int colsdst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		int cdst = tid % colsdst;
		int rdst = tid / colsdst;
		int rsrc = cdst;
		int csrc = rdst;
		dst[tid] = src[rsrc * colssrc + csrc];
		tid += stride;
	}
}

__global__ void cu_sigmoid(const float* src, float* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float tmp = __fmul_rd(src[tid], -1.0);
		tmp = __expf(tmp);
		tmp = __fadd_rd(tmp, 1.0);
		dst[tid] = __fdividef(1.0, tmp);
		tid += stride;
	}
}

__global__ void cu_dsigmoid(const float* src, float* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float tmp = __expf(src[tid]);
		float tmp2 = __fadd_rd(tmp, 1.0);
		tmp2 = __fmul_rd(tmp2, tmp2);
		dst[tid] = fdividef(tmp, tmp2);
		tid += stride;
	}
}

__global__ void cu_dsigmoid_a(const float* src, float* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float tmp = __fsub_rd(1.0, src[tid]);
		dst[tid] = __fmul_rd(tmp, src[tid]);
		tid += stride;
	}
}

__global__ void cu_relu(const float* src, float* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] > 0.0) dst[tid] = src[tid];
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_drelu(const float* src, float* dst, int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		if(src[tid] > 0.0) dst[tid] = 1.0;
		else dst[tid] = 0.0;
		tid += stride;
	}
}

__global__ void cu_leaky_relu(const float* src, float* dst, int n){
	const float leaky_relu_alpha = 100.0;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float p = 0.0;
		float n = 0.0;
		if(src[tid] > 0.0) p = src[tid];
		if(src[tid] < 0.0) n = src[tid];
		n = fdividef(n, leaky_relu_alpha);
		dst[tid] = __fadd_rd(p, n);
		tid += stride;
	}
}

__global__ void cu_dleaky_relu(const float* src, float* dst, int n){
	const float leaky_relu_alpha = 100.0;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float p = 0.0;
		float n = 0.0;
		if(src[tid] > 0.0) p = 1;
		if(src[tid] < 0.0) n = 1;
		n = fdividef(n, leaky_relu_alpha);
		dst[tid] = __fadd_rd(p, n);
		tid += stride;
	}
}

__global__ void cu_tanh(const float* src, float* dst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		dst[tid] = tanhf(src[tid]);
		tid += stride;
	}
}

__global__ void cu_dtanh(const float* src, float* dst, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		float tmp = __fmul_rd(src[tid], src[tid]);
		dst[tid] = __fsub_rd(1.0, tmp);
		tid += stride;
	}
}










