//TODO kernel implementation
#define BLOCK_SIZE 32
#include <cmath>

// copy input data
__global__ void copy_data(float* din, float* dout){
    unsigned int i = blockIdx . x* blockDim . x + threadIdx . x;
    dout[i] = din[i] * 6.0f;
}

__global__ void compute_lines(const float *din, float* dout, int x, int y, float* sums){
    unsigned int line = blockIdx . x * blockDim.x + threadIdx.x;
    float z = sqrtf(3.f) - 2.f;
    float z1;
    float* myLine = dout + (line * x);
    const float* myLineIn = din + (line * x);

    // compute 'sum'
    float sum = (myLineIn[0] + powf(z, x)
                             * myLineIn[x - 1]) * 6 * (1.f + z) / z;
    sum += sums[line] * 6;
//    if (line == 0){
//        printf("%f", sum);
//    }

    // iterate back and forth
    myLine[0] = sum * z / (1.f - powf(z, 2 * x));
    for (int j = 1; j < x; ++j) {
        myLine[j] = myLineIn[j]* 6 + z * myLine[j - 1] ;
    }
    myLine[x - 1] =  myLine[x - 1] * z / (z - 1.f);
    for (int j = x - 2; 0 <= j; --j) {
        myLine[j] = z * (myLine[j + 1] - myLine[j]);
    }
}

__global__ void compute_cols(const float *din, float* dout, int x, int y, float * sums){
    unsigned int col = blockIdx . x * blockDim.x + threadIdx.x;
    const float z = sqrtf(3.f) - 2.f;
//    float z1;
    float *myCol = dout + col;
    const float *myColIn = din + col;

    // compute 'sum'
    float sum = (myColIn[0*x] + powf(z, y)
                              * myColIn[(y - 1)*x]) * 6 * (1.f + z) / z;

    sum += sums[col] * 6;
    // iterate back and forth
    myCol[0*x] = sum * z / (1.f - powf(z, 2 * y));
    for (int j = 1; j < y; ++j) {
        myCol[j*x] = myColIn[j*x] * 6 + z * myCol[(j - 1)*x];
    }
    myCol[(y - 1)*x] *= z / (z - 1.f);
    for (int j = y - 2; 0 <= j; --j) {
        myCol[j*x] = z * (myCol[(j + 1)*x] - myCol[j*x]);
    }
}

__global__ void com(const float* presums, float* sums, const float * in, int x){
    __shared__ float cache[128];
    float sum = 0;
    int tid = threadIdx.x;
    int a = x / blockDim.x;
    int block_pos_x = blockIdx.x % a;
    int block_pos_y = blockIdx.x / a;
    cache[tid] = presums[block_pos_x * blockDim.x + tid];

    __syncthreads();
    for (int i = 0; i < 128; ++i){
        int cur = x * (block_pos_y * blockDim.x + tid) + block_pos_x * blockDim.x + i;
        sum += cache[i] * in[cur];
    }

    atomicAdd(&sums[tid + block_pos_y * blockDim.x], sum);
}

__global__ void com2(const float* presums, float* sums, const float * in, int x){
    __shared__ float cache[128];
    float sum = 0;
    int tid = threadIdx.x;
    int a = x / blockDim.x;
    int block_pos_x = blockIdx.x % a;
    int block_pos_y = blockIdx.x / a;
    cache[tid] = presums[block_pos_y * blockDim.x + tid];

    __syncthreads();
    for (int i = 0; i < 128; ++i){
        int cur = x * block_pos_y * blockDim.x + block_pos_x * blockDim.x + tid + i * x;
        sum += cache[i] * in[cur];
    }

    atomicAdd(&sums[tid + block_pos_x * blockDim.x], sum);
}

__global__ void  compute_presum( float * cpresums, int x){
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1 = z;
    float z2 = powf(z, 2 * x - 2);
    float iz = 1.f / z;

    cpresums[0] = 0;
    cpresums[x-1] = 0;
    for (int j = 1; j < (x - 1); ++j) {
        cpresums[j] = (z2 + z1);
        z1 *= z;
        z2 *= iz;
    }
}

__device__ float* gpresums;
__device__ float* sums;
__device__ float* sums2;

void solveGPU(float *in, float *out, int x, int y) {

//    copy_data<<<(x*y)/256, 256>>>(in, out);

    cudaMalloc(&gpresums, x * sizeof(float));
    cudaMalloc(&sums, x * sizeof(float));
    cudaMalloc(&sums2, x * sizeof(float));
//    cudaMemset(sums, 0, x * sizeof(float));
    compute_presum<<<1,1>>>(gpresums, x);
    com<<<(x*y)/(128 * 128), 128>>>(gpresums, sums, in, x);

    compute_lines<<<y/BLOCK_SIZE, BLOCK_SIZE>>>(in, out, x, y, sums);

//    copy_data<<<(x*y)/256, 256>>>(in, out);
    com2<<<(x*y)/(128 * 128), 128>>>(gpresums, sums2, out, x);
    compute_cols<<<x/BLOCK_SIZE, BLOCK_SIZE>>>(out, out, x, y, sums2);

}

