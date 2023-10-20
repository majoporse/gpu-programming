//TODO kernel implementation
#define BLOCK_SIZE 32
#include <cmath>

// copy input data
__global__ void copy_data(float* din, float* dout){
    unsigned int i = blockIdx . x* blockDim . x + threadIdx . x;
    dout[i] = din[i] * 6.0f;
}

__global__ void compute_lines(const float *din, float* dout, int x, int y){
    unsigned int line = blockIdx . x;
    float z = sqrtf(3.f) - 2.f;
    float z1;
    int tid = threadIdx.x;
    float* myLine = dout + (line * x);

    __shared__ float sm[BLOCK_SIZE];
    sm[tid] = dout[line * x + tid];
    __syncthreads();

    // compute 'sum'
    float sum = (sm[0] + powf(z, x) * myLine[x - 1]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * x - 2);
    float iz = 1.f / z;

    int iters = 0;
    while (iters * BLOCK_SIZE < x) {
        if (tid == 0){
            for (int j = 0; j < BLOCK_SIZE && j + iters * BLOCK_SIZE < (x - 1); ++j){ //j < (x - 1); ++j) {
                if (iters == 0 && j == 0)
                    continue;
                sum += (z2 + z1) * sm[j];
                z1 *= z;
                z2 *= iz;
            }
        }
        ++iters;
        __syncthreads();
        if (line *x + BLOCK_SIZE * iters < x)
            sm[tid] = dout[line * x + BLOCK_SIZE * iters + tid];
        __syncthreads();
    }
    if (tid == 0){
        // iterate back and forth
        myLine[0] = sum * z / (1.f - powf(z, 2 * x));
        for (int j = 1; j < x; ++j) {
            myLine[j] += z * myLine[j - 1];
        }
        myLine[x - 1] *= z / (z - 1.f);
        for (int j = x - 2; 0 <= j; --j) {
            myLine[j] = z * (myLine[j + 1] - myLine[j]);
        }
    }


}

__global__ void compute_cols(const float *din, float* dout, int x, int y){
    unsigned int col = blockIdx . x;
    const float z = sqrtf(3.f) - 2.f;
    float z1;
    int a = 0;
    float *myCol = dout + col;
    int tid = threadIdx.x;

    // compute 'sum'
    __shared__ float sm[BLOCK_SIZE];
    sm[tid] = din[col + tid * x];
    __syncthreads();

    // compute 'sum'
    float sum = (sm[0] + powf(z, y) * myCol[(y - 1) * x]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * y - 2);
    float iz = 1.f / z;

    int iters = 0;
    while (iters * BLOCK_SIZE < y) {
        if (tid == 0){
            for (int j = 0; j < BLOCK_SIZE && j + iters * BLOCK_SIZE < (y - 1); ++j){
                if (iters == 0 && j == 0)
                    continue;
                sum += (z2 + z1) * sm[j];
                z1 *= z;
                z2 *= iz;
            }
        }
        ++iters;
        __syncthreads();
        if (BLOCK_SIZE * iters < y)
            sm[tid] = din[col + x * (BLOCK_SIZE * iters + tid)];
        __syncthreads();
    }

    if (tid == 0){
        // iterate back and forth
        myCol[0] = sum * z / (1.f - powf(z, 2 * y));
        for (int j = 1; j < y; ++j) {
            myCol[j * x] += z * myCol[(j - 1) * x];
        }
        myCol[(y - 1) * x] *= z / (z - 1.f);
        for (int j = y - 2; 0 <= j; --j) {
            myCol[j * x] = z * (myCol[(j + 1) * x] - myCol[j * x]);
        }

    }
}


void solveGPU(float *in, float *out, int x, int y) {
//    const float gain = 6.0f;
//    const float z = sqrtf(3.f) - 2.f;
//    float z1;
    copy_data<<<(x*y)/256, 256>>>(in, out);
    compute_lines<<<y, BLOCK_SIZE>>>(in, out, x, y);

//    copy_data<<<(x*y)/256, 256>>>(out, out);
//    compute_cols<<<x, BLOCK_SIZE>>>(in, out, x, y);
//
//    for (int line = 0; line < y; line++) {
//        float* myLine = out + (line * x);
//
//        // copy input data
//        for (int i = 0; i < x; i++) {
//            myLine[i] = in[i + (line * x)] * gain;
//        }
//
//        // compute 'sum'
//        float sum = (myLine[0] + powf(z, x)
//                                 * myLine[x - 1]) * (1.f + z) / z;
//        z1 = z;
//        float z2 = powf(z, 2 * x - 2);
//        float iz = 1.f / z;
//        for (int j = 1; j < (x - 1); ++j) {
//            sum += (z2 + z1) * myLine[j];
//            z1 *= z;
//            z2 *= iz;
//        }
//
//        // iterate back and forth
//        myLine[0] = sum * z / (1.f - powf(z, 2 * x));
//        for (int j = 1; j < x; ++j) {
//            myLine[j] += z * myLine[j - 1];
//        }
//        myLine[x - 1] *= z / (z - 1.f);
//        for (int j = x - 2; 0 <= j; --j) {
//            myLine[j] = z * (myLine[j + 1] - myLine[j]);
//        }
//    }
//
//    // process columns
//    for (int col = 0; col < x; col++) {
//        float *myCol = out + col;
//
//        // multiply by gain (input data are already copied)
//        for (int i = 0; i < y; i++) {
//            myCol[i * x] *= gain;
//        }
//
//        // compute 'sum'
//        float sum = (myCol[0 * x] + powf(z, y)
//                                    * myCol[(y - 1) * x]) * (1.f + z) / z;
//        z1 = z;
//        float z2 = powf(z, 2 * y - 2);
//        float iz = 1.f / z;
//        for (int j = 1; j < (y - 1); ++j) {
//            sum += (z2 + z1) * myCol[j * x];
//            z1 *= z;
//            z2 *= iz;
//        }
//
//        // iterate back and forth
//        myCol[0 * x] = sum * z / (1.f - powf(z, 2 * y));
//        for (int j = 1; j < y; ++j) {
//            myCol[j * x] += z * myCol[(j - 1) * x];
//        }
//        myCol[(y - 1) * x] *= z / (z - 1.f);
//        for (int j = y - 2; 0 <= j; --j) {
//            myCol[j * x] = z * (myCol[(j + 1) * x] - myCol[j * x]);
//        }
//    }
}

