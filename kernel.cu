//TODO kernel implementation
#define BLOCK_SIZE 32
#include <cmath>

__global__ void compute_lines(const float * __restrict__ din, float* __restrict__ dout, int x, int y){
    unsigned int line = blockIdx . x * blockDim.x + threadIdx.x;
    float z = 1.73205080756887729f - 2.f;
    float z1;
    float* myLine = dout + (line * x);
    const float* myLineIn = din + (line * x);

    // compute 'sum'
    float sum = (myLineIn[0] + powf(z, x)
                             * myLineIn[x - 1]) * 6 * (1.f + z) / z;
    sum += dout[line * x] * 6;


    // iterate back and forth
    float last = sum * z / (1.f - powf(z, 2 * x));
    myLine[0] = last;
    float cur;
    for (int j = 1; j < x; ++j) {
        cur = myLineIn[j]* 6 + z * last;
        myLine[j] = cur;
        last = cur;
    }
    last = myLine[x - 1] * z / (z - 1.f);
    myLine[x - 1] =  last;
    for (int j = x - 2; 0 <= j; --j) {
        cur = z * (last - myLine[j]);
        myLine[j] = cur;
        last = cur;
    }
}

__global__ void compute_cols(const float * __restrict__ din, float* __restrict__ dout, int x, int y){

    unsigned int col = blockIdx . x * blockDim.x + threadIdx.x;
    const float z = 1.73205080756887729f - 2.f; //sqrt(3)
    float *myCol = dout + col;
    const float *myColIn = dout + col;

    // compute 'sum'
    float sum = (myColIn[0*x] + powf(z, y)
                              * myColIn[(y - 1)*x]) * 6 * (1.f + z) / z;

    sum += din[col] * 6;
//    if (col == 10){
//        printf("%f", sum);
//    }

    // iterate back and forth
    float cur;
    float last = sum * z / (1.f - powf(z, 2 * y));
    myCol[0] = last;
    for (int j = 1; j < y; ++j) {
        cur = myColIn[j*x] * 6 + z * last;
        myCol[j*x] = cur;
        last = cur;
    }
    last = myCol[(y - 1)*x] * z / (z - 1.f);
    myCol[(y - 1)*x] = last;
    for (int j = y - 2; 0 <= j; --j) {
        cur = z * (last - myCol[j*x]);
        myCol[j*x] = cur;
        last = cur;
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

template<unsigned int blockSize>
__device__ inline void computesumrec(volatile float* sv, int tid){

    if ( tid < blockSize )
        sv [ tid ] += sv [ tid + blockSize / 2 ];
    __syncthreads();

    computesumrec<blockSize / 2>(sv, tid);
}

template<>
__device__ inline void computesumrec<1>(volatile float* sv, int tid) {}
#define SUMBLOCK 64
__device__ float gpresums[X];

__global__ void compute_line_sum(const float * __restrict__ in, float* out, int x, bool first, int to_sum){
    __shared__ float sv [SUMBLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.y * x + blockDim.x * blockIdx.x + tid;

    float c = first ? __ldg(&gpresums[blockDim.x * blockIdx.x + tid]) : 1;

    //copy to memory with gain if first
    float a =  blockDim.x * blockIdx.x + tid < to_sum ? __ldg(&in[i]) : 0;
    sv [ tid ] = a * c;
    __syncthreads();

    computesumrec<SUMBLOCK/2>(sv, tid);
    if (tid == 0)
        out[blockIdx.y * x + blockIdx.x] = sv[0];
}

__global__ void compute_col_sum(const float * __restrict__ in, float* out, int x, bool first, int to_sum){
    __shared__ float sv [SUMBLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i =  blockIdx.x + (blockIdx.y * blockDim.x + tid) * x;

    float c = first ? __ldg(&gpresums[blockIdx.y * blockDim.x + tid]) : 1;
    float a = blockIdx.y * blockDim.x + tid < to_sum ? __ldg(&in[i]) : 0;
    //copy to memory with gain if first
    sv [ tid ] = a * c;
    __syncthreads();

    computesumrec<SUMBLOCK/2>(sv, tid);
    if (tid == 0)
        out[blockIdx.x + blockIdx.y * x] = sv[0];
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



float arr[X];

__global__ void  compute_presum(float *  cpresums, int x){
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

void solveGPU(float *in, float *out, int x, int y) {



//    cudaMalloc(&gpresums, x * sizeof(float));
    const float gain = 6.0f;
    const float z = 1.73205080756887729f - 2.f;
    float z1 = z;
    float z2 = powf(z, 2 * X - 2);
    float iz = 1.f / z;
    arr[0] = 0;
    arr[X-1] = 0;
    for (int j = 1; j < (X - 1); ++j) {
        arr[j] = (z2 + z1);
        z1 *= z;
        z2 *= iz;
    }
    cudaMemcpyToSymbol(gpresums, arr, x * sizeof(float));
//    compute_presum<<<1, 1>>>(gpresums, x);

    float * cur = in;
    bool first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
        dim3 dimgrid = dim3(aa, y);

        compute_line_sum<<<dimgrid, SUMBLOCK>>>(cur, out, x, first, a);
//        printf("%d\n", aa);
        cur = out;
        first = false;
    }

    compute_lines<<<y/BLOCK_SIZE, BLOCK_SIZE>>>(in, out, x, y);

    cur = out;
    first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
        dim3 dimgrid = dim3(y, aa);

        compute_col_sum<<<dimgrid, SUMBLOCK>>>(cur, in, x, first, a);
//        printf("%d\n", aa);
        cur = in;
        first = false;
    }
    compute_cols<<<x/BLOCK_SIZE, BLOCK_SIZE>>>(in, out, x, y);

}

