//TODO kernel implementation
#define BLOCK_SIZE 32
#include <cmath>
__device__ float BUF[X*Y];
__device__ float gpresums[X];

__global__ void compute_cols(const float * __restrict__ din, float* dout, int x, int y, bool in_col){

    unsigned int col = blockIdx . x * blockDim.x + threadIdx.x;
    const float z = 1.73205080756887729f - 2.f; //sqrt(3)
    float *myCol = dout + col;

    // compute 'sum'
    float sum = (myCol[0*x] + powf(z, y)
                              * myCol[(y - 1)*x]) * 6 * (1.f + z) / z;
    int i = in_col ? col * x : col;
    sum += BUF[i] * 6;

    // iterate back and forth
    float cur;
    float last = sum * z / (1.f - powf(z, 2 * y));
    myCol[0] = last;
    for (int j = 1; j < y; ++j) {
        __syncthreads();
        cur = myCol[j*x] * 6 + z * last;
        myCol[j*x] = cur;
        last = cur;

    }
    __syncthreads();
    last = myCol[(y - 1)*x] * z / (z - 1.f);
    myCol[(y - 1)*x] = last;
    for (int j = y - 2; 0 <= j; --j) {
        __syncthreads();
        cur = z * (last - myCol[j*x]);
        myCol[j*x] = cur;
        last = cur;

    }
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
#define SUMBLOCK 128


__global__ void compute_line_sum(const float * __restrict__ in, int x, bool first, int to_sum){
    __shared__ float sv [SUMBLOCK];
    if (!first)
        in = BUF;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.y * x + blockDim.x * blockIdx.x + tid;

    float c = first ? gpresums[blockDim.x * blockIdx.x + tid] : 1;

    //copy to memory with gain if first
    float a =  blockDim.x * blockIdx.x + tid < to_sum ? in[i] : 0;
    sv [ tid ] = a * c;
    __syncthreads();

    computesumrec<SUMBLOCK/2>(sv, tid);
    if (tid == 0)
        BUF[blockIdx.y * x + blockIdx.x] = sv[0];
}


float arr[X];

#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void transposeCoalesced(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void solveGPU(float *in, float *out, int x, int y) {

//    cudaMalloc(&buf, x * y * sizeof(float));
    dim3 a = dim3(x/32, y/32);
    dim3 b = dim3(32, 8);
    transposeCoalesced<<<a, b>>>(out, in);
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

    bool first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
        dim3 dimgrid = dim3(aa, y);

        compute_line_sum<<<dimgrid, SUMBLOCK>>>(in, x, first, a);
//        printf("%d\n", aa);
        first = false;
    }
    compute_cols<<<x/BLOCK_SIZE, BLOCK_SIZE>>>(in, out, x, y, true);


    first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
        dim3 dimgrid = dim3(aa, y);

        compute_line_sum<<<dimgrid, SUMBLOCK>>>(out, x, first, a);
//        printf("%d\n", aa);
        first = false;
    }
    transposeCoalesced<<<a, b>>>(in, out);
    compute_cols<<<x/BLOCK_SIZE, BLOCK_SIZE>>>(in, in, x, y, true);
    cudaMemcpy(out, in, x*y* sizeof(float), cudaMemcpyDeviceToDevice);
}

