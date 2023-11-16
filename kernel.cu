//TODO kernel implementation
#define BLOCK_SIZE 16
#include <cmath>
__device__ float BUF_SUM[X*Y];
__device__ float gpresums[X > Y ? X : Y];

//__device__ inline void parallel_sum(int x, float sm[BLOCK_SIZE][BLOCK_SIZE + 10], const float *myColIn, float * myCol){
//    const float z = 1.73205080756887729f - 2.f; //sqrt(3) - 2
//    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
//    unsigned int tidx = threadIdx.x;
//    unsigned int tidy = threadIdx.y;
//
//    if (threadIdx.y == 0){
//        //load prev 10
//        for (int i = 0; i <= 10; ++i)
//            sm[tidx][10 - i] = myColIn[(row - i) * x] * 6;
//    } else{
//        sm[tidx][tidy + 10] = myColIn[row * x] * 6;
//    }
//    __syncthreads();
//
//    float total = sm[tidx][tidy + 10];
//    float pow = z;
//
//    for (int i = 1; i < 10; ++i){
//        total += sm[tidx][tidy + 10 - i] * pow;
//        pow = pow * z;
//    }
//    myCol[row * x] = total;
//}

__device__ inline void parallel_diff(int x, float sm[BLOCK_SIZE][BLOCK_SIZE + 10],const float* myColIn, float * myCol){
    const float z = 1.73205080756887729f - 2.f; //sqrt(3) - 2
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    if (threadIdx.y == blockDim.y - 1){
        //load 10 forward
        for (int i = 0; i <= 10; ++i){
            sm[threadIdx.x][threadIdx.y + i] = myColIn[(row + i) * x];
        }
    } else{
        sm[threadIdx.x][threadIdx.y] = myColIn[row * x];
    }
    __syncthreads();

    float total = 0;
    double pow = z;
    for (int i = 0; i < 10; ++i){
        total += pow * -sm[tidx][tidy + i];
//        if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.y == 11 && threadIdx.x == 0)
//            printf("| %f |",total);
        pow *= z;
    }
    myCol[row * x] = total;
}

__device__ inline void sequential_sum(int x, int y, const float* myColIn, float* myCol, bool is_trans){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const float z = 1.73205080756887729f - 2.f; //sqrt(3) - 2

    float p = y < 10 ? powf(z, y) : 0;
    float sum = (myColIn[0] + p * myColIn[(y - 1)*x]) * 6 * (1.f + z) / z;
    sum += BUF_SUM[ col * (is_trans ? X : Y) ] * 6;

    float cur;
    float last = sum * z / (1.f - p*p);
    myCol[0] = last;
    for (int j = 1; j < BLOCK_SIZE; ++j) {
        cur = myColIn[j * x] * 6 + z * last;
        myCol[j * x] = cur;
        last = cur;
    }
}

__device__ inline void sequential_diff(int x, int y, const float *myColIn, float* myCol, bool is_trans){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const float z = -0.2679491924; //sqrt(3) - 2

    float p = y < 10 ? powf(z, y) : 0;
    float sum = (myColIn[0] + p * myColIn[(y - 1)*x]) * 6 * (1.f + z) / z;
    sum += BUF_SUM[ col * (is_trans ? X : Y) ] * 6;

    float cur;
    float last = myColIn[(y - 1)*x] * z / (z - 1.f);
    myCol[(y - 1)*x] = last;
    for (int j = y - 2; j > y - 2 - BLOCK_SIZE; --j) {
        cur = z * (last - myColIn[j*x]);
        myCol[j*x] = cur;
        last = cur;
    }
}

__global__ void compute_cols(float* din, float* dout, int x, int y, bool is_trans) {
    __shared__ float sm[BLOCK_SIZE][BLOCK_SIZE + 10];
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float *myCol = dout + col;
    float *myColIn = din + col;

    // iterate back and forth
    if (blockIdx.y == 0 && threadIdx.y == 0) {
        sequential_sum(x, y, myColIn, myCol, is_trans);
    } else if (blockIdx.y != 0) {
        parallel_sum(x, sm, myColIn, myCol);
    }
}

__global__ void compute_cols_back(float* din, float* dout, int x, int y, bool is_trans) {
    __shared__ float sm[BLOCK_SIZE][BLOCK_SIZE + 10];
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float *myCol = dout + col;
    float *myColIn = din + col;

    if (blockIdx.y == gridDim.y - 1 && threadIdx.y == 0){
        sequential_diff(x, y, myColIn, myCol, is_trans);
    } else if (blockIdx.y != gridDim.y -1){
        parallel_diff(x, sm, myColIn, myCol);
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


__global__ void compute_line_sum(const float * in, int x, bool first, int to_sum){
    __shared__ float sv [SUMBLOCK];
    if (!first)
        in = BUF_SUM;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.y * x + blockDim.x * blockIdx.x + tid;

    float c = first ? gpresums[blockDim.x * blockIdx.x + tid] : 1;

    //copy to memory with gain if first
    float a =  blockDim.x * blockIdx.x + tid < to_sum ? in[i] : 0;
    sv [ tid ] = a * c;
    __syncthreads();

    computesumrec<SUMBLOCK/2>(sv, tid);
    if (tid == 0)
        BUF_SUM[blockIdx.y * x + blockIdx.x] = sv[0];
}


float arr[X > Y ? X : Y];

#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void transposeCoalesced(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y + j];
}

void solveGPU(float *in, float *out, int x, int y) {

    transposeCoalesced<<<dim3(x/32, y/32), dim3(32, 8)>>>(out, in);

    const float z = 1.73205080756887729f - 2.f; //sqrt(3)
    const int xx = (X > Y ? X : Y);
    float z1 = z;
    float z2 = powf(z, 2 * xx - 2);
    float iz = 1.f / z;
    arr[0] = 0;
    arr[xx-1] = 0;
    for (int j = 1; j < (xx - 1); ++j) {
        arr[j] = (z2 + z1);
        z1 *= z;
        z2 *= iz;
    }
    cudaMemcpyToSymbol(gpresums, arr, xx * sizeof(float));

    bool first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;

        compute_line_sum<<<dim3(aa, y), SUMBLOCK>>>(in, x, first, a);
        first = false;
    }

    //compute on transposed matrix

    compute_cols<<<dim3(Y/BLOCK_SIZE, X/BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(out, in, y, x, true);
    compute_cols_back<<<dim3(Y/BLOCK_SIZE, X/BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(in, out, y, x, true);

    first = true;
    //still transposed matrix
//    for (int a = y; a > 0; a /= SUMBLOCK) {
//        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
//
//        compute_line_sum<<<dim3(aa, x), SUMBLOCK>>>(out, y, first, a);
//        first = false;
//    }

//    cudaMemcpy(out, in, x*y* sizeof(float), cudaMemcpyDeviceToDevice);
    transposeCoalesced<<<dim3(y/32, x/32), dim3(32, 8)>>>(in, out);

    //compute on normal matrix
    compute_cols<<<dim3(X/BLOCK_SIZE, Y/BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(out, in, x, y, true);
    compute_cols_back<<<dim3(X/BLOCK_SIZE, Y/BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(in, out, x, y, true);
    cudaMemcpy(out, in, x*y* sizeof(float), cudaMemcpyDeviceToDevice);
}