//TODO kernel implementation
#define BLOCK_SIZE 32
#include <cmath>
__device__ float BUF[X*Y];
__device__ float gpresums[(X > Y) ? X : Y];

__global__ void compute_cols(float* dout, int x, int y){
    float buf[32];

    unsigned int col = blockIdx . x * blockDim.x + threadIdx.x;
    const float z = 1.73205080756887729f - 2.f; //sqrt(3)
    float *myCol = dout + col;
    float a = powf(z, y);

    // compute 'sum'
    float sum = (myCol[0] + a
                            * myCol[(y - 1)*x]) * 6 * (1.f + z) / z;
    sum += BUF[col*x] * 6;

    // iterate back and forth
    float cur;
    float last = sum * z / (1.f - a * a);
    myCol[0] = last;
    for (int j = 0; j < y; j += 32) {

        for (int b = 0; b < 32 && j+b < y; ++b){
            __syncthreads();
            buf[b] = myCol[(j + b)*x];
        }


        for (int b = 0; b < 32 && j+b < y; ++b){
            if (j+b == 0)
                continue;

            cur = buf[b] * 6 + z * last;
            buf[b] = cur;
            last = cur;
        }

        for (int b = 0; b < 32 && j+b < y; ++b){
            __syncthreads();
            myCol[(j+b) * x] = buf[b];
        }
    }

    __syncthreads();
    last = myCol[(y - 1)*x] * z / (z - 1.f);
    myCol[(y - 1)*x] = last;
    for (int j = y - 2 - 31; 0 <= j + 32; j-=32) {

        for (int b = 31; b >= 0 && 0 <= j+ b; --b){
            __syncthreads();
            buf[b] = myCol[(j + b)*x];
        }

        for (int b = 31; b >= 0 && 0 <= j+ b; --b){

            cur = z * (last - buf[b]);
            buf[b] = cur;
            last = cur;
        }

        for (int b = 31; b >= 0 && 0 <= j+ b; --b){
            __syncthreads();
            myCol[(j+b) * x] = buf[b];
        }

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

#define SUMBLOCK 16


__global__ void compute_line_sum(const float * __restrict__ in, int x, bool first, int to_sum){
    __shared__ float sv [SUMBLOCK][SUMBLOCK + 1];
    __shared__ float presums[SUMBLOCK];

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    if (!first)
        in = BUF;

    if (first && tidy == 0)
        presums[tidx] = gpresums[blockIdx.x * blockDim.x + tidx];

    __syncthreads();
    unsigned int i = (blockDim.y * blockIdx.y + tidy) * x + blockDim.x * blockIdx.x + tidx;

    float c = first ? presums[tidx] : 1;

    //copy to memory with gain if first
    float a =  (blockDim.x * blockIdx.x + tidx < to_sum) ? in[i] : 0;
    sv[tidy][tidx] = a * c;
    __syncthreads();

    computesumrec<SUMBLOCK/2>(sv[tidy], tidx);
    if (tidx == 0)
        BUF[(blockIdx.y * blockDim.y + tidy) * x + blockIdx.x] = sv[tidy][0];
}


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

__global__ void compute_sums(){
    const float z = 1.73205080756887729f - 2.f;
    float z1 = z;
    float z2 = powf(z, 2 * X - 2);
    float iz = 1.f / z;
    gpresums[0] = 0;
    gpresums[X-1] = 0;
    for (int j = 1; j < (X - 1); ++j) {
        gpresums[j] = (z2 + z1);
        z1 *= z;
        z2 *= iz;
    }
}

void solveGPU(float *in, float *out, int x, int y) {

    dim3 a = dim3(x/32, y/32);
    dim3 b = dim3(32, 8);
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    transposeCoalesced<<<a, b, TILE_DIM*(TILE_DIM + 1), s1>>>(out, in);
    compute_sums<<<1,1, 0, s2>>>();
    dim3 block = dim3(SUMBLOCK, SUMBLOCK);
    bool first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
        dim3 dimgrid = dim3(aa, y / SUMBLOCK);

        compute_line_sum<<<dimgrid, block>>>(in, x, first, a);
        first = false;
    }

    compute_cols<<<x/32, 32>>>( out, x, y);


    transposeCoalesced<<<a, b, TILE_DIM*(TILE_DIM + 1), s1>>>(in, out);
    first = true;
    for (int a = x; a > 0; a /= SUMBLOCK) {
        int aa = (a + SUMBLOCK - 1) / SUMBLOCK;
        dim3 dimgrid = dim3(aa, y / SUMBLOCK);

        compute_line_sum<<<dimgrid, block>>>(out, x, first, a);
        first = false;
    }
    compute_cols<<<x/32, 32>>>(in, x, y);
    cudaMemcpy(out, in, x*y* sizeof(float), cudaMemcpyDeviceToDevice);
}

