#include <cmath>

void solveCPU(float *in, float *out, int x, int y) {
    const float gain = 6.0f;
    const float z = sqrtf(3.f) - 2.f;
    float z1;

    // process lines
    for (int line = 0; line < y; line++) {
        float* myLine = out + (line * x);

        // copy input data
        for (int i = 0; i < x; i++) {
            myLine[i] = in[i + (line * x)] * gain;
        }

        // compute 'sum'
        float sum = (myLine[0] + powf(z, x)
            * myLine[x - 1]) * (1.f + z) / z;
        z1 = z;
        float z2 = powf(z, 2 * x - 2);
        float iz = 1.f / z;
        for (int j = 1; j < (x - 1); ++j) {
            sum += (z2 + z1) * myLine[j];
            z1 *= z;
            z2 *= iz;
        }
//        if (line == 10){
//            printf("%f", sum);
//        }

//        // iterate back and forth
        myLine[0] = sum * z / (1.f - powf(z, 2 * x));
//        int j = 1;
//        myLine[j] += z * myLine[j - 1];
        for (int j = 1; j < x; ++j) {
            myLine[j] += z * myLine[j - 1];
        }
        myLine[x - 1] *= z / (z - 1.f);
        for (int j = x - 2; 0 <= j; --j) {
            myLine[j] = z * (myLine[j + 1] - myLine[j]);
        }


    }

    // process columns
    for (int col = 0; col < x; col++) {
        float* myCol = out + col;

        // multiply by gain (input data are already copied)
        for (int i = 0; i < y; i++) {
            myCol[i*x] *= gain;
        }

        // compute 'sum'
        float sum = (myCol[0*x] + powf(z, y)
            * myCol[(y - 1)*x]) * (1.f + z) / z;
        z1 = z;
        float z2 = powf(z, 2 * y - 2);
        float iz = 1.f / z;
        for (int j = 1; j < (y - 1); ++j) {
            sum += (z2 + z1) * myCol[j*x];
            z1 *= z;
            z2 *= iz;
        }
//        if (col == 10){
//            printf("%f", sum);
//        }

        // iterate back and forth
        myCol[0*x] = sum * z / (1.f - powf(z, 2 * y));
        for (int j = 1; j < y; ++j) {
            myCol[j*x] += z * myCol[(j - 1)*x];
        }
        myCol[(y - 1)*x] *= z / (z - 1.f);
        for (int j = y - 2; 0 <= j; --j) {
            myCol[j*x] = z * (myCol[(j + 1)*x] - myCol[j*x]);
        }
    }
}

