#include "stdio.h"
#include "time.h"
#include "conv_core.h"

#define IN_WIDTH 28
#define IN_HEIGHT 28
#define IN_CH 1

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5
#define X_STRIDE 1
#define Y_STRIDE 1

#define RELU_EN 0
#define MODE 0 //0:VALID, 1:SAME
#define X_PADDING (MODE ? (KERNEL_WIDTH - 1) / 2 : 0)
#define Y_PADDING (MODE ? (KERNEL_HEIGHT - 1) / 2 : 0)

#define OUT_CH 1
#define OUT_WIDTH ((IN_WIDTH + 2 * X_PADDING - KERNEL_WIDTH) / X_STRIDE + 1)
#define OUT_HEIGHT ((IN_HEIGHT + 2 * Y_PADDING - KERNEL_HEIGHT) / Y_STRIDE + 1)

int main(void)
{
    float feature_in[IN_HEIGHT][IN_WIDTH][IN_CH];
    float W[KERNEL_HEIGHT][KERNEL_WIDTH][IN_CH][OUT_CH];
    float bias[OUT_CH];
    float feature_out[OUT_HEIGHT][OUT_WIDTH][OUT_CH];

    for (int i = 0; i < IN_HEIGHT; i++)
        for (int j = 0; j < IN_WIDTH; j++)
            for (int cin = 0; cin < IN_CH; cin++)
                feature_in[i][j][cin] = i * IN_WIDTH + j;

    for (int i = 0; i < KERNEL_HEIGHT; i++)
        for (int j = 0; j < KERNEL_WIDTH; j++)
            for (int cin = 0; cin < IN_CH; cin++)
                for (int cout = 0; cout < OUT_CH; cout++)
                    W[i][j][cin][cout] = 1;

    for (int cout = 0; cout < OUT_CH; cout++)
        bias[cout] = 0;

    double start_time = clock();
    Conv(IN_CH, OUT_CH, IN_HEIGHT, IN_WIDTH,
        KERNEL_WIDTH, Y_STRIDE, RELU_EN, MODE,
        feature_in[0][0], W[0][0][0], bias, feature_out[0][0]
    );
    double end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Function execution time: %f seconds\n", time_taken);

    for (int i = 0; i < OUT_HEIGHT; i++)
        for (int j = 0; j < OUT_WIDTH; j++)
            for (int cout = 0; cout < OUT_CH; cout++)
            {
                printf("OUT[%d][%d][%d]=%f\n", i, j, cout, feature_out[i][j][cout]);
            }

    return 0;
}
