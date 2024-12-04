#include "stdio.h"
#include "time.h"
#include "pooling_core.h"

#define MODE 0 // 0 : MAX ; 1 : MEAN
#define IN_WIDTH 4
#define IN_HEIGHT 4
#define IN_CH 3

#define KERNEL_WIDTH 2
#define KERNEL_HEIGHT 2

#define OUT_WIDTH (IN_WIDTH / KERNEL_WIDTH)
#define OUT_HEIGHT (IN_HEIGHT / KERNEL_HEIGHT)

int main(void)
{
    float feature_in[IN_HEIGHT][IN_WIDTH][IN_CH];
    float feature_out[OUT_HEIGHT][OUT_WIDTH][IN_CH];

    for (int i = 0; i < IN_HEIGHT; i++)
        for (int j = 0; j < IN_WIDTH; j++)
            for (int cin = 0; cin < IN_CH; cin++)
                feature_in[i][j][cin] = i * IN_HEIGHT * IN_WIDTH + j * IN_HEIGHT + cin;


    clock_t start_time = clock();
    Pooling(IN_CH, IN_HEIGHT, IN_WIDTH,
        KERNEL_WIDTH, MODE,
        feature_in[0][0], feature_out[0][0]
    );
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Function execution time: %f seconds\n", time_taken);
    for (int i = 0; i < OUT_HEIGHT; i++)
        for (int j = 0; j < OUT_WIDTH; j++)
            for (int cout = 0; cout < IN_CH; cout++)
            {
                printf("OUT[%d][%d][%d]=%f\n", i, j, cout, feature_out[i][j][cout]);
            }
}
