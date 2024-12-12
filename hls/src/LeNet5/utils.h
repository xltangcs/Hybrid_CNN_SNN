#ifndef __UTILS_H__
#define __UTILS_H__

#include "ap_fixed.h"


// model
/*************************
 * lenet5
 * layer1 : 28*28*1 -> 24*24*6 -> 12*12*6
 * layer2 : 12*12*6 -> 8*8*16  -> 4*4*16
 * flatten: 256
 * layer3 : 256 -> 120
 * layer4 : 120 -> 84
 * layer5 : 84  -> 10
 */

#define LAYER1_INPUT_SIZE       28
#define LAYER1_CONV_SIZE        24
#define LAYER1_POOL_SIZE        LAYER1_CONV_SIZE/2
#define LAYER1_INPUT_CHANNELS   1
#define LAYER1_OUTPUT_CHANNELS  6
#define LAYER2_INPUT_SIZE       12
#define LAYER2_CONV_SIZE        8
#define LAYER2_POOL_SIZE        LAYER2_CONV_SIZE/2
#define LAYER2_INPUT_CHANNELS   6
#define LAYER2_OUTPUT_CHANNELS  16
#define LAYER3_INPUT_SIZE       4*4*16
#define LAYER3_OUTPUT_SIZE      120
#define LAYER4_INPUT_SIZE       120
#define LAYER4_OUTPUT_SIZE      84
#define LAYER5_INPUT_SIZE       84
#define LAYER5_OUTPUT_SIZE      10
#define KERNEL_SIZE             5

// define data size
typedef ap_uint<1>      data_bool;
typedef ap_int<8>       data_i8;
typedef ap_uint<8>      data_u8;
typedef ap_int<16>      data_i16;
typedef ap_uint<16>     data_u16;
typedef float           data_f32;

#ifdef QUANT
    typedef float           weight_size;
    typedef float           bias_size;
#else
    typedef float           weight_size;
    typedef float           bias_size;
#endif


#endif
