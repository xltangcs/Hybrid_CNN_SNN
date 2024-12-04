#ifndef __CONV_CORE_H__
#define __CONV_CORE_H__

#include "ap_fixed.h"

typedef ap_uint<8>  data_u8;
typedef ap_uint<16> data_u16;
typedef float       data_f32;
typedef ap_uint<1>  data_bool;


void Conv(data_u16 CHin, data_u16 CHout, data_u16 Hin, data_u16 Win, data_u8 K, data_u8 S, data_bool relu, data_u8 pd, data_f32 feature_in[], data_f32 conv_w[], data_f32 bias[], data_f32 feature_out[]);

#endif
