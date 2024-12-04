#ifndef __POOLING_CORE_H__
#define __POOLING_CORE_H__


#include "ap_fixed.h"

typedef ap_uint<8>  data_u8;
typedef ap_uint<16> data_u16;
typedef float       data_f32;
typedef ap_uint<1>  data_bool;

void Pooling(data_u16 CHin, data_u16 Hin, data_u16 Win, data_u8 K, data_bool mode, float feature_in[],float feature_out[]);



#endif
