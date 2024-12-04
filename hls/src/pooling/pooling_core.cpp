#include "pooling_core.h"

#define max(a,b) ((a>b)?a:b)

typedef ap_uint<8>  data_u8;
typedef ap_uint<16> data_u16;
typedef float       data_f32;
typedef ap_uint<1>  data_bool;

void Pooling(data_u16 CHin, data_u16 Hin, data_u16 Win, data_u8 K, data_bool mode, float feature_in[],float feature_out[])
{
#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_in offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=Win
#pragma HLS INTERFACE s_axilite port=K
#pragma HLS INTERFACE s_axilite port=Hin
#pragma HLS INTERFACE s_axilite port=mode
#pragma HLS INTERFACE s_axilite port=CHin

    data_u16 pool_size = K * K; // 池化窗口大小
    data_u16 hout = Hin / K; // 输出高度
    data_u16 wout = Win / K; // 输出宽度

    // 遍历每个输出特征图
    for (data_u16 c = 0; c < CHin; ++c) {
        // 遍历每个输出像素
        for (data_u16 oh = 0; oh < hout; ++oh) {
            for (data_u16 ow = 0; ow < wout; ++ow) {

            	data_f32 max_val = 0;
            	if (mode == 0) max_val = -9999999;

                for (data_u8 kh = 0; kh < K; ++kh) {
                    for (data_u8 kw = 0; kw < K; ++kw) {

                        data_u16 ih = oh * K + kh;
                        data_u16 iw = ow * K + kw;

                        data_f32 in_val = feature_in[ih * Win * CHin + iw * CHin + c];

                        if (mode == 0) { // 最大池化
                            max_val = max(max_val, in_val);
                        } else { // 平均池化
                            max_val += in_val;
                        }

                    }
                }
                if (mode == 1) { // 平均池化
                    max_val /= pool_size;
                }
                feature_out[oh * wout * CHin + ow * CHin + c] = max_val;
            }
        }
    }
}
