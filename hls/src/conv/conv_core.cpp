#include "conv_core.h"


void Conv(data_u16 CHin, data_u16 CHout, data_u16 Hin, data_u16 Win, data_u8 K, data_u8 S, data_bool relu, data_u8 pd, data_f32 feature_in[], data_f32 conv_w[], data_f32 bias[], data_f32 feature_out[])
{
#pragma HLS INTERFACE m_axi depth=4294967295 port=conv_w offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=bias offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_in offset=slave
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=CHout
#pragma HLS INTERFACE s_axilite port=S
#pragma HLS INTERFACE s_axilite port=relu
#pragma HLS INTERFACE s_axilite port=CHin
#pragma HLS INTERFACE s_axilite port=Hin
#pragma HLS INTERFACE s_axilite port=K
#pragma HLS INTERFACE s_axilite port=Win
#pragma HLS INTERFACE s_axilite port=pd


    data_u16 Hout = (Hin + 2 * pd - K) / S + 1;
    data_u16 Wout = (Win + 2 * pd - K) / S + 1;

    // 循环遍历输出特征图的每个点
    for(data_u16 hout = 0; hout < Hout; ++hout)
 {
        for(data_u16 wout = 0; wout < Wout; ++wout) {
            for(data_u16 chout = 0; chout < CHout; ++chout){

                data_f32 sum = bias[chout]; // 初始化为偏置值
                // 循环遍历卷积核的每个点
                for(data_u8 ky = 0; ky < K; ++ky) {
                    for(data_u8 kx = 0; kx < K; ++kx) {
                        for(data_u16 chin = 0; chin < CHin; ++chin) {
#pragma HLS UNROLL
                            // 计算输入特征图的坐标
                            data_u16 h = hout * S - pd + ky;
                            data_u16 w = wout * S - pd + kx;

                            // 确保坐标在输入特征图范围内
                            if(h < Hin && w < Win) {
                                // 计算卷积和
                                sum += feature_in[h * CHin * Win + w * CHin + chin] * conv_w[ky * K * CHin * CHout +  kx * CHin * CHout + chin * CHout + chout];

                            }
                        }
                    }
                }

                // 应用ReLU激活函数
                if(relu) {
                    sum = (sum > 0) ? sum : 0;
                }

                // 写入输出特征图
                feature_out[hout * Wout * CHout + wout * CHout+ chout] = sum;
            }
        }
    }
}
