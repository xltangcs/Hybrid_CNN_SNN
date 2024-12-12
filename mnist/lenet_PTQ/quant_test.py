import torch
import os

# load model
model = torch.load("./result/32bit_best_model.pth")

# print(model["conv2.bias"])
# for name, param in model.items():
#     v = name.replace('.', '_')
#     print(f"{v} = model['{name}']")
# load params
conv1_weights = model['conv1.weight']
conv1_bias = model['conv1.bias']
conv2_weights = model['conv2.weight']
conv2_bias = model['conv2.bias']
fc1_weights = model['full3.weight']
fc1_bias = model['full3.bias']
fc2_weights = model['full4.weight']
fc2_bias = model['full4.bias']
fc3_weights = model['full5.weight']
fc3_bias = model['full5.bias']

# quant
import numpy as np

# quantization
s_x = 1 / 255

# manually set activation range by looking at the histogram
conv1_rst_max = 2
conv2_rst_max = 6
fc1_rst_max = 15
fc2_rst_max = 15

# manually set weight range by looking at the histogram
conv1_weights_max = 0.5
conv2_weights_max = 0.45
fc1_weights_max = 0.4
fc2_weights_max = 0.3
fc3_weights_max = 0.4

conv1_weight_max = torch.quantile(conv1_weights, 0.99)
print(conv1_weight_max)
# or set weight range by maximum, subject to outliers
# conv1_weights_max = torch.max(torch.abs(conv1_weights))
# conv2_weights_max = torch.max(torch.abs(conv2_weights))
# fc1_weights_max = torch.max(torch.abs(fc1_weights))
# fc2_weights_max = torch.max(torch.abs(fc2_weights))
# fc3_weights_max = torch.max(torch.abs(fc3_weights))

# conv1
s_conv1_weights = conv1_weights_max / 127

s_conv1_bias = s_conv1_weights * s_x

print(f"bias {conv1_weights / s_conv1_weights}\n");
print(f"bias {conv1_bias/s_conv1_bias}\n");

conv1_weights_q = torch.round(conv1_weights / s_conv1_weights).to(torch.int8)
conv1_bias_q = torch.round(conv1_bias / s_conv1_bias).to(torch.int32)

# conv2
s_conv2_weights = conv2_weights_max / 127
s_conv1_a = conv1_rst_max / 127
s_conv2_bias = s_conv2_weights * s_conv1_a

conv2_weights_q = torch.round(conv2_weights / s_conv2_weights).to(torch.int8)
conv2_bias_q = torch.round(conv2_bias / s_conv2_bias).to(torch.int32)

# fc1
s_fc1_weights = fc1_weights_max / 127
s_conv2_a = conv2_rst_max / 127
s_fc1_bias = s_fc1_weights * s_conv2_a

fc1_weights_q = torch.round(fc1_weights / s_fc1_weights).to(torch.int8)
fc1_bias_q = torch.round(fc1_bias / s_fc1_bias).to(torch.int32)

# fc2
s_fc2_weights = fc2_weights_max / 127
s_fc1_a = fc1_rst_max / 127
s_fc2_bias = s_fc2_weights * s_fc1_a

fc2_weights_q = torch.round(fc2_weights / s_fc2_weights).to(torch.int8)
fc2_bias_q = torch.round(fc2_bias / s_fc2_bias).to(torch.int32)

# fc3
s_fc3_weights = fc3_weights_max / 127
s_fc2_a = fc2_rst_max / 127
s_fc3_bias = s_fc3_weights * s_fc2_a

fc3_weights_q = torch.round(fc3_weights / s_fc3_weights).to(torch.int8)
fc3_bias_q = torch.round(fc3_bias / s_fc3_bias).to(torch.int32)

m_conv1 = s_conv1_weights * s_x / s_conv1_a
m_conv2 = s_conv2_weights * s_conv1_a / s_conv2_a
m_fc1 = s_fc1_weights * s_conv2_a / s_fc1_a
m_fc2 = s_fc2_weights * s_fc1_a / s_fc2_a

def tensor2txt(path, t):
    with open(path, 'w') as f:
        # 将每个参数张量的内容转换为numpy数组，并转换为字符串格式
        param_str = " ".join(map(str, t.numpy().flatten()))  # flatten展开张量
        f.write(param_str)
def save_tensor(path, value):
    with open(path, 'w') as f:
        # 将每个参数张量的内容转换为numpy数组，并转换为字符串格式
        value_str = " ".join(map(str, value.numpy().flatten()))  # flatten展开张量
        f.write(value_str)


def multiply(N, M, P):
    print("------------------------------")
    for n in range(1, N):
        result = M * P
        Mo = round(2 ** n * M)
        approx_result = (Mo * P) >> n
        error = result- approx_result
        if (error < 1 and error >-1):
            print("n = %d, Mo = %d, approx = %f, error = %f"%(n, Mo, approx_result, error))
    return n, Mo

print("----------------")
p_conv1 = round(conv1_rst_max / (s_conv1_weights * s_x))
print(torch.is_tensor(conv1_rst_max / (s_conv1_weights * s_x)))  

p_conv2 = round(conv2_rst_max / (s_conv2_weights * s_conv1_a))
p_fc1 = round(fc1_rst_max / (s_fc1_weights * s_conv2_a))
p_fc2 = round(fc2_rst_max / (s_fc2_weights * s_fc1_a))

N = 31
n_conv1, m0_conv1 = multiply(N, m_conv1, p_conv1)
n, m = np.frexp(m_conv1)
print(f"MM: {m_conv1}\tnn: {n}\tmm: {m}\n")
print(2**float(m)*n)
print(2**(-17.0)*129)
print(m_conv1 / (2**float(m)))
n_conv2, m0_conv2 = multiply(N, m_conv2, p_conv2)
n_fc1, m0_fc1 = multiply(N, m_fc1, p_fc1)
n_fc2, m0_fc2 = multiply(N, m_fc2, p_fc2)

save_tensor("./result/quant2/conv1_weights.txt", conv1_weights_q)
save_tensor("./result/quant2/conv2_weights.txt", conv2_weights_q)
save_tensor("./result/quant2/full3_weights.txt", fc1_weights_q)
save_tensor("./result/quant2/full4_weights.txt", fc2_weights_q)
save_tensor("./result/quant2/full5_weights.txt", fc3_weights_q)
save_tensor("./result/quant2/conv1_bias.txt",    conv1_bias_q)
save_tensor("./result/quant2/conv2_bias.txt",    conv2_bias_q)
save_tensor("./result/quant2/full3_bias.txt",    fc1_bias_q)
save_tensor("./result/quant2/full4_bias.txt",    fc2_bias_q)
save_tensor("./result/quant2/full5_bias.txt",    fc3_bias_q)