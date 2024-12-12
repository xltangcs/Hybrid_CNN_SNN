import torch
import os
import numpy as np
from torchvision import datasets, transforms
import datetime
from models import *


relu_activations = {}

def cal_activation_max():
    relu_max = []
    for layer_name, activations in relu_activations.items():
        # 将所有样本的激活值拼接成一个大的张量
        activations = torch.cat(activations, dim=0)  # 拼接所有样本的激活值
        activations_numpy = activations.cpu().numpy()
        activations_numpy_99 = np.quantile(activations_numpy, 0.999)
        # 计算最小值和最大值
        min_val = activations.min().item()
        max_val = activations.max().item()
        # 计算99%百分位数
        # percentile_99 = torch.quantile(activations, 0.99).item()
        # 打印每个ReLU层的统计值
        relu_max.append(activations_numpy_99)
        print(f"{layer_name} - Min: {min_val:.4f}, Max: {max_val:.4f}, 99:{activations_numpy_99}")
    return relu_max

def get_activation(name):
    def hook(model, input, output):
        if name not in relu_activations:
            relu_activations[name] = []
        # min_val = output.min()
        # max_val = torch.quantile(output, 0.99)

        relu_activations[name].append(output)   
    return hook

def save_tensor(path, value):
    with open(path, 'w') as f:
        # 将每个参数张量的内容转换为numpy数组，并转换为字符串格式
        value_str = " ".join(map(str, value.numpy().flatten()))  # flatten展开张量
        f.write(value_str)

#cal M0 n ---> MP --> (Mo * P) >> n
def multiply(name, N, M, P, rewrite = False): 
    path = "./result/8bit/n_m0.txt"
    if rewrite:
        with open(path, "w") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S\n")
            f.write(f"{current_time}\n")
    # print(f"-----{name}----------")
    for n in range(1, N):
        result = M * P
        Mo = round(2 ** n * M)
        approx_result = (Mo * P) >> n
        error = result- approx_result
        if (error < 0.01 and error >-0.01):
            with open(path, "a") as f:
                n_m0_str = "{0}_n = {1}\t{0}_Mo = {2}\tapprox = {3}, error = {4}\n".format(name, n, Mo, approx_result, error)
                f.write(n_m0_str)
            return
            # print("n = %d, Mo = %d, approx = %f, error = %f"%(n, Mo, approx_result, error))

    return
if __name__ == '__main__':

    model = LeNet5()
    float_model_path = "./result/32bit_best_model.pth"
    model_state_dict = torch.load(float_model_path)
    model.load_state_dict(model_state_dict)

    model.relu1.register_forward_hook(get_activation("relu1"))
    model.relu2.register_forward_hook(get_activation("relu2"))
    model.relu3.register_forward_hook(get_activation("relu3"))
    model.relu4.register_forward_hook(get_activation("relu4"))

    # 加载测试数据集
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    # 执行前向传播，保存所有样本的激活值
    model.eval()  # 设置模型为评估模式
    for data, target in test_dataloader:
        with torch.no_grad():  # 不计算梯度
            _ = model(data)  # 前向传播，只用来收集激活值

    conv1_weights = model_state_dict['conv1.weight']
    conv2_weights = model_state_dict['conv2.weight']
    full3_weights = model_state_dict['full3.weight']
    full4_weights = model_state_dict['full4.weight']
    full5_weights = model_state_dict['full5.weight']
    conv1_bias    = model_state_dict['conv1.bias']
    conv2_bias    = model_state_dict['conv2.bias']
    full3_bias    = model_state_dict['full3.bias']
    full4_bias    = model_state_dict['full4.bias']
    full5_bias    = model_state_dict['full5.bias']

    # conv1_weights_max = torch.quantile(conv1_weights, 0.9999).item()
    # conv2_weights_max = torch.quantile(conv2_weights, 0.9999).item()
    # full3_weights_max = torch.quantile(full3_weights, 0.9999).item()
    # full4_weights_max = torch.quantile(full4_weights, 0.9999).item()
    # full5_weights_max = torch.quantile(full5_weights, 0.9999).item()

    conv1_weights_max = torch.max(conv1_weights).item()
    conv2_weights_max = torch.max(conv2_weights).item()
    full3_weights_max = torch.max(full3_weights).item()
    full4_weights_max = torch.max(full4_weights).item()
    full5_weights_max = torch.max(full5_weights).item()
    relu1_max, relu2_max, relu3_max, relu4_max = cal_activation_max()
    # print(relu1_max)

    s_x = 1/255
    # conv1
    scale_conv1_weights = conv1_weights_max / 127
    scale_conv1_bias = scale_conv1_weights * s_x

    q_conv1_weights = torch.round(conv1_weights / scale_conv1_weights).to(torch.int8)
    q_conv1_bias    = torch.round(conv1_bias    / scale_conv1_bias).to(torch.int8)

    #conv2
    scale_conv2_weights = conv2_weights_max / 127
    active_conv1 = relu1_max / 127
    scale_conv2_bias = scale_conv2_weights * relu1_max / 127

    q_conv2_weights = torch.round(conv2_weights / scale_conv2_weights).to(torch.int8)
    q_conv2_bias    = torch.round(conv2_bias    / scale_conv2_bias).to(torch.int8)

    #full3
    scale_full3_weights = full3_weights_max / 127
    active_conv2 = relu2_max / 127
    scale_full3_bias = scale_full3_weights * relu2_max / 127

    q_full3_weights = torch.round(full3_weights / scale_full3_weights).to(torch.int8)
    q_full3_bias    = torch.round(full3_bias    / scale_full3_bias).to(torch.int8)

    #full4
    scale_full4_weights = full4_weights_max / 127
    active_full3 = relu3_max / 127
    scale_full4_bias = scale_full4_weights * relu3_max / 127

    q_full4_weights = torch.round(full4_weights / scale_full4_weights).to(torch.int8)
    q_full4_bias    = torch.round(full4_bias    / scale_full4_bias).to(torch.int8)

    #full5
    scale_full5_weights = full5_weights_max / 127
    active_full4 = relu4_max / 127
    scale_full5_bias = scale_full5_weights * relu4_max / 127

    q_full5_weights = torch.round(full5_weights / scale_full5_weights).to(torch.int8)
    q_full5_bias    = torch.round(full5_bias    / scale_full5_bias).to(torch.int8)

    save_tensor("./result/8bit/conv1_weights.txt", q_conv1_weights)
    save_tensor("./result/8bit/conv2_weights.txt", q_conv2_weights)
    save_tensor("./result/8bit/full3_weights.txt", q_full3_weights)
    save_tensor("./result/8bit/full4_weights.txt", q_full4_weights)
    save_tensor("./result/8bit/full5_weights.txt", q_full5_weights)
    save_tensor("./result/8bit/conv1_bias.txt",    q_conv1_bias   )
    save_tensor("./result/8bit/conv2_bias.txt",    q_conv2_bias   )
    save_tensor("./result/8bit/full3_bias.txt",    q_full3_bias   )
    save_tensor("./result/8bit/full4_bias.txt",    q_full4_bias   )
    save_tensor("./result/8bit/full5_bias.txt",    q_full5_bias   )

    m_conv1 = scale_conv1_weights * s_x          / active_conv1 
    m_conv2 = scale_conv2_weights * active_conv1 / active_conv2  
    m_full3 = scale_full3_weights * active_conv2 / active_full3  
    m_full4 = scale_full4_weights * active_full3 / active_full4

    p_conv1 = round(relu1_max / (scale_conv1_weights * s_x         ))
    p_conv2 = round(relu2_max / (scale_conv2_weights * active_conv1))
    p_full3 = round(relu3_max / (scale_full3_weights * active_conv2))
    p_full4 = round(relu4_max / (scale_full4_weights * active_full3))

    N = 31
    multiply("conv1", N, m_conv1, p_conv1, rewrite = True)
    multiply("conv2", N, m_conv2, p_conv2)
    multiply("full3", N, m_full3, p_full3)
    multiply("full4", N, m_full4, p_full4)
