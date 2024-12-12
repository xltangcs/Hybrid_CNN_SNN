import torch
import os
import time
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from models import *

conv1_weights = np.zeros((6, 1, 5, 5))
conv2_weights = np.zeros((16, 6, 5, 5))
full3_weights = np.zeros((120, 4*4*16))
full4_weights = np.zeros((84, 120))
full5_weights = np.zeros((10, 84))
conv1_bias    = np.zeros((6))
conv2_bias    = np.zeros((16))
full3_bias    = np.zeros((120))
full4_bias    = np.zeros((84))
full5_bias    = np.zeros((10))

conv1_n = 15	
conv1_Mo = 33
conv2_n = 16	
conv2_Mo = 93
full3_n = 15	
full3_Mo = 65
full4_n = 15	
full4_Mo = 105
      
# conv1_n  = 17
# conv2_n  = 16
# full3_n  = 15
# full4_n  = 14

# conv1_Mo = 129
# conv2_Mo = 69
# full3_Mo = 31
# full4_Mo = 39

def read_tensor(file, num):
    # 读取文件中的数据
    with open(file, 'r') as f:
        data = f.read()
    data = list(map(float, data.split()))
    
    if len(data) != np.prod(num.shape):
        raise ValueError(f"Data size {len(data)} does not match tensor size {np.prod(tensor.shape)}.")

    num[:] = np.array(data, dtype=int).reshape(num.shape)

def conv(input, weights, bias):
    weights_tensor = torch.tensor(weights, dtype=torch.float32) 
    bias_tensor = torch.tensor(bias, dtype=torch.float32)
    output_tensor = F.conv2d(input, weights_tensor, bias_tensor, stride=1, padding=0)
    output_tensor = F.relu(output_tensor)
    return output_tensor

def full(input, weights, bias, relu = True):
    weights_tensor = torch.tensor(weights, dtype=torch.float32)  
    bias_tensor = torch.tensor(bias, dtype=torch.float32)
    output_tensor = F.linear(input, weights_tensor, bias_tensor)
    if relu:
        output_tensor = F.relu(output_tensor)
    return output_tensor

def maxpool(input):
    output_tensor = F.max_pool2d(input.float(), kernel_size=2, stride=2)
    return output_tensor

def test(input):
    input = input * 255

    convrelu1_output = conv(input, conv1_weights, conv1_bias)
    convrelu1_output = (convrelu1_output * conv1_Mo).to(torch.int32)>> int(conv1_n)
    maxpool1_output = maxpool(convrelu1_output)

    convrelu2_output = conv(maxpool1_output, conv2_weights, conv2_bias)
    convrelu2_output = (convrelu2_output * conv2_Mo).to(torch.int32)>> int(conv2_n)
    maxpool2_output = maxpool(convrelu2_output)

    maxpool2_output = maxpool2_output.view(-1).float()

    fullrelu3_output = full(maxpool2_output, full3_weights, full3_bias)
    fullrelu3_output = (fullrelu3_output * full3_Mo).to(torch.int32)>> int(full3_n)

    fullrelu4_output = full(fullrelu3_output.float(), full4_weights, full4_bias)
    fullrelu4_output = (fullrelu4_output * full4_Mo).to(torch.int32)>> int(full4_n)

    full5_output = full(fullrelu4_output.float(), full5_weights, full5_bias, relu = False)

    # print(full5_output)

    return full5_output

read_tensor("./result/8bit/conv1_weights.txt", conv1_weights)
read_tensor("./result/8bit/conv2_weights.txt", conv2_weights)
read_tensor("./result/8bit/full3_weights.txt", full3_weights)
read_tensor("./result/8bit/full4_weights.txt", full4_weights)
read_tensor("./result/8bit/full5_weights.txt", full5_weights)
read_tensor("./result/8bit/conv1_bias.txt",    conv1_bias   )
read_tensor("./result/8bit/conv2_bias.txt",    conv2_bias   )
read_tensor("./result/8bit/full3_bias.txt",    full3_bias   )
read_tensor("./result/8bit/full4_bias.txt",    full4_bias   )
read_tensor("./result/8bit/full5_bias.txt",    full5_bias   )

# 加载测试数据集
data_transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True)


# data_iter = iter(test_dataloader)
# data, labels = next(data_iter)
# y = test(data)
# print(f"pred = {torch.argmax(y)}\t labels = {labels}")


acc = 0
st = time.time()
for data, target in test_dataloader:
    y = test(data)
    pred = torch.argmax(y)
    acc += (pred == int(target))

    print(f"pred = {pred}\t labels = {int(target)}")
et = time.time()
print(et - st)
print("After quantization, Accuracy = {0:.2f}%".format(acc/len(test_dataloader)*100))

    