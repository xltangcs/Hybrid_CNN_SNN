import torch
import os
import numpy as np

def save_state_dict(state_dict, folder):
    # 如果文件夹不存在则创建文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 遍历state_dict中的每一个参数
    for name, param in state_dict.items():
        # 为每个参数生成文件路径
        param_path = os.path.join(folder, f"{name}.txt")
        
        # 将参数保存为文本文件
        with open(param_path, 'w') as f:
            # 将每个参数张量的内容转换为numpy数组，并转换为字符串格式
            param_str = " ".join(map(str, param.cpu().numpy().flatten()))  # flatten展开张量
            f.write(param_str)
        
        print(f"Saved parameter {name} to {param_path}")

if __name__ == '__main__':
    float_model_path = "./result/32bit_best_model.pth"
    model_state_dict = torch.load(float_model_path)
    save_state_dict(model_state_dict, "./result/32bit")