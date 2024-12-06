import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.quantization import get_default_qconfig, quantize_qat, QConfig
from torch.ao.quantization.quantize import quantize, prepare_qat, convert
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver
from models import *

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss, train_acc = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch} Loss: {train_loss/len(train_loader)}')

def test(model, test_loader):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_acc += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * test_acc / len(test_loader.dataset)
    print(f'Current model acc is {accuracy:0.2f}')
    return accuracy

if __name__ == '__main__':
    # 数据转化为tensor格式
    data_transform = transforms.Compose([transforms.ToTensor()])

    # 加载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # 加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet5_quant().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    float_model_path = './result/32bit_best_model.pth'
    if float_model_path:
        model.load_state_dict(torch.load(float_model_path, map_location=device))

    epochs = 5
    for epoch in range(1, epochs+1):
        train(model, train_dataloader, optimizer, epoch)

    model.eval()
    test_acc = test(model, test_dataloader)
    print(f'Float model acc is {test_acc:0.2f}')

    model.train()

    # 量化
    model.qconfig = get_default_qconfig('fbgemm') 
    # 定义自定义的量化配置
    custom_qconfig = QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                        quant_min=0,
                                        quant_max=255,  # 4-bit 量化的范围
                                        dtype=torch.quint8),  # 注意: 目前 PyTorch 不直接支持 4-bit 的 dtype, 这里仅作为示例
        weight=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                    quant_min=-128,
                                    quant_max=127,
                                    dtype=torch.qint8)
    )
    model.qconfig = custom_qconfig

    model = torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                                    ['conv2', 'relu2'],
                                                    ['full3', 'relu3'],
                                                    ['full4', 'relu4']])

    model = torch.quantization.prepare_qat(model, inplace=True)

    epochs = 10
    for epoch in range(1, epochs+1):
        train(model, train_dataloader, optimizer, epoch)

    model.eval()
    model_8bit = torch.quantization.convert(model.to('cpu'))

    test_8bit_acc = test(model_8bit, test_dataloader)
    print(f'8bit model acc is {test_8bit_acc:0.2f}')
    torch.save(model_8bit.state_dict(), './result/8bit_best_model.pth')