import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import *

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss, train_acc = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    best_acc = 0
    for epoch in range(1, epochs+1):
        train(model, train_dataloader, optimizer, epoch)
        test_acc = test(model, test_dataloader)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './result/32bit_best_model.pth')
        if epoch == epochs:
            torch.save(model.state_dict(), './result/32bit_checkpoint.pth')

        print(f'Epoch: {epoch}/{epochs}\t Best acc: {best_acc:.3f}%')


