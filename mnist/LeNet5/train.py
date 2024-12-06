import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
from data import MNIST
from model import LeNet5



# 数据转化为tensor格式
data_transform = transforms.Compose([transforms.ToTensor()])

# 加载训练数据集
train_dataset = datasets.MNIST(root='../MNIST_data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='../MNIST_data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)

# MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)

# TRAIN MODEL
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())

def train(dataloader, model, loss_func, optimizer, epoch):
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		y_hat = model(X)
		loss = loss_func(y_hat, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss, current = loss.item(), batch * len(X)
	print(f'epoch {epoch+1}: \tloss: {loss:>7f}', end='\t')


# Test model
def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, test_acc = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	test_acc /= size
	print(f'Accuracy: {(100 * test_acc):>0.1f}%, Average loss: {test_loss:>8f}')
	return test_acc, test_loss

if __name__ == '__main__':
	epoches = 50
	best_acc = 0

	for epoch in range(epoches):
		train(train_dataloader, model, loss_func, optimizer, epoch)
		test_acc, test_loss = test(test_dataloader, model, loss_func)

		if test_acc >= best_acc:
			torch.save(model.state_dict(), '../Result/best_model.pth')
		if epoch == epoches -1:
			torch.save(model.state_dict(), '../Result/last_model.pth')
