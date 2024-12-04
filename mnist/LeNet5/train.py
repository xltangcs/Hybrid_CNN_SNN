import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
from data import MNIST
from model import LeNet5


# DATASET
train_data = MNIST(
	root='../MNIST_data',
	train=True, 
	transform=ToTensor()
)

test_data = MNIST(
	root='../MNIST_data',
	train=False,
	transform=ToTensor()
)

# PREPROCESS
batch_size = 256
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)
print(len(train_data))

# MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)

# TRAIN MODEL
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())

def train(dataloader, model, loss_func, optimizer, epoch):
	model.train()
	data_size = len(dataloader.dataset)
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
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f'Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}')

if __name__ == '__main__':
	epoches = 100
	for epoch in range(epoches):
		train(train_dataloader, model, loss_func, optimizer, epoch)
		test(test_dataloader, model, loss_func)

	# Save models
	torch.save(model.state_dict(), '../Result/model.pth')
	print('Saved PyTorch LeNet5 State to model.pth')
