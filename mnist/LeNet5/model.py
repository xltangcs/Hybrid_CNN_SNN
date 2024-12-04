from torch.nn import Module
from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                          nn.BatchNorm2d(6),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 16, 120),
                                       nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84),
                                       nn.ReLU())
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
if __name__ == '__main__':
    LeNet = LeNet5()
    print(LeNet)