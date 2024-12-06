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
        self.layer3 = nn.Sequential(nn.Linear(4 * 4 * 16, 120),
                                       nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(120, 84),
                                       nn.ReLU())
        self.layer5 = nn.Linear(84, 10)
        self.flat = nn.Flatten(2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flat(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
if __name__ == '__main__':
    LeNet = LeNet5()
    print(LeNet)