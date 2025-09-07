import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3) #(in_channels,out_channels,kernel_size)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3) #输出尺寸 = (输入尺寸 - 卷积核尺寸 + 2 × padding) ÷ 步长 + 1 
        self.fc1 = nn.Linear(64*12*12,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1,64*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x,dim=1) #在多分类中使用log_softmax更稳定
    
#初始化模型
model = SimpleCNN()
