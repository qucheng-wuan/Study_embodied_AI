import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

#定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor
    transforms.Normalize((0.5,),(0.5,)) #标准化
])

#加载MINIST数据集
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader =torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)

#这个就是先加载数据 然后构建一个数据加载器 然后再把数据下载下来

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

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

for epoch in range(10): #训练10个epoch
   for batch_idx,(data,target) in enumerate(train_loader):
         optimizer.zero_grad()  #梯度清零
         output = model(data)   #前向传播
         loss = criterion(output,target) #计算损失
         loss.backward()       #反向传播
         optimizer.step()      #更新参数

         if batch_idx % 100 ==0:
             print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

