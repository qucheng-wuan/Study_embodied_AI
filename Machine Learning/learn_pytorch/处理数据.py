import torch
from torchvision import datasets,transforms

#定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor
    transforms.Normalize((0.5,),(0.5,)) #标准化
])

#加载MINIST数据集
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader =torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)

