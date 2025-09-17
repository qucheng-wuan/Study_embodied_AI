import torch
#print(torch.backends.mps.is_built())
#device_maps = torch.device("mps"). #显式创建device对象

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print({device})

class Model(torch.nn.Module):
 #一般模型都继承torch.nn.Module. 这个是所有神经网络的基类 所有都要继承她
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10,2)
        #线性层的数学表达式为：y = x·W + b，其中W是权重矩阵（形状为 2×10），b是偏置向量（形状为 2）

    def forward(self,x):
        return self.linear(x)
    
#创造模型实例
model = Model()
#迁移模型
model = model.to(device)

#加载数据
data = torch.randn(5,10).to(device)

output = model(data)

print("输出后的张量：",{output.device})


