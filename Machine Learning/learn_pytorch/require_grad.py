import torch

x=torch.rand(2,3,requires_grad = True)
print(x)

y = x*2 
z = y.mean()

print(z) #输出是个标量 求平均值

z.backward()

print(x.grad)