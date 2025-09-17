import torch

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])

c = torch.dot(a,b)
print(c)