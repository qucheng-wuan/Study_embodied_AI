import torch

a = torch.tensor([1,2,3])
b = torch.tensor([[2],
                 [3],
                 [4]])

result = torch.matmul(a,b)
print(result)