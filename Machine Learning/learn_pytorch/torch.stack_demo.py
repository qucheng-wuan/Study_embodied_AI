import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

c = torch.tensor([[13, 14, 15],
                  [16, 17, 18]])

print(a.shape)
#torch.stack会在新维度上面提升
stack_0 = torch.stack((a,b,c),dim=0)
print(stack_0.shape)
print(stack_0)