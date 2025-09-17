import torch

# 创建一个2行3列的张量（共6个元素）
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("原始张量x的形状：", x.shape)  # 输出：torch.Size([2, 3])
print("原始张量x：")
print(x)

# 用view(3, 2)重塑为3行2列
y = x.view(3, 2)
print("\n重塑后的张量y的形状：", y.shape)  # 输出：torch.Size([3, 2])
print("重塑后的张量y：")
print(y)

# 验证数据是否共享（修改x会影响y，因为view不复制数据）
x[0, 0] = 100
print("\n修改x后，y的值也会变：")
print(y)
