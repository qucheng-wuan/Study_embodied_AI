import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(SimpleModel,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )

    def forward(self,x):
            return self.layers(x)
    
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 超参数
input_size = 784
hidden_size = 256
output_size = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

model = SimpleModel(input_size,hidden_size,output_size).to(device)

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

train_data = torch.randn(1000,input_size)
train_labels = torch.randint(0,output_size,(1000,))

total_steps = len(train_data)// batch_size
for epoch in range(num_epochs):
    for i in range(total_steps):
        #获取当前数据
        batch_x = train_data[i*batch_size:(i+1)*batch_size].to(device)
        batch_y = train_labels[i*batch_size:(i+1)*batch_size].to(device)

        #前向传播
        outputs = model(batch_x)
        loss = criterion(outputs,batch_y)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 ==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')




