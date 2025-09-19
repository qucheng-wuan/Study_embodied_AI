import numpy as np
import matplotlib.pyplot as plt


#配置四层感知机参数
INPUT_DIM = 4
HIDDEN1 = 16 #隐藏层1单位数
HIDDEN2 = 8 #隐藏层2单位数
NUM_CLASSES = 3

LEARNING_RATE = 0.01
EPOCHS = 300
BATCH_SIZE = None  #none表示全梯度下降

#处理数据集
data = np.genfromtxt("/Users/quchengzou/编程/pytorch/iris/iris.data",delimiter = ",", dtype = object,encoding = "utf-8")
#过滤空行
data = [row for row in data
               if row is not None or len(row)== 5 ]

X = np.array([row[:4] 
               for row in data], dtype=float)   # 特征矩阵 (150, 4)
y_str = np.array([row[4] 
                   for row in data])   
#字符串到整型编码
label_map = {"Iris-setosa": 0,"Iris-versicolor":1, "Iris-virginica":2}
y_int = np.array([label_map[label.decode('utf-8')] 
                  if isinstance(label, bytes)
                    else label_map[label] 
                    for label in y_str], dtype=int)

#整形编码到字符串
num_classes= 3 #三类花--长度为3的独向量
y_onehot = np.zeros((len(y_int),num_classes),dtype=float)
y_onehot[np.arange(len(y_int)),y_int] = 1

#  特征标准化
def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean)/std, (X_val - mean)/std, (X_test - mean)/std 

#标签映射
y_str = np.array([
    row[4].decode('utf-8') if isinstance(row[4], bytes) else row[4] 
    for row in data
])
y = np.array([label_map[label] for label in y_str],dtype = int )

#打乱 然后分类
np.random.seed(42)
datasets = np.arange(len(X))
np.random.shuffle(datasets)
#这个要先索引数字 然后再shuttle 要不然特征和数据对不上

# 按打乱的索引重新排列特征和One-Hot标签
X = X[datasets]
y_onehot= y_onehot[datasets]

n = len(X)
n_train = int(0.8*n)
n_val = int(0.1*n)

#切片分数据
X_train, y_train_onehot = X[:n_train],y_onehot[:n_train]
X_val , y_val_onehot= X[n_train:n_train+n_val],y_onehot[n_train:n_train+n_val]
X_test,y_test_onehot = X[n_train+n_val:],y_onehot[n_train+n_val:]

X_train, X_val, X_test = standardize(X_train, X_val, X_test)

#print("训练集大小:", X_train.shape[0])
#print("验证集大小:", X_val.shape[0])
#print("测试集大小:", X_test.shape[0])

#模型参数初始化
np.random.seed(42)
W1 = np.random.randn(INPUT_DIM,HIDDEN1) #4,16
b1 = np.zeros((1,HIDDEN1))              #1,16
W2 = np.random.randn(HIDDEN1,HIDDEN2)   #16,8
b2 = np.zeros((1,HIDDEN2))              #1,8
W3 = np.random.randn(HIDDEN2,NUM_CLASSES)#8,4
b3 = np.zeros((1,NUM_CLASSES))          #1,4

#前向传播（numpy） RELU函数激活隐藏层 输出层softmax
def forward(X,W1,b1,W2,b2,W3,b3):
    #前向传播：X--A3
    #返回中间变量，用于反向传播 Z1(A1),Z1(A2),Z1(A3)
    Z1 = np.dot(X,W1)+b1    #(m,4)*(4,16)=(m,16)
    A1 = np.maximum(0,Z1)

    Z2 = np.dot(A1,W2)+b2
    A2 = np.maximum(0,Z2) #RELU函数激活

    Z3 = np.dot(A2,W3)+b3
    exp_Z3 = np.exp(Z3-np.max(Z3,axis=1,keepdims=True))
    A3 = exp_Z3 / np.sum(exp_Z3,axis=1,keepdims=True)
    #手动实现softmax 稳定版本 避免溢出
    return Z1,A1,Z2,A2,Z3,A3  

def compute_cross_entropy(A3,y_onehot):
    #计算交叉熵
    m =A3.shape[0] #A3.shape是[m,3]
    loss = - np.mean(np.sum(y_onehot*np.log(A3+1e-10),axis=1))
    return loss

def backward(X, y_onehot, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = X.shape[0]  # 样本数
    #输出层
    dZ3 = A3 - y_onehot
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis = 0,keepdims = True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 *(Z2>0)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis = 0, keepdims =True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 *(Z1>0)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1,axis = 0,keepdims = True)

    return dW1,db1,dW2,db2,dW3,db3

def update_parameters(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3, lr):
    #梯度下降--参数更新
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * db3
    b3 -= lr * db3
    return W1, b1, W2, b2, W3, b3

def compute_accuracy(A3,y_onehot):
    #将A3转换为输出类别，与真实的onehot(独热标签转索引）对比计算正确率
    y_pred = np.argmax(A3,axis = 1)
    y_real = np.argmax(y_onehot,axis = 1) #(m,)
    accuracy = np.mean(y_pred == y_real )
    return accuracy

# 初始化训练记录
train_loss_history = []    # 训练集损失
val_loss_history = []      # 验证集损失
train_acc_history = []     # 训练集准确率
val_acc_history = []       # 验证集准确率

# 复制初始参数（用于后续第一个样本梯度计算）
W1_init = W1.copy()
b1_init = b1.copy()
W2_init = W2.copy()
b2_init = b2.copy()
W3_init = W3.copy()
b3_init = b3.copy()

# 训练循环
for epoch in range(EPOCHS):
    # -------------------------- 训练集前向传播 --------------------------
    Z1_train, A1_train, Z2_train, A2_train, Z3_train, A3_train = forward(
        X_train, W1, b1, W2, b2, W3, b3
    )
    # 计算训练集损失和准确率
    train_loss = compute_cross_entropy(A3_train, y_train_onehot)
    train_acc = compute_accuracy(A3_train, y_train_onehot)
    
    # -------------------------- 反向传播+参数更新 --------------------------
    dW1, db1, dW2, db2, dW3, db3 = backward(
        X_train, y_train_onehot, Z1_train, A1_train, Z2_train, A2_train, Z3_train, A3_train, W1, W2, W3
    )
    # 更新参数
    W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, LEARNING_RATE)
    
    # -------------------------- 验证集评估 --------------------------
    Z1_val, A1_val, Z2_val, A2_val, Z3_val, A3_val = forward(
        X_val, W1, b1, W2, b2, W3, b3
    )
    # 计算验证集损失和准确率
    val_loss = compute_cross_entropy(A3_val, y_val_onehot)
    val_acc = compute_accuracy(A3_val, y_val_onehot)
    
    # -------------------------- 记录结果 --------------------------
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    
    # 每20轮打印一次进度
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")





    





