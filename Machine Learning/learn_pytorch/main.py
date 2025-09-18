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

    Z3 = np.dot(A3,W3)+b3
    exp_Z3 = np.exp(Z3-np.max(Z3,axis=1,keepdim=True))
    A3 = exp_Z3 / np.sum(exp_z3,axis=1,keepdims=True)
    #手动实现softmax 稳定版本 避免溢出
    return Z1,A1,Z2,A2,Z3,A3  

def compute_cross_entropy(A3,y_onehot):
    #计算交叉熵
    m =A3.shape[0] #A3.shape是[m,3]
    loss = - np.mean(y_onehot*np.log(A3+1e-10,axis=1))
    return loss

def backward(X,y_oneshot,Z1,A1,Z2,A2,Z3,A3,W1,W2,W3)
    





