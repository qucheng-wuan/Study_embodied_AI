from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# 加载加州房价数据集
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
data_X = housing.data
data_y = housing.target

# 使用 SimpleImputer 填充缺失值
imputer = SimpleImputer(strategy="mean")  # 使用均值填充
data_X = imputer.fit_transform(data_X)

# 创建并训练模型
model = LinearRegression()
model.fit(data_X, data_y)

# 预测并打印结果
#print(model.predict(data_X[:4, :]))
#print(data_y[:4])

#print(model.coef_) #y=0.1x+0.3
#print(model.intercept_)

#print(model.get_params())  

print(model.score(data_X, data_y))  # R²评分