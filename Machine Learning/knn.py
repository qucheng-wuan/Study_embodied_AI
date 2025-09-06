import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#print(iris_X[:2,:] )
#print(iris_y)

X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)
#print(y_test)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.predict(X_test))
print(y_test)

#我理解了 也就是knn.fit这一步就是将数据train了 然后knn.predict这一步就是用test数据来预测
#对应过来和y_test对比 看看预测的准不准
