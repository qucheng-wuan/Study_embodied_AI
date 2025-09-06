from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100, n_features=3, n_targets=3, noise=1)
plt.scatter(X,y)
plt.show()
