from sklearn import svm
from sklearn import datasets

clf=svm.SVC()
iris = datasets.load_iris()
X,y = iris.data, iris.target
clf.fit(X,y)

#method:pickle
#import pickle
#with open("save/clf.pickle","wb") as f:
#    pickle.dump(clf, f )  #保存model
#with open("save/clf.pickle","rb") as f:
#    clf2= pickle.load(f)
#    print(clf2.predict(X[0:1])) 
     #预测

     #method 2:joblib
import joblib
  #save
joblib.dump(clf,'save/clf2.pkl')
  #restore
clf3=joblib.load('save/clf2.pkl')
print(clf3.predict(X[0:1]))