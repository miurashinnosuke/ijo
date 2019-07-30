import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from scipy import stats

X1 = np.random.normal(0.0, 1.0, (60,2))
X2 = np.random.normal(3.0, 1.0, (60,2))
X = np.vstack([X1, X2])

X_train = preprocessing.scale(X)

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)

th = stats.scoreatpercentile(y_pred_train, 100 * 0.05) 

xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), th, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, Z, levels=[th], linewidths=2, colors='green')
plt.scatter(X_train[y_pred_train==1][:,0], X_train[y_pred_train==1][:,1], color="red")
plt.scatter(X_train[y_pred_train==-1][:,0], X_train[y_pred_train==-1][:,1], color="blue")
plt.show()
