import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GMM
from sklearn.mixture import BayesianGaussianMixture

df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

X = pd.concat([weight, height], axis=1)
X_train = X.drop(11)
X = X.values
X_train = X_train.values

gmm = BayesianGaussianMixture(n_components=2, verbose=1, max_iter=10000)
gmm.fit(X_train)
X_pred = gmm.predict(X_train)

clusters_set = set(X_pred)
print("clusters_set:", clusters_set)
print("len(clusters_set):", len(clusters_set))

cluster0 = X_train[X_pred == 0]
cluster1 = X_train[X_pred == 1]
plt.scatter(cluster0[:,0], cluster0[:,1], color="red")
plt.scatter(cluster1[:,0], cluster1[:,1], color="blue")
plt.show()
