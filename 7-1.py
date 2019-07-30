import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("qtdbsel102.txt", delimiter="\t")
X = df.ix[:,1]
X = X.values

w = 100
nk = 1
T = 3000

Xtr = X[:T]
X = X[T+1:2*T+1]

def create_slide_data(data, w):
        D = []
        T = len(data)
        N = T - w + 1
        for i in range(N):
            D.append(data[i:i+w])
        return D

def dist(pt1, pt2):
    return np.sqrt(((pt2 - pt1) ** 2).sum())

def score(dist_list, k):
    dist_list.sort()
    return sum(dist_list[:k]) / k

Dtr = create_slide_data(Xtr, w)
D = create_slide_data(X, w)

a_list = []
for x in D:
    dist_list = []
    for xtr in Dtr:
        dist_list.append(dist(x, xtr))
    a = score(dist_list, nk)
    a_list.append(a)
                                                             
plt.plot(range(len(a_list)), a_list, linestyle="solid", color="red")
plt.title("Anomaly score")
plt.xlabel("Index")
plt.ylabel("Anomaly score")
plt.show()
