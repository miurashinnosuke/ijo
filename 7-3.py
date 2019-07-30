import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1)
tt = 0.1
x1 = np.arange(0,10.1,tt)
x2 = np.arange(10.1,20.1,tt)
x3 = np.arange(20.2,30.1,tt)
pi = np.pi
y1 = np.sin(pi*x1) + np.random.randn(len(x1))
y2 = np.sin(2*pi*x2) + np.random.randn(len(x2))
y3 = np.sin(pi*x3) + np.random.randn(len(x3))
xii = np.hstack((y1,y2))
xi = np.hstack((xii,y3))

w = 10
k = 10
L = 5
m = 2
Tt = 302
print(len(xi))

def create_slide_data(data, w):
        D = []
        T = len(data)
        N = T - w + 1
        for i in range(N):
            D.append(data[i:i+w])
        return D

a_list = []

for t in range(w+k-1, Tt-L):
    begin_at = t-w-k+1
    end_at = t
    X1 = pd.DataFrame(create_slide_data(xi[begin_at:end_at],w)).T
    X1 = X1.iloc[::-1]

    begin_at = t-w-k+1+L
    end_at = t+L
    X2 = pd.DataFrame(create_slide_data(xi[begin_at:end_at],w)).T
    X2 = X2.iloc[::-1]

    U1, s1, V1 = np.linalg.svd(X1, full_matrices=False)
    U1 = U1[:, 0:m]
    U2, s2, V2 = np.linalg.svd(X2, full_matrices=False)
    U2 = U2[:, 0:m]

    U3, s3, V3 = np.linalg.svd(np.dot(U1.T , U2))
    sig1 = s3[0]
    a = 1 - sig1 * sig1
    a_list.append(a)
plt.plot(range(len(a_list)), a_list, linestyle="solid", color="red")
plt.title("Anomaly score")
plt.xlabel("Index")
plt.ylabel("Anomaly score")
plt.show()
