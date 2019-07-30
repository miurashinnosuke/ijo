import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("qtdbsel102.txt", delimiter="\t")
X = df.ix[:,1]
X = X.values

w = 50
k = w / 2
L = k / 2
m = 2
Tt = 3000

xi = X[Tt+1:2*Tt+1]

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
