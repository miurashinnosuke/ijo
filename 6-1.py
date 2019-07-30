import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv("UScrime.csv")
mask = ["M", "Ed", "Po1", "Po2", "LF", "M.F", "Pop", "NW", "U1", "U2","GDP", "Ineq","Prob", "Time"]
X = df[mask]
y = df["y"]
N = len(y)

clf = linear_model.RidgeCV()
clf.fit(X,y)

coefs = clf.coef_
npcoefs = np.array(coefs)
lam = clf.alpha_
ypred = clf.predict(X)

sig2 = (lam* (npcoefs**2).sum() +sum(y-ypred)**2) /N

X = X - X.mean() 
X = X.T
Xc = X.values
M = len(Xc)
H = (Xc.T).dot(np.linalg.solve(Xc.dot(Xc.T)+lam*(np.identity(M)),Xc))
TrHN = np.trace(H) / N
a = (ypred - y)**2 / ((1-TrHN)**2*70000)
sorted_a = sorted(a,reverse = True)
th = sorted_a[(int(N * 0.05 - 1))]

plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.plot(range(len(a)), a, "bo")
plt.axhline(y=th,color='red')
plt.show()
