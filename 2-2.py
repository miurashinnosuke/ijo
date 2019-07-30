import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy import linalg as la


df = pd.read_csv('Davis.csv').values

x = df[: ,2:4]
mx = x.mean(axis = 0)

xc = x - mx
xc.shape

sx = (xc.T.dot(xc) / x[:,0].size).astype(float)

ap = np.dot(xc, np.linalg.inv(sx)) * xc
a = ap[:,0] + ap[:,1]

th = stats.chi2.ppf(0.99,2)

plt.scatter(np.arange(a.size), a, color='b')
plt.plot([0,200],[th,th],'k-',color='r',linewidth=2)
plt.show()
