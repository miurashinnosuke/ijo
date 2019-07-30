import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('road.csv',index_col=0)

X = df.drop("drivers", axis=1).values
drivers = df.as_matrix(columns=["drivers"])

X = X / drivers
X = np.log(X + 1)
mx = X.mean(axis=0)
Xc = X-mx
Sx = np.cov(X,rowvar=0,bias=1)

xc_prime = Xc[4,:]  
SN1 = 10 * np.log10(xc_prime**2 / np.diag(Sx))

plt.bar(range(len(SN1)), SN1, tick_label=["deaths","popden","rural","temp","fuel"], align="center")
plt.title("SN ratio")
plt.show()
