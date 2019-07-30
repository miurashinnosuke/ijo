import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("Davis.csv")
mu = df.weight.mean()
s2 = ((df.weight-mu) ** 2).mean() 

a = ((df.weight-mu)**2)/s2
th = stats.chi2.interval(0.99,1)[1]

plt.scatter(np.arange(df.weight.size), a, color = 'b')
plt.plot([0,200], [th,th],'k-', color = "r", linewidth=2)
plt.show()
