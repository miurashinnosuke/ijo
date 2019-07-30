import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import sys
weight = pd.read_csv('Davis.csv')['weight'].values

m = weight.mean()

s2 = ((weight - m) ** 2).mean()

a = (weight - m) ** 2 / s2

th = sp.stats.chi2.ppf(0.99,1)

plt.scatter( np.arange(weight.size), a, color = 'g')
plt.plot([0,200], [th,th] , color='b', linestyle='-', linewidth=2)
plt.show()
