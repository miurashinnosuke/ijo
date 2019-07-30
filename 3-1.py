import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv("Davis.csv")
mu = df.weight.mean()
si = np.std(df.weight)*(len(df.weight) - 1)/len(df.weight)

kmo = mu ** 2 /si ** 2
smo = si ** 2 / mu

fit_alpha, fit_loc, fit_beta = stats.gamma.fit(np.array(df.weight), floc=0)

a = df.weight / smo - (kmo - 1) * np.log(df.weight / smo)    
sorted_a = sorted(a,reverse = True)
th = sorted_a[(int(len(df.weight) * 0.01 - 1))]  

plt.plot(range(len(a)), a, "bo")
plt.axhline(y=th,color='red')
plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.show()
