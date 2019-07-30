import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

score = np.array([0.19,0.86,0.17,0.12,0.04,0.78,0.16,0.51,0.57,0.27])
anomaly =np.array([0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0])

data0 = np.vstack((score,anomaly))
data1 = data0[:, np.argsort(data0[0])[::-1]]
score_sorted = data1[0]
anomaly_sorted = data1[1]
n_total = len(anomaly)
n_norm = sum(anomaly)
n_anom = n_total - n_norm
coverage = [0] * n_total
detection = [0] * n_total

for i in range (n_total):
    n_detectedAnom = sum(anomaly_sorted[0:i+1])
    n_detectedNorm = (n_total-i-1) - sum(anomaly_sorted[i+1:10])
    coverage[i] = n_detectedAnom / n_norm
    detection[i] = n_detectedNorm / n_anom

plt.plot(score_sorted,coverage,marker = "o",linestyle="dashed")
plt.plot(score_sorted,detection,marker = "o")
plt.show()
