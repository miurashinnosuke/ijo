import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

values = np.vstack([weight, height])
kernel = gaussian_kde(values)

a = -kernel.logpdf(values)
plt.plot(range(len(a)), a, "bo")
plt.show()
