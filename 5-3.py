import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("Cars93.csv")
mask = ["Min.Price", "Price", "Max.Price", "MPG.city", "MPG.highway", "EngineSize", "Horsepower", "RPM", "Rev.per.mile", "Fuel.tank.capacity","Length", "Wheelbase","Width", "Turn.circle", "Weight"]
X = df[mask]
X = (X - X.mean()) / X.std()
X = X.T
Xc = X.values

G = (Xc.T).dot(Xc)

evd, v = np.linalg.eig(G)

plt.plot(range(len(evd)), evd, color="red")
plt.title("Eigen values")
plt.xlabel("Eigen value number")
plt.ylabel("Eigen value")
plt.show()

m = 3
Lam_12 = np.diag(evd[0:m] ** (-0.5))

xx2 = Lam_12.dot(v[:,0:m].T).dot(Xc.T).dot(Xc)
aa1 = (Xc * Xc).sum(axis=0) - (xx2 * xx2).sum(axis=0)

result = pd.DataFrame({ "name": df["Make"], "a": aa1 })
print(result.sort_values(by='a', ascending=False)[:3])
