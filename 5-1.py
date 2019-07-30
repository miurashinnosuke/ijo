import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("Cars93.csv")
mask = ["Min.Price", "Price", "Max.Price", "MPG.city", "MPG.highway", "EngineSize", "Horsepower", "RPM", "Rev.per.mile", "Fuel.tank.capacity","Length", "Wheelbase","Width", "Turn.circle", "Weight"]
X = df[mask]
X = (X - X.mean()) / X.std()    
X = X.T                         
Xc = X.values

S = Xc.dot(Xc.T)  

evd, v = np.linalg.eig(S)

plt.plot(range(len(evd)), evd, color="red")
plt.title("Eigen values")
plt.xlabel("Eigen value number")
plt.ylabel("Eigen value")
plt.show()

m = 2                       
x2 = v[:,0:m].T.dot(Xc)     
a1 = (Xc * Xc).sum(axis=0) - (x2 * x2).sum(axis=0)  
print(x2)
result = pd.DataFrame({ "name": df["Make"], "a": a1 })
print(result.sort_values(by='a', ascending=False)[:6])               

plt.scatter(x2[0,:], x2[1,:], color="red")
for i in range(len(x2[0,:])):
        plt.text(x2[0,i], x2[1,i], i+1, ha="center", va="top", size=15)

plt.title("Cars93")
plt.xlabel("Second element")
plt.ylabel("First element")
plt.show()
