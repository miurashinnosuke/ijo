import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GMM

df = pd.read_csv("Davis.csv")
weight = df["weight"]
height = df["height"]

X = pd.concat([weight, height], axis=1)
X_train = X.drop(11)
X = X.values
X_train = X_train.values

clf = GMM(n_components=2)
clf.fit(X_train)
X_pred = clf.predict(X_train)

anomaly_scores = -clf.score_samples(X)[0]

plt.title("Anomaly score")
plt.xlabel("Sample number")
plt.ylabel("Anomaly score")
plt.plot(range(len(anomaly_scores)), anomaly_scores, "bo")
plt.show()
