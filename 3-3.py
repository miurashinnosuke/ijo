import numpy as np
from scipy.stats import norm
import math

n0 = norm.rvs(3.0, 0.5, size=600)
n1 = norm.rvs(0, 3.0, size=400)
n = np.concatenate([n0, n1])
np.random.shuffle(n)
N = 1000
pi0 = 0.5       
pi1 = 0.5
mu0 = 5.0
mu1 = -5.0
sig0 = 1.0
sig1 = 5.0

ite = range(10)

for i in ite:
    piN0 = norm.pdf(x=n, loc=mu0, scale=sig0)
    piN1 = norm.pdf(x=n, loc=mu1, scale=sig1)
    qn0 = piN0 / (piN0 + piN1)
    qn1 = piN1 / (piN0 + piN1)

    pi0 = qn0.sum() / N
    pi1 = qn1.sum() / N
    mu0 = (qn0 * n).sum() / (N * pi0)
    mu1 = (qn1 * n).sum() / (N * pi1)
    sig0 = math.sqrt((qn0 * (n - mu0) * (n - mu0)).sum() / (N * pi0))
    sig1 = math.sqrt((qn1 * (n - mu1) * (n - mu1)).sum() / (N * pi1))

print(pi0, mu0, sig0)
print(pi1, mu1, sig1)
