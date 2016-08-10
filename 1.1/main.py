import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

N = 10
M = 4

X = np.linspace(0, 1, N, endpoint=True)
S = np.sin(2*np.pi*X) + np.random.rand(N)
z = np.linspace(0, 1, N*10, endpoint=True)


a = np.zeros([M,M])
b = np.zeros(M)

for i in range(0, M):
    for j in range(0, M):
        a[i,j] = math.fsum(map(lambda x: x**(i+j), X))
    b[i] = sum([m*n for (m, n) in zip(map(lambda x: x**i, X), S)])

w = solve(a, b)

plt.plot(z, np.polyval(w[::-1], z))
plt.plot(X, S, 'o')
plt.show()
