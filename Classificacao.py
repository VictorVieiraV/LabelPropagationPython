import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power

n = 20
n_labeled = 10
alpha = 0.99
sigma = 0.1

matrizX, matrizY = make_moons(n, shuffle=True, noise=0.1, random_state=None)

Y_input = np.concatenate(((matrizY[:n_labeled,None] == np.arange(2)).astype(float), np.zeros((n-n_labeled,2))))

dm = cdist(matrizX, matrizX, 'euclidean')
rbf = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
vfunc = np.vectorize(rbf)
W = vfunc(dm, sigma)
np.fill_diagonal(W, 0)

sum_lines = np.sum(W,axis=1)
D = np.diag(sum_lines)

D = fractional_matrix_power(D, -0.5)
S = np.dot(np.dot(D,W), D)

n_iter = 400

F = np.dot(S, Y_input)*alpha + (1-alpha)*Y_input
for t in range(n_iter):
    F = np.dot(S, F)*alpha + (1-alpha)*Y_input

Y_result = np.zeros_like(F)
Y_result[np.arange(len(F)), F.argmax(1)] = 1

Y_v = [1 if x == 0 else 0 for x in Y_result[0:,0]]

color = ['red' if l == 0 else 'blue' for l in Y_v]

print(f"\n")
print(f"\n")
print(f'Sucesso!')

plt.scatter(matrizX[0:,0], matrizX[0:,1])
plt.show()

plt.scatter(matrizX[0:,0], matrizX[0:,1], color=color)
plt.show()
