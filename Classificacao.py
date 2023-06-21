import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power

# Configurações gerais
numTotalPontos = 200  # Número total de pontos de dados
numtPontosRotulados = 10  # Número de pontos de dados rotulados
fatorSuavizacao = 0.99  # Fator de suavização
sigmaCalculoDistancia = 0.1  # Parâmetro sigma para o cálculo da distância

# Geração dos dados
matrizX, matrizY = make_moons(numTotalPontos, shuffle=True, noise=0.1, random_state=None)

# Preparação dos rótulos de entrada
rotolosEntrada = np.concatenate(((matrizY[:numtPontosRotulados,None] == np.arange(2)).astype(float), np.zeros((numTotalPontos-numtPontosRotulados,2))))

# Cálculo das distâncias euclidianas
distanciaEuclidiana = cdist(matrizX, matrizX, 'euclidean')

# Função RBF (Função de Base Radial)
baseRadial = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
vfunc = np.vectorize(baseRadial)
W = vfunc(distanciaEuclidiana, sigmaCalculoDistancia)
np.fill_diagonal(W, 0) # Zerando os elementos diagonais da matriz de similaridade

# Cálculo da soma das linhas da matriz de similaridade
somaLinhasMatrizSimilaridade = np.sum(W,axis=1)

# Construção da matriz diagonal D
matrizDiagonal = np.diag(somaLinhasMatrizSimilaridade)

# Cálculo da matriz D^(-0.5)
matrizDiagonal = fractional_matrix_power(matrizDiagonal, -0.5)

# Construção da matriz de similaridade normalizada S
matrizSimilaridadeNormalizada = np.dot(np.dot(matrizDiagonal,W), matrizDiagonal)

# Configurações para iterações
numIteracoes = 400

# Propagação dos rótulos
rotulosPropagados = np.dot(matrizSimilaridadeNormalizada, rotolosEntrada)*fatorSuavizacao + (1-fatorSuavizacao)*rotolosEntrada
for t in range(numIteracoes):
    rotulosPropagados = np.dot(matrizSimilaridadeNormalizada, rotulosPropagados)*fatorSuavizacao + (1-fatorSuavizacao)*rotolosEntrada

# Atribuição dos rótulos resultantes
rotulosResultantes = np.zeros_like(rotulosPropagados)
rotulosResultantes[np.arange(len(rotulosPropagados)), rotulosPropagados.argmax(1)] = 1

# Preparação para visualização
listaRotulosEmBinarios = [1 if x == 0 else 0 for x in rotulosResultantes[0:,0]] # Converter os rótulos para uma lista de valores binários (0 ou 1)
color = ['red' if l == 0 else 'blue' for l in listaRotulosEmBinarios] # Atribuir cores com base nos rótulos

print(f"\n")
print(f'Sucesso!')

# Imprimir dados originais
plt.scatter(matrizX[0:,0], matrizX[0:,1])
plt.show()

# Imprimir dados com cores dos rótulos
plt.scatter(matrizX[0:,0], matrizX[0:,1], color=color)
plt.show()