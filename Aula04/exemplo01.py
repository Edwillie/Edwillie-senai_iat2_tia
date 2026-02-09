import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# buscar base de dados
np.random.seed(42)

# loc=[x,y] x= Renda | y= Nota do clientes
# scale=[i, j] i = valor de X pode variar i +/- | y = valor de Y pode variar j +/-
# size = (w, t) w = linhas | t = colunas

#Grupo 1- Economicos
x1 = np.random.normal(loc=[30,20], scale=[10, 10], size=(50, 2))

#Grupo 2- Medianos
x2 = np.random.normal(loc=[60,50], scale=[15, 15], size=(100, 2))

#Grupo 3- Ricos
x3 = np.random.normal(loc=[100,80], scale=[15, 10], size=(50, 2))

x = np.concatenate([x1, x2, x3])

df = pd.DataFrame(x, columns=['Renda Anual (K)', 'Score de gastos (0-100)'])

lkmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
lkmeans.fit(x) #Treinando o modelo

df['Cluster'] = lkmeans.labels_ #Resultados

plt.figure(figsize=(10, 6))
cores = ['red', 'green', 'blue']

for i in range(3):
    grupo = df[df['Cluster'] == i]
    plt.scatter(grupo['Renda Anual (K)'], grupo['Score de gastos (0-100)'], s=100, c=cores[i], label=f'Grupo {i}', alpha=0.6, edgecolors='w')

centroide = lkmeans.cluster_centers_
plt.scatter(centroide[:, 0], centroide[:, 1], s=300, c="black", marker="X", label='Cliente Tipico')
plt.title("Segmentação de Mercado com IA")
plt.xlabel("Renda Anual (R$)")
plt.ylabel("Score de gastos")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()