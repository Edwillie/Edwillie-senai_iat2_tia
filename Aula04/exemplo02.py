import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# buscar base de dados
np.random.seed(42)

# loc=[x, y] x= Peso | y= Diametro
# scale=[i, j] i = valor de X pode variar i +/- | y = valor de Y pode variar j +/-
# size = (w, t) w = linhas | t = colunas

#Grupo 1- Boas
x1 = np.random.normal(loc=[12, 15], scale=[0.02, 0.02], size=(300, 2))

#Grupo 2- Atenção
x2 = np.random.normal(loc=[12.15, 15.10], scale=[0.05, 0.05], size=(30, 2))

#Grupo 3- Lixo
x3 = np.random.normal(loc=[11.8, 14.8], scale=[0.1, 0.1], size=(10, 2))

x = np.concatenate([x1, x2, x3])

df = pd.DataFrame(x, columns=['Peso', 'Diametro'])

lkmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
lkmeans.fit(x) #Treinando o modelo

df['Cluster'] = lkmeans.labels_ #Resultados

plt.figure(figsize=(10, 6))
cores = ['red', 'green', 'blue']
nomes = ['Sucata', 'Gordinhas', 'Perfeitas']

for i in range(3):
    grupo = df[df['Cluster'] == i]
    plt.scatter(grupo['Peso'], grupo['Diametro'], s=100, c=cores[i], label=f'Grupo {nomes[i]}', alpha=0.6, edgecolors='w')

plt.xlabel("Diâmetro")
plt.ylabel("Peso")
plt.axvline(x=12.0, color='red', linestyle='--', alpha=0.5, label='Meta Diâmetro 12mm')
plt.axhline(y=15.0, color='red', linestyle='--', alpha=0.5, label='Meta Peso 15g')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()