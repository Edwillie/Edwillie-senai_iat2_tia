import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
np.random.seed(42)

x_perfeito = np.random.normal(loc=[12.0, 15.0], \
                              scale=[0.02, 0.02],\
                                size=(300,2))
x_atencao = np.random.normal(loc=[12.15, 15.10], \
                              scale=[0.05, 0.05],\
                                size=(30,2))
x_critico = np.random.normal(loc=[11.8, 14.8], \
                              scale=[0.1, 0.1],\
                                size=(10,2))
x = np.concatenate([x_perfeito, x_atencao, x_critico])
df = pd.DataFrame(x, columns=['Diâmetro(mm)', 'Peso(g)'])
print("Base feita com ia antes do processo")
print(df.sample(5))
kmeans = KMeans(n_clusters=3, random_state=42, \
                n_init=10)
kmeans.fit(x)
df['Status_IA'] = kmeans.labels_
media_pro_grupo = \
    df.groupby('Status_IA')['Diâmetro(mm)'].mean()
grupo_bom = media_pro_grupo.sub(12.0).abs().idxmin()
grupo_ruim = media_pro_grupo.sub(12.0).abs().idxmax()
todos_grupos = [0 , 1, 2]
todos_grupos.remove(grupo_bom)
todos_grupos.remove(grupo_ruim)
grupo_alerta = todos_grupos[0]
print("Lote bom: ", grupo_bom)
print("Lote alerta: ", grupo_alerta)
print("Lote ruim: ", grupo_ruim)
plt.figure(figsize=(10,6))
plt.scatter(df['Diâmetro(mm)'], df['Peso(g)'],\
            c=df['Status_IA'], cmap='viridis', s=60, \
                alpha=0.7, edgecolors='black')
plt.axvline(x=12.0, color='red', linestyle='--', \
            alpha=0.5, label='Meta Diâmetro (12mm)')
plt.axhline(y=15.0, color='red', linestyle='--', \
            alpha=0.5, label='Meta Peso (15g)')
centroides = kmeans.cluster_centers_
plt.scatter(centroides[:, 0], centroides[:, 1], \
            s=200, c='red', marker='x', label='Centro')
plt.title('Controle de qualidade com AI')
plt.xlabel('Diâmetro(mm)')
plt.ylabel('Peso(g)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show(block=False) # para continuar o código

# Pegamos 20 peças aleatórias para simular a passagem na esteira agora
amostra_esteira = df.sample(20).reset_index(drop=True)

for i, peca in amostra_esteira.iterrows():
    diam = peca['Diâmetro(mm)']  
    peso = peca['Peso(g)']        
    decisao_ia = int(peca['Status_IA'])
    
    print(f"PRODUTO #{i+1:02d} [D:{diam:.2f}mm | P:{peso:.2f}g] -> ", end='')
    
    # LÓGICA DO FLIPPER (ATUADOR MECÂNICO)
    if decisao_ia == grupo_bom:
        print("✅ APROVADO     | AÇÃO: FLIPPER RETO (Siga para Caixa Verde)")
    elif decisao_ia == grupo_alerta:
        print("⚠️ ALERTA       | AÇÃO: FLIPPER 45º (Desvio Suave para Revisão)")
        time.sleep(0.5) # O flipper demora um pouquinho para atuar
    else: # grupo_ruim
        print("❌ SUCATA       | AÇÃO: FLIPPER 90º (Desvio Brusco para Lixo!)")
        time.sleep(1.0) # Mais tempo para garantir que a peça caiu
        
    time.sleep(0.2) # Velocidade da esteira

print("\n--- FIM DO TURNO ---")
input("Pressione Enter para fechar...")