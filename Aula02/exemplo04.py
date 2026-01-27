from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Desafio: Diagnóstico...")
dados = load_breast_cancer()
x = dados.data
y = dados.target

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.transform(x_teste)

modelo = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, random_state=42)

print("Treinando a Rede Neural")
modelo.fit(x_treino, y_treino)
previsoes = modelo.predict(x_teste)
acuracia = accuracy_score(y_teste, previsoes)
print("Acuracia do Diagnóstico: ", acuracia*100, "%")

erros = (y_teste != previsoes).sum()
print("Erros encontrados: ", erros, "pessoas. ")


print("*"*60)
print("E a Joaninha???")
exames_paciente_joana = [[
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]]

exames_paciente_joana = scaler.transform(exames_paciente_joana)
previsao_joana = modelo.predict(exames_paciente_joana)
if previsao_joana[0] == 0:
    resultado = "Alerta: Fazer mais exames"
else:
    resultado = "Sucesso: Nada Detectado" 

print("Resultado do exame da paciente - ", previsao_joana[0] )
print("Diagnostico da IA: ", resultado)

print("*"*60)
print("E a paciente novo???")
exames_paciente_novo = [[
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]]

exames_paciente_novo = scaler.transform(exames_paciente_novo)
previsao_outro = modelo.predict(exames_paciente_novo)

if previsao_outro[0] == 0:
    resultado = "Alerta: Fazer mais exames"
else:
    resultado = "Sucesso: Nada Detectado" 

print("Resultado do exame da paciente - ", previsao_outro[0] )
print("Diagnostico da IA: ", resultado)