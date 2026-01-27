from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Desafio: Diagnóstico...")
dados = load_breast_cancer()
x = dados.data
y = dados.target

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

print("Treinando com ", len(x_treino), "pacientes. ")
modelo = Perceptron(random_state=42)
modelo.fit(x_treino, y_treino)
previsoes = modelo.predict(x_teste)
acuracia = accuracy_score(y_teste, previsoes)
print("Acuracia do Diagnóstico: ", acuracia*100, "%")

print("*"*60)
print("E a Joaninha???")
exames_paciente_joana = [[
    0.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    1.2699, 0.7886, 2.058, 23.56, 1.008462, 0.0146, 1.02387, 0.01315, 1.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 300.2977, 0.07259
]]

previsao_joana = modelo.predict(exames_paciente_joana)
if previsao_joana[0] == 0:
    resultado = "Alerta: Fazer mais exames"
else:
    resultado = "Sucesso: Nada Detectado" 

print("Resultado do exame da paciente - ", previsao_joana[0] )
print("Diagnostico da IA: ", resultado)