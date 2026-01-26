import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

dados_treino = [
    [[2, 3], 0],
    [[3, 4], 0],
    [[4, 5], 0],
    [[5, 4], 1],
    [[6, 7], 1],
    [[7, 8], 1],
    [[8, 7], 1],
    [[9, 9], 1],
    [[3, 3], 0],
    [[7, 6], 1],
]

x = np.array([entrada for entrada, _ in dados_treino])
y = np.array([resultado for _, resultado in dados_treino])

print("Numero de alunos (linhas): ", x.shape[0])
print("Numero de colunas ", x.shape[1])
print("Numero (Y): ", y.shape[0], "resultados")

print("Criando e treinando....")
print("*"*60)
modelo_aprovacao = Perceptron(max_iter=1000, random_state=42)
modelo_aprovacao.fit(x, y)
print("Treinamento Concluido!")

print()
print("Pesos aprendidos ", modelo_aprovacao.coef_[0])
print("Vies aprendido ", modelo_aprovacao.intercept_[0])

print("Testando o modelo")
predicoes = modelo_aprovacao.predict(x)

for i, (entrada, resultado_esperado, predicao) in enumerate(zip(x, y, predicoes), 1):
    acertou = "ok" if predicao == 1 else "reprovado"
    status = "Aprovado" if predicao == 1 else "Reprovado"
    print("Aluno", i)
    print("Estudo: ", entrada[0], "h")
    print("Trabalho:", entrada[1])
    print("Predição:", status, predicao)
    print("Esperado", resultado_esperado, acertou)
    print("-"*60)

acuracia = accuracy_score(y, predicoes)
print("Acuracia: ", acuracia*len(y)/len(y))
print("Em porcentagem: ", 100*acuracia, "%")

print("Testando com novos alunos....")
novos_alunos = [
    [5,5],
    [8,8],
    [2,2],
    [6,6],
    [3,9],
    [9,3],
    [8,2],
]

predicoes_novos = modelo_aprovacao.predict(novos_alunos)
for i, (entrada, predicao) in enumerate(zip(novos_alunos, predicoes_novos), 1):
    status = "Aprovado" if predicao == 1 else "Reprovado"
    print("Novo Aluno", i)
    print("Estudo: ", entrada[0], "h")
    print("Trabalho:", entrada[1])
    print("Status:", status)
    print("-"*60)
