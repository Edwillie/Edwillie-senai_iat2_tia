import numpy as np
import time

class NeuronioHebbiano:
    def __init__(self):
        # Pesos
        # Peso 0 (Comida)
        # Peso 1 (Campainha)
        self.weights = np.array([1.0, 0.0])
        self.learning_rate = 0.5

    def ativar (self, entradas):
        soma = np.dot(entradas, self.weights)

        return 1 if soma > 0 else 0
    
    def treinar_passo(self, entradas):
        # Novo peso = peso atual + (taxa * Entrada * Saida)
        saida_real = self.ativar(entradas)

        print("Entradas", entradas)
        print("Pesos atuais", self.weights)
        print("O neuronio disparou?")

        if saida_real==1:
            print("Sim")
            delta_w = self.learning_rate * entradas * saida_real
            self.weights += delta_w 
            print("Aprendizado Hebb", delta_w)
            print("Novos Pesos", self.weights)
        else:
            print("Não")
            print("Nenhum novo aprendizado")

brain = NeuronioHebbiano()
# vetores de entrada: [comida, campainha]
# 1 = presente, 0 = ausente

# Situação: Apenas Campainha
SITUACAO_APENAS_CAMPAINHA = np.array([0,1])

print("Testando a campainha antes do treinamento")
resposta = brain.ativar(SITUACAO_APENAS_CAMPAINHA)
if resposta == 1:
    print("Resultado: Salivou e não deveria")
else:
    print("Resultado: Campainha ignorada. Conforme esperado")    

# Situação: Apenas Treino
SITUACAO_TREINO = np.array([1,1])

print("Testando a campainha o treinamento")
resposta = brain.ativar(SITUACAO_TREINO)
if resposta == 1:
    print("Resultado: Campainha compreendida.")
else:
    print("Resultado: Campainha ignorada. Algo de errado ocorreu")
