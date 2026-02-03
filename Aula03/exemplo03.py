import numpy as np

class LinearAssociator:
    def __init__(self, input_size, output_size) -> None:
        self.W = np.zeros((output_size, input_size))

    def train(self, inputs, targets):
        print("Treinando o associador com")
        print(len(inputs), "pares")

        for x, y in zip(inputs, targets):
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            self.W += np.dot(y, x.T)

        print("Matriz de memória calculada", self.W)
        print("="*40)

    def predict(self, input_pattern):
        # Fórmula = Y = W*X
        input_pattern = input_pattern.reshape(-1, 1)
        output = np.dot(self.W, input_pattern)
        return output.flatten()
    

# Entradas [Esporte, Tecnologia, Culinária]
CLIENTE_ESPORTISTA = np.array([1, 0, 0])
CLIENTE_NERD = np.array([0, 1, 0])
CLIENTE_CHEF = np.array([0, 0, 1])

# Saidas [Roupas, Eletronicos, Alimenticios]
PRODUTOS_ESPORTE = np.array([1, 0, 0])
PRODUTOS_TECH = np.array([0, 1, 0])
PRODUTOS_COZINHA = np.array([0, 0, 1])

rede = LinearAssociator(input_size=3, output_size=3)
inputs = [CLIENTE_ESPORTISTA, CLIENTE_NERD, CLIENTE_CHEF]
targets = [PRODUTOS_ESPORTE, PRODUTOS_TECH, PRODUTOS_COZINHA]

rede.train(inputs, targets)
print("Teste 1")
print("Entrada", CLIENTE_ESPORTISTA)
resultado = rede.predict(CLIENTE_ESPORTISTA)
print("Saida: ", resultado)
print("Esperado: ", PRODUTOS_ESPORTE)

print("Cliente Hibrido")
CLIENTE_HIBRIDO = CLIENTE_ESPORTISTA + CLIENTE_CHEF

print("Cliente Hibrido: ", CLIENTE_HIBRIDO)
resultado = rede.predict(CLIENTE_HIBRIDO)
print("Saida: ", resultado)

