import numpy as np

bloco = " ■ "
line += bloco

class HopfieldNetwork:
    def __init__(self, size) -> None:
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        # W = soma(x*x.T)
        print("Treinando a rede com", len(patterns), "padrões")

        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += np.dot(p, p.T)

        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size
        print("Treinamento concluído")