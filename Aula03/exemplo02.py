import numpy as np

bloco = " ■ "

class HopfieldNetwork:
    def __init__(self, size) -> None:
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        # W = soma(x*x.T)
        print("Treinando a rede com", len(patterns), "padrões")

        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += np.dot(p, p.T) # O .T é a transposição do array numpy

        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size
        print("Treinamento concluído")

    def predict(self, input_pattern, max_iter=100, early_stopping=True):
        current_state = input_pattern.copy()
        for i in range(max_iter):
            activation = np.dot(self.weights, current_state)
            new_state = np.sign(activation)
            new_state[new_state == 0] = 1

            if early_stopping and np.array_equal(current_state, new_state):
                print("Convergencia Atingida", i+1)
                return new_state
            
            current_state = new_state

        return current_state
    
def print_grid(pattern, rows, cols, title="Padrão"):
    print("---", title, "---")
    grid = pattern.reshape(rows, cols)
    for row in grid:
        line = ""

        for val in row:
            if val == 1:
                line += bloco
            else:
                line += "."
        
        print(line)
    print()