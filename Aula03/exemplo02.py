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
                line += " . "
        
        print(line)
    print()

print("="*50)
ROWS, COLS = 5, 5
N_NEURONS = ROWS * COLS

P1 = np.array([
     1, -1, -1, -1,  1,
    -1,  1, -1,  1, -1,
    -1, -1,  1, -1, -1,
    -1,  1, -1,  1, -1,
     1, -1, -1, -1,  1,
])

print_grid(P1, ROWS, COLS, "Padrão X")

P2 = np.array([
    -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1,
     1,  1,  1,  1,  1,
    -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1,
])

print_grid(P2, ROWS, COLS, "Padrão +")

hopfield = HopfieldNetwork(size=N_NEURONS)
hopfield.train([P1, P2])

print("="*50)
noisy_input = P1.copy()
noisy_input[0]  = -1
noisy_input[12] = -1
noisy_input[1] = 1

print_grid(noisy_input, ROWS, COLS, "Danificado")

restored_pattern = hopfield.predict(noisy_input)

print_grid(restored_pattern, ROWS, COLS, "Recuperado")

noisy_input[0] = -1
noisy_input[1] = -1
noisy_input[2] = 1
noisy_input[4] = -1
noisy_input[6] = -1
noisy_input[7] = 1
noisy_input[8] = -1
noisy_input[10] = 1
noisy_input[11] = 1
noisy_input[13] = 1

print_grid(noisy_input, ROWS, COLS, "Danificado")

restored_pattern = hopfield.predict(noisy_input)

print_grid(restored_pattern, ROWS, COLS, "Recuperado")
