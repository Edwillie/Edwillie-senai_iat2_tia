import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(15, 35, 200)
print(x)
agradavel = fuzz.trimf(x, [18, 24, 30])

temp_real = float(input("Digite a temperatura: "))
grau = fuzz.interp_membership(x, agradavel, temp_real)

print("Temperatura Real: ", temp_real)
print("Grau: ", round(grau, 2))

if grau == 1.0:
    acao = "Perfeito, Ar em espera"
elif grau >= 0.5:
    acao = "Razoavel, Diminuir potencia do ar"
elif grau >= 0.0:
    acao = "Quase fora. Aumentar potencia do ar"
else:
    acao = "Fora do conforto! Ligar o ar no máximo!"

print("Ação: ", acao) 

