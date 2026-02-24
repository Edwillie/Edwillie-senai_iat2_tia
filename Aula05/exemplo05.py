import numpy as np
import skfuzzy as fuzz
import requests

x = np.linspace(15, 35, 200)
agradavel = fuzz.trimf(x, [18, 24, 30])

cidade = "Pindamonhangaba"
url = f"http://wttr.in/{cidade}?format=j1"

reposta = requests.get(url)
dados = reposta.json()
temp_real = float(dados["current_condition"][0]["temp_C"])

grau = fuzz.interp_membership(x, agradavel, temp_real)

print("Cidade: ", cidade)
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

