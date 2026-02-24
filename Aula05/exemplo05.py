import numpy as np
import skfuzzy as fuzz
import requests

x = np.linspace(0, 100, 500)
agradavel = fuzz.trimf(x, [30, 50, 70])

cidade = input("Informe o nome da cidade: ")
url = f"http://wttr.in/{cidade}?format=j1"

reposta = requests.get(url)
dados = reposta.json()
humidity_real = float(dados["current_condition"][0]["humidity"])

grau = fuzz.interp_membership(x, agradavel, humidity_real)

print("Cidade: ", cidade)
print("Umidade Real: ", humidity_real)
print("Grau: ", round(grau, 2))

if grau == 1.0:
    acao = "Excelente Umidade"
elif grau >= 0.5:
    acao = "Baixa Umidade"
elif grau >= 0.0:
    acao = "Umidade acima do esperado"
else:
    acao = "Umidade fora de padrão"

print("Ação: ", acao) 

