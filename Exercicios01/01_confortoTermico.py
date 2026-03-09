import numpy as np
import skfuzzy as fuzz

temp_univ = np.linspace(0, 40, 401)
umid_univ = np.linspace(0, 100, 101)

temp_agradavel = fuzz.trimf(temp_univ, [15, 22, 30])
umid_adequada = fuzz.trimf(umid_univ, [30, 50, 70])

cenarios = {
    "A": {"temp": 22, "umid": 50},
    "B": {"temp": 28, "umid": 65},
    "C": {"temp": 35, "umid": 20},
}

print("=== Exercício 1 - Conforto Térmico (AND) ===")
for nome, c in cenarios.items():
    temp_grau = fuzz.interp_membership(temp_univ, temp_agradavel, c["temp"])
    umid_grau = fuzz.interp_membership(umid_univ, umid_adequada, c["umid"])
    conforto = np.fmin(temp_grau, umid_grau)

    print(f"Cenário {nome}: T={c['temp']}°C, U={c['umid']}%")
    print(f"  Grau de 'temperatura agradável': {temp_grau:.3f}")
    print(f"  Grau de 'umidade adequada':     {umid_grau:.3f}")
    print(f"  Grau de conforto (AND):         {conforto:.3f}\n")