import numpy as np
import skfuzzy as fuzz

dor_univ = np.linspace(0, 10, 101)
press_univ = np.linspace(80, 180, 101)

dor_intensa = fuzz.trimf(dor_univ, [5, 8, 10])
press_elevada = fuzz.trimf(press_univ, [130, 155, 180])

pacientes = {
    1: {"dor": 3, "press": 160},
    2: {"dor": 9, "press": 120},
    3: {"dor": 7, "press": 150},
}

print("=== Exercício 2 - Triagem Hospitalar (OR) ===")
for pid, p in pacientes.items():
    dor_grau = fuzz.interp_membership(dor_univ, dor_intensa, p["dor"])
    press_grau = fuzz.interp_membership(press_univ, press_elevada, p["press"])
    prioridade = np.fmax(dor_grau, press_grau)

    print(f"Paciente {pid}: Dor={p['dor']}, Pressão={p['press']} mmHg")
    print(f"  Grau de 'dor intensa':        {dor_grau:.3f}")
    print(f"  Grau de 'pressão elevada':    {press_grau:.3f}")
    print(f"  Grau de prioridade (OR):      {prioridade:.3f}\n")