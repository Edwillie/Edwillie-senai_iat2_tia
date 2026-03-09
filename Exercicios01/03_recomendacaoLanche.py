import numpy as np
import skfuzzy as fuzz

sac_univ = np.linspace(0, 10, 101)
saciedade = fuzz.trimf(sac_univ, [0, 5, 10])

usuarios = {
    "A": 2,
    "B": 5,
    "C": 8,
}

print("=== Exercício 3 - Recomendação de Lanche (NOT) ===")
for nome, s in usuarios.items():
    sac_grau = fuzz.interp_membership(sac_univ, saciedade, s)

    recomendacao = 1 - sac_grau

    print(f"Usuário {nome}: Saciedade={s}")
    print(f"  Grau de 'saciado':                 {sac_grau:.3f}")
    print(f"  Grau de recomendação de lanche:    {recomendacao:.3f}\n")