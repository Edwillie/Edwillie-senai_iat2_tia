import numpy as np
import skfuzzy as fuzz

chuva_univ = np.linspace(0, 10, 101)
solo_univ = np.linspace(0, 10, 101)
vento_univ = np.linspace(0, 10, 101)

chuva = fuzz.trimf(chuva_univ, [0, 3, 6])
solo_seco = fuzz.trimf(solo_univ, [0, 3, 6])
vento_forte = fuzz.trimf(vento_univ, [5, 7, 10])

cenarios = {
    1: {"chuva": 7, "solo": 2, "vento": 3},
    2: {"chuva": 1, "solo": 8, "vento": 2},
    3: {"chuva": 2, "solo": 3, "vento": 8},
}

print("=== Exercício 6 - Irrigação Automática (AND + OR + NOT) ===")
for cid, c in cenarios.items():
    chuva_g = fuzz.interp_membership(chuva_univ, chuva, c["chuva"])
    solo_g = fuzz.interp_membership(solo_univ, solo_seco, c["solo"])
    vento_g = fuzz.interp_membership(vento_univ, vento_forte, c["vento"])

    not_vento = 1 - vento_g

    solo_and_not_vento = np.fmin(solo_g, not_vento)

    sprinkler = np.fmax(chuva_g, solo_and_not_vento)

    print(f"Cenário {cid}: Chuva={c['chuva']}, Solo={c['solo']}, Vento={c['vento']}")
    print(f"  Grau de 'chovendo':                 {chuva_g:.3f}")
    print(f"  Grau de 'solo seco':               {solo_g:.3f}")
    print(f"  Grau de 'vento forte':             {vento_g:.3f}")
    print(f"  Grau de 'NÃO vento forte':         {not_vento:.3f}")
    print(f"  Grau de (solo seco AND NOT vento): {solo_and_not_vento:.3f}")
    print(f"  Grau de acionamento (OR):          {sprinkler:.3f}\n")