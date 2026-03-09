import numpy as np
import skfuzzy as fuzz

fluxo_univ = np.linspace(0, 100, 101)
vel_univ = np.linspace(0, 120, 121)

fluxo_alto = fuzz.trimf(flow_univ := fluxo_univ, [50, 75, 100])
velocidade_boa = fuzz.trimf(vel_univ, [40, 80, 120])

situacoes = {
    "A": {"fluxo": 80, "vel": 30},
    "B": {"fluxo": 40, "vel": 90},
    "C": {"fluxo": 70, "vel": 50},
}

print("=== Exercício 5 - Alerta de Trânsito (OR + NOT) ===")
for nome, s in situacoes.items():
    fluxo_g = fuzz.interp_membership(fluxo_univ, fluxo_alto, s["fluxo"])
    vel_boa_g = fuzz.interp_membership(vel_univ, velocidade_boa, s["vel"])

    not_vel_boa = 1 - vel_boa_g

    alerta = np.fmax(fluxo_g, not_vel_boa)

    print(f"Situação {nome}: Fluxo={s['fluxo']}, Velocidade={s['vel']} km/h")
    print(f"  Grau de 'fluxo alto':          {fluxo_g:.3f}")
    print(f"  Grau de 'velocidade boa':      {vel_boa_g:.3f}")
    print(f"  Grau de 'NÃO velocidade boa':  {not_vel_boa:.3f}")
    print(f"  Grau de alerta (OR):           {alerta:.3f}\n")