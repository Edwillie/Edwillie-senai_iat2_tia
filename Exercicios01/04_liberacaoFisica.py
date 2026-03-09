import numpy as np
import skfuzzy as fuzz

cond_univ = np.linspace(0, 10, 101)
lesao_univ = np.linspace(0, 10, 101)

cond_bom = fuzz.trimf(cond_univ, [4, 7, 10])
lesao_grau = fuzz.trimf(lesao_univ, [0, 4, 8])

alunos = {
    1: {"cond": 8, "lesao": 1},
    2: {"cond": 6, "lesao": 7},
    3: {"cond": 9, "lesao": 4},
}

print("=== Exercício 4 - Liberação Atividade Física (AND + NOT) ===")
for aid, a in alunos.items():
    cond_g = fuzz.interp_membership(cond_univ, cond_bom, a["cond"])
    lesao_g = fuzz.interp_membership(lesao_univ, lesao_grau, a["lesao"])

    not_lesao = 1 - lesao_g
    liberacao = np.fmin(cond_g, not_lesao)

    print(f"Aluno {aid}: Cond={a['cond']}, Lesão={a['lesao']}")
    print(f"  Grau de 'condicionamento bom': {cond_g:.3f}")
    print(f"  Grau de 'lesionado':           {lesao_g:.3f}")
    print(f"  Grau de 'NÃO lesionado':       {not_lesao:.3f}")
    print(f"  Grau de liberação (AND):       {liberacao:.3f}\n")