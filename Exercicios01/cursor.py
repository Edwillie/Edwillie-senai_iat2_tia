import numpy as np
import skfuzzy as fuzz


def exercicio_1():
    """
    Conforto térmico (AND entre temperatura agradável e umidade adequada)
    """
    # Universos
    temp_univ = np.linspace(0, 40, 401)
    umid_univ = np.linspace(0, 100, 101)

    # Funções de pertinência (triangulares)
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


def exercicio_2():
    """
    Triagem hospitalar (OR entre dor intensa e pressão elevada)
    """
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


def exercicio_3():
    """
    Recomendação de lanche (NOT do nível de saciedade)
    """
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
        # Quanto MENOR a saciedade, MAIOR a recomendação: NOT(saciedade)
        recomendacao = 1 - sac_grau

        print(f"Usuário {nome}: Saciedade={s}")
        print(f"  Grau de 'saciado':                 {sac_grau:.3f}")
        print(f"  Grau de recomendação de lanche:    {recomendacao:.3f}\n")


def exercicio_4():
    """
    Liberação de atividade física (AND entre condicionamento bom e NOT(lesão))
    """
    cond_univ = np.linspace(0, 10, 101)
    lesao_univ = np.linspace(0, 10, 101)

    cond_bom = fuzz.trimf(cond_univ, [4, 7, 10])
    lesao_grau = fuzz.trimf(lesao_univ, [0, 4, 8])  # 'lesionado'

    alunos = {
        1: {"cond": 8, "lesao": 1},
        2: {"cond": 6, "lesao": 7},
        3: {"cond": 9, "lesao": 4},
    }

    print("=== Exercício 4 - Liberação Atividade Física (AND + NOT) ===")
    for aid, a in alunos.items():
        cond_g = fuzz.interp_membership(cond_univ, cond_bom, a["cond"])
        lesao_g = fuzz.interp_membership(lesao_univ, lesao_grau, a["lesao"])
        # NOT(lesão)
        not_lesao = 1 - lesao_g
        liberacao = np.fmin(cond_g, not_lesao)

        print(f"Aluno {aid}: Cond={a['cond']}, Lesão={a['lesao']}")
        print(f"  Grau de 'condicionamento bom': {cond_g:.3f}")
        print(f"  Grau de 'lesionado':           {lesao_g:.3f}")
        print(f"  Grau de 'NÃO lesionado':       {not_lesao:.3f}")
        print(f"  Grau de liberação (AND):       {liberacao:.3f}\n")


def exercicio_5():
    """
    Alerta de trânsito (OR entre fluxo alto e NOT(velocidade boa))
    """
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

        # NOT(velocidade boa) => velocidade ruim / não boa
        not_vel_boa = 1 - vel_boa_g

        alerta = np.fmax(fluxo_g, not_vel_boa)

        print(f"Situação {nome}: Fluxo={s['fluxo']}, Velocidade={s['vel']} km/h")
        print(f"  Grau de 'fluxo alto':          {fluxo_g:.3f}")
        print(f"  Grau de 'velocidade boa':      {vel_boa_g:.3f}")
        print(f"  Grau de 'NÃO velocidade boa':  {not_vel_boa:.3f}")
        print(f"  Grau de alerta (OR):           {alerta:.3f}\n")


def exercicio_6():
    """
    Irrigação automática:
    Regra: ligar se CHOVENDO OR (SOLO SECO AND NOT(VENTO FORTE))
    """
    chuva_univ = np.linspace(0, 10, 101)
    solo_univ = np.linspace(0, 10, 101)
    vento_univ = np.linspace(0, 10, 101)

    # Assumindo formas iguais para chuva e solo seco, conforme enunciado
    chuva = fuzz.trimf(chuva_univ, [0, 3, 6])       # 'chovendo'
    solo_seco = fuzz.trimf(solo_univ, [0, 3, 6])    # 'solo seco'
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

        # 1. NOT(vento forte)
        not_vento = 1 - vento_g

        # 2. AND(solo seco, NOT(vento))
        solo_and_not_vento = np.fmin(solo_g, not_vento)

        # 3. OR(chovendo, AND)
        sprinkler = np.fmax(chuva_g, solo_and_not_vento)

        print(f"Cenário {cid}: Chuva={c['chuva']}, Solo={c['solo']}, Vento={c['vento']}")
        print(f"  Grau de 'chovendo':                 {chuva_g:.3f}")
        print(f"  Grau de 'solo seco':               {solo_g:.3f}")
        print(f"  Grau de 'vento forte':             {vento_g:.3f}")
        print(f"  Grau de 'NÃO vento forte':         {not_vento:.3f}")
        print(f"  Grau de (solo seco AND NOT vento): {solo_and_not_vento:.3f}")
        print(f"  Grau de acionamento (OR):          {sprinkler:.3f}\n")


if __name__ == "__main__":
    exercicio_1()
    exercicio_2()
    exercicio_3()
    exercicio_4()
    exercicio_5()
    exercicio_6()