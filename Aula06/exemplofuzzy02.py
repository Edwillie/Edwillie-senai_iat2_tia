alpha = 0.4

def and_fuzzy_com_alpha(a, b, alpha):
    resultado = min(a, b)
    return resultado if resultado >= alpha else 0.0

def and_fuzzy_alpha_individual(a, b, alpha):
    a_cortado = a if a>= alpha else 0.0
    b_cortado = b if b>= alpha else 0.0
    return min(a_cortado, b_cortado)


#cliente1
aprovacao1 = and_fuzzy_com_alpha(0.67, 0.50, alpha)
print("Cliente 1", aprovacao1)

#cliente2
aprovacao2 = and_fuzzy_com_alpha(0.67, 0.0, alpha)
print("Cliente 2", aprovacao2)

#cliente3
aprovacao3 = and_fuzzy_alpha_individual(0.67, 0.50, alpha)
print("Cliente 3", aprovacao3)

#cliente4
aprovacao4 = and_fuzzy_alpha_individual(0.67, 0.0, alpha)
print("Cliente 4", aprovacao4)