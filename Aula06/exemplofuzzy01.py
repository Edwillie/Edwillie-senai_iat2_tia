import numpy as np
import skfuzzy as fuzz


x_renda = np.arange(0, 11, 1)
x_historico = np.arange(0, 11, 1)

renda_boa = fuzz.trimf(x_renda, [4, 7, 10])
historico_bom = fuzz.trimf(x_historico, [5, 8, 10])

renda_val = fuzz.interp_membership(x_renda, renda_boa, 4)
hist_val = fuzz.interp_membership(x_historico, historico_bom, 5)

aprovacao = np.fmin(renda_val, hist_val)
print("Crédito Aprovado (AND)")
print(round(aprovacao, 2))
print("min (", renda_val,", ",hist_val,")")