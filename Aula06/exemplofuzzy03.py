import numpy as np
import skfuzzy as fuzz

x_fumaca = np.arange(0, 101, 1)
x_temp = np.arange(20, 81, 1)

fumaca_alta = fuzz.trapmf(x_fumaca, [30, 60, 100, 100])
temp_alta = fuzz.trapmf(x_temp, [40, 55, 80, 80])

fumaca_val1 = fuzz.interp_membership(x_fumaca, fumaca_alta, 10)
temp_val1 = fuzz.interp_membership(x_temp, temp_alta, 65)

alarme = np.fmax(fumaca_val1, temp_val1)
print("Alarme (OR)")
print(round(alarme, 2))
print("max (", fumaca_val1,", ",temp_val1,")")