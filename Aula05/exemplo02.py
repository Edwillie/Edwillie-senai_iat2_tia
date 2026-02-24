import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 160, 300)
adequada = fuzz.trapmf(x, [60, 80, 100, 120])

cenarios = {
    "A": {"velocidade": 90, "desc":"Velocidade adequada"},
    "B": {"velocidade": 30, "desc":"Acelera ae!!!!"},
    "C": {"velocidade": 170, "desc":"Pqp pisa no freio zé"}        
}

print("="*50)
print("Funcção Trapezio")
print("="*50)

for nome, info in cenarios.items():
    grau = fuzz.interp_membership(x, adequada, info["velocidade"])
    print("\nCenario", nome, "(", info["velocidade"], "km/h): ")
    print("miA = ", round(grau, 1))
    print("->", info["desc"])


#5. Plotagem
fig, ax = plt.subplots(figsize=(10, 5))

# Curva principal
ax.plot(x, adequada, 'b', linewidth=2.5, label='Adequada [60, 80, 100, 120]')
ax.fill_between(x, adequada, alpha=0.1, color='blue')

# Marcando o platô
ax.axvspan(80, 100, alpha=0.08, color='green', label='Platô (totalmente adequada)')

# Marcando cada cenário no gráfico
cores   = {"A": "green",  "B": "orange",  "C": "red"}
markers = {"A": "o",      "B": "s",       "C": "X"}

for nome, info in cenarios.items():
    grau = fuzz.interp_membership(x, adequada, info["velocidade"])
    ax.vlines(info["velocidade"], 0, grau, colors=cores[nome], linestyles="--", linewidth=1.5)
    ax.hlines(grau, 15, info["velocidade"], colors=cores[nome], linestyles="--", linewidth=1.5)
    ax.plot(info["velocidade"], grau,
            marker=markers[nome], color=cores[nome], markersize=12,
            label="Cenário {}: {}°C → μ = {:.1f}".format(nome, info["velocidade"], grau),
            zorder=5)

# Anotando os parâmetros a, b, c
for val, label in zip([60, 80, 100, 120], ['a = 60', 'b = 80', 'c = 100', 'd = 120']):
    ax.axvline(val, color='gray', linestyle=':', linewidth=1)
    ax.text(val, 1.02, "{} = {}".format(label[0], val), ha='center', fontsize=9, color='gray')

ax.set_title('Função de Pertinência Fuzzy — Velocidade Adequada', fontsize=13)
ax.set_xlabel('Velocidade (Km/h)')
ax.set_ylabel('Grau de Pertinência  μ(x)')
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(0, 160)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()
