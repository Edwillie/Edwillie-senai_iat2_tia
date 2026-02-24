import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(15, 35, 200)
agradavel = fuzz.trimf(x, [18, 24, 30])

cenarios = {
    "A": {"temp": 24, "desc":"Agradavel: Ar em espera!"},
    "B": {"temp": 21, "desc":"Meio-Agradavel - Diminuir o Ar"},
    "C": {"temp": 32, "desc":"Fora - Ligar Ar Maximo"}        
}

print("="*50)
print("Funcção Triangular")
print("="*50)

for nome, info in cenarios.items():
    grau = fuzz.interp_membership(x, agradavel, info["temp"])
    print("\nCenario", nome, "(", info["temp"], "C): ")
    print("miA = ", round(grau, 1))
    print("->", info["desc"])


#5. Plotagem
fig, ax = plt.subplots(figsize=(10, 5))

# Curva principal
ax.plot(x, agradavel, 'b', linewidth=2.5, label='Agradável [18, 24, 30]')
ax.fill_between(x, agradavel, alpha=0.1, color='blue')

# Marcando cada cenário no gráfico
cores   = {"A": "green",  "B": "orange",  "C": "red"}
markers = {"A": "o",      "B": "s",       "C": "X"}

for nome, info in cenarios.items():
    grau = fuzz.interp_membership(x, agradavel, info["temp"])
    ax.vlines(info["temp"], 0, grau,
              colors=cores[nome], linestyles="--", linewidth=1.5)
    ax.hlines(grau, 15, info["temp"],
              colors=cores[nome], linestyles="--", linewidth=1.5)
    ax.plot(info["temp"], grau,
            marker=markers[nome], color=cores[nome], markersize=12,
            label="Cenário {}: {}°C → μ = {:.1f}".format(nome, info["temp"], grau),
            zorder=5)

# Anotando os parâmetros a, b, c
for val, label in zip([18, 24, 30], ['a = 18', 'b = 24', 'c = 30']):
    ax.axvline(val, color='gray', linestyle=':', linewidth=1)
    ax.text(val, 1.02, "{} = {}".format(label[0], val), ha='center', fontsize=9, color='gray')

ax.set_title('Função de Pertinência Fuzzy — Temperatura Agradável', fontsize=13)
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Grau de Pertinência  μ(x)')
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(15, 35)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()
