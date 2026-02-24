import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(80, 200, 300)
normal = fuzz.gaussmf(x, 120, 10)

cenarios = {
    "A": {"pressao": 120, "desc":"Pressao Ideal"},
    "B": {"pressao": 60,  "desc":"Pressao Baixa"},
    "C": {"pressao": 180, "desc":"Pressao Alta"}        
}

print("="*50)
print("Funcção Trapezio")
print("="*50)

for nome, info in cenarios.items():
    grau = fuzz.interp_membership(x, normal, info["pressao"])
    print("\nCenario", nome, "(", info["pressao"], "mmHg ): ")
    print("miA = ", round(grau, 4))
    print("->", info["desc"])


#Plotagem
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, normal, 'purple', linewidth=2.5, label='Pressão Normal (gaussmf: média=120, σ=10)')
ax.fill_between(x, normal, alpha=0.1, color='purple')

# Marcando os sigmas (±1σ e ±2σ)
ax.axvspan(110, 130, alpha=0.08, color='green', label='±1σ  (110 a 130 mmHg)')
ax.axvspan(100, 140, alpha=0.05, color='blue',  label='±2σ  (100 a 140 mmHg)')

# Marcando cada cenário
cores   = {"A": "green", "B": "orange", "C": "red"}
markers = {"A": "o",     "B": "s",      "C": "X"}

for nome, info in cenarios.items():
    grau = fuzz.interp_membership(x, normal, info["pressao"])
    ax.vlines(info["pressao"], 0, grau, colors=cores[nome], linestyles="--", linewidth=1.5)
    ax.hlines(grau, 80, info["pressao"], colors=cores[nome], linestyles="--", linewidth=1.5)
    ax.plot(info["pressao"], grau,
            marker=markers[nome], color=cores[nome], markersize=12,
            label="Cenário " + nome + ": " + str(info["pressao"]) + " mmHg → μ = " + str(round(grau, 2)),
            zorder=5)

# Anotando a média e os sigmas
ax.axvline(120, color='gray', linestyle=':', linewidth=1)
ax.text(120, 1.03, "média = 120", ha='center', fontsize=9, color='gray')
ax.text(110, 0.65, "-1σ", ha='center', fontsize=8, color='green')
ax.text(130, 0.65, "+1σ", ha='center', fontsize=8, color='green')

ax.set_title('Função de Pertinência Fuzzy — Pressão Arterial Normal', fontsize=13)
ax.set_xlabel('Pressão Arterial Sistólica (mmHg)')
ax.set_ylabel('Grau de Pertinência  μ(x)')
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(80, 200)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()