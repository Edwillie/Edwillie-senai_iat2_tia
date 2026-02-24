import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random  # substitua por sensor ou API real

x = np.linspace(15, 35, 200)
print(x)
agradavel = fuzz.trimf(x, [18, 24, 30])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, agradavel, 'b', linewidth=2, label='Agradável')
ax.fill_between(x, agradavel, alpha=0.1, color='blue')
ponto, = ax.plot([], [], 'ro', markersize=12, label='Temperatura atual')
linha_v = ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
linha_h = ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
titulo = ax.set_title('')

ax.set_xlim(15, 38)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('μ(x)')
ax.grid(True, alpha=0.3)
ax.legend()

def atualizar(frame):
    # Substitua esta linha por sensor ou API real
    temp_atual = random.uniform(16, 34)

    grau = fuzz.interp_membership(x, agradavel, temp_atual)
    ponto.set_data([temp_atual], [grau])
    linha_v.set_xdata([temp_atual])
    linha_h.set_ydata([grau])
    titulo.set_text(f"Temperatura: {temp_atual:.1f}°C  |  μ = {grau:.2f}")
    return ponto, linha_v, linha_h, titulo

ani = animation.FuncAnimation(fig, atualizar, interval=1000)  # atualiza a cada 1s
plt.tight_layout()
plt.show()