import numpy as np
import matplotlib.pyplot as plt

# 1) Variables (étiquettes) et séries
labels = [f"Cam{i}" for i in range(1, 16)]
no_adapt = np.array([48,70,55,60,50,62,75,40,42,20,18,35,33,38,45])
ours     = np.array([52,72,50,60,45,58,72,40,50,28,20,48,30,50,55])
upper    = np.array([65,55,50,60,45,60,80,55,60,30,28,58,52,65,70])

# 2) Angles + fermeture des polygones
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
angles = np.r_[angles, angles[0]]
def close_loop(v): return np.r_[v, v[0]]

# 3) Figure polaire
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111, polar=True)

ax.set_rlabel_position(0)   # orientation des labels radiaux
ax.set_ylim(20, 80)
ax.set_yticks([40,50, 60,70])

# 4) Traces (couleurs par défaut de matplotlib)
ax.plot(angles, close_loop(no_adapt), linestyle="--", label="No-adapt")
ax.fill(angles, close_loop(no_adapt), alpha=0.1)

ax.plot(angles, close_loop(ours), label="Ours")
ax.fill(angles, close_loop(ours), alpha=0.1)

ax.plot(angles, close_loop(upper), label="Upper Bound Model")
ax.fill(angles, close_loop(upper), alpha=0.1)

# 5) Étiquettes angulaires et légende
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.title("Radar chart – exemple")
plt.tight_layout()
plt.savefig("circle_figure.png", dpi=300)
plt.show()