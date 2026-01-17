import numpy as np
import matplotlib.pyplot as plt

# 1) Variables (étiquettes) et séries
labels = ["1%", "5%", "10%", "20%", "50%", "70%", "90%", "100%"]

no_adapt = np.array([48, 55, 60, 62, 75, 40, 45, 50])
ours     = np.array([52, 50, 60, 58, 72, 40, 55, 55])

# 2) Angles + fermeture des polygones
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
angles = np.r_[angles, angles[0]]
def close_loop(v): return np.r_[v, v[0]]

# 3) Figure polaire
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)   # ⬆️ start at top
ax.set_theta_direction(-1)       # ⏱ clockwise

ax.set_rlabel_position(0)
ax.set_ylim(10, 90)
ax.set_yticks([10,20,30,40, 50, 60, 70,80,90])

# 4) Traces (couleurs par défaut de matplotlib)
ax.plot(angles, close_loop(no_adapt), linestyle="--", label="Source only")
ax.fill(angles, close_loop(no_adapt), alpha=0.1)

ax.plot(angles, close_loop(ours), label="Unlearning")
ax.fill(angles, close_loop(ours), alpha=0.1)


# 5) Étiquettes angulaires et légende
ax.set_xticklabels(labels, fontsize=10)
ax.tick_params(axis='y', labelsize=9)

ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
#plt.title("Radar chart – exemple")
plt.tight_layout()
plt.savefig("circle_figure.png", dpi=300)
plt.show()