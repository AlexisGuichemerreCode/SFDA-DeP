import torch
import matplotlib.pyplot as plt


def compute_energy(logits):
    return -torch.logsumexp(logits, dim=-1)


logits = torch.randn(28, 28, 2)
energy = compute_energy(logits)
print(energy)

plt.imshow(energy.detach().numpy(), cmap="inferno")
plt.colorbar(label="Energy")
plt.title("Pixel-wise Energy Map")
plt.savefig("energy_map.png", dpi=300, bbox_inches='tight')
