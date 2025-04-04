import torch
from torchvision import transforms
import torchstain
import cv2
import numpy as np
from scipy.spatial.distance import cosine



target = cv2.cvtColor(cv2.imread("target.png"), cv2.COLOR_BGR2RGB)
to_transform = cv2.cvtColor(cv2.imread("source.png"), cv2.COLOR_BGR2RGB)

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

normalizer_target = torchstain.normalizers.MacenkoNormalizer(backend='torch')
normalizer_target.fit(T(target))

normalizer_source = torchstain.normalizers.MacenkoNormalizer(backend='torch')
normalizer_source.fit(to_transform))

HE_target, _, _ = normalizer_target._TorchMacenkoNormalizer__compute_matrices(I=T(target), Io=240, alpha=1, beta=0.15)
HE_source, _, _ = normalizer_source._TorchMacenkoNormalizer__compute_matrices(I=T(to_transform), Io=240, alpha=1, beta=0.15)

t_to_transform = T(to_transform)
norm, H, E = normalizer_target.normalize(I=t_to_transform, stains=True)

HE1 = HE_target.cpu().numpy()
HE2 = HE_source.cpu().numpy()
frobenius_dist = np.linalg.norm(HE1 - HE2) 



# Fonction pour sauvegarder les images
def save_image(tensor, filename):
    if len(tensor.shape) == 3 and tensor.shape[2] == 3:  # Image RGB
        img = tensor.cpu().numpy()  # Convertir (C, H, W) → (H, W, C)
    elif len(tensor.shape) == 2:  # Image en niveaux de gris
        img = tensor.cpu().numpy()
    else:
        raise ValueError(f"Format inattendu pour l'image : {tensor.shape}")

    # Vérifier les valeurs
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Sauvegarder avec OpenCV
    cv2.imwrite(filename,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Sauvegarder les images normalisées
save_image(norm, "normalized.png")
save_image(H, "hematoxylin.png")
save_image(E, "eosin.png")

