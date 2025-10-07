import cv2
import os
import re

import cv2
import os
import re
import numpy as np
import subprocess

# --- Dossier contenant les images ---
img_folder = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/CAMELYON512/resnet50/STD_CL/PixelCAM/id_test_unlearning_pxcam_ent_mix-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/entropy_hist"
output_file = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/video_cancer_normal_limited.mp4"

print("Dossier courant :", os.getcwd())


output_mp4 = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/video_cancer_normal_pxcam_ent_mix.mp4"
output_avi = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/video_cancer_normal_pxcam_ent_mix.avi"

# --- Fonction pour extraire (epoch, batch) ---
def extract_numbers(filename):
    match = re.search(r"ep(\d+)_b(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)

# --- Séparer images cancer et normal ---
images_cancer = [f for f in os.listdir(img_folder) if "cancer" in f and f.endswith(".png")]
images_normal = [f for f in os.listdir(img_folder) if "normal" in f and f.endswith(".png")]

images_cancer = sorted(images_cancer, key=extract_numbers)
images_normal = sorted(images_normal, key=extract_numbers)

if not images_cancer or not images_normal:
    raise RuntimeError("❌ Pas trouvé d'images 'cancer' ou 'normal' dans le dossier.")

num_frames = min(len(images_cancer), len(images_normal))

# Lire une première image pour définir dimensions
first_cancer = cv2.imread(os.path.join(img_folder, images_cancer[0]))
first_normal = cv2.imread(os.path.join(img_folder, images_normal[0]))

height = max(first_cancer.shape[0], first_normal.shape[0])
width = first_cancer.shape[1] + first_normal.shape[1]

# Downscale pour éviter fichiers trop gros
scale = 0.5
out_width = int(width * scale) // 2 * 2
out_height = int(height * scale) // 2 * 2

fps = 5
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI universel
video = cv2.VideoWriter(output_avi, fourcc, fps, (out_width, out_height))

if not video.isOpened():
    raise RuntimeError(f"❌ Impossible d'ouvrir VideoWriter avec dimensions {out_width}x{out_height}")

print(f"✅ Création AVI {output_avi} ({out_width}x{out_height}, {fps} fps)")

# --- Boucle d'écriture ---
for i in range(num_frames):
    img_cancer = cv2.imread(os.path.join(img_folder, images_cancer[i]))
    img_normal = cv2.imread(os.path.join(img_folder, images_normal[i]))

    img_cancer = cv2.resize(img_cancer, (first_cancer.shape[1], first_cancer.shape[0]))
    img_normal = cv2.resize(img_normal, (first_normal.shape[1], first_normal.shape[0]))

    combined = np.zeros((height, width, 3), dtype=np.uint8)
    combined[:img_cancer.shape[0], :img_cancer.shape[1]] = img_cancer
    combined[:img_normal.shape[0], img_cancer.shape[1]:] = img_normal

    combined = cv2.resize(combined, (out_width, out_height))
    video.write(combined)

    if (i+1) % 10 == 0 or i == num_frames - 1:
        print(f"   → Frame {i+1}/{num_frames}")

video.release()
cv2.destroyAllWindows()

# Vérifier la taille du fichier AVI
if os.path.exists(output_avi):
    size_mb = os.path.getsize(output_avi) / (1024*1024)
    print(f"🎥 AVI sauvegardé : {output_avi} ({size_mb:.2f} MB)")

    # --- Conversion AVI → MP4 avec ffmpeg ---
    try:
        print("🔄 Conversion en MP4 standard avec ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y", "-i", output_avi,
            "-vcodec", "libx264", "-crf", "23", output_mp4
        ], check=True)
        size_mp4 = os.path.getsize(output_mp4) / (1024*1024)
        print(f"🎉 MP4 sauvegardé : {output_mp4} ({size_mp4:.2f} MB)")
    except Exception as e:
        print("⚠️ Erreur pendant la conversion ffmpeg :", e)
        print(f"👉 Tu peux la faire manuellement : ffmpeg -i {output_avi} -vcodec libx264 -crf 23 {output_mp4}")
else:
    print("❌ Erreur : le fichier AVI n'a pas été créé.")