import pickle
import os

def load_pseudo_labels(directory):
    file_path = os.path.join(directory, 'pseudo_labels.pkl')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    with open(file_path, 'rb') as file:
        pseudo_labels = pickle.load(file)

    return pseudo_labels

# Exemple d'utilisation
directory = '/export/livia/home/vision/Aguichemerre/WACV_NOWARMUP/CAMELYON/ADAPTED/id_target_b_cl_sfde_6_CAMELYON512_PixelCAM_lr0p001_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50'
labels = load_pseudo_labels(directory)

print(labels)