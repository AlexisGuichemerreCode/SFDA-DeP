import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

# Exemple d'utilisation
path = "/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/GLAS/resnet50/STD_CL/PixelCAM/id_test_unlearning_38-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/metrics_history.pickle"
data = load_pickle(path)

print(type(data))   # Pour voir le type d'objet
print(len(data))    # Si c'est une liste ou un dict
print(data)         # Pour afficher le contenu (si pas trop gros)


