import pickle

with open("/export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/exps/CAMELYON17_512/resnet50/STD_CL/PixelCAM/id_unlearning_c17_24-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50/unlearning_metrics/train/F1_results_train_LR1em05_STEP5_K10p0_P0p1_LRETAIN1p0_LFORGET1p0.pickle", "rb") as f:
    data = pickle.load(f)

print(type(data))

