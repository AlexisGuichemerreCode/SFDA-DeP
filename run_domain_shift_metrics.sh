#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology_clone

# =====================================================================
# ALL get_domain_shift.py evaluations
# 24 source models / 159 explicit source-target configurations
# =====================================================================


# ---------------------------------------------------------------------
# MODEL 01: EnergyCAM | CAMELYON512 fold 0
# id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 02: PixelCAM | EBHI fold 0
# id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 5 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 6 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 7 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 8 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 9 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 10 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 11 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_EBHI_PixelCAM_pixelcam_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 03: SAT | CAMELYON17_512 fold 2
# id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 04: SAT | CAMELYON512 fold 0
# id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 05: DEEPMIL | EBHI fold 0
# id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 5 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 6 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 7 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 8 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 9 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 10 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 11 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_EBHI_DEEPMIL_deepmil_low_res_50-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 06: SAT | GLAS fold 0
# id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset GLAS --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 07: DEEPMIL | CAMELYON17_512 fold 2
# id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 08: DEEPMIL | CAMELYON17_512 fold 3
# id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 09: DEEPMIL | CAMELYON17_512 fold 1
# id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 10: PixelCAM | CAMELYON17_512 fold 3
# id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 11: PixelCAM | CAMELYON17_512 fold 4
# id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 12: SAT | EBHI fold 0
# id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 5 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 6 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 7 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 8 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 9 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 10 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 11 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset EBHI --target_dataset EBHI --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_EBHI_SAT_sat_low_res_100-tsk_STD_CL-ds_EBHI-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 13: SAT | CAMELYON17_512 fold 1
# id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 14: PixelCAM | CAMELYON17_512 fold 0
# id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 15: SAT | CAMELYON17_512 fold 4
# id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 16: DEEPMIL | GLAS fold 0
# id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset GLAS --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 17: DEEPMIL | CAMELYON17_512 fold 4
# id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 4 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 18: PixelCAM | CAMELYON17_512 fold 1
# id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 1 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 19: DEEPMIL | CAMELYON17_512 fold 0
# id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 20: PixelCAM | CAMELYON17_512 fold 2
# id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 2 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 21: SAT | CAMELYON17_512 fold 0
# id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 22: SAT | CAMELYON17_512 fold 3
# id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 3 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --wsol_method SAT --sfda_method source --source_dataset CAMELYON17_512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224


# ---------------------------------------------------------------------
# MODEL 23: DEEPMIL | CAMELYON512 fold 0
# id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON512 --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method DEEPMIL --sfda_method source --source_dataset CAMELYON512 --target_dataset GLAS --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50


# ---------------------------------------------------------------------
# MODEL 24: PixelCAM | GLAS fold 0
# id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50
# ---------------------------------------------------------------------

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 1 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 4 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset GLAS --target_dataset CAMELYON17_512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

python get_domain_shift.py --cudaid 0 --split train --fold_src_dataset 0 --fold_trg_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --wsol_method PixelCAM --sfda_method source --source_dataset GLAS --target_dataset CAMELYON512 --path_pre_trained_source /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50

wait
