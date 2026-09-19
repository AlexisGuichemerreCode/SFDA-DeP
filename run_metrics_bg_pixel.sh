#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh

conda deactivate
conda activate SFDA_env_wsol_histology_clone

mkdir -p results_background

COMMON_ARGS="--cudaid 0 --split train --checkpoint_type best_classification"

echo "============================================================"
echo "GLAS -> CAMELYON17 centers 0 1 2 3 4"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset GLAS \
    --source_fold 0 \
    --target_dataset CAMELYON17_512 \
    --target_folds 0 1 2 3 4 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/GLAS_to_C17.png \
    --results_pickle results_background/GLAS_to_C17.pkl

echo "============================================================"
echo "CAMELYON17 center 0 -> centers 1 2 3 4"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON17_512 \
    --source_fold 0 \
    --target_dataset CAMELYON17_512 \
    --target_folds 1 2 3 4 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_fold0_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C17_0_to_C17.png \
    --results_pickle results_background/C17_0_to_C17.pkl

echo "============================================================"
echo "CAMELYON17 center 1 -> centers 0 2 3 4"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON17_512 \
    --source_fold 1 \
    --target_dataset CAMELYON17_512 \
    --target_folds 0 2 3 4 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_7_CAMELYON17_512_fold1_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_4_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_1-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C17_1_to_C17.png \
    --results_pickle results_background/C17_1_to_C17.pkl

echo "============================================================"
echo "CAMELYON17 center 2 -> centers 0 1 3 4"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON17_512 \
    --source_fold 2 \
    --target_dataset CAMELYON17_512 \
    --target_folds 0 1 3 4 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_fold2_PixelCAM_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_1_CAMELYON17_512_SAT_sat_50-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C17_2_to_C17.png \
    --results_pickle results_background/C17_2_to_C17.pkl

echo "============================================================"
echo "CAMELYON17 center 3 -> centers 0 1 2 4"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON17_512 \
    --source_fold 3 \
    --target_dataset CAMELYON17_512 \
    --target_folds 0 1 2 4 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold3_PixelCAM_50-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_DEEPMIL_resnet50_100-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C17_3_to_C17.png \
    --results_pickle results_background/C17_3_to_C17.pkl

echo "============================================================"
echo "CAMELYON17 center 4 -> centers 0 1 2 3"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON17_512 \
    --source_fold 4 \
    --target_dataset CAMELYON17_512 \
    --target_folds 0 1 2 3 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_3_CAMELYON17_512_fold4_PixelCAM_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_5_CAMELYON17_512_SAT_sat_100-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_6_CAMELYON17_512_DEEPMIL_resnet50_150-tsk_STD_CL-ds_CAMELYON17_512-fold_4-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C17_4_to_C17.png \
    --results_pickle results_background/C17_4_to_C17.pkl

echo "============================================================"
echo "CAMELYON16 -> GLAS"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON512 \
    --source_fold 0 \
    --target_dataset GLAS \
    --target_folds 0 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C16_to_GLAS.png \
    --results_pickle results_background/C16_to_GLAS.pkl

echo "============================================================"
echo "CAMELYON16 -> CAMELYON17 centers 0 1 2 3 4"
echo "============================================================"

python get_all_cam_metric_background.py \
    $COMMON_ARGS \
    --source_dataset CAMELYON512 \
    --source_fold 0 \
    --target_dataset CAMELYON17_512 \
    --target_folds 0 1 2 3 4 \
    --pixelcam_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_0_CAMELYON512_EnergyCAM_layercam_low_res_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 \
    --sat_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_2_CAMELYON512_SAT_sat_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 \
    --deepmil_path /export/livia/home/vision/Aguichemerre/src_models_ICLR_2027/id_source_8_CAMELYON512_DEEPMIL_2-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 \
    --output results_background/C16_to_C17.png \
    --results_pickle results_background/C16_to_C17.pkl

echo "============================================================"
echo "All CAM background-precision experiments finished."
echo "============================================================"
