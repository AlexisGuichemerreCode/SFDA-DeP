export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology

python get_cam.py --cudaid 0 --split train --checkpoint_type best_classification --encoder_name resnet50 --pixel_wise_classification False --source_dataset CAMELYON17_512 --path_cam /export/livia/home/vision/Aguichemerre/Energy_based_Adaptation/data_cams/resnet50-pixelcam-bcl-glas-camelyon17_512_2_cams_train --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPRW_Histology_Project_2024/model_benchmark_sfuda/source_models/BCL/GLAS/id_test_unlearning_43-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 