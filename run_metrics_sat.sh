export CUDA_VISIBLE_DEVICES=1

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology

python get_accuracy.py --cudaid 0 --split test --fold_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --sfda_method SAT_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/SAT/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 &
python get_accuracy.py --cudaid 0 --split train --fold_dataset 3 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --sfda_method SAT_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/SAT/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 &

python get_accuracy.py --cudaid 0 --split test --fold_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --sfda_method SAT_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/SAT/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 &
python get_accuracy.py --cudaid 0 --split train --fold_dataset 2 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --sfda_method SAT_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/SAT/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 &

python get_accuracy.py --cudaid 0 --split test --fold_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --sfda_method SAT_classic --source_dataset CAMELYON512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/SAT/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 &
python get_accuracy.py --cudaid 0 --split train --fold_dataset 0 --checkpoint_type best_classification --encoder_name deit_sat_tiny_patch16_224 --sfda_method SAT_classic --source_dataset CAMELYON512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/SAT/id_source_2_GLAS_SAT_multiple_model_entropy_10_350-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_SAT-spooling_GAP-arch_SATClassifier-ecd_deit_sat_tiny_patch16_224 &


wait
