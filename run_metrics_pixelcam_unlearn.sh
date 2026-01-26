export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology

python get_accuracy.py --cudaid 0 --split test --fold_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method PixelCAM_unlearn --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/3/PIXELCAM/BCL/id_target_b_cl_esfda_pixelcam_3_23_CAMELYON17_512_PixelCAM_lr0p0001_20-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 &
python get_accuracy.py --cudaid 0 --split train --fold_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method PixelCAM_unlearn --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/3/PIXELCAM/BCL/id_target_b_cl_esfda_pixelcam_3_23_CAMELYON17_512_PixelCAM_lr0p0001_20-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 &

python get_accuracy.py --cudaid 0 --split test --fold_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method PixelCAM_unlearn --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/2/PIXELCAM/BCL/id_target_b_cl_esfda_pixelcam_2_21_CAMELYON17_512_PixelCAM_lr0p0001_20-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 &
python get_accuracy.py --cudaid 0 --split train --fold_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method PixelCAM_unlearn --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/2/PIXELCAM/BCL/id_target_b_cl_esfda_pixelcam_2_21_CAMELYON17_512_PixelCAM_lr0p0001_20-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 &

python get_accuracy.py --cudaid 0 --split test --fold_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method PixelCAM_unlearn --source_dataset CAMELYON512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON512/PIXELCAM/BCL/id_target_b_cl_esfda_pixelcam_20_CAMELYON512_PixelCAM_lr0p0001_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 &
python get_accuracy.py --cudaid 0 --split train --fold_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method PixelCAM_unlearn --source_dataset CAMELYON512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON512/PIXELCAM/BCL/id_target_b_cl_esfda_pixelcam_20_CAMELYON512_PixelCAM_lr0p0001_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50 &


wait
