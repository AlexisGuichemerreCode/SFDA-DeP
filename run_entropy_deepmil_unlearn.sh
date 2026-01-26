export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology

python get_accuracy_entropy.py --cudaid 0 --split test --fold_dataset 3 --checkpoint_type B-UNLEARNING_valcl --encoder_name resnet50 --sfda_method DEEPMIL_PixelCAM_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/3/DEEPMIL/BCL/id_target_b_cl_esfda_deepmil_3_22_CAMELYON17_512_PixelCAM_lr0p0001_20-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &
python get_accuracy_entropy.py --cudaid 0 --split train --fold_dataset 3 --checkpoint_type B-UNLEARNING_valcl --encoder_name resnet50 --sfda_method DEEPMIL_PixelCAM_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/3/DEEPMIL/BCL/id_target_b_cl_esfda_deepmil_3_22_CAMELYON17_512_PixelCAM_lr0p0001_20-tsk_STD_CL-ds_CAMELYON17_512-fold_3-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &

python get_accuracy_entropy.py --cudaid 0 --split test --fold_dataset 2 --checkpoint_type B-UNLEARNING_valcl --encoder_name resnet50 --sfda_method DEEPMIL_PixelCAM_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/2/DEEPMIL/BCL/id_target_b_cl_esfda_deepmil_2_11_CAMELYON17_512_PixelCAM_lr1e-05_20-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &
python get_accuracy_entropy.py --cudaid 0 --split train --fold_dataset 2 --checkpoint_type B-UNLEARNING_valcl --encoder_name resnet50 --sfda_method DEEPMIL_PixelCAM_classic --source_dataset CAMELYON17_512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON17_512/2/DEEPMIL/BCL/id_target_b_cl_esfda_deepmil_2_11_CAMELYON17_512_PixelCAM_lr1e-05_20-tsk_STD_CL-ds_CAMELYON17_512-fold_2-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &

python get_accuracy_entropy.py --cudaid 0 --split test --fold_dataset 0 --checkpoint_type B-UNLEARNING_valcl --encoder_name resnet50 --sfda_method DEEPMIL_PixelCAM_classic --source_dataset CAMELYON512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON512/DEEPMIL/BCL/id_target_b_cl_esfda_deepmil_31_CAMELYON512_PixelCAM_lr0p0001_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &
python get_accuracy_entropy.py --cudaid 0 --split train --fold_dataset 0 --checkpoint_type B-UNLEARNING_valcl --encoder_name resnet50 --sfda_method DEEPMIL_PixelCAM_classic --source_dataset CAMELYON512 --wsol_method PixelCAM --path_pre_trained_source /export/livia/home/vision/Aguichemerre/unlearn_models/GLAS/CAMELYON512/DEEPMIL/BCL/id_target_b_cl_esfda_deepmil_31_CAMELYON512_PixelCAM_lr0p0001_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_PixelCAM-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &


wait
