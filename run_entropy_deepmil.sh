export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology

python get_accuracy_entropy.py --cudaid 0 --split test --fold_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method DEEPMIL --source_dataset CAMELYON17_512 --wsol_method DEEPMIL --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/DEEPMIL/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &
python get_accuracy_entropy.py --cudaid 0 --split train --fold_dataset 3 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method DEEPMIL --source_dataset CAMELYON17_512 --wsol_method DEEPMIL --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/DEEPMIL/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &

python get_accuracy_entropy.py --cudaid 0 --split test --fold_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method DEEPMIL --source_dataset CAMELYON17_512 --wsol_method DEEPMIL --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/DEEPMIL/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &
python get_accuracy_entropy.py --cudaid 0 --split train --fold_dataset 2 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method DEEPMIL --source_dataset CAMELYON17_512 --wsol_method DEEPMIL --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/DEEPMIL/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &

python get_accuracy_entropy.py --cudaid 0 --split test --fold_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method DEEPMIL --source_dataset CAMELYON512 --wsol_method DEEPMIL --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/DEEPMIL/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &
python get_accuracy_entropy.py --cudaid 0 --split train --fold_dataset 0 --checkpoint_type best_classification --encoder_name resnet50 --sfda_method DEEPMIL --source_dataset CAMELYON512 --wsol_method DEEPMIL --path_pre_trained_source /export/livia/home/vision/Aguichemerre/CVPR_2026_SFDA_WSOL/source_model/GLAS/DEEPMIL/id_source_5_GLAS_DEEPMIL_250-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50 &


wait
