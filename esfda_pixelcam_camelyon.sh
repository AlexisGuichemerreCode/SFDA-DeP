export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate SFDA_env_wsol_histology

python main.py --task STD_CL --encoder_name resnet50 --arch STDClassifier --pixel_wise_classification True --runmode search-mode --opt__name_optimizer sgd --batch_size 128 --eval_batch_size 128 --eval_checkpoint_type best_localization --opt__step_size 5 --opt__gamma 0.9 --max_epochs 10 --freeze_classifier_sfda False --freeze_encoder_sfda False --support_background True --method PixelCAM --mil_gated False --spatial_pooling WGAP --dataset CAMELYON512 --fold 0 --debug_subfolder None --amp True --opt__lr 0.001 --sf_uda True --sf_uda_source_ds GLAS --sf_uda_source_ds_fold 0 --sf_uda_source_encoder_name resnet50 --sf_uda_source_checkpoint_type best_classification --sf_uda_source_wsol_method PixelCAM --sf_uda_source_wsol_arch STDClassifier --sf_uda_source_wsol_spatial_pooling WGAP --esfda True --esfda_select_imgs True --esfda_select_imgs_ratio 0.2 --select_distance False --select_entropy True --random_select_ratio 1.0 --esfda_reverse_imgs True --esfda_entropy_partial True --esfda_entropy_partial_lambda 0.01 --esfda_weight_entropy linear --esfda_flip_labels True --CEFlipLoss_lambda 0.5 --esfda_notflip_labels True --CENotFlipLoss_lambda 0.5 --esfda_flip_labels_weight True --unlearning_models False --m_unlearning_models 1 --target_domain_ds_to_compute_stats CAMELYON512 --ds_to_compute_acc_trainset_source_target GLAS --cmpt_batch 50 --cl_train_models False --exp_id COUNT_unlearning_50

wait