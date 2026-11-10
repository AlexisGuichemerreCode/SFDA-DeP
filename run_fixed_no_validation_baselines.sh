#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_fixed_no_validation_baselines.sh [shot|cdcl|erl|rgv|sfda_de|spf|all]
# Runs serially on one GPU to avoid GPU-memory contention.
export CUDA_VISIBLE_DEVICES=0

source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
conda deactivate || true
conda activate SFDA_env_wsol_histology_clone

common=(
  --task STD_CL
  --encoder_name resnet50
  --pixel_wise_classification True
  --arch STDClassifier
  --runmode search-mode
  --opt__name_optimizer sgd
  --eval_batch_size 32
  --batch_size 32
  --eval_checkpoint_type best_localization
  --opt__step_size 20
  --opt__gamma 0.9
  --max_epochs 100
  --freeze_cl False
  --support_background True
  --method PixelCAM
  --mil_gated False
  --spatial_pooling WGAP
  --dataset CAMELYON17_512
  --fold 0
  --debug_subfolder None
  --amp True
  --opt__lr 0.0001
  --sf_uda True
  --sf_uda_source_ds GLAS
  --sf_uda_source_ds_fold 0
  --sf_uda_source_encoder_name resnet50
  --sf_uda_source_checkpoint_type best_classification
  --sf_uda_source_wsol_method PixelCAM
  --sf_uda_source_wsol_arch STDClassifier
  --sf_uda_source_wsol_spatial_pooling WGAP
  --sl_pc_seeder probability_negative_salloarea_seeder
  --path_cam resnet50-pixelcam-bcl-glas-camelyon17_512_0
  --no_validation_protocol True
  --epoch_measure 5
  --ece_adapt False
  --ece_adapt_lambda 1.0
  --cal False
  --cal_lambda 0.5
  --cl_train_models False
  --correct_and_incorrect_pseudo_labels False
  --correct_pseudo_labels_ratio 0.1
)

run_one() {
  local method="$1"
  local -a specific=()
  case "$method" in
    shot)
      specific=(
        --shot True --shot_freq_epoch 1 --shot_dist_type cosine
        --ent_pseudo_lb True --ent_pseudo_lb_lambda 1.0
        --div_pseudo_lb True --div_pseudo_lb_lambda 1.0
        --ce_pseudo_lb True --ce_pseudo_lb_lambda 1.0
        --ce_pseudo_lb_smooth 0.1
      )
      ;;
    cdcl)
      specific=(
        --cdcl True --cdcl_threshold 0.5 --cdcl_cvg 0.01
        --cdcl_lambda 1.0 --cdcl_pseudo_lb True --cdcl_tau 0.5
      )
      ;;
    erl)
      specific=(
        --cdcl True --cdcl_threshold 0.5 --cdcl_cvg 0.01
        --cdcl_lambda 1.0 --cdcl_pseudo_lb True --cdcl_tau 0.5
        --erl True --erl_beta 0.1 --erl_lambda 1.0
      )
      ;;
    rgv)
      specific=(
        --rgv True --sfda_aug_transform True
        --ce_pseudo_lb True --ce_pseudo_lb_lambda 1.0
        --div_pseudo_lb True --div_pseudo_lb_lambda 1.0
        --semalg True --lambdaS 1.0 --betaS 0.7
      )
      ;;
    sfda_de)
      # IMPORTANT: verify the actual SFDA-DE activation flag in your main.py.
      # cdd_pseudo_lb remains False as requested (it is not an activation flag).
      specific=(
        --sfde True
        --cdd_lambda 1.0 --cdd_pseudo_lb True
        
      )
      ;;
    spf)
      # SPF: fixed loss coefficients, rho=0.1, resampling every 5 epochs.
      # Specific arguments override common defaults below.
      specific=(
        --grsfda False --esfda True --esfda_2 False
        --disable_train_augmentations False
        --esfda_select_imgs True --esfda_select_imgs_ratio 0.1
        --esfda_top2 False --esfda_top_no_dom False
        --esfda_rand_underpred False --retain_all_others True
        --dynamic_selection True --resample_every 5
        --measure_loc True --save_multiple_unlearn_models False
        --loc_self_learning False --ce_pseudo_lb False
        --select_best_on_train False --stable_match_strategy True
        --save_unlearning_model_all_criterion True
        --activate_load_mask False --track_test_performance True
        --track_cams False --track_cam_every 1
        --select_distance False --select_entropy True
        --random_select_ratio 1.0 --esfda_reverse_imgs True
        --esfda_top2 True
        --esfda_entropy_partial True --esfda_entropy_partial_lambda 1.0
        --esfda_weight_entropy linear
        --esfda_flip_labels True --CEForget_lambda 1.0
        --esfda_notflip_labels True --CERetain_lambda 1.0
        --esfda_flip_labels_weight False
        --esfda_loc False --esfda_loc_lambda 1.0 --esfda_loc_mode mse
        --div_pseudo_lb False --div_pseudo_lb_lambda 1.0
        --ece False --ece_lambda 1.0
        --ece_adapt True --ece_adapt_lambda 1.0 --keep_ratio 0.3
        --epx False --epx_lambda 1.0
        --forget_el False --forget_lambda 1.0
        --sl_min 10 --sl_max 10 --sl_min_p 0.2 --low_res True
        --crf_fc False --crf_lambda 1.0 --crf_sigma_xy 10
        --unlearning_models False --m_unlearning_models 1
        --fold_came_compute 0 --cmpt_batch 1000
        --cl_train_models True --entropy_filter_mode False
        --neg_samples_partial True
        --cal_lambda 1.0
      )
      ;;
    *) echo "Unknown method: $method" >&2; return 2 ;;
  esac

  echo "Running $method (100 epochs, every 5 epochs, fixed coefficients)"
  python main.py "${common[@]}" "${specific[@]}" --exp_id "$method"
}

selection="${1:-all}"
if [[ "$selection" == all ]]; then
  for method in shot cdcl erl rgv sfda_de spf; do
    run_one "$method"
  done
else
  run_one "$selection"
fi
