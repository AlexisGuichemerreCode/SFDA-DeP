#!/usr/bin/env bash
set -euo pipefail

# Rebuttal baselines: GLAS (source) -> CAMELYON17_512 center/fold 0 (target).
#   - litrico : Litrico et al., CVPR 2023, "Guiding Pseudo-labels with
#               Uncertainty Estimation for SFUDA" (dlib/sf_uda/litrico.py)
#   - ucon    : UCon-SFDA, ICLR 2025, "Revisiting SFDA: a New Perspective via
#               Uncertainty Control" (dlib/sf_uda/ucon.py)
#
# Same protocol as run_fixed_no_validation_baselines.sh (PixelCAM / ResNet50,
# 100 epochs, no validation, measure every 5 epochs, fixed coefficients).
#
# Usage: bash run_litrico_ucon_glas_c17_0.sh [litrico|ucon|all] [GPU_ID]
# Runs serially on one GPU.
# Optional env overrides (quick tests):
#   MAX_EPOCHS=6 EPOCH_MEASURE=3 EXP_SUFFIX=_quick bash run_litrico_ucon_glas_c17_0.sh ucon 0

selection="${1:-all}"
export CUDA_VISIBLE_DEVICES="${2:-0}"

cd "$(dirname "$0")"

# Env: repo-local uv venv if present (servers set up with setup_uv_env.sh),
# otherwise the local conda env.
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  source /export/livia/home/vision/Aguichemerre/anaconda3/etc/profile.d/conda.sh
  conda deactivate || true
  conda activate SFDA_env_wsol_histology_clone
fi

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
  --max_epochs "${MAX_EPOCHS:-100}"
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
  --epoch_measure "${EPOCH_MEASURE:-5}"
  --ece_adapt False
  --ece_adapt_lambda 1.0
  --cal False
  --cal_lambda 0.5
  --cl_train_models False
  --correct_and_incorrect_pseudo_labels False
  --correct_pseudo_labels_ratio 0.1
  # all other SFUDA methods OFF
  --shot False --faust False --nrc False --cdcl False --sfde False
  --rgv False --erl False --esfda False --semalg False
  --ce_pseudo_lb False --ent_pseudo_lb False --div_pseudo_lb False
  --sfda_aug_transform False
)

run_one() {
  local method="$1"
  local -a specific=()
  case "$method" in
    litrico)
      # Official defaults (train_target.py): k=10 neighbors, temporal
      # length 5, MoCo m=0.999 / T=0.07, queue 16384, label refinement,
      # negative learning + uncertainty reweighting, ctr + div losses.
      specific=(
        --litrico True
        --litrico_n_strong_views 2
        --litrico_num_neighbors 10
        --litrico_temporal_length 5
        --litrico_queue_size 16384
        --litrico_moco_m 0.999
        --litrico_moco_t 0.07
        --litrico_label_refinement True
        --litrico_neg_learning True
        --litrico_reweighting True
        --litrico_cls True --litrico_cls_lambda 1.0
        --litrico_ctr True --litrico_ctr_lambda 1.0
        --litrico_div True --litrico_div_lambda 1.0
      )
      ;;
    ucon)
      # Official VisDA-C config (configs_pp/visda2017_config.yml):
      # K=5, alpha=1 decayed with beta=5, lambda_dc=1 (fixed),
      # lambda_pl=0.01, k_pl=1 (fixed), tau=1.1 (fixed).
      specific=(
        --ucon True
        --ucon_n_strong_views 1
        --ucon_k 5
        --ucon_alpha 1.0 --ucon_beta 5.0 --ucon_alpha_decay True
        --ucon_pos True --ucon_pos_lambda 1.0
        --ucon_neg True --ucon_neg_lambda 1.0
        --ucon_dc True --ucon_dc_lambda 1.0 --ucon_dc_temp 1.0
        --ucon_pl True --ucon_pl_lambda 0.01
        --ucon_partial_k_type fixed --ucon_partial_k 1
        --ucon_tau_type fixed --ucon_tau 1.1
      )
      ;;
    *) echo "Unknown method: $method" >&2; return 2 ;;
  esac

  echo "Running $method: GLAS -> CAMELYON17_512 fold 0 (${MAX_EPOCHS:-100} epochs, measure every 5)"
  python main.py "${common[@]}" "${specific[@]}" --exp_id "${method}_glas_c17_0${EXP_SUFFIX:-}"
}

if [[ "$selection" == all ]]; then
  for method in litrico ucon; do
    run_one "$method"
  done
else
  run_one "$selection"
fi
