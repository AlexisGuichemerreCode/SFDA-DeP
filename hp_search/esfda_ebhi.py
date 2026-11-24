"""
HP search: ESFDA (ours) on EBHI, source fold 0 (PixelCAM / ResNet50,
best_localization source checkpoint) -> target folds 1, 6 and 8.

Grid, exp_ids and main.py arguments reproduce
configs_cc/python_script/create_pxcam_esfda_EBHI.py exactly (same key order,
so the same config index), only target folds change.
72 configs per fold: resample_every x CEForget x CERetain x ratio.

Same modes as hp_search/litrico_ucon_c17.py (list / local / serve / slurm /
collect), e.g.:
  python hp_search/esfda_ebhi.py list
  python hp_search/esfda_ebhi.py slurm --cluster nibi --ids-file ids.txt \
      --accounts def-egranger rrg-josedolz --time 1:00:00 --submit
  python hp_search/esfda_ebhi.py serve --queue-file q.txt --gpus 0 1 2 3 \
      --per-gpu 3
  python hp_search/esfda_ebhi.py collect [--exps-root $SCRATCH/benchmark_wsol_histo]
"""
import itertools
import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(abspath(__file__)))
import litrico_ucon_c17 as base  # noqa: E402

SOURCE_DS = 'EBHI'
SOURCE_FOLD = 0
TARGET_FOLDS = [1, 6, 8]
METHOD = 'PixelCAM'
STEP = 20

# same insertion order as create_pxcam_esfda_EBHI.py (method hps, common hps,
# then esfda hps) -> identical itertools.product order / config indices.
HPS = {
    'spatial_pooling': ['WGAP'],
    'support_background': [True],
    'pixel_wise_classification': [True],
    'opt__lr': [0.0001],
    'opt__weight_decay': [0.0001],
    'opt__gamma': [0.9],
    'esfda': [True],
    'esfda_select_imgs': [True],
    'stable_match_strategy': [True],
    'select_entropy': [True],
    'esfda_entropy_partial': [True],
    'esfda_reverse_imgs': [True],
    'esfda_weight_entropy': ['linear'],
    'retain_all_others': [True],
    'dynamic_selection': [True],
    'resample_every': [5, 10, 20],
    'measure_loc': [True],
    'esfda_flip_labels_v4': [True],
    'esfda_notflip_labels': [True],
    'CEForget_lambda': [0.1, 0.3],
    'CERetain_lambda': [0.1, 0.5, 1.0],
    'ece_adapt': [True],
    'ece_adapt_lambda': [1.0],
    'loc_self_learning': [True],
    'keep_ratio': [0.1],
    'neg_samples_partial': [False],
    'esfda_select_imgs_ratio': [0.10, 0.20, 0.30, 0.40],
    'low_res': [True],
    'sl_pc_seeder': ['probability_negative_salloarea_seeder'],
    'unlearning_models': [False],
    'm_unlearning_models': [1],
    'fold_came_compute': [0],
    'cmpt_batch': [1000],
    'cl_train_models': [False],
    'freeze_classifier_sfda': [False],
    'freeze_encoder_sfda': [False],
}

SEARCHED = ['resample_every', 'CEForget_lambda', 'CERetain_lambda',
            'esfda_select_imgs_ratio']


def jobs(methods=None, folds=None):
    folds = TARGET_FOLDS if folds is None else folds
    keys, values = zip(*HPS.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    out = []
    for fold in folds:
        for ind, hp in enumerate(grid):
            lr_str = str(hp['opt__lr']).replace('.', 'p')
            ratio_str = str(hp['esfda_select_imgs_ratio']).replace('.', 'p')
            exp_id = (f"target_b_cl_esfda_ebhi{SOURCE_FOLD}_to_ebhi{fold}_"
                      f"ratio{ratio_str}_resample{hp['resample_every']}_"
                      f"forget{hp['CEForget_lambda']}_"
                      f"retain{hp['CERetain_lambda']}_"
                      f"config{ind}_{METHOD}_lr{lr_str}_step{STEP}")
            args = [
                '--task', 'STD_CL', '--encoder_name', 'resnet50',
                '--arch', 'STDClassifier', '--runmode', 'search-mode',
                '--opt__name_optimizer', 'sgd', '--batch_size', '32',
                '--eval_checkpoint_type', 'best_classification',
                '--opt__step_size', str(STEP), '--max_epochs', '100',
                '--freeze_cl', 'False', '--method', METHOD,
                '--dataset', 'EBHI', '--fold', str(fold),
                '--cudaid', '0', '--debug_subfolder', 'None', '--amp', 'True',
                '--sf_uda', 'True',
                '--sf_uda_source_ds', SOURCE_DS,
                '--sf_uda_source_ds_fold', str(SOURCE_FOLD),
                '--sf_uda_source_encoder_name', 'resnet50',
                '--sf_uda_source_checkpoint_type', 'best_localization',
                '--sf_uda_source_wsol_method', METHOD,
                '--sf_uda_source_wsol_arch', 'STDClassifier',
                '--sf_uda_source_wsol_spatial_pooling', 'WGAP',
                '--exp_id', exp_id,
            ]
            for k, v in hp.items():
                args += [f'--{k}', str(v)]
            out.append({'method': 'esfda', 'fold': fold, 'config': ind,
                        'hp': {k: hp[k] for k in SEARCHED},
                        'exp_id': exp_id, 'args': args})
    return out


# plug the EBHI experiment into the generic runner.
base.TAG = 'ebhi_esfda_f1_f8'
base.SLURM_TAG = 'ebhi_esfda_f1_f8'
base.DATASET = 'EBHI'
base.METHODS = ['esfda']
base.TARGET_FOLDS = TARGET_FOLDS
base.jobs = jobs

if __name__ == '__main__':
    base.main()
