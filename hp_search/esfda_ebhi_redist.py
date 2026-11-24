"""
ESFDA on EBHI (source fold 0 -> target folds 5, 6): forget-loss probability
redistribution study (dlib/losses/unlearning_sfda.py).

Same fixed arguments as hp_search/esfda_ebhi.py except: one fixed config
(ratio 0.2, resample 10, lr 1e-4, CEForget = CERetain = ece_adapt = 1.0),
esfda_flip_labels_v4 off, and the forget loss varies:
  v3       : esfda_flip_labels_v3 (previous EBHI forget loss, reference)
  uniform  : esfda_forget_redistribution uniform
  hybrid_a : esfda_forget_redistribution hybrid, alpha in ALPHAS

  python hp_search/esfda_ebhi_redist.py list
  python hp_search/esfda_ebhi_redist.py local --gpus 0 1 --per-gpu 2
  python hp_search/esfda_ebhi_redist.py collect
Forget/retain selection ratio (default 0.2) via env ESFDA_RATIO, e.g.
  ESFDA_RATIO=0.4 python hp_search/esfda_ebhi_redist.py local --folds 5 ...
Subset of variants via env ESFDA_VARIANTS (comma-separated names), e.g.
  ESFDA_VARIANTS=uniform python hp_search/esfda_ebhi_redist.py local \
      --folds 8 9 11 ...
"""
import os
import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(abspath(__file__)))
import litrico_ucon_c17 as base  # noqa: E402
import esfda_ebhi  # noqa: E402

TARGET_FOLDS = [5, 6]
ALPHAS = [0.0, 0.2, 0.7]
RATIO = float(os.environ.get('ESFDA_RATIO', '0.2'))
RATIO_STR = str(RATIO).replace('.', 'p')

FIXED = dict(esfda_ebhi.HPS)
FIXED = {k: v[0] for k, v in FIXED.items()}
FIXED.update({
    'resample_every': 10,
    'esfda_select_imgs_ratio': RATIO,
    'CEForget_lambda': 1.0,
    'CERetain_lambda': 1.0,
    'ece_adapt_lambda': 1.0,
    'esfda_flip_labels_v4': False,
})

VARIANTS = [('v3', {'esfda_flip_labels_v3': True,
                    'esfda_forget_redistribution': 'none'}),
            ('uniform', {'esfda_flip_labels_v3': False,
                         'esfda_forget_redistribution': 'uniform'})]
for a in ALPHAS:
    VARIANTS.append((f"hybrid_a{str(a).replace('.', 'p')}",
                     {'esfda_flip_labels_v3': False,
                      'esfda_forget_redistribution': 'hybrid',
                      'esfda_forget_alpha': a}))

if os.environ.get('ESFDA_VARIANTS'):
    _keep = os.environ['ESFDA_VARIANTS'].split(',')
    assert set(_keep) <= {n for n, _ in VARIANTS}, _keep
    VARIANTS = [(n, v) for n, v in VARIANTS if n in _keep]


def jobs(methods=None, folds=None):
    folds = TARGET_FOLDS if folds is None else folds
    out = []
    for fold in folds:
        for ind, (name, var) in enumerate(VARIANTS):
            hp = dict(FIXED, **var)
            exp_id = (f"target_b_cl_esfda_redist_{name}_ebhi"
                      f"{esfda_ebhi.SOURCE_FOLD}_to_ebhi{fold}_ratio{RATIO_STR}_"
                      f"resample10_forget1.0_retain1.0_loc1.0_"
                      f"{esfda_ebhi.METHOD}_lr0p0001_step{esfda_ebhi.STEP}")
            args = [
                '--task', 'STD_CL', '--encoder_name', 'resnet50',
                '--arch', 'STDClassifier', '--runmode', 'search-mode',
                '--opt__name_optimizer', 'sgd', '--batch_size', '32',
                '--eval_checkpoint_type', 'best_classification',
                '--opt__step_size', str(esfda_ebhi.STEP),
                '--max_epochs', '100',
                '--freeze_cl', 'False', '--method', esfda_ebhi.METHOD,
                '--dataset', 'EBHI', '--fold', str(fold),
                '--cudaid', '0', '--debug_subfolder', 'None', '--amp', 'True',
                '--sf_uda', 'True',
                '--sf_uda_source_ds', esfda_ebhi.SOURCE_DS,
                '--sf_uda_source_ds_fold', str(esfda_ebhi.SOURCE_FOLD),
                '--sf_uda_source_encoder_name', 'resnet50',
                '--sf_uda_source_checkpoint_type', 'best_localization',
                '--sf_uda_source_wsol_method', esfda_ebhi.METHOD,
                '--sf_uda_source_wsol_arch', 'STDClassifier',
                '--sf_uda_source_wsol_spatial_pooling', 'WGAP',
                '--exp_id', exp_id,
            ]
            for k, v in hp.items():
                args += [f'--{k}', str(v)]
            out.append({'method': 'esfda', 'fold': fold, 'config': ind,
                        'hp': {'variant': name}, 'exp_id': exp_id,
                        'args': args})
    return out


base.TAG = 'ebhi_esfda_redist_f5_f6'
base.SLURM_TAG = 'ebhi_esfda_redist_f5_f6'
base.DATASET = 'EBHI'
base.METHODS = ['esfda']
base.TARGET_FOLDS = TARGET_FOLDS
base.jobs = jobs

if __name__ == '__main__':
    base.main()
