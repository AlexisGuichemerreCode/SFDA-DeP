"""
HP search: ESFDA on EBHI (source fold 0 -> target folds 5, 6) with the
forget-loss probability redistribution (dlib/losses/unlearning_sfda.py).

Same 72-config grid as hp_search/esfda_ebhi.py (create_pxcam_esfda_EBHI.py:
resample_every x CEForget x CERetain x ratio, same config indices). The
forget loss esfda_flip_labels_v4 is replaced by
esfda_forget_redistribution, one of 4 variants:
  uniform, uniform_all, hybrid (alpha 0), hybrid_all (alpha 0)
-> 4 x 72 = 288 runs per target fold.

  python hp_search/esfda_ebhi_redist_grid.py list --folds 5
  python hp_search/esfda_ebhi_redist_grid.py slurm --cluster nibi --folds 6 \
      --accounts def-egranger rrg-josedolz --time 1:30:00 --submit
  python hp_search/esfda_ebhi_redist_grid.py collect \
      --exps-root $SCRATCH/benchmark_wsol_histo
Subset of variants via env ESFDA_VARIANTS (comma-separated names).
"""
import os
import sys
from os.path import dirname, abspath

sys.path.insert(0, dirname(abspath(__file__)))
import litrico_ucon_c17 as base  # noqa: E402
import esfda_ebhi  # noqa: E402

TARGET_FOLDS = [5, 6]

VARIANTS = [
    ('uniform', {'esfda_forget_redistribution': 'uniform'}),
    ('uniform_all', {'esfda_forget_redistribution': 'uniform_all'}),
    ('hybrid_a0p0', {'esfda_forget_redistribution': 'hybrid',
                     'esfda_forget_alpha': 0.0}),
    ('hybrid_all_a0p0', {'esfda_forget_redistribution': 'hybrid_all',
                         'esfda_forget_alpha': 0.0}),
]
if os.environ.get('ESFDA_VARIANTS'):
    _keep = os.environ['ESFDA_VARIANTS'].split(',')
    assert set(_keep) <= {n for n, _ in VARIANTS}, _keep
    VARIANTS = [(n, v) for n, v in VARIANTS if n in _keep]


def jobs(methods=None, folds=None):
    folds = TARGET_FOLDS if folds is None else folds
    out = []
    for name, var in VARIANTS:
        for j in esfda_ebhi.jobs(folds=folds):
            args = list(j['args'])
            # redistribution replaces the v4 forget loss.
            args[args.index('--esfda_flip_labels_v4') + 1] = 'False'
            for k, v in var.items():
                args += [f'--{k}', str(v)]
            exp_id = j['exp_id'].replace('target_b_cl_esfda_',
                                         f'target_b_cl_esfda_redist_{name}_')
            args[args.index('--exp_id') + 1] = exp_id
            out.append({'method': name, 'fold': j['fold'],
                        'config': j['config'], 'hp': dict(j['hp']),
                        'exp_id': exp_id, 'args': args})
    return out


base.TAG = 'ebhi_esfda_redist_grid_f5_f6'
base.SLURM_TAG = 'ebhi_esfda_redist_grid_f5_f6'
base.DATASET = 'EBHI'
base.METHODS = [n for n, _ in VARIANTS]
base.TARGET_FOLDS = TARGET_FOLDS
base.jobs = jobs

if __name__ == '__main__':
    base.main()
