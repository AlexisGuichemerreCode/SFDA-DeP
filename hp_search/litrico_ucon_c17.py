"""
HP search: Litrico (CVPR'23) and UCon-SFDA (ICLR'25),
CAMELYON17_512 center 0 (source, PixelCAM / ResNet50) -> centers 1, 2, 3, 4.

Same protocol as configs_cc/python_script/generate_nrc.py (Compute Canada):
  SGD lr 1e-4, weight decay 1e-4, gamma 0.9, step 20, batch 32, 100 epochs,
  AMP, validation-based model selection (--eval_checkpoint_type
  best_classification), single seed.
Grid: every method loss weight in {0.5, 1.0}; UCon PL weight in
  {0.01, 0.5, 1.0} (0.01 = official UCon-SFDA value).
  -> Litrico 8 configs, UCon 24 configs, per target center.

One grid definition, used by all modes (identical exp_ids everywhere):

  list     print jobs.
  local    run a GPU queue on a plain server (taylor3, ...):
             python hp_search/litrico_ucon_c17.py local --folds 1 2 \
                 --gpus 0 1 2 3 --per-gpu 3
           Skips jobs already finished (passed.txt). Resumable.
  slurm    write sbatch scripts on a Compute Canada cluster (and submit
           with --submit). Accounts are used round-robin (--accounts).
             python hp_search/litrico_ucon_c17.py slurm --cluster nibi \
                 --folds 3 --accounts def-egranger rrg-josedolz --submit
  collect  per (method, target): pick config with best validation
           classification, report its test metrics (csv + table).
"""
import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from os.path import join, dirname, abspath, isfile, isdir

REPO = dirname(dirname(abspath(__file__)))

SOURCE_FOLD = 0
TARGET_FOLDS = [1, 2, 3, 4]
METHODS = ['litrico', 'ucon']
WSOL = 'PIXELCAM'

COMMON = [
    '--task', 'STD_CL', '--encoder_name', 'resnet50',
    '--arch', 'STDClassifier', '--runmode', 'search-mode',
    '--opt__name_optimizer', 'sgd', '--batch_size', '32',
    '--eval_checkpoint_type', 'best_classification',
    '--opt__step_size', '20', '--max_epochs', '100', '--freeze_cl', 'False',
    '--method', 'PixelCAM', '--dataset', 'CAMELYON17_512',
    '--cudaid', '0', '--debug_subfolder', 'None', '--amp', 'True',
    '--sf_uda', 'True',
    '--sf_uda_source_ds', 'CAMELYON17_512',
    '--sf_uda_source_ds_fold', str(SOURCE_FOLD),
    '--sf_uda_source_encoder_name', 'resnet50',
    '--sf_uda_source_checkpoint_type', 'best_classification',
    '--sf_uda_source_wsol_method', 'PixelCAM',
    '--sf_uda_source_wsol_arch', 'STDClassifier',
    '--sf_uda_source_wsol_spatial_pooling', 'WGAP',
    '--spatial_pooling', 'WGAP', '--support_background', 'True',
    '--mil_gated', 'False', '--pixel_wise_classification', 'True',
    '--opt__lr', '0.0001', '--opt__weight_decay', '0.0001',
    '--opt__gamma', '0.9',
]

# fixed (official) method settings; only loss weights are searched.
FIXED = {
    'litrico': [
        '--litrico', 'True', '--litrico_n_strong_views', '2',
        '--litrico_num_neighbors', '10', '--litrico_temporal_length', '5',
        '--litrico_queue_size', '16384', '--litrico_moco_m', '0.999',
        '--litrico_moco_t', '0.07', '--litrico_label_refinement', 'True',
        '--litrico_neg_learning', 'True', '--litrico_reweighting', 'True',
        '--litrico_cls', 'True', '--litrico_ctr', 'True',
        '--litrico_div', 'True',
    ],
    'ucon': [
        '--ucon', 'True', '--ucon_n_strong_views', '1', '--ucon_k', '5',
        '--ucon_alpha', '1.0', '--ucon_beta', '5.0',
        '--ucon_alpha_decay', 'True',
        '--ucon_pos', 'True', '--ucon_neg', 'True', '--ucon_dc', 'True',
        '--ucon_dc_temp', '1.0', '--ucon_pl', 'True',
        '--ucon_partial_k_type', 'fixed', '--ucon_partial_k', '1',
        '--ucon_tau_type', 'fixed', '--ucon_tau', '1.1',
    ],
}

GRID = {
    'litrico': {
        'litrico_cls_lambda': [0.5, 1.0],
        'litrico_ctr_lambda': [0.5, 1.0],
        'litrico_div_lambda': [0.5, 1.0],
    },
    'ucon': {
        'ucon_pos_lambda': [0.5, 1.0],
        'ucon_neg_lambda': [0.5, 1.0],
        'ucon_dc_lambda': [0.5, 1.0],
        'ucon_pl_lambda': [0.01, 0.5, 1.0],
    },
}

# Compute Canada: MIG slices used by the existing generated scripts.
CLUSTERS = {
    'nibi': {'gres': 'gpu:nvidia_h100_80gb_hbm3_2g.20gb:1',
             'root': '/home/aguich25/projects/def-egranger/aguich25/SFDA-DeP'},
    'rorqual': {'gres': 'gpu:nvidia_h100_80gb_hbm3_2g.20gb:1',
                'root': '/home/aguich25/links/projects/def-egranger/'
                        'aguich25/SFDA-DeP'},
    'narval': {'gres': 'gpu:a100_3g.20gb:1',
               'root': '/home/aguich25/projects/def-egranger/aguich25/SFDA-DeP'},
}

SLURM_HEADER = """#!/bin/sh

#SBATCH --account={account}
#SBATCH --output={logdir}/{name}_%J.o
#SBATCH --error={logdir}/{name}_%J.e
#SBATCH --tasks-per-node=1
#SBATCH --gres={gres}
#SBATCH --cpus-per-task=3
#SBATCH --time={time}
#SBATCH --mem=15000M

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load StdEnv/2023
module load python/3.12.4
module load cuda/12.6
module load swig/4.1.1
module load gcc/13.3
module load opencv/4.11.0
module load scipy-stack/2025a

source ~/Venvs/wsol_ted_loc/bin/activate

echo 'Start moving dataset into the current node....'
mkdir -p $SLURM_TMPDIR/datasets/wsol-done-right
cp ~/scratch/CAMELYON17_512.tar.gz $SLURM_TMPDIR/datasets/wsol-done-right/
tar -xf $SLURM_TMPDIR/datasets/wsol-done-right/CAMELYON17_512.tar.gz -C $SLURM_TMPDIR/datasets/wsol-done-right/
rm $SLURM_TMPDIR/datasets/wsol-done-right/CAMELYON17_512.tar.gz
echo 'Finished moving/extracting dataset into the current node....'

cd {root}
"""


def fmt(v) -> str:
    return str(v).replace('.', 'p')


def jobs(methods, folds):
    out = []
    for m in methods:
        keys = list(GRID[m].keys())
        for fold in folds:
            for ci, vals in enumerate(itertools.product(
                    *[GRID[m][k] for k in keys])):
                hp = dict(zip(keys, vals))
                tag = '_'.join(f"{k.split('_')[-2]}{fmt(v)}"
                               for k, v in hp.items())
                exp_id = (f'target_b_cl_{m}_c17_src{SOURCE_FOLD}_trg{fold}_'
                          f'{ci}_CAMELYON17_512_{WSOL}_lr0p0001_20_{tag}')
                args = COMMON + ['--fold', str(fold)] + FIXED[m]
                for k, v in hp.items():
                    args += [f'--{k}', str(v)]
                args += ['--exp_id', exp_id]
                out.append({'method': m, 'fold': fold, 'config': ci,
                            'hp': hp, 'exp_id': exp_id, 'args': args})
    return out


def exp_dir(exp_id, fold):
    base = join(REPO, 'exps', 'CAMELYON17_512', 'resnet50', 'STD_CL',
                'PixelCAM')
    return join(base, f'id_{exp_id}-tsk_STD_CL-ds_CAMELYON17_512-fold_{fold}'
                      f'-mag_None-runmode_search-mode-mth_PixelCAM-spooling_'
                      f'WGAP-arch_STDClassifier-ecd_resnet50')


def is_done(job) -> bool:
    return isfile(join(exp_dir(job['exp_id'], job['fold']), 'passed.txt'))


# ------------------------------------------------------------------ local
def run_local(js, gpus, per_gpu):
    logdir = join(REPO, 'outputjobs', 'hp_c17_litrico_ucon')
    os.makedirs(logdir, exist_ok=True)
    todo = [j for j in js if not is_done(j)]
    print(f'{len(js)} jobs, {len(js) - len(todo)} already done, '
          f'{len(todo)} to run on gpus {gpus} x {per_gpu}')
    slots = {(g, s): None for g in gpus for s in range(per_gpu)}
    failed = []
    while todo or any(p is not None for p in slots.values()):
        for key, cur in list(slots.items()):
            if cur is not None:
                p, j, fh = cur
                rc = p.poll()
                if rc is None:
                    continue
                fh.close()
                status = 'OK' if (rc == 0 and is_done(j)) else f'FAIL({rc})'
                if status != 'OK':
                    failed.append(j['exp_id'])
                print(f"[{time.strftime('%H:%M:%S')}] {status} gpu{key[0]} "
                      f"{j['exp_id']}", flush=True)
                slots[key] = None
            if slots[key] is None and todo:
                j = todo.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(key[0]))
                fh = open(join(logdir, j['exp_id'] + '.log'), 'w')
                p = subprocess.Popen([sys.executable, 'main.py'] + j['args'],
                                     cwd=REPO, env=env, stdout=fh,
                                     stderr=subprocess.STDOUT)
                slots[key] = (p, j, fh)
                print(f"[{time.strftime('%H:%M:%S')}] START gpu{key[0]} "
                      f"{j['exp_id']}", flush=True)
                time.sleep(20)  # stagger startup (copy_code, data loading).
        time.sleep(15)
    print(f'ALL DONE. failed={len(failed)}')
    for f in failed:
        print('  FAILED', f)


# ------------------------------------------------------------------ slurm
def run_slurm(js, cluster, accounts, wall, submit):
    c = CLUSTERS[cluster]
    root = c['root']
    outdir = join(root, 'configs_cc', 'generated_scripts_hp_c17_0_litrico_ucon',
                  cluster)
    logdir = join(root, 'configs_cc', 'slurm_logs', 'hp_c17_0_litrico_ucon')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    n = 0
    for i, j in enumerate([j for j in js if not is_done(j)]):
        acc = accounts[i % len(accounts)]
        name = j['exp_id']
        script = SLURM_HEADER.format(account=acc, logdir=logdir, name=name,
                                     gres=c['gres'], time=wall, root=root)
        script += 'python main.py ' + ' '.join(j['args']) + '\n'
        path = join(outdir, name + '.sh')
        with open(path, 'w') as f:
            f.write(script)
        n += 1
        if submit:
            r = subprocess.run(['sbatch', path], capture_output=True,
                               text=True)
            print(acc, r.stdout.strip() or r.stderr.strip())
    print(f'{n} scripts in {outdir} (submitted={submit})')


# ---------------------------------------------------------------- collect
def read_val_best(log_txt):
    """Best validation classification (model-selection score) from log.txt."""
    import re
    best = None
    if not isfile(log_txt):
        return None
    for m in re.finditer(r'split: valcl/metric classification: '
                         r'([0-9.]+)_best', open(log_txt).read()):
        best = float(m.group(1))
    return best


def read_test_at_best(perf_txt):
    """Test metrics of the checkpoint selected on validation."""
    import re
    d = {}
    if not isfile(perf_txt):
        return d
    txt = open(perf_txt).read()
    for k in ['classification', 'F1', 'localization']:
        m = re.search(rf'split: test/metric {k}: ([0-9.]+) ', txt)
        if m:
            d[k] = float(m.group(1))
    for k in ['PXAP', 'Dice foreground', 'MIOU']:
        m = re.search(rf'^{k}: ([0-9.]+)', txt, flags=re.M)
        if m:
            d[k] = float(m.group(1))
    return d


def collect(js, out_csv):
    rows = []
    for j in js:
        d = exp_dir(j['exp_id'], j['fold'])
        if not is_done(j):
            rows.append({'method': j['method'], 'fold': j['fold'],
                         'config': j['config'], 'status': 'missing', **j['hp']})
            continue
        val = read_val_best(join(d, 'log.txt'))
        tst = read_test_at_best(join(d, 'performance_log_best_classification.txt'))
        rows.append({'method': j['method'], 'fold': j['fold'],
                     'config': j['config'], 'status': 'done', **j['hp'],
                     'val_cls': val,
                     'test_cls': tst.get('classification'),
                     'test_f1': tst.get('F1'),
                     'test_pxap': tst.get('PXAP', tst.get('localization')),
                     'test_dice_fg': tst.get('Dice foreground'),
                     'test_miou': tst.get('MIOU'),
                     'dir': d})
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out_csv}')
    print(f"{'method':8s} {'trg':>3s} {'done':>6s} {'cfg':>4s} {'val_cls':>8s}"
          f" {'test_cls':>8s} {'F1':>6s} {'PxAP':>6s}")
    for m in METHODS:
        for fold in TARGET_FOLDS:
            rs = [r for r in rows if r['method'] == m and r['fold'] == fold]
            if not rs:
                continue
            done = [r for r in rs if r['status'] == 'done' and
                    r.get('val_cls') is not None]
            if not done:
                print(f'{m:8s} {fold:3d} {0:>3d}/{len(rs):<2d}')
                continue
            b = max(done, key=lambda r: r['val_cls'])

            def f(x):
                return f'{x:6.2f}' if isinstance(x, float) else '   n/a'
            print(f"{m:8s} {fold:3d} {len(done):>3d}/{len(rs):<2d} "
                  f"{b['config']:4d} {f(b['val_cls']):>8s} "
                  f"{f(b['test_cls']):>8s} {f(b['test_f1'])} "
                  f"{f(b['test_pxap'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['list', 'local', 'slurm', 'collect'])
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--folds', nargs='+', type=int, default=TARGET_FOLDS)
    ap.add_argument('--gpus', nargs='+', type=int, default=[0])
    ap.add_argument('--per-gpu', type=int, default=1)
    ap.add_argument('--cluster', choices=list(CLUSTERS))
    ap.add_argument('--accounts', nargs='+', default=['def-egranger'])
    ap.add_argument('--time', default='3:00:00')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--csv', default=join(REPO, 'hp_c17_litrico_ucon.csv'))
    a = ap.parse_args()

    js = jobs(a.methods, a.folds)
    if a.mode == 'list':
        for j in js:
            print(('DONE ' if is_done(j) else 'TODO ') + j['exp_id'])
        print(len(js), 'jobs')
    elif a.mode == 'local':
        run_local(js, a.gpus, a.per_gpu)
    elif a.mode == 'slurm':
        assert a.cluster, '--cluster required'
        run_slurm(js, a.cluster, a.accounts, a.time, a.submit)
    else:
        collect(js, a.csv)


if __name__ == '__main__':
    main()
