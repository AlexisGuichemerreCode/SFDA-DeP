# [SFDA-DeP: Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions]


## Abstract

Weakly Supervised Object Localization (WSOL) models enable joint classification and region-of-interest localization in histology images using only image-class supervision. When deployed in a target domain, distributions shift remains a major cause of performance degradation, especially when applied on new organs or institutions with different staining protocols and scanner characteristics. Under stronger cross-domain shifts, WSOL predictions can become biased toward dominant classes, producing highly skewed pseudo-label distributions in the target domain.  Source-Free (Unsupervised) Domain Adaptation (SFDA) methods are commonly employed to address domain shift. However, because they rely on self-training, the initial bias is reinforced over training iterations, degrading both classification and localization tasks. We identify this amplification of prediction bias as a primary obstacle to the SFDA of WSOL models in histopathology. This paper introduces  SFDA-DeP, a method inspired by machine unlearning that formulates SFDA as an iterative process of identifying and correcting prediction bias. It periodically identifies target images from over-predicted classes and selectively reduces the predictive confidence for uncertain (high entropy) images, while preserving confident predictions. This process reduces the drift of decision boundaries and bias toward dominant classes. A jointly optimized pixel-level classifier further restores discriminative localization features under distribution shift. 
%
Extensive experiments on cross-organ and -center histopathology benchmarks (GLAS, CAMEYON16, CAMELYON17) with several WSOL models show that SFDA-DeP consistently improves classification and localization over state-of-the-art SFDA baselines.
### Issues:
Please create a github issue.

### Content:
* [View](#view)
* [Requirements](#re2q)
* [Datasets](#datasets)
* [Run code](#run)

#### <a name='view'> Method</a>:

Implemented WSOL methods:
- DEEPMIL
- SAT
- LayerCAM
- PixelCAM

#### <a name='reqs'> Requirements</a>:

Quick installation to create a virtual environment using conda:
```bash
./make_venv.sh NAME_OF_YOUR_VENV
```

* [Full dependencies](dependencies/requirements.txt)
* Build and install CRF:
    * Install [Swig](http://www.swig.org/index.php)
    * CRF (not used in this work, but it is part of the code.)

```shell
cdir=$(pwd)
cd dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
cd $cdir
cd dlib/crf/crfwrapper/colorbilateralfilter
swig -python -c++ colorbilateralfilter.i
python setup.py install
```

#### <a name="datasets"> Download datasets </a>:
#### 2.1. Links to dataset:
* [GlaS](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest)
* [Camelyon16](https://github.com/jeromerony/survey_wsl_histology)


#### 2.2. Download datasets:

* GlaS: [./download-glas-dataset.sh](./download-glas-dataset.sh).

You find the splits in [./folds](./folds).


### 2.3 Code for datasets split/sampling (+ patches sampling from WSI):
* See [datasets-split](https://github.com/jeromerony/survey_wsl_histology/tree/init-branch/datasets-split).
* Detailed documentation: [datasets-split/README.md](https://github.com/jeromerony/survey_wsl_histology/blob/init-branch/datasets-split/README.md).

#### <a name="run"> Run code </a>:

E.g. PixelCAM method via CAM WSOL method.

1- Train on data GlaS:

* LayerCAM-method: LayerCAM over GlaS using ResNet50:

```shell
#!/usr/bin/env bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate da

# ==============================================================================
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

export OMP_NUM_THREADS=50
python main.py \
       --task STD_CL \
       --encoder_name resnet50 \
       --arch STDClassifier \
       --spatial_dropout 0.1 \
       --runmode search-mode \
       --opt__name_optimizer sgd \
       --batch_size 32 \
       --eval_batch_size 64 \
       --eval_checkpoint_type best_localization \
       --opt__step_size 5 \
       --opt__gamma 0.1 \
       --max_epochs 1000 \
       --freeze_cl False \
       --support_background True \
       --method LayerCAM \
       --spatial_pooling WGAP \
       --dataset GLAS \
       --fold 0 \
       --cudaid $cudaid \
       --debug_subfolder None \
       --amp True \
       --opt__lr 0.001 \
       --sf_uda False \
       --exp_id 01_12_2024_09_23_52_900932__4871059
```

From the folder of the experiment `01_02_2024_09_23_52_900932__4871059`,
copy one of the checkpoints folders to the folder `pretrained`
`GlaS-0-resnet50-LayerCAM-WGAP-cp_best_localization` or
`GlaS-0-resnet50-LayerCAM-WGAP-cp_best_classification`.

Store the CAMs of the training dataset in the folder data_cams

2- Train PixelCAM on data GlaS:

```shell
#!/usr/bin/env bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate da

# ==============================================================================
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

export OMP_NUM_THREADS=50
python main.py \
       --task STD_CL \
       --encoder_name resnet50 \
       --arch STDClassifier \
       --spatial_dropout 0.1 \
       --runmode search-mode \
       --opt__name_optimizer sgd \
       --batch_size 32 \
       --eval_batch_size 64 \
       --eval_checkpoint_type best_localization \
       --opt__step_size 5 \
       --opt__gamma 0.1 \
       --max_epochs 1000 \
       --freeze_cl False \
       --support_background True \
       --method PixelCAM \
       --spatial_pooling WGAP \
       --dataset GLAS \
       --fold 0 \
       --cudaid $cudaid \
       --debug_subfolder None \
       --amp True \
       --opt__lr 0.001 \
       --ece True \
       --ece_lambda 1.0 \
       --neg_samples_partial False \
       --sl_pc_seeder probability_negative_salloarea_seeder \
       --sl_min 5 \
       --sl_max 5 \
       --sl_min_p 0.4 \
       --path_cam  resnet50-layercam-bloc-glas \
       --path_pre_trained_model_cl GLAS-0-resnet50-LayerCAM-WGAP-cp_best_classification \
       --exp_id 01_12_2024_09_25_14_467534__5485897
```

3- Adapt PixelCAM on data CAMELYON16:

```shell
#!/usr/bin/env bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate da

# ==============================================================================
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

export OMP_NUM_THREADS=50
python main.py \
       --task STD_CL \
       --encoder_name resnet50 \
       --arch STDClassifier \
       --spatial_dropout 0.1 \
       --runmode search-mode \
       --opt__name_optimizer sgd \
       --batch_size 32 \
       --eval_batch_size 64 \
       --eval_checkpoint_type best_localization \
       --opt__step_size 5 \
       --opt__gamma 0.1 \
       --max_epochs 1000 \
       --freeze_cl False \
       --support_background True \
       --method PixelCAM \
       --spatial_pooling WGAP \
       --dataset GLAS \
       --fold 0 \
       --cudaid $cudaid \
       --debug_subfolder None \
       --amp True \
       --opt__lr 0.001 \
       --sf_uda_source_ds CAMELYON16 \
       --sf_uda_source_ds_fold 0 \
       --sf_uda_source_encoder_name resnet50 \
       --sf_uda_source_checkpoint_type best_classification \
       --sf_uda_source_wsol_method PixelCAM \
       --sf_uda_source_wsol_arch STDClassifier \
       --sf_uda_source_wsol_spatial_pooling WGAP \
       --esfda True \
       --esfda_select_imgs True \
       --esfda_select_imgs_ratio 0.15 \
       --retain_all_others True \
       --dynamic_selection False \
       --resample_every 2 \
       --esfda_flip_labels True \
       --CEForget_lambda 1.0 \
       --esfda_notflip_labels True \
       --CERetain_lambda 1.0 \
       --ece_adapt True \
       --ece_adapt_lambda 1.0 \
       --neg_samples_partial True \
       --sl_pc_seeder probability_negative_salloarea_seeder \
       --sl_min 5 \
       --sl_max 5 \
       --sl_min_p 0.4 \
       --entropy_filter_mode True \
       --keep_ratio 0.15 \
       --exp_id 019_03_2026_09_25_14_467534__5485897
```



