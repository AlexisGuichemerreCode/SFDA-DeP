from calendar import EPOCH
from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import subprocess
from pathlib import Path
import datetime as dt
import argparse
# import more_itertools as mit
from dlib.sf_uda import adadsa
import torch.nn as nn
import pandas as pd
import csv


import numpy as np
import numpy
from tqdm import tqdm
# import pretrainedmodels.utils
import yaml
import munch
import pickle
# from texttable import Texttable
from dlib.datasets.wsol_data_core import get_mask
from PIL import Image
import torch
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import seaborn as sns



from collections import defaultdict



# root_dir = dirname(dirname(dirname(abspath(__file__))))
root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)
import torch.nn.functional as F

from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from  dlib.configure import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_tag
from dlib.utils.tools import Dict2Obj
# from dlib.utils.tools import log_device
from dlib.configure import config

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model, get_pretrainde_classifier


from dlib.datasets.wsol_loader_natural import get_data_loader_natural

from dlib.datasets.wsol_loader import get_data_loader
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets.wsol_loader import get_class_labels
from dlib.datasets.wsol_loader import get_image_ids
from dlib.learning.train_wsol import Basic, PerformanceMeter
from dlib.utils.tools import get_cpu_device
from dlib.process.parseit import str2bool
from dlib.utils.tools import t2n
import cv2
import json
from glob import glob

class IgnoreKeyLoader(yaml.SafeLoader):
    def ignore_keys(self, node):
        ignore_key = 'best_valid_tau_cl'
        if isinstance(node, yaml.MappingNode):
            i = 0
            while i < len(node.value):
                if node.value[i][0].value == ignore_key:
                    del node.value[i]
                else:
                    i += 1
        return self.construct_yaml_map(node)
    
    def ignore_numpy_scalars(self, node):
        return None  # or any other dummy value
    
IgnoreKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, IgnoreKeyLoader.ignore_keys)
IgnoreKeyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', IgnoreKeyLoader.ignore_numpy_scalars)


def cl_forward(args, model, images):

    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits

def _compute_accuracy_distrib(args, model, loader):
    import torch.nn.functional as F
    from collections import Counter

    num_correct = 0
    num_images = 0

    num_correct_normal = 0
    num_images_normal = 0

    num_correct_cancer = 0
    num_images_cancer = 0

    entropies = []
    class_pred_counts = Counter()  # pour compter les classes prédictes

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

            probs = F.softmax(cl_logits, dim=1)  # [B, C]
            log_probs = torch.log_softmax(cl_logits, dim=1)

            entropy = - torch.sum(probs * log_probs, dim=1)  # [B]
            entropies.extend(entropy.cpu().tolist())

            # Comptage des classes prédites
            for p in pred.cpu().tolist():
                class_pred_counts[p] += 1

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

        for j in range(len(targets)):
            if targets[j] == 0:
                num_images_normal += 1
                if pred[j] == targets[j]:
                    num_correct_normal += 1
            elif targets[j] == 1:
                num_images_cancer += 1
                if pred[j] == targets[j]:
                    num_correct_cancer += 1
            else:
                raise ValueError("Unknown class label")

    classification_acc_normal = num_correct_normal / float(num_images_normal) * 100 if num_images_normal > 0 else 0
    classification_acc_cancer = num_correct_cancer / float(num_images_cancer) * 100 if num_images_cancer > 0 else 0
    classification_acc = num_correct / float(num_images) * 100

    # Distribution prédite normalisée
    total_preds = sum(class_pred_counts.values())
    class_distribution = [
        class_pred_counts[i] / total_preds if total_preds > 0 else 0.0
        for i in range(args.num_classes)
    ]

    return classification_acc, classification_acc_normal, classification_acc_cancer, entropies, class_distribution

def _compute_accuracy(args, model, loader):
    num_correct = 0
    num_images = 0

    num_correct_normal = 0
    num_images_normal = 0

    num_correct_cancer = 0
    num_images_cancer = 0

    entropies = []

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

            probs = F.softmax(cl_logits, dim=1)  # [B, C]
            log_probs = torch.log_softmax(cl_logits, dim=1)

            entropy = - torch.sum(probs * log_probs, dim=1)  # [B]
            entropies.extend(entropy.cpu().tolist())

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

        # Compute accuracy for each class
        for j in range(len(targets)):
            if targets[j] == 0:
                num_images_normal += 1
                if pred[j] == targets[j]:
                    num_correct_normal += 1
            elif targets[j] == 1:
                num_images_cancer += 1
                if pred[j] == targets[j]:
                    num_correct_cancer += 1
            else:
                raise ValueError("Unknown class label")
            
    # Compute accuracy for each class
    classification_acc_normal = num_correct_normal / float(num_images_normal) * 100 if num_images_normal > 0 else 0
    classification_acc_cancer = num_correct_cancer / float(num_images_cancer) * 100 if num_images_cancer > 0 else 0


    classification_acc = num_correct / float(num_images) * 100
    
    return classification_acc, classification_acc_normal, classification_acc_cancer, entropies


def img_entropy(args, model, loader):
    model.eval()
    entropies = []
    #predictions = []
    #labels = []


    for i, (images, targets, *_) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)  # [B, C]
            probs = F.softmax(cl_logits, dim=1)  # [B, C]
            log_probs = torch.log_softmax(cl_logits, dim=1)

            entropy = - torch.sum(probs * log_probs, dim=1)  # [B]
            entropies.extend(entropy.cpu().tolist())
            #predictions.extend(cl_logits.argmax(dim=1).cpu().tolist())
            #labels.extend(targets.cpu().tolist())

    return entropies

def load_model(exp_path,dataset,checkpoint_type, cudaid, tmp_outd='tmp_outd', parsedargs=None):
    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['model']['folder_pre_trained_cl'] = None
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    os.makedirs(args.outd, exist_ok=True)

    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    tag = get_tag(args, checkpoint_type=checkpoint_type)

    msg = 'Task: {} \t box_v2_metric: {} \t' \
        'Dataset: {} \t Method: {} \t ' \
        'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
                                args.method, args.model['encoder_name'])
    encoder_name = args.model['encoder_name']
    method_name = args.method
    source_dataset = dataset
    
    # DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(cudaid))

    tag = get_tag(args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)
    with open(join(path_cl, 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        # args_dict['model']['freeze_encoder'] = False
        args_dict['pixel_wise_classification'] = args_dict.get('pixel_wise_classification', False)
        args_dict['model']['folder_pre_trained_cl'] = None
        args_dict['multiple_layer_pixel_classifier'] = False
        args_dict['anchors_ortogonal'] = False
        args_dict['detach_pixel_classifier'] = False
        args_dict['batch_norm_pixel_classifier'] = False
        args_dict['one_layer_pixel_classifier'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    args.sf_uda = False

    model = get_model(args)[0]

    print(f'Loading model for {method_name}-{encoder_name} from {path_cl}')
    if parsedargs.external_model == None:
        if "tscam" in encoder_name:
            model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

            model.load_state_dict(model_tscam, strict=True)
        elif "sat" in encoder_name:
            model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

            model.load_state_dict(model_tscam, strict=True)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)

            if method_name == constants.METHOD_PIXELCAM:    #'EnergyCAM': constants.METHOD_ENERGY:
                header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
                                map_location=get_cpu_device())
                model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)
    else:
        path_eternal_cl = parsedargs.external_model
        encoder_w = torch.load(join(path_eternal_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_eternal_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)



        if method_name == constants.METHOD_PIXELCAM:    #'EnergyCAM': constants.METHOD_ENERGY:
            header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
                            map_location=get_cpu_device())
            model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)

    return model


def load_loader(exp_path,dataset,checkpoint_type, cudaid, split, tmp_outd='tmp_outd', parsedargs=None):
    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['model']['folder_pre_trained_cl'] = None
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    os.makedirs(args.outd, exist_ok=True)

    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    tag = get_tag(args, checkpoint_type=checkpoint_type)

    msg = 'Task: {} \t box_v2_metric: {} \t' \
        'Dataset: {} \t Method: {} \t ' \
        'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
                                args.method, args.model['encoder_name'])
    encoder_name = args.model['encoder_name']
    method_name = args.method
    source_dataset = dataset
    
    # DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(cudaid))

    tag = get_tag(args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)
    with open(join(path_cl, 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        # args_dict['model']['freeze_encoder'] = False
        args_dict['pixel_wise_classification'] = args_dict.get('pixel_wise_classification', False)
        args_dict['model']['folder_pre_trained_cl'] = None
        args_dict['multiple_layer_pixel_classifier'] = False
        args_dict['anchors_ortogonal'] = False
        args_dict['detach_pixel_classifier'] = False
        args_dict['batch_norm_pixel_classifier'] = False
        args_dict['one_layer_pixel_classifier'] = False
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    args.sf_uda = False

    basic_config = config.get_config(ds=source_dataset, fold=args.fold, magnification=args.magnification)

    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']
    args.cam_curve_interval = basic_config['cam_curve_interval']

    ####################################################################################
    ###############load performance log file from orignal exp checkpoint ###############
    ####################################################################################
    assert split == constants.TESTSET or split == constants.VALIDSET or split == constants.TRAINSET
    #split == constants.VALIDSET
    #split = constants.VALIDSET
    log_file_path_best_loc = os.path.join(exp_path, f'performance_log_{checkpoint_type}.pickle')
    if os.path.isfile(log_file_path_best_loc):
        with open(log_file_path_best_loc, 'rb') as f:
            results = pickle.load(f)
        # if split == constants.TESTSET:
        #     clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][-1]
        #     loc_acc_from_orginal_exp_path = results[split]['localization_IOU_50']['value_per_epoch'][-1]
        # else:
        best_epoch = -1 if split == constants.TESTSET else results[split][checkpoint_type.replace('best_', '')]['best_epoch']
        #clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][best_epoch]
        #loc_acc_from_orginal_exp_path = results[split]['localization']['value_per_epoch'][best_epoch]


    ####################################################################################
    ####################################################################################
    DLLogger.flush()
    
    metadata_root = join(constants.RELATIVE_META_ROOT, source_dataset, f"fold-{args.fold}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    domain_data_paths = config.configure_data_paths(args_dict, source_dataset)

    #target_metadata_root = join('./folds/wsol-done-right-splits', source_dataset, f"fold-{args.fold}")
    # args_dict['data_root'] = '/export/gauss/vision/Aguichemerre/datasets'
    #target_domain_data_paths = config.configure_data_paths(args_dict, source_dataset)

    loaders = get_data_loader(
            data_roots=domain_data_paths,
            metadata_root=metadata_root,
            batch_size=32,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[split],
            #constants.TRAINSET
            eval_batch_size = 32#args.eval_batch_size,
        )


    return loaders, metadata_root, domain_data_paths, args



def measure_model_diff(exp_path_source,exp_path_target, checkpoint_type, source_dataset,target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None):

    source_model = load_model(exp_path_source, source_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    target_model = load_model(exp_path_target, target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)

    device = torch.device('cuda:{}'.format(cudaid))
    source_model.to(device)
    target_model.to(device)

    state_dict_source_model = source_model.state_dict()
    state_dict_target_model = target_model.state_dict()
    
    layer_diffs = {}
    for name in state_dict_source_model:
        if name in state_dict_target_model:
            weight1 = state_dict_source_model[name]
            weight2 = state_dict_target_model[name]

            if weight1.dtype in (torch.float32, torch.float64) and weight2.dtype in (torch.float32, torch.float64):
                diff = torch.norm(weight1 - weight2, p=2).item()
                layer_diffs[name] = diff
            else:
                print(f"Skipping {name} (dtype: {weight1.dtype})")

    for layer_name, l2_norm in layer_diffs.items():
        print(f"{layer_name}: L2 diff = {l2_norm:.4f}")

    for k, v in layer_diffs.items():
        if not (isinstance(k, str) and isinstance(v, (int, float))):
            print(f"Invalid Input : {k} -> type: {type(v)}")


    grouped = defaultdict(list)

    for param_name, l2_diff in layer_diffs.items():
        try:
            layer = param_name.split('.')[1]  # ex: 'layer1'
            grouped[layer].append((param_name, l2_diff))
        except Exception as e:
            print(f"Erreur dans le nom: {param_name} ({e})")


    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    source_model_name = parsedargs.source_model_name
    target_model_name = parsedargs.target_model_name

    output_dir = os.path.join('visualization', 'model_difference', source_model_name, source_dataset)
    os.makedirs(output_dir, exist_ok=True)

    for i, (layer_name, param_list) in enumerate(grouped.items()):
        param_names = [p[0] for p in param_list]
        diffs = [p[1] for p in param_list]

        plt.figure(figsize=(16, 5))
        plt.bar(param_names, diffs, color=colors[i % len(colors)])
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("L2 Difference")
        plt.title(f"L2 difference of parameters in {layer_name}")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir,f"{source_dataset}_with_{source_model_name}_to_{target_dataset}_on_{target_model_name}_diff_{layer_name}.png"), dpi=300)
        plt.close()

    return 0


def measure_entropy(exp_path_source,exp_path_target, checkpoint_type, source_dataset,target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, args = None, multiple_model = None):

    source_model = load_model(exp_path_source, source_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    #target_model = load_model(exp_path_target, target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)

    new_model = deepcopy(source_model)

    device = torch.device('cuda:{}'.format(cudaid))
    source_model.to(device)
    #target_model.to(device)
    new_model.to(device)

    


    source_model.eval()
    new_model.eval()


    #target_loader, target_metadata_root, target_domain_data_paths, args = load_loader(exp_path_target, target_dataset, checkpoint_type, cudaid, split, tmp_outd=tmp_outd, parsedargs=parsedargs)
    source_loader, source_metadata_root, source_domain_data_paths, args = load_loader(exp_path_source, source_dataset, checkpoint_type, cudaid, split, tmp_outd=tmp_outd, parsedargs=parsedargs)

    cl_performance, class_0_performance, class_1_performance, entropies = _compute_accuracy(args, source_model, source_loader[split])
    print(f"[Initial Classification performance: {cl_performance}")
    print(f"[Initial Normal Classification performance: {class_0_performance}")
    print(f"[Initial Cancer Classification performance: {class_1_performance}")
    print(f"Initial Entropy performance: {np.mean(entropies)}")

    #entropies = img_entropy(args, source_model, source_loader[split])

    #print(f"Initial Entropy performance: {np.mean(entropies)}")

    return 0


def find_worst_rank1_entropy(base_path):
    worst_entropy = -1.0  # On veut la plus haute entropie
    worst_folder = None

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        file_path = os.path.join(folder_path, 'best_models_entropy.txt')
        if not os.path.isfile(file_path):
            print(f"[⚠️] Fichier manquant dans : {folder_name}")
            continue

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Skip header and parse the first line (Rank = 1)
            rank1_line = lines[1].strip().split()
            if len(rank1_line) < 3:
                print(f"[⚠️] Format invalide dans {file_path}")
                continue

            entropy = float(rank1_line[2])

            if entropy > worst_entropy:
                worst_entropy = entropy
                worst_folder = folder_name

        except Exception as e:
            print(f"[❌] Erreur avec {file_path}: {e}")

    if worst_folder:
        print(f"\n📁 Dossier avec la plus haute entropie (Rank 1): {worst_folder}")
        print(f"🔺 Entropie = {worst_entropy:.6f}")
        return worst_folder, worst_entropy
    else:
        print("Aucun fichier valide trouvé.")
        return None, None
    

def measure_entropy_all_checkpoints(
    exp_path_source,
    source_dataset,
    cudaid,
    split,
    tmp_outd='tmp_outd',
    parsedargs=None,
    #args=None,
    output_csv='entropy_performance.csv',
    max_epoch=20
):
    # Liste complète des checkpoints à évaluer
    checkpoint_types = ['best_classification', 'best_localization']
    checkpoint_types += [f'B-EPOCH{i}' for i in range(1, 11)]
    #checkpoint_types += [f'B-EPOCH{i}' for i in range(1, 2)]

    # Charger une seule fois le loader
    source_loader, _, _, args = load_loader(
        exp_path_source, source_dataset, checkpoint_types[0], cudaid, split,
        tmp_outd=tmp_outd, parsedargs=parsedargs
    )

    # Coolect
    results = []

    for checkpoint_type in checkpoint_types:
        try:
            model = load_model(
                exp_path_source, source_dataset, checkpoint_type, cudaid,
                tmp_outd=tmp_outd, parsedargs=parsedargs
            )
        except Exception as e:
            print(f"[⚠️] Erro while loading checkpoint {checkpoint_type}: {e}")
            continue

        device = torch.device(f'cuda:{cudaid}')
        model = deepcopy(model).to(device).eval()

        try:
            cl_perf, norm_perf, cancer_perf, entropies = _compute_accuracy(
                args, model, source_loader[split]
            )
        except Exception as e:
            print(f"[⚠️] Échec du calcul des métriques pour {checkpoint_type}: {e}")
            continue

        mean_entropy = float(np.mean(entropies))
        print(f"[{checkpoint_type}] CL: {cl_perf:.2f}, Normal: {norm_perf:.2f}, Cancer: {cancer_perf:.2f}, Entropy: {mean_entropy:.4f}")

        results.append({
            'checkpoint': checkpoint_type,
            'classification': cl_perf,
            'normal_class': norm_perf,
            'cancer_class': cancer_perf,
            'entropy': mean_entropy
        })

    # Écriture des résultats dans un fichier CSV
    csv_path = os.path.join(exp_path_source, output_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'classification', 'normal_class', 'cancer_class', 'entropy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Résultats enregistrés dans : {csv_path}")





def average_state_dicts(state_dicts):
    """Calcule la moyenne pondérée (uniforme) de plusieurs state_dicts."""
    avg_state_dict = {}
    num_models = len(state_dicts)

    for key in state_dicts[0]:
        avg_state_dict[key] = sum(d[key] for d in state_dicts) / num_models

    return avg_state_dict


def measure_ensembled_entropy_from_checkpoints(
    exp_path_source,
    source_dataset,
    cudaid,
    split,
    tmp_outd='tmp_outd',
    parsedargs=None,
    output_csv='ensembled_entropy.csv',
    max_epoch=10
):
    # Chargement du loader (une fois)
    loader, _, _, args = load_loader(
        exp_path_source, source_dataset, 'B-EPOCH1', cudaid, split,
        tmp_outd=tmp_outd, parsedargs=parsedargs
    )

    device = torch.device(f'cuda:{cudaid}')
    model_ref = load_model(
        exp_path_source, source_dataset, 'B-EPOCH1', cudaid,
        tmp_outd=tmp_outd, parsedargs=parsedargs
    ).to(device)

    model_class = type(model_ref)
    results = []
    loaded_state_dicts = []

    for j in range(1, max_epoch + 1):
        checkpoint_name = f'B-EPOCH{j}'

        try:
            model_j = load_model(
                exp_path_source, source_dataset, checkpoint_name, cudaid,
                tmp_outd=tmp_outd, parsedargs=parsedargs
            )
            state_dict_j = model_j.state_dict()
            loaded_state_dicts.append(state_dict_j)
        except Exception as e:
            print(f"[⚠️] Échec chargement {checkpoint_name}: {e}")
            continue

        # Moyennage progressif
        averaged_dict = average_state_dicts(loaded_state_dicts)

        # Charger dans un nouveau modèle
        ensembled_model = model_ref.to(device)
        ensembled_model.load_state_dict(averaged_dict)
        ensembled_model.eval()

        try:
            cl_perf, norm_perf, cancer_perf, entropies = _compute_accuracy(
                args, ensembled_model, loader[split]
            )
        except Exception as e:
            print(f"[⚠️] Échec évaluation de l'ensemble 1..{j}: {e}")
            continue

        mean_entropy = float(np.mean(entropies))
        print(f"[Ensemble B-EPOCH1..{j}] CL: {cl_perf:.2f}, Normal: {norm_perf:.2f}, Cancer: {cancer_perf:.2f}, Entropy: {mean_entropy:.4f}")

        results.append({
            'ensemble_until': f'B-EPOCH1_to_{j}',
            'classification': cl_perf,
            'normal_class': norm_perf,
            'cancer_class': cancer_perf,
            'entropy': mean_entropy
        })

    # Sauvegarde CSV
    csv_path = os.path.join(exp_path_source, output_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ensemble_until', 'classification', 'normal_class', 'cancer_class', 'entropy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Résultats de l'ensembling progressif enregistrés dans : {csv_path}")


import csv

def load_alpha_weights_csv(csv_path):
    alpha_weights = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                checkpoint = row['checkpoint']
                kl = float(row['kl_target_uniform'])
                if kl > 0:
                    weight = 1.0 / kl
                    alpha_weights[checkpoint] = weight
            except Exception as e:
                print(f"[⚠️] Erreur dans la ligne du CSV : {e}")
                continue

    # Normalisation
    total = sum(alpha_weights.values())
    for k in alpha_weights:
        alpha_weights[k] /= total

    return alpha_weights



def evaluate_weighted_ensemble_model(
    exp_path_source,
    source_dataset,
    target_dataset,
    cudaid,
    split='valid',
    alpha_csv_path='entropy_performance_combined.csv',
    merged_model_name='merged_model.pt',
    tmp_outd='tmp_outd',
    parsedargs=None
):
    import os
    import torch
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    import csv

    # === Charger les poids alpha depuis le CSV ===
    alpha_dict = {}
    epsilon = 1e-6
    
    with open(os.path.join(exp_path_source, alpha_csv_path), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ckpt = row['checkpoint']
                acc = float(row['classification'])              # accuracy source
                kl = float(row['kl_target_uniform'])            # KL target
                if kl > 0 and acc > 0:
                    score = acc * (1.0 / (kl + epsilon))
                    alpha_dict[ckpt] = score
            except Exception as e:
                print(f"[⚠️] Erreur lecture ligne {ckpt} : {e}")
    # === Trier les checkpoints par score décroissant ===
    sorted_checkpoints = sorted(alpha_dict.items(), key=lambda x: -x[1])
    checkpoint_list = [ckpt for ckpt, _ in sorted_checkpoints]

    device = torch.device(f'cuda:{cudaid}')

    results = []

    cumulative_weights = []
    cumulative_states = []

    # === Préparer la sauvegarde CSV ===
    csv_path = os.path.join(exp_path_source, 'progressive_merge_evaluation.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'n_models', 'merged_checkpoints',
            'src_classification', 'src_normal', 'src_cancer', 'src_entropy',
            'tgt_classification', 'tgt_normal', 'tgt_cancer'
        ])
        writer.writeheader()

    for i in range(len(checkpoint_list)):
        ckpt = checkpoint_list[i]
        print(f"[🔁] Chargement checkpoint : {ckpt}")
        try:
            model = load_model(
                exp_path_source, source_dataset, ckpt, cudaid,
                tmp_outd=tmp_outd, parsedargs=parsedargs
            )
            state_dict = model.state_dict()
            cumulative_states.append(state_dict)
            cumulative_weights.append(alpha_dict[ckpt])

            # === Fusion pondérée cumulative ===
            weight_list = [w / sum(cumulative_weights) for w in cumulative_weights]
            avg_state_dict = {}
            for key in state_dict:
                avg_state_dict[key] = sum(w * sd[key] for w, sd in zip(weight_list, cumulative_states))

            # === Charger modèle de référence pour structure ===
            model_ref = deepcopy(model).to(device)
            model_ref.load_state_dict(avg_state_dict)
            model_ref.eval()

            # === Évaluation sur la source ===
            source_loader, _, _, args = load_loader(
                exp_path_source, source_dataset, ckpt, cudaid, split,
                tmp_outd=tmp_outd, parsedargs=parsedargs
            )
            cl_src, norm_src, cancer_src, entropies = _compute_accuracy(args, model_ref, source_loader[split])
            mean_entropy_src = float(np.mean(entropies))

            # === Évaluation sur la target ===
            target_loader, _, _, _ = load_loader(
                exp_path_source, target_dataset, ckpt, cudaid, split,
                tmp_outd=tmp_outd, parsedargs=parsedargs
            )
            cl_tgt, norm_tgt, cancer_tgt, _ = _compute_accuracy(args, model_ref, target_loader[split])

            print(f"✅ Merge {i+1} modèles: {checkpoint_list[:i+1]}")
            print(f"→ Source: CL={cl_src:.2f}, Normal={norm_src:.2f}, Cancer={cancer_src:.2f}, Entropy={mean_entropy_src:.4f}")
            print(f"→ Target: CL={cl_tgt:.2f}, Normal={norm_tgt:.2f}, Cancer={cancer_tgt:.2f}")

            writer.writerow({
                'n_models': i + 1,
                'merged_checkpoints': '|'.join(checkpoint_list[:i+1]),
                'src_classification': cl_src,
                'src_normal': norm_src,
                'src_cancer': cancer_src,
                'src_entropy': mean_entropy_src,
                'tgt_classification': cl_tgt,
                'tgt_normal': norm_tgt,
                'tgt_cancer': cancer_tgt
            })

        except Exception as e:
            print(f"[❌] Échec pour {ckpt} : {e}")
            continue

def compute_kl_divergence(pred_dist, epsilon=1e-12):
    """
    Calcule KL(pred_dist || uniforme)
    :param pred_dist: liste ou tableau numpy des fréquences par classe, ex. [0.3, 0.7]
    :param epsilon: petite valeur pour éviter log(0)
    :return: float (KL divergence)
    """
    pred_dist = np.array(pred_dist) + epsilon  # pour éviter log(0)
    pred_dist = pred_dist / pred_dist.sum()    # normalisation de sécurité

    num_classes = len(pred_dist)
    uniform = np.ones(num_classes) / num_classes

    kl = np.sum(pred_dist * np.log(pred_dist / uniform))
    return float(kl)

def measure_entropy_all_checkpoints_dual(
    exp_path_source,
    source_dataset,
    target_dataset,
    cudaid,
    split='valid',
    tmp_outd='tmp_outd',
    parsedargs=None,
    output_csv='entropy_performance_combined.csv',
    max_epoch=10
):
    import os, csv
    import numpy as np
    import torch
    from copy import deepcopy

    # ⚠️ Tu dois avoir ces deux fonctions dans utils ou localement

    checkpoint_types = ['best_classification', 'best_localization']
    checkpoint_types += [f'B-EPOCH{i}' for i in range(1, max_epoch + 1)]

    # Loaders
    source_loader, _, _, args = load_loader(
        exp_path_source, source_dataset, checkpoint_types[0], cudaid, split,
        tmp_outd=tmp_outd, parsedargs=parsedargs
    )
    target_loader, _, _, _ = load_loader(
        exp_path_source, target_dataset, checkpoint_types[0], cudaid, split,
        tmp_outd=tmp_outd, parsedargs=parsedargs
    )

    results = []

    for checkpoint_type in checkpoint_types:
        try:
            model = load_model(
                exp_path_source, source_dataset, checkpoint_type, cudaid,
                tmp_outd=tmp_outd, parsedargs=parsedargs
            )
        except Exception as e:
            print(f"[⚠️] Erreur au chargement du checkpoint {checkpoint_type}: {e}")
            continue

        device = torch.device(f'cuda:{cudaid}')
        model = deepcopy(model).to(device).eval()

        try:
            cl_perf, norm_perf, cancer_perf, _ = _compute_accuracy(
                args, model, source_loader[split]
            )
        except Exception as e:
            print(f"[⚠️] Échec du calcul des performances source pour {checkpoint_type}: {e}")
            continue

        # ➤ Distribution prédite sur la target (sans labels)
        try:
            _, _, _, _, pred_dist = _compute_accuracy_distrib(
                args, model, target_loader[split]
            )
        except Exception as e:
            print(f"[⚠️] Échec du calcul de la distribution sur la cible pour {checkpoint_type}: {e}")
            pred_dist = None

        # ➤ KL divergence  
        try:
            kl_score = compute_kl_divergence(pred_dist)
        except Exception as e:
            print(f"[⚠️] Échec du calcul de la KL pour {checkpoint_type}: {e}")
            kl_score = None

        print(f"[{checkpoint_type}] Acc. source: {cl_perf:.2f}, KL(target ∥ uniform): {kl_score:.4f}" if kl_score is not None else "")

        results.append({
            'checkpoint': checkpoint_type,
            'classification': cl_perf,
            'normal_class': norm_perf,
            'cancer_class': cancer_perf,
            'kl_target_uniform': kl_score if kl_score is not None else -1
        })

    csv_path = os.path.join(exp_path_source, output_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'classification', 'normal_class', 'cancer_class', 'kl_target_uniform'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Résultats enregistrés dans : {csv_path}")


def avering_models(exp_path_source,exp_path_target, checkpoint_type, source_dataset,target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, args = None, multiple_model=None):

    model_0 = load_model(multiple_model[0], source_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    model_1 = load_model(multiple_model[1], target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    model_2 = load_model(multiple_model[2], target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    model_3 = load_model(multiple_model[3], target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    model_4 = load_model(multiple_model[4], target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)

    models = [model_0, model_1, model_2, model_3, model_4]

    device = torch.device('cuda:{}'.format(cudaid))

    avg_state_dict = deepcopy(models[0].state_dict())


    for key in avg_state_dict.keys():
        for i in range(1, len(models)):
            avg_state_dict[key] += models[i].state_dict()[key]
        avg_state_dict[key] = avg_state_dict[key] / len(models)

    
    new_model = deepcopy(model_0)
    new_model.load_state_dict(avg_state_dict)

    new_model.to(device)

    new_model.eval()

    #target_loader, target_metadata_root, target_domain_data_paths, args = load_loader(exp_path_target, target_dataset, checkpoint_type, cudaid, split, tmp_outd=tmp_outd, parsedargs=parsedargs)
    source_loader, source_metadata_root, source_domain_data_paths, args = load_loader(exp_path_source, source_dataset, checkpoint_type, cudaid, split, tmp_outd=tmp_outd, parsedargs=parsedargs)

    cl_performance, class_0_performance, class_1_performance, entropies = _compute_accuracy(args, new_model, source_loader[split])
    print(f"[Initial Classification performance: {cl_performance}")
    print(f"[Initial Normal Classification performance: {class_0_performance}")
    print(f"[Initial Cancer Classification performance: {class_1_performance}")
    print(f"Initial Entropy performance: {np.mean(entropies)}")

    #entropies = img_entropy(args, source_model, source_loader[split])

    #print(f"Initial Entropy performance: {np.mean(entropies)}")

    return 0




    
def fast_eval():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default=None)
    # parser.add_argument("--exp_path", type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')
    parser.add_argument('--noise_level_for_eval_with_noisy_bbox', nargs='+',
                        type=int, default=[5, 10, 15, 20, 25, 30, 35 ,40, 45, 50])
    parser.add_argument("--target_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--image_ids_to_draw_target', nargs='+', type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, default=None, help="Source dataset")
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_target", type=str, default=None, help="Path to the pre-trained target model.")

    parser.add_argument("--path_pre_trained_source_1", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_source_2", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_source_3", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_source_4", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument("--path_pre_trained_source_5", type=str, default=None, help="Path to the pre-trained source model.")

    parser.add_argument("--external_model", type=str, default=None, help="Path to the external bb+cl.")
    parser.add_argument("--source_model_name", type=str, default=None, help="Name of source model.")
    parser.add_argument("--target_model_name", type=str, default=None, help="Name of target model.")

    parser.add_argument("--path_folder_models", type=str, default=None, help="path for multiple models entropy with best cl or loc.")


    parsedargs = parser.parse_args()
    
    os.makedirs('tmp_outd', exist_ok=True)  
    log_backends = [
                # ArbJSONStreamBackend(Verbosity.VERBOSE,
                #                     join(args.outd, "log.json")),
                ArbTextStreamBackend(Verbosity.VERBOSE,
                                    join('tmp_outd', f"split_{parsedargs.split}_log.txt")),
            ]
    
    _VERBOSE = True
    if _VERBOSE:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
        
    DLLogger.GLOBAL_LOGGER = DLLogger.NotInitializedObject()
        
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())
    ##########

    base_checkpoint_types = [parsedargs.checkpoint_type]
        
    for checkpoint_type_extended in base_checkpoint_types:
        checkpoint_type = checkpoint_type_extended
        
        split = parsedargs.split
        assert split == constants.TESTSET or split == constants.VALIDSET or split == constants.TRAINSET
        
        _CODE_FUNCTION = 'fast_eval_{}'.format(split)

        #target_methods = ['SOURCE', 'SFDE', 'SHOT', 'CDCL', 'ADADSA']
        target_methods = ['SOURCE']

        exp_path_source = parsedargs.path_pre_trained_source
        exp_path_target = parsedargs.path_pre_trained_target

        multiple_model = [parsedargs.path_pre_trained_source_1,
                          parsedargs.path_pre_trained_source_2,
                          parsedargs.path_pre_trained_source_3,
                          parsedargs.path_pre_trained_source_4,
                          parsedargs.path_pre_trained_source_5]
        
        # exp_path_source = parsedargs.path_pre_trained_source

        #measure_entropy(exp_path_source=exp_path_source, exp_path_target = exp_path_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs, multiple_model=multiple_model)

        #measure_entropy(exp_path_source=exp_path_source, exp_path_target = exp_path_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs, multiple_model=multiple_model)
        measure_entropy_all_checkpoints(exp_path_source=exp_path_source, source_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)  
        #measure_entropy_all_checkpoints_dual(exp_path_source=exp_path_source, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)  
        


        #evaluate_weighted_ensemble_model(exp_path_source=exp_path_source, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split,  alpha_csv_path='entropy_performance_combined.csv', merged_model_name='merged_model.pt',tmp_outd='tmp_outd', parsedargs=parsedargs)  
        #measure_ensembled_entropy_from_checkpoints(exp_path_source=exp_path_source, source_dataset=parsedargs.source_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)  

        #find_worst_rank1_entropy(exp_path_source)
        #overlay_images, input_images, method_name, gt_masks = measure_model_diff(exp_path_source=exp_path_source, exp_path_target = exp_path_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)

if __name__ == '__main__':
    fast_eval()