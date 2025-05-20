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

def _compute_accuracy(args, model, loader):
    num_correct = 0
    num_images = 0

    num_correct_normal = 0
    num_images_normal = 0

    num_correct_cancer = 0
    num_images_cancer = 0

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

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
    
    return classification_acc, classification_acc_normal, classification_acc_cancer

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


def measure_accuracy_diff_model(exp_path_source,exp_path_target, checkpoint_type, source_dataset,target_dataset, cudaid, split, tmp_outd='tmp_outd', parsedargs=None, args = None):

    source_model = load_model(exp_path_source, source_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)
    target_model = load_model(exp_path_target, target_dataset, checkpoint_type, cudaid, tmp_outd=tmp_outd, parsedargs=parsedargs)

    new_model = deepcopy(source_model)

    device = torch.device('cuda:{}'.format(cudaid))
    source_model.to(device)
    target_model.to(device)
    new_model.to(device)

    state_dict_source_model = source_model.state_dict()
    state_dict_target_model = target_model.state_dict()

    cl_perf_list = []
    cam_perf_list = []
    modified_param_names = []


    source_model.eval()
    target_model.eval()
    new_model.eval()

    # keys_to_copy = [k for k in state_dict_source_model.keys() if k.startswith("encoder.layer4") or k.startswith("classification_head")]

    # for k in keys_to_copy:
    #     if k in state_dict_target_model:
    #         state_dict_target_model[k] = state_dict_source_model[k]
    #     else:
    #         print(f"[WARNING] Key {k} mot found in target_model — Ignored.")

    #target_model.load_state_dict(state_dict_target_model)

    target_loader, target_metadata_root, target_domain_data_paths, args = load_loader(exp_path_target, target_dataset, checkpoint_type, cudaid, split, tmp_outd=tmp_outd, parsedargs=parsedargs)

    cl_performance, class_0_performance, class_1_performance = _compute_accuracy(args, source_model, target_loader[split])
    print(f"[Initial Classification performance: {cl_performance}")
    print(f"[Initial Normal Classification performance: {class_0_performance}")
    print(f"[Initial Cancer Classification performance: {class_1_performance}")

    # Evaluation of localisation CAM
    cam_computer = CAMComputer(
        args=deepcopy(args),
        model=source_model,
        loader=target_loader[split],
        metadata_root=os.path.join(target_metadata_root, split),
        mask_root=target_domain_data_paths[split],
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=target_dataset,
        split=split,
        cam_curve_interval=args.cam_curve_interval,
        multi_contour_eval=args.multi_contour_eval,
        out_folder=args.outd,
    )

    cam_performance = cam_computer.compute_and_evaluate_cams()
    print(f"[Initial CAM performance: {cam_performance}")

    cl_perf_list.append(cl_performance)
    cam_perf_list.append(cam_performance)
    modified_param_names.append("Source_model")


    base_keys = ["classification_head", "pixel_wise_classification_head"]
    replacement_stages = [base_keys.copy()]

    layer4_keys = sorted([k for k in state_dict_source_model.keys() if k.startswith("encoder.layer4")])
    #blocks = list(set(k.split('.')[2] for k in layer4_keys))  # ex: '0', '1', '2'
    #blocks = sorted(blocks, reverse=True)

    # for i in range(1, len(blocks)+1):
    #     stage_keys = ["classification_head", "pixel_wise_classification_head"]
    #     selected_blocks = blocks[:i]  
    #     for blk in selected_blocks:
    #         blk_keys = [k for k in layer4_keys if f".{blk}." in k or k.endswith(f".{blk}")]
    #         stage_keys += blk_keys
    #     replacement_stages.append(stage_keys)

    for i in range(len(layer4_keys)):
        next_stage = replacement_stages[-1].copy()
        next_stage.append(layer4_keys[i])
        replacement_stages.append(next_stage)

    for stage_id, keys_to_copy_prefix in enumerate(replacement_stages):
        print(f"\n--- Step {stage_id + 1}/{len(replacement_stages)}: replacement of {len(keys_to_copy_prefix)} blocs ---")

        # Clone target_state_dict
        modified_state_dict = source_model.state_dict().copy()

        for k in state_dict_source_model:
            if any(k.startswith(prefix) for prefix in keys_to_copy_prefix):
                if k in modified_state_dict:
                    modified_state_dict[k] = state_dict_target_model[k]

        # Load new state_dict into target_model
        new_model.load_state_dict(modified_state_dict)

        # Evaluation of classification accuracy
        cl_performance = _compute_accuracy(args, new_model, target_loader[split])
        print(f"[Step {stage_id + 1}] Classification performance: {cl_performance}")

        # Evaluation of localisation CAM
        cam_computer = CAMComputer(
            args=deepcopy(args),
            model=new_model,
            loader=target_loader[split],
            metadata_root=os.path.join(target_metadata_root, split),
            mask_root=target_domain_data_paths[split],
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=target_dataset,
            split=split,
            cam_curve_interval=args.cam_curve_interval,
            multi_contour_eval=args.multi_contour_eval,
            out_folder=args.outd,
        )

        cam_performance = cam_computer.compute_and_evaluate_cams()
        print(f"[Step {stage_id + 1}] CAM performance: {cam_performance}")
    
        cl_perf_list.append(cl_performance)
        cam_perf_list.append(cam_performance)

        # Dernier paramètre modifié (hors classifieurs)
        non_classifier_keys = [k for k in replacement_stages[stage_id] if not any(k.startswith(p) for p in base_keys)]
        last_key = non_classifier_keys[-1] if non_classifier_keys else "only_classifiers"
        modified_param_names.append(last_key)


    fig, ax = plt.subplots(figsize=(20, 6))
    bars = ax.bar(modified_param_names, cam_perf_list, color='blue', label="CAM performance")
    ax.set_xticks(range(len(modified_param_names)))
    ax.set_xticklabels(modified_param_names, rotation=90, fontsize=7)
    ax.set_ylabel("Localization metric", fontsize=12)
    ax.set_xlabel("Last param modified", fontsize=12)
    ax.set_title("Evolution of localization performance", fontsize=14)
    ax.bar_label(bars, fmt='%.1f', fontsize=10, rotation=90, label_type='edge')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('visualization', 'model_difference', parsedargs.source_model_name, source_dataset, f"{source_dataset}_with_{parsedargs.source_model_name}_to_{target_dataset}_on_{parsedargs.target_model_name}_cam_perf.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 6))
    bars = ax.bar(modified_param_names, cl_perf_list, color='red', label="Classification accuracy")
    ax.set_xticks(range(len(modified_param_names)))
    ax.set_xticklabels(modified_param_names, rotation=90, fontsize=7)
    ax.set_ylabel("Classification metric", fontsize=12)
    ax.set_xlabel("Last param modified", fontsize=12)
    ax.set_title("Evolution of classification performance", fontsize=14)
    ax.bar_label(bars, fmt='%.1f', fontsize=10, rotation=90, label_type='edge')  # valeurs au-dessus
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('visualization', 'model_difference', parsedargs.source_model_name, source_dataset, f"{source_dataset}_with_{parsedargs.source_model_name}_to_{target_dataset}_on_{parsedargs.target_model_name}_cl_perf.png"), dpi=300)
    plt.close()

    # cl_performance =_compute_accuracy(args, target_model, target_loader[split])
    # print(f"cl_performance: {cl_performance}")


    # cam_computer = CAMComputer(
    #                         args=deepcopy(args),
    #                         model=target_model,
    #                         loader=target_loader[split],
    #                         metadata_root=os.path.join(target_metadata_root, split),
    #                         mask_root=target_domain_data_paths[split],
    #                         iou_threshold_list=args.iou_threshold_list,
    #                         dataset_name=target_dataset,
    #                         split= split,
    #                         cam_curve_interval=args.cam_curve_interval,
    #                         multi_contour_eval=args.multi_contour_eval,
    #                         out_folder=args.outd,
    #                     )

    # cam_performance = cam_computer.compute_and_evaluate_cams()

    # print(f"cam_performance: {cam_performance}")





    
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
    parser.add_argument("--external_model", type=str, default=None, help="Path to the external bb+cl.")
    parser.add_argument("--source_model_name", type=str, default=None, help="Name of source model.")
    parser.add_argument("--target_model_name", type=str, default=None, help="Name of target model.")

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


        measure_accuracy_diff_model(exp_path_source=exp_path_source, exp_path_target = exp_path_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)


        #overlay_images, input_images, method_name, gt_masks = measure_model_diff(exp_path_source=exp_path_source, exp_path_target = exp_path_target, checkpoint_type=checkpoint_type, source_dataset=parsedargs.source_dataset,target_dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)

if __name__ == '__main__':
    fast_eval()