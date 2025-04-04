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

import torchstain
from torchvision import transforms
from PIL import Image


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

def cl_forward(args, model, images):

    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits

def t2n(t):
    return t.detach().cpu().numpy().astype(float)

def modify_stain(args, model, loader, path_staining, dist_staining):
    num_correct = 0
    num_images = 0

    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
    ])
    
    he_matrices_list = []
    maxC_list = []

    for file in Path(path_staining).glob("*.pt"):

        staining_data = torch.load(file)


        he_matrices_list.append(staining_data["he_matrix"])
        maxC_list.append(staining_data["maxC"])

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()

        device = images
        adjusted_images = []
        for image in images:
            normalizer_target = torchstain.normalizers.MacenkoNormalizer(backend='torch')

            IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
            IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


            mean = torch.tensor(IMAGE_MEAN_VALUE).view(3, 1, 1).to(image.device)
            std = torch.tensor(IMAGE_STD_VALUE).view(3, 1, 1).to(image.device)
            image_denorm = image * std + mean
            image_denorm = (image_denorm * 255).clamp(0, 255).byte()

            image_cv2 = image_denorm.permute(1, 2, 0).cpu().numpy()

            normalizer_target.fit(T(image_cv2))
            target_he_matrix, _, _ = normalizer_target._TorchMacenkoNormalizer__compute_matrices(
                I=T(image_cv2), Io=240, alpha=1, beta=0.15
            )

            # find stain
            staining_matrices_tensor = torch.stack(he_matrices_list).to(target_he_matrix.device)
            distances = np.linalg.norm(staining_matrices_tensor - target_he_matrix, axis=(1, 2))


            #dist_staining = 100
            distance_treshold = np.percentile(distances, dist_staining)

            best_match_index = (np.abs(distances - distance_treshold)).argmin()

            #best_match_index = np.argmax(distances)
            closest_staining = he_matrices_list[best_match_index]  # (3,2)
            closest_maxC = maxC_list[best_match_index] 

            normalizer_shift = torchstain.normalizers.MacenkoNormalizer(backend='torch')

            normalizer_shift.HERef = closest_staining 
            normalizer_shift.maxCRef = closest_maxC

            # Apply normalization
            normalized_image, H, E = normalizer_shift.normalize(I=T(image_cv2), stains=True)

            # Convert in Tensor PyTorch
            normalized_image_tensor = torch.tensor(normalized_image).permute(2, 0, 1).to(device)
            adjusted_images.append(normalized_image_tensor)

            cv2.imwrite("initial_image.png", cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR))

            # Convertir l'image normalisée en format OpenCV (BGR) et sauvegarder
            normalized_image_cv2 = normalized_image_tensor.permute(1, 2, 0).cpu().numpy()
            cv2.imwrite("normalized_image.png", cv2.cvtColor(normalized_image_cv2, cv2.COLOR_RGB2BGR))

        # Convertir toutes les images du batch en un Tensor PyTorch
        adjusted_images_tensor = torch.stack(adjusted_images)

        # Passer les images modifiées au modèle
        with torch.no_grad():
            cl_logits = cl_forward(args, model, adjusted_images_tensor)
            pred = cl_logits.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc
    
def _compute_accuracy(args, model, loader, list_stain, maxC_list):
    num_correct = 0
    num_images = 0
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
    ])
    

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()

        device = images
        adjusted_images = []
        for image in images:
            normalizer_target = torchstain.normalizers.MacenkoNormalizer(backend='torch')

            IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
            IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


            mean = torch.tensor(IMAGE_MEAN_VALUE).view(3, 1, 1).to(image.device)
            std = torch.tensor(IMAGE_STD_VALUE).view(3, 1, 1).to(image.device)
            image_denorm = image * std + mean
            image_denorm = (image_denorm * 255).clamp(0, 255).byte()

            image_cv2 = image_denorm.permute(1, 2, 0).cpu().numpy()
            cv2.imwrite("initial_image.png", cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR))

            # convert image
            #image_np = image_cv2.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

            # extract HE matrix
            normalizer_target.fit(T(image_cv2))
            target_he_matrix, _, _ = normalizer_target._TorchMacenkoNormalizer__compute_matrices(
                I=T(image_cv2), Io=240, alpha=1, beta=0.15
            )

            # find stain
            staining_matrices_tensor = torch.stack(list_stain).to(target_he_matrix.device)
            distances = np.linalg.norm(staining_matrices_tensor - target_he_matrix, axis=(1, 2))
            best_match_index = np.argmin(distances)
            closest_staining = list_stain[best_match_index]  # (3,2)
            closest_maxC = maxC_list[best_match_index] 

            normalizer_shift = torchstain.normalizers.MacenkoNormalizer(backend='torch')

            normalizer_shift.HERef = closest_staining 
            normalizer_shift.maxCRef = closest_maxC

            # Apply normalization
            normalized_image, H, E = normalizer_shift.normalize(I=T(image_cv2), stains=True)

            # Convert in Tensor PyTorch
            normalized_image_tensor = torch.tensor(normalized_image).permute(2, 0, 1).to(device)
            adjusted_images.append(normalized_image_tensor)

            cv2.imwrite("initial_image.png", cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR))

            # Convertir l'image normalisée en format OpenCV (BGR) et sauvegarder
            normalized_image_cv2 = normalized_image_tensor.permute(1, 2, 0).cpu().numpy()
            cv2.imwrite("normalized_image.png", cv2.cvtColor(normalized_image_cv2, cv2.COLOR_RGB2BGR))

        # Convertir toutes les images du batch en un Tensor PyTorch
        adjusted_images_tensor = torch.stack(adjusted_images)

        # Passer les images modifiées au modèle
        with torch.no_grad():
            cl_logits = cl_forward(args, model, adjusted_images_tensor)
            pred = cl_logits.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_cam(mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# def numpy_scalar_constructor(loader, node):
#     return float(loader.construct_scalar(node))

# yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', numpy_scalar_constructor)

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

def get_visaualization(exp_path, target_method, sf_uda_source_folder, checkpoint_type, dataset, cudaid, image_ids_to_draw, split='test', tmp_outd='tmp_outd', parsedargs=None):
    # config_model.yaml
    # if parsedargs.draw_vis_with_best_source_classifier:
    #     print('wait')
    #     exp_path = exp_path.replace('models_benchmark', 'model_benchmarks_classification')
    # if target_method != 'SOURCE':
    #     tmp_exp_path = glob(f'{exp_path}/*/')
    #     assert len(tmp_exp_path) == 1
    #     exp_path = tmp_exp_path[0]
    if target_method == 'SOURCE' and parsedargs.draw_vis_with_best_source_classifier:
        checkpoint_type = constants.BEST_CL

    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['pixel_wise_classification'] = False
        #args_dict['spatial_dropout'] = 0.0
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type
        
        # assert dataset == args.dataset, f"dataset name in config file is {args.dataset} but you passed {dataset}"
    
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

    
    # DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(cudaid))
    
    tag = get_tag(args, checkpoint_type=checkpoint_type)
    # path_cl = join(exp_path, tag)
    # args.sf_uda_source_folder = path_cl
    # config_model.yaml
    #tag = get_tag(args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)
    with open(join(path_cl, 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # if target_method == 'EnergyCAM':
        if 'PixelCAM' in target_method:
            args_dict['pixel_wise_classification'] = True
            args_dict['anchors_ortogonal'] = False
            args_dict['batch_norm_pixel_classifier'] = False
            args_dict['low_res'] = False
            args_dict['multiple_layer_pixel_classifier'] = False
            args_dict['detach_pixel_classifier'] = False
            args_dict['one_layer_pixel_classifier'] = False
        else:
            args_dict['pixel_wise_classification'] = False
            args_dict['anchors_ortogonal'] = False
            args_dict['batch_norm_pixel_classifier'] = False
            args_dict['low_res'] = False
            args_dict['multiple_layer_pixel_classifier'] = False
            args_dict['detach_pixel_classifier'] = False
            args_dict['one_layer_pixel_classifier'] = False
        args_dict['model']['spatial_dropout'] = 0.0
        if target_method == 'NEGEV':
            args_dict['model']['folder_pre_trained_cl'] = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/pretrained/GLAS-0-resnet50-DEEPMIL-DeepMil-cp_best_classification"
        # args_dict = yaml.safe_load(fy)
        # args_dict['model']['freeze_encoder'] = False
        #args_dict['model']['support_background'] = True
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type
        #args.model['folder_pre_trained_cl'] = None

    args.sf_uda = False
    # args.sf_uda_source_folder = '/export/livia/home/vision/Aguichemerre/models_benchmark/source_models/BLOC/CAMELYON512/id_source_12_CAMELYON512_DEEPMIL_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50/CAMELYON512-0-resnet50-DEEPMIL-DeepMil-cp_best_localization'
    #load source model
    if target_method == 'ADADSA':
        # sf_uda_source_folder = '/export/livia/home/vision/Aguichemerre/models_benchmark/source_models/BLOC/CAMELYON512/id_source_12_CAMELYON512_DEEPMIL_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50/CAMELYON512-0-resnet50-DEEPMIL-DeepMil-cp_best_localization'
        if dataset == constants.GLAS:
            tag_targ_method = tag.replace(constants.GLAS, constants.CAMELYON512)
        else:
            tag_targ_method = tag.replace(constants.CAMELYON512, constants.GLAS)
        with open(join(sf_uda_source_folder, tag_targ_method, 'config_model.yaml'), 'r') as fy:
            args_dict_source = yaml.load(fy, Loader=IgnoreKeyLoader)
            # args_dict = yaml.safe_load(fy)
            # args_dict['model']['freeze_encoder'] = False
            args_source = Dict2Obj(args_dict_source)
            args_source.outd = tmp_outd
            args_source.distributed = False
            args_source.eval_checkpoint_type = checkpoint_type
        args_source.model['folder_pre_trained_cl'] = os.path.join(sf_uda_source_folder, tag_targ_method)
        model_soruce = get_pretrainde_classifier(args_source)

        s_model = deepcopy(model_soruce)
        t_model = deepcopy(model_soruce)
        t_model = t_model.to(get_cpu_device())
        model = deepcopy(model_soruce)

        adadsa.freeze_all_params(s_model)
        adadsa.freeze_all_params(t_model)
        adadsa.freeze_all_params(model)
        _device = next(model.parameters()).device

        for batch_norm_cl in [nn.BatchNorm1d,
                                nn.BatchNorm2d,
                                nn.BatchNorm3d]:
            adadsa.replace_all_bn_with_adadsa_bn(model=model,
                                                    s_model=s_model,
                                                    t_model=t_model,
                                                    batch_norm_cl=batch_norm_cl,
                                                    device=_device
                                                    )

        model = adadsa.adadsa_freeze_all_model_except_bn_a(model)
    else:
        model = get_model(args)[0]

    print(f'Loading model for {method_name}-{encoder_name} from {path_cl}')
    # if "tscam" in encoder_name:
    #     model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

    #     model.load_state_dict(model_tscam, strict=True)
    # else:
    #if target_method == 'EnergyCAM':
    if 'PixelCAM' in target_method:
        if "deit" in encoder_name:
            model_sat = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

            model.load_state_dict(model_sat, strict=True)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)

            header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
                                map_location=get_cpu_device())
            model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)

    elif target_method == 'NEGEV':
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                            map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

        decoder_w = torch.load(join(path_cl, 'decoder.pt'),
                            map_location=get_cpu_device())
        model.decoder.super_load_state_dict(decoder_w, strict=True)

        seg_head_w = torch.load(join(path_cl, 'segmentation_head.pt'),
                            map_location=get_cpu_device())
        model.segmentation_head.load_state_dict(seg_head_w, strict=True)

    else:
        if "deit" in encoder_name:
            model_sat = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

            model.load_state_dict(model_sat, strict=False)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)



    DLLogger.log(fmsg("Model checkpoint Loaded from {}".format(path_cl)))
        
    model.to(device)
    model.eval()

    # basic_config = config.get_config(ds=args.dataset, fold=args.fold, magnification=args.magnification)
    basic_config = config.get_config(ds=constants.GLAS, fold=args.fold, magnification=args.magnification)

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
        best_epoch = -1 
        #if split == constants.TESTSET else results[split][checkpoint_type.replace('best_', '')]['best_epoch']
        #clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][best_epoch]
        #loc_acc_from_orginal_exp_path = results[split]['localization']['value_per_epoch'][best_epoch]
        
    ####################################################################################
    ####################################################################################
    DLLogger.flush()
    
    metadata_root = join(constants.RELATIVE_META_ROOT, dataset, f"fold-{args.fold}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    target_domain_data_paths = config.configure_data_paths(args_dict, dataset)
    second_domain_data_paths = config.configure_data_paths(args_dict, 'GLAS')
    second_metadata_root = join(constants.RELATIVE_META_ROOT, 'GLAS', f"fold-{args.fold}")


    loaders = get_data_loader(
            data_roots=target_domain_data_paths,
            metadata_root=metadata_root,
            batch_size=256,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[split],
            eval_batch_size = 256, #args.eval_batch_size,
            chg_staining= parsedargs.chg_staining,
            path_staining=parsedargs.path_staining,
            dist_staining=parsedargs.dist_staining
        )
    
    loaders2 = get_data_loader(
            data_roots=second_domain_data_paths,
            metadata_root=second_metadata_root,
            batch_size=256,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[constants.TESTSET],
            eval_batch_size = 256#args.eval_batch_size,
        )

    # t0 = dt.datetime.now()
    # accuracy = _compute_accuracy(args, model, loaders[split])
    # fmsg_tmp = f'Results using split {split} and using best checkpoint that was selected using {checkpoint_type}\n'
    # fmsg_tmp += '\nClassification accuracy from current eval: {}'.format(accuracy)
    # fmsg_tmp += "\nClcassifier's evalaition time of {} split: {}".format(split, dt.datetime.now() - t0)
    # DLLogger.log(fmsg(fmsg_tmp))
    # DLLogger.flush()
    
    # cam_computer = CAMComputer(
    #         args=deepcopy(args),
    #         model=model,
    #         loader=loaders[constants.TESTSET],
    #         metadata_root=os.path.join(metadata_root, constants.TESTSET),
    #         mask_root=args.mask_root,
    #         iou_threshold_list=args.iou_threshold_list,
    #         dataset_name=args.dataset,
    #         split=constants.TESTSET,
    #         cam_curve_interval=args.cam_curve_interval,
    #         multi_contour_eval=args.multi_contour_eval,
    #         out_folder=args.outd,
    #     )

    overlay_images = {}
    input_images = {}
    gt_masks = {}


    staining_matrices_list = []
    maxC_list = []

    path_staining = parsedargs.path_staining
    dist_staining = parsedargs.dist_staining

    modify_stain(args, model, loaders[split], path_staining, dist_staining)

    for batch_idx, (images, targets, _, image_ids, _, _, _, _) in tqdm(
        enumerate(loaders[split]), ncols=constants.NCOLS,
        total=len(loaders[split])):
        image_size = images.shape[2:]
        images = images.to(device)
        targets = targets.to(device)
        


        #with torch.no_grad():
        #    out = model(images.cuda())
        #    pixel_features = model.encoder_last_features

        for image, target, image_id in zip(images, targets, image_ids):
            IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
            IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

            mean = torch.tensor(IMAGE_MEAN_VALUE).view(3, 1, 1).to(image.device)
            std = torch.tensor(IMAGE_STD_VALUE).view(3, 1, 1).to(image.device)
            image_denorm = image * std + mean
            image_denorm = (image_denorm * 255).clamp(0, 255).byte()

            image_reformat = image_denorm.permute(1, 2, 0).cpu().numpy()

            normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
            T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
            ])

            #if isinstance(image, torch.Tensor):
            #    image = image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

            normalizer.fit(T(image_reformat))
            he_matrix, _, maxC = normalizer._TorchMacenkoNormalizer__compute_matrices(I=T(image_reformat), Io=240, alpha=1, beta=0.15)
            
            #staining_matrices_list.append(he_matrix)
            #maxC_list.append(maxC)

            #modify_stain(args, model, loaders2[constants.TESTSET], path_staining)

            staining_data = {
            "he_matrix": he_matrix,  # Matrice HE
            "maxC": maxC  # Matrice maxC
        }

            tmp = str(Path(image_id).with_suffix(''))
            file_wo_bmp = tmp.replace('/', '_')
            file_pt = f'{file_wo_bmp}.pt'
            output_path = path_staining + '/' + file_pt
            torch.save(staining_data, output_path)
            # staining_data = torch.load(output_path)
            # test= staining_data["he_matrix"]
            # test2 = staining_data["maxC"]

            #torch.save(cam, output_path)

        print("end")

        #_compute_accuracy(args, model, loaders[constants.TESTSET], staining_matrices_list, maxC_list)
    return 0

def fast_eval():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--pixel_wise_classification", type=str2bool, default=False)
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--multiple_layer", type=str2bool, default=False)
    parser.add_argument("--one_layer", type=str2bool, default=False)
    parser.add_argument("--anchors_ortogonal", type=str2bool, default=False)
    parser.add_argument("--detach_pixel_classifier", type=str2bool, default=False)
    # parser.add_argument("--exp_path", type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')
    parser.add_argument('--noise_level_for_eval_with_noisy_bbox', nargs='+',
                        type=int, default=[5, 10, 15, 20, 25, 30, 35 ,40, 45, 50])
    parser.add_argument("--target_dataset", type=str, default=None,
                        help="Name of the dataset.", required=True, choices=[constants.CAMELYON512, constants.GLAS])
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument('--target_domain_exp_path', type=json.loads, default={})
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--draw_vis_with_best_source_classifier', type=str2bool, default=False)
    parser.add_argument("--path_staining", type=str, default=None, help="Path to store stainings.")
    parser.add_argument("--dist_staining", type=float, default=None, help="dist stainings.")
    parser.add_argument("--chg_staining", type=str2bool, default=None, help="chg stainings.")

    parsedargs = parser.parse_args()


    if not os.path.exists(parsedargs.path_staining):
        # If the path doesn't exist, create it
        os.makedirs(parsedargs.path_staining)
    
    # exp_path = parsedargs.exp_path
    # checkpoint_type = parsedargs.checkpoint_type
    # tmp_outd = join(parsedargs.tmp_outd, os.path.split(exp_path)[-1])#, 'split_'+split+'_'+checkpoint_type)
    # tmp_outd = join(exp_path, '0_re-eval_log')#, 'split_'+split+'_'+checkpoint_type)
    # os.makedirs(tmp_outd, exist_ok=True)
    # # tmp_outd = parsedargs.tmp_outd
    # assert os.path.isdir(exp_path)
    
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
        # exp_path = parsedargs.exp_path
        # # checkpoint_type = parsedargs.checkpoint_type
        # # tmp_outd = join(parsedargs.tmp_outd, os.path.split(exp_path)[-1])#, 'split_'+split+'_'+checkpoint_type)
        # tmp_outd = join(exp_path, '0_re-eval_log')#, 'split_'+split+'_'+checkpoint_type)
        # # tmp_outd = parsedargs.tmp_outd
        # assert os.path.isdir(exp_path)
        assert split == constants.TESTSET or split == constants.VALIDSET or split == constants.TRAINSET
        
        _CODE_FUNCTION = 'fast_eval_{}'.format(split)

        #target_methods = ['DeepMIL', 'EnergyCAM DL', 'GradCAMpp', 'EnergyCAM GC', 'LayerCAM', 'EnergyCAM LC', 'SAT', 'EnergyCAM SAT']
        target_methods = ['LayerCAM']
        #'CAM', 'GradCAMpp', 'NEGEV',
        # target_methods = ['ADADSA']GradCAMpp'EnergyCAM', 'NEGEV', 
        #create fig len(parsedargs.image_ids_to_draw) row and len(target_methods) columns
        fig, axs = plt.subplots(len(parsedargs.image_ids_to_draw), len(target_methods)+2, figsize=((len(target_methods)+2)*1.9, 2*len(parsedargs.image_ids_to_draw)),squeeze=False)
        

        method_name_lst = []
        for ind_method, target_method in enumerate(target_methods):
            ind_method+= 2
            # if target_method == 'ADADSA':
            #     for i, image_id in enumerate(parsedargs.image_ids_to_draw):
            #         axs[i, ind_method].axis('off')
            #         if ind_method == 1:
            #             axs[i, ind_method].set_title(f'{target_method}')
            #     continue

            exp_path = parsedargs.target_domain_exp_path[target_method]
            overlay_images, input_images, method_name, gt_masks = get_visaualization(exp_path=exp_path, target_method=target_method, sf_uda_source_folder=parsedargs.path_pre_trained_source, checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split=split, tmp_outd='tmp_outd', parsedargs=parsedargs)
            

    
if __name__ == '__main__':
    fast_eval()