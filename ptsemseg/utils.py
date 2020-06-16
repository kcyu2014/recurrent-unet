"""
Misc Utility functions
"""
import argparse
import os
import logging
import datetime
import sys

import IPython
import numpy as np

from collections import OrderedDict

from torch.nn import functional as F


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir, mode='run', level=logging.INFO):

    # logger = logging.getLogger('{}-{}'.format(logdir, mode))
    # Clean the logger and set to root.
    logger = logging.getLogger()
    logger.handlers = []

    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    if not os.path.exists(logdir):
        os.makedirs(logdir, mode=744)

    file_path = os.path.join(logdir, '{}_{}.log'.format(mode,ts))
    hdlr = logging.FileHandler(file_path)
    if mode == 'eval':
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(level)
    return logger


def clean_logger(logger):
    print("Clean logger.")
    # IPython.embed()
    # for hdlr in logger.handlers:
    #     logger.removeHandler(hdlr)
    #     del hdlr
    logger.handlers = []


def get_argparser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/rcnn2_gteahand.yml',
        # unet_eythhand, vanillaRNNunet_eythhand, rcnn2_eythhand
        # unet_gteahand, rcnn2_gteahand, vanillaRNNunet_gteahand
        # rcnn2_egohand, vanillaRNNunet_egohand, unet_egohand, fcn8s_egohand, unet_hand, fcn8s_hand, psp_egohand
        # vanillaRNNunet_epflhand, unet_epflhand, rcnn2_epflhand
        # unet_hofhand, vanillaRNNunet_hofhand, rcnn2_hofhand
        help="Configuration file to use"
    )
    parser.add_argument("--device", nargs="?", type=str, default="cuda:0", help="GPU or CPU to use")
    parser.add_argument("--steps", nargs="?", type=int, default=3, help="Recurrent Steps")
    parser.add_argument("--clip", nargs="?", type=float, default=10., help="gradient clip threshold")
    parser.add_argument("--hidden_size", nargs="?", type=int, default=32, help="hidden size")
    parser.add_argument("--initial", nargs="?", type=int, default=1, help="initial value of hidden state")
    parser.add_argument("--scale_weight", nargs="?", type=float, default=-1., help="loss decay after recurrent steps")
    # args = parser.parse_args()
    return parser


def handle_input_target_mismatch(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    return input, target