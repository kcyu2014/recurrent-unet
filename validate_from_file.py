"""
Serve as a convenient wrapper for validate.py.

"""
import glob
import os
import re
import IPython
import yaml

from ptsemseg.loader import get_data_path
from utils import validate_parser
from validate import validate, result_root

ROOT = os.getcwd()
M_ROOT = "runs/"
os.chdir(ROOT)


def final_run_dirs(args):
    if args.dataset == "drive":
        run_dirs = [
        ]
    elif args.dataset == "epfl":
        run_dirs = [
        ]
    elif args.dataset == "eyth":
        run_dirs = [
        ]
    elif args.dataset == 'egohand':
        run_dirs = [
        ]
    elif args.dataset == 'gtea':
        run_dirs = [
        ]
    elif args.dataset == 'hofhand':
        run_dirs = [
        ]

    elif args.dataset == 'road':
        run_dirs = [
        ]

    elif args.dataset == 'cityscapes':
        run_dirs = [
        ]
    elif args.dataset == 'epflhand-new':
        run_dirs = [
        ]
    else:
        raise ValueError(f"Dataset not supported {args.dataset}")
    return run_dirs


def process_args_from_loaded_cfg(cfg, args):
    """
    process arguments from loaded CFG.
    :param cfg:
    :param args:
    :return:
    """
    # If calling this, it will use CFG for sure.
    # if 'runet' in cfg['model']['arch']:
    #     args.hidden_size = cfg['model']['hidden_size']

    #
    r = re.compile('-h\d+-')
    r_d = re.compile('\d+')
    _h = r.findall(cfg['logdir'])
    if len(_h) > 0:
        res = int(r_d.findall(_h[0])[0])
        if res > 1: # avoid the represent h=init
            args.hidden_size = res
        else:
            res = args.hidden_size

        if cfg['model'].get('hidden_size'):
            assert cfg['model']['hidden_size'] == res
        else:
            cfg['model']['hidden_size'] = res

    args.gate = cfg['model'].get('gate') or args.gate

    if args.is_recurrent is not None:
        args.is_recurrent = cfg['training']['loss']['name'] \
                            in ['multi_step_cross_entropy']

    out_path = args.out_path or os.path.join(
        'results', cfg['data']['dataset'],
        os.path.basename(os.path.dirname(cfg['logdir'])))
    cfg['eval_out_path'] = out_path

    # Process for unet_level
    if cfg['model']['arch'] == "runet":
        if cfg['model'].get('unet_level'):
            args.unet_level = cfg['model']['unet_level']
        else:
            unet_level = args.hidden_size // 32
            args.unet_level = unet_level

        cfg['model']['unet_level'] = args.unet_level
        cfg['model']['recurrent_level'] = args.recurrent_level

    if not os.path.exists(cfg['data']['path']):
        cfg['data']['path'] = get_data_path(
            cfg['data']['path'],
            config_file=f"configs/dataset/{cfg['data']['dataset'].replace('_', '')}.yml")
    IPython.embed()
    return cfg, args


def load_complete_info_from_dir(run_dir, args):
    """
    Load model from given directory.
    Must work with the complete config.yaml given in the folder.

    :param run_dir:
    :return:
    """
    run_dir = os.path.join(M_ROOT, run_dir)
    # Starting from this, should be reused.
    try:
        config_path = glob.glob(run_dir + '/config*')[0]
    except IndexError as e:
        try:
            config_path = glob.glob(run_dir + '/*.yaml')[0]
        except IndexError as e:
            try:
                config_path = glob.glob(run_dir + '/*.yml')[0]
            except IndexError as e:
                raise FileNotFoundError(f"nothing is found in {run_dir}.")
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)

    if os.path.exists(run_dir):
        cfg['logdir'] = run_dir
    else:
        assert os.path.exists(cfg['logdir'])
    # IPython.embed()
    best_path = os.path.join(run_dir, cfg['training']['resume'])
    if not os.path.exists(best_path):
        # try to search best_model
        best_path = glob.glob(f"{run_dir}/*best_model*")[0]
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No file found at {run_dir}")

        cfg['training']['resume'] = os.path.basename(best_path)

    # for EPFL-hand, set to -1 by force. do not use the croping.
    if cfg['data']['dataset'] == "epfl_hand_roi":
        cfg['data']['dataset'] = 'epfl_hand'
    cfg['data']['void_class'] = -1

    # if 'road' in cfg['data']['dataset']:
    if 'road' or 'cityscapes' in cfg['data']['dataset']:
        cfg['training']['validate_batch_size'] = 1
        cfg['training']['batch_size'] = 1

    cfg, args = process_args_from_loaded_cfg(cfg, args)
    cfg['model']['steps'] = args.steps
    return cfg, args


def run_validate(args, run_dirs, result_dir=None, roi_only=False):
    """
    run multiple validation here.
    """

    for i, r_dir in enumerate(run_dirs):

        cfg, args = load_complete_info_from_dir(r_dir, args)
        result_dir = result_dir or os.path.join('results', cfg['data']['dataset'])
        results = validate(cfg, args, roi_only=roi_only)
        logdir = result_dir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        result_path = logdir + f"{cfg['model']['arch']}" + '.yml'
        with open(result_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)



if __name__ == '__main__':
    parser = validate_parser()
    args = parser.parse_args()
    args.eval_flip = True
    args.measure_time = True
    args.structure = 'baseline'
    args.batch_size = 1
    roi_only = False
    run_dirs = final_run_dirs(args)
    del args.dataset
    run_validate(args, run_dirs, roi_only=roi_only)
