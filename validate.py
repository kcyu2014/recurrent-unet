import logging
import os
import random
import argparse
import IPython
import yaml
import torch
import timeit
import numpy as np

from os.path import join as pjoin
from torch.backends import cudnn
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_void_class
from ptsemseg.utils import get_logger, clean_logger
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from utils import validate_parser
torch.backends.cudnn.benchmark = True

# def process_pred_roi(gt, pred, void_class):
#     new_pred = np.where(gt == void_class, void_class, pred)
#     return new_pred


def wrap_str(*inputs):
    res = ""
    for i in inputs:
        res += str(i)
    return res


def result_root(cfg, create=False, appe=None):
    train_dir = cfg['logdir']
    train_id = train_dir.replace("runs/{}".format(cfg['training']['prefix']), '').replace('/', '-')[1:]
    logdir = os.path.join('results', cfg['training']['prefix'], train_id)
    if create:
        os.makedirs(os.path.dirname(logdir), exist_ok=True)
    logdir += appe if appe is not None else ''
    return logdir


def load_model_and_preprocess(cfg, args, n_classes, device):
    if 'NoParamShare' in cfg['model']['arch']:
        args.steps = cfg['model']['steps']
    model = get_model(cfg['model'], n_classes, args).to(device)
    if os.path.exists(args.model_path):
        model_path = args.model_path
    else:
        model_path = pjoin(cfg['logdir'],cfg['training']['resume'])
    # print(model)
    state = convert_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)["model_state"])

    # IPython.embed()

    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return model, model_path


def validate(cfg, args, roi_only=False):
    # Make sure this flag is on.
    torch.backends.cudnn.benchmark = True
    logger = get_logger(cfg['logdir'], 'eval', level=logging.WARN if args.prefix == 'benchmark' else logging.INFO)
    results = {}
    results['valid'] = {}
    results['test'] = {}
    result_tags = ['valid', 'test']

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    is_void_class = get_void_class(cfg['data']['dataset'])
    logger.info("{} is void? {}".format(cfg['data'], is_void_class))

    if cfg['data']['dataset'] not in ['cityscapes']:
        update_raw = True
        img_norm = True
    else:
        update_raw = False
        img_norm = False

    loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
        img_norm=img_norm
    )

    test_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['test_split'],
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
        img_norm=img_norm
    )

    IPython.embed()

    n_classes = loader.n_classes
    if roi_only:
        # assert n_classes > 2
        assert cfg['data']['void_class'] > 0
        assert loader.void_classes == cfg['data']['void_class']
    validate_batch_size = cfg['training'].get('validate_batch_size') or cfg['training']['batch_size']

    valloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=8)

    testloader = data.DataLoader(test_loader,
                                 batch_size=validate_batch_size,
                                 num_workers=8)

    if cfg['training']['loss']['name'] in ['multi_step_cross_entropy'] and cfg['model']['arch'] not in ['pspnet']:
        running_metrics = None
    else:
        running_metrics = runningScore(n_classes, void=is_void_class) if not roi_only else \
            runningScore(n_classes + 1, roi_only)

    # Setup Model

    model, model_path = load_model_and_preprocess(cfg, args, n_classes, device)
    logger.info("Loading model {} from {}".format(cfg['model']['arch'], model_path))

    with torch.no_grad():
        if cfg['training']['loss']['name'] in ['multi_step_cross_entropy'] and cfg['model']['arch'] not in ['pspnet']:
            for loader_type, myloader in enumerate([valloader, testloader]):

                computation_time = 0
                img_no = 0
                if args.benchmark and loader_type == 0:
                    continue
                # For all the images in this loader.
                for i, (images, labels) in enumerate(myloader):
                    if args.benchmark:
                        if i > 100:
                            break
                    start_time = timeit.default_timer()
                    images = images.to(device)
                    if args.eval_flip:
                        # Flip images in numpy (not support in tensor)
                        flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
                        flipped_images = torch.from_numpy(flipped_images).float().to(device)

                        if cfg['model']['arch'] in ['reclast']:
                            h0 = torch.ones([images.shape[0], args.hidden_size, images.shape[2], images.shape[3]],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0)
                            outputs_flipped = model(flipped_images, h0)

                        elif cfg['model']['arch'] in ['recmid']:
                            W, H = images.shape[2], images.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0)
                            outputs_flipped = model(flipped_images, h0)

                        elif cfg['model']['arch'] in ['dru', 'sru']:
                            W, H = images.shape[2], images.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            s0 = torch.ones([images.shape[0], n_classes, W, H],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0, s0)
                            outputs_flipped = model(flipped_images, h0, s0)

                        elif cfg['model']['arch'] in ['druvgg16', 'druresnet50', 'druresnet50syncedbn']:
                            W, H = images.shape[2], images.shape[3]
                            w, h = int(W / 2 ** 4), int(H / 2 ** 4)
                            if cfg['model']['arch'] in ['druresnet50', 'druresnet50syncedbn']:
                                w, h = int(W / 2 ** 5), int(H / 2 ** 5)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            s0 = torch.zeros([images.shape[0], n_classes, W, H],
                                             dtype=torch.float32, device=device)
                            outputs = model(images, h0, s0)
                            outputs_flipped = model(flipped_images, h0, s0)

                        else:
                            outputs = model(images)
                            outputs_flipped = model(flipped_images)

                        outputs_list = [output.data.cpu().numpy() for output in outputs]
                        outputs_flipped_list = [output_flipped.data.cpu().numpy() for output_flipped in outputs_flipped]
                        outputs_list = [(outputs + outputs_flipped[:, :, :, ::-1]) / 2.0 for
                                        outputs, outputs_flipped in zip(outputs_list, outputs_flipped_list)]

                    else:
                        if cfg['model']['arch'] in ['reclast']:
                            h0 = torch.ones([images.shape[0], args.hidden_size, images.shape[2], images.shape[3]],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0)

                        elif cfg['model']['arch'] in ['recmid']:
                            W, H = images.shape[2], images.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0)

                        elif cfg['model']['arch'] in ['dru', 'sru']:
                            W, H = images.shape[2], images.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            s0 = torch.ones([images.shape[0], n_classes, W, H],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0, s0)

                        elif cfg['model']['arch'] in ['druvgg16', 'druresnet50', 'druresnet50syncedbn']:
                            W, H = images.shape[2], images.shape[3]
                            w, h = int(W / 2 ** 4), int(H / 2 ** 4)
                            if cfg['model']['arch'] in ['druresnet50', 'druresnet50syncedbn']:
                                w, h = int(W / 2 ** 5), int(H / 2 ** 5)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            s0 = torch.zeros([images.shape[0], n_classes, W, H],
                                             dtype=torch.float32, device=device)
                            outputs = model(images, h0, s0)

                        else:
                            outputs = model(images)

                        outputs_list = [output.data.cpu().numpy() for output in outputs]

                    pred = [np.argmax(outputs, axis=1) for outputs in outputs_list]

                    gt = labels.numpy()

                    if roi_only:
                        """ Process for ROI, basically, mask the Pred based on GT"""
                        # IPython.embed()
                        for k in range(len(pred)):
                            pred[k] = np.where(gt == loader.void_classes, loader.void_classes, pred[k])

                    if args.measure_time:
                        elapsed_time = timeit.default_timer() - start_time
                        computation_time += elapsed_time
                        img_no += pred[-1].shape[0]
                        if (i + 1) % 5 == 0:
                            logger.warning(
                                "Inference time \
                                  (iter {0:5d}): {1:3.5f} fps".format(
                                    i + 1, pred[-1].shape[0] / elapsed_time
                                )
                            )
                    for k in range(len(pred)):
                        if running_metrics is None:
                            running_metrics = []
                            for _ in range(len(pred)):
                                running_metrics = running_metrics + [runningScore(n_classes, void=is_void_class)
                                                                     if not roi_only else runningScore(n_classes + 1, roi_only)]
                        else:
                            running_metrics[k].update(gt, pred[k], step=k)
                            if update_raw and k == len(pred) - 1:
                                running_metrics[k].update_raw(gt, outputs_list[k], step=k)

                if args.measure_time:
                    logger.warning(f'{computation_time}, {img_no}')
                    logger.warning("Overall Inference time {} fps".format(img_no*1. / computation_time))

                if loader_type == 0:
                    logger.info('validation set performance :')

                elif loader_type == 1:
                    logger.info('test set performance')
                results[result_tags[loader_type]] = {}

                for j in range(len(running_metrics)):
                    score, class_iou, class_f1 = running_metrics[j].get_scores()
                    logger.info('RNN step {}'.format(j+1))
                    results[result_tags[loader_type]][j] = {}
                    for k, v in score.items():
                        v = float(v)
                        logger.info(wrap_str(k, v))
                        results[result_tags[loader_type]][j][k] = v

                    for i in range(n_classes - 1 if is_void_class else n_classes):
                        logger.info(wrap_str("iou class {}: \t".format(i), class_iou[i]))
                        results[result_tags[loader_type]][j]['iou{}'.format(i)] = float(class_iou[i])

                    for i in range(n_classes - 1 if is_void_class else n_classes):
                        logger.info(wrap_str("f1 class {}: \t".format(i), class_f1[i]))
                        results[result_tags[loader_type]][j]['f1{}'.format(i)] = float(class_f1[i])

                    # if j == 2:
                    #     p, r, thresh = running_metrics[j].compute_break_even()
                    #     logger.info(f"P/R Break even at {(p + r) / 2}.")
                # running_metrics = None
        else:
            for loader_type, myloader in enumerate([valloader, testloader]):
                start = timeit.default_timer()
                computation_time = 0
                img_no = 0
                if args.benchmark and loader_type == 0:
                    continue
                for i, (images, labels) in enumerate(myloader):
                    if args.benchmark:
                        if i > 100:
                            break
                    start_time = timeit.default_timer()
                    images = images.to(device)
                    if args.eval_flip:
                        outputs = model(images)
                        # Flip images in numpy (not support in tensor)
                        outputs = outputs.data.cpu().numpy()
                        flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
                        flipped_images = torch.from_numpy(flipped_images).float().to(device)
                        outputs_flipped = model(flipped_images)
                        outputs_flipped = outputs_flipped.data.cpu().numpy()
                        outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0
                        pred = np.argmax(outputs, axis=1)
                    else:
                        outputs = model(images)
                        outputs = outputs.data.cpu().numpy()
                        pred = np.argmax(outputs, axis=1)

                    gt = labels.numpy()

                    if roi_only:
                        pred = np.where(gt == loader.void_classes, loader.void_classes, pred)

                    if args.measure_time:
                        elapsed_time = timeit.default_timer() - start_time
                        computation_time += elapsed_time
                        img_no += pred.shape[0]
                        if (i + 1) % 5 == 0:
                            logging.warning(
                                "Inference time \
                                  (iter {0:5d}): {1:3.5f} fps".format(
                                    i + 1, pred.shape[0] / elapsed_time
                                )
                            )
                    running_metrics.update(gt, pred)
                    if update_raw:
                        running_metrics.update_raw(gt, outputs)

                if args.measure_time:
                    logging.warning("{computation_time}, {img_no}")
                    logging.warning("Overall Inference time {} fps".format(img_no * 1. / computation_time))
                    logging.warning("Inference time with data loading {} fps".format(img_no * 1. / (timeit.default_timer()-start)))
                    # running_metrics = None

                score, class_iou, class_f1 = running_metrics.get_scores()
                if update_raw:
                    p, r, thres = running_metrics.compute_break_even()
                    logger.warning("P/R Break even at {}.".format((p + r) / 2))

                results[result_tags[loader_type]] = {}
                if loader_type == 0:
                    logger.warning('validation set performance :')

                elif loader_type == 1:
                    logger.warning('test set performance')

                for k, v in score.items():
                    logger.info(wrap_str(k, v))
                    results[result_tags[loader_type]][k] = v

                for i in range(n_classes - 1 if is_void_class else n_classes):
                    logger.info(wrap_str("iou class {}: \t".format(i), class_iou[i]))
                    results[result_tags[loader_type]]['iou{}'.format(i)] = class_iou[i]

                for i in range(n_classes - 1 if is_void_class else n_classes):
                    logger.info(wrap_str("f1 class {}: \t".format(i), class_f1[i]))
                    results[result_tags[loader_type]]['f1{}'.format(i)] = class_f1[i]

    result_path = result_root(cfg, create=True) + '.yml'
    with open(result_path, 'w') as f:
        yaml.dump(results, f,  default_flow_style=False)

    print(model_path)
    # IPython.embed()
    clean_logger(logger)
    return results


if __name__ == "__main__":
    # parser = validate_parser()
    # args = parser.parse_args()
    #
    # with open(args.config) as fp:
    #     cfg = yaml.load(fp)
    # # set the log DIR.
    #
    # logdir = os.path.join('results',  args.prefix,
    #                       f"{cfg['data']['dataset']}-{args.model}", str(random.randint(1, 10000)))
    # cfg['logdir'] = logdir
    #
    # validate(cfg, args)

    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/dataset/cityscapes.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                                  True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                                  True by default",
    )
    parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                                  True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                                  True by default",
    )
    parser.set_defaults(measure_time=True)

    parser.add_argument("--dataset", nargs="?", type=str, default="cityscapes", help="dataset")
    parser.add_argument("--img_rows", nargs="?", type=int, default=1025, help="img_rows")
    parser.add_argument("--img_cols", nargs="?", type=int, default=2049, help="img_cols")
    parser.add_argument("--no-img_norm", nargs="?", type=bool, default=True, help="no-img_norm")
    parser.add_argument("--batch_size", nargs="?", type=int, default=2, help="batch_size")
    parser.add_argument("--split", nargs="?", type=str, default="val", help="split")
    parser.add_argument("--device", nargs="?", type=str, default="cuda:0", help="GPU or CPU to use")

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    cfg['logdir'] = 'runs/cityscapes/pspnet'
    validate(cfg, args)
