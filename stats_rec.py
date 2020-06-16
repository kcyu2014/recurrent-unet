import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import random

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

# torch.backends.cudnn.benchmark = True
# def process_pred_roi(gt, pred, void_class):
#     new_pred = np.where(gt == void_class, void_class, pred)
#     return new_pred


def stats(iou, diff, stage):
    print('num_img, num_rec', iou.shape)
    img_id = np.arange(len(diff))
    max_id = np.argmax(iou, 1)
    hist = np.bincount(max_id)
    t = np.arange(2, diff.shape[1] + 1)
    tick_iou = np.arange(1, diff.shape[1] + 1)
    curves_diff = []
    curves_miou = []
    for i in range(diff.shape[1]):
        sel_id = img_id[max_id == i]
        if not sel_id.size:
            print('empty rec no detected {}'.format(i))
        else:
            diff_aver = diff[sel_id].mean(axis=0)
            iou_aver = iou[sel_id, ::].mean(axis=0)
            curves_diff.append(diff_aver)
            curves_miou.append(iou_aver)
    for i in range(len(curves_diff)):
        fig, ax1 = plt.subplots()
        ax1.plot(t, curves_diff[i][1::], 'bo-', label='Best-RecNo-{}-InstanceNo-{}'.format(i+1, hist[i]))
        # print('threshold value stop-now for rec {}'.format(i+1), curves[i][i])
        # if j < 17:
        #     print('threshold value stop-previous for rec {}'.format(i+1), curves[i][i+1])
        # if j < 16:
        #     print('threshold value stop-previous-previous for rec {}'.format(i+1), curves[i][i+2])
        # if i==0:
        #     plt.ylim(0., 30)
        # elif i==1:
        #     plt.ylim(0., 4)
        # else:
        #     plt.ylim(0., 0.5)
        ax1.set_xlabel('Test-Rec-Number(instance-No-{}/{})'.format(hist[i], len(diff)), fontsize=16)
        ax1.set_ylabel('Distance between Rt & Rt-1', fontsize=16, color='b')
        ax1.set_xticks(t)
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params('y', colors='b')
        ax1.set_ylim(0, 12)
        ax2 = ax1.twinx()
        ax2.plot(tick_iou, curves_miou[i], 'r*-')
        ax2.set_ylabel('mIOU', fontsize=16, color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_ylim(0.4, 1)
        plt.legend(loc='upper right', prop={'size': 8})
        # plt.show()
        plt.savefig('runs/eyth_hand/iccvablation/eyth_{}_dru_stats_{}.pdf'.format(stage, i+1), bbox_inches='tight')
    print('plot done!')

    values = np.max(iou, 1)

    print('img Number with optimal predictions at recurrence from 1 to  iou.shape[1] \n', hist)


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
    model = get_model(cfg['model'], n_classes, args).to(device)
    if os.path.exists(args.model_path):
        model_path = args.model_path
    else:
        model_path = pjoin(cfg['logdir'], cfg['training']['resume'])
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
    logger = get_logger(cfg['logdir'], 'eval')
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

    train_loader = data_loader(
        data_path,
        split=cfg['data']['train_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_rows']),
    )

    val_loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_rows']),
    )

    test_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['test_split'],
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
    )
    update_raw = False
    n_classes = val_loader.n_classes
    if roi_only:
        # assert n_classes > 2
        assert cfg['data']['void_class'] > 0
        assert val_loader.void_classes == cfg['data']['void_class']

    validate_batch_size = cfg['training'].get('validate_batch_size') or cfg['training']['batch_size']
    # IPython.embed()
    trainloader = data.DataLoader(train_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=8)

    valloader = data.DataLoader(val_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=8)

    testloader = data.DataLoader(test_loader,
                                 batch_size=validate_batch_size,
                                 num_workers=8)

    if cfg['training']['loss']['name'] in ['multi_step_cross_entropy']:
        running_metrics = None
    else:
        running_metrics = runningScore(n_classes, void=is_void_class) if not roi_only else \
            runningScore(n_classes + 1, roi_only)

    # Setup Model

    model, model_path = load_model_and_preprocess(cfg, args, n_classes, device)
    logger.info("Loading model {} from {}".format(cfg['model']['arch'], model_path))

    result_path = result_root(cfg, create=True) + '.yml'
    mysoftmax = torch.nn.Softmax2d()
    with torch.no_grad():
        if cfg['training']['loss']['name'] in ['multi_step_cross_entropy']:
            for loader_type, myloader in enumerate([trainloader, valloader, testloader]):
                iou_list = []
                diff_list = []
                # For all the images in this loader.
                for i, (images, labels) in enumerate(myloader):

                    # TODO DEBUGING here.
                    # if i > 2:
                    #     break
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

                        elif cfg['model']['arch'] in ['dru']:
                            W, H = images.shape[2], images.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            s0 = torch.ones([images.shape[0], n_classes, W, H],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0, s0)
                            outputs_flipped = model(flipped_images, h0)

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

                        elif cfg['model']['arch'] in ['dru']:
                            W, H = images.shape[2], images.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32, device=device)
                            s0 = torch.ones([images.shape[0], n_classes, W, H],
                                            dtype=torch.float32, device=device)
                            outputs = model(images, h0, s0)

                        else:
                            outputs = model(images)

                        outputs_list = [output.data.cpu().numpy() for output in outputs]
                        pred = [np.argmax(output, axis=1) for output in outputs_list]

                    gt = labels.numpy()

                    if roi_only:
                        """ Process for ROI, basically, mask the Pred based on GT"""
                        # IPython.embed()
                        for k in range(len(pred)):
                            pred[k] = np.where(gt == valloader.void_classes, valloader.void_classes, pred[k])

                    if args.measure_time:
                        elapsed_time = timeit.default_timer() - start_time
                        if (i + 1) % 50 == 0:
                            print(
                                "Inference time \
                                  (iter {0:5d}): {1:3.5f} fps".format(
                                    i + 1, pred[-1].shape[0] / elapsed_time
                                )
                            )

                    pred_ious = np.zeros(len(pred), np.float32)
                    diff_of_batch = np.zeros(len(pred), np.float32)
                    for k in range(len(pred)):
                        if running_metrics is None:
                            running_metrics = []
                            for _ in range(len(pred)):
                                running_metrics = running_metrics + [runningScore(n_classes, void=is_void_class)
                                                                     if not roi_only else runningScore(n_classes + 1,
                                                                                                       roi_only)]
                        else:
                            running_metrics[k].update(gt, pred[k], step=k)
                            score, cls_iu = running_metrics[k].get_scores()
                            pred_ious[k] = score['Mean IoU : \t']
                            running_metrics[k].reset()
                            if update_raw and k == len(pred) - 1:
                                running_metrics[k].update_raw(gt, outputs_list[k], step=k)
                        if k > 0:
                            # distance = np.linalg.norm(outputs_list[k]-outputs_list[k-1])
                            distance = np.linalg.norm(mysoftmax(torch.from_numpy(outputs_list[k])) -
                                                      mysoftmax(torch.from_numpy(outputs_list[k - 1])))
                            diff_of_batch[k] = distance
                            # print('recurrence {}, diff {}'.format(k, distance))

                    iou_list.append(pred_ious)
                    diff_list.append(diff_of_batch)

                if loader_type == 0:
                    logger.info('training set performance :')
                    train_iou = np.asarray(iou_list)
                    train_diff = np.asarray(diff_list)
                    stats(train_iou, train_diff, 'train')
                elif loader_type == 1:
                    logger.info('validation set performance :')
                    val_iou = np.asarray(iou_list)
                    val_diff = np.asarray(diff_list)
                    stats(val_iou, val_diff, 'val')
                elif loader_type == 2:
                    logger.info('test set performance')
                    test_iou = np.asarray(iou_list)
                    test_diff = np.asarray(diff_list)
                    stats(test_iou, test_diff, 'test')
                # results[result_tags[loader_type]] = {}

                # for j in range(len(running_metrics)):
                #     score, class_iou = running_metrics[j].get_scores()
                #     logger.info('RNN step {}'.format(j+1))
                #     results[result_tags[loader_type]][j] = {}
                #     for k, v in score.items():
                #         v = float(v)
                #         logger.info(wrap_str(k, v))
                #         results[result_tags[loader_type]][j][k] = v
                #
                #     for i in range(n_classes - 1 if is_void_class else n_classes):
                #         logger.info(wrap_str(f"iou class {i}: \t", class_iou[i]))
                #         results[result_tags[loader_type]][j][f'iou{i}'] = float(class_iou[i])
                #
                #     # if j == 2:
                #     #     p, r, thresh = running_metrics[j].compute_break_even()
                #     #     logger.info(f"P/R Break even at {(p + r) / 2}.")
                # running_metrics = None
        else:
            for loader_type, myloader in enumerate([valloader, testloader]):
                for i, (images, labels) in enumerate(myloader):
                    start_time = timeit.default_timer()

                    images = images.to(device)

                    if args.eval_flip:
                        # Flip images in numpy (not support in tensor)
                        flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
                        flipped_images = torch.from_numpy(flipped_images).float().to(device)

                        outputs = model(images)
                        outputs_flipped = model(flipped_images)

                        outputs = outputs.data.cpu().numpy()
                        outputs_flipped = outputs_flipped.data.cpu().numpy()
                        outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

                        pred = np.argmax(outputs, axis=1)
                    else:
                        outputs = model(images)
                        outputs = outputs.data.cpu().numpy()
                        pred = np.argmax(outputs, axis=1)

                    gt = labels.numpy()

                    if roi_only:
                        pred = np.where(gt == valloader.void_classes, valloader.void_classes, pred)

                    if args.measure_time:
                        elapsed_time = timeit.default_timer() - start_time
                        if (i + 1) % 50 == 0:
                            print(
                                "Inference time \
                                  (iter {0:5d}): {1:3.5f} fps".format(
                                    i + 1, pred.shape[0] / elapsed_time
                                )
                            )
                    running_metrics.update(gt, pred)
                    if update_raw:
                        running_metrics.update_raw(gt, outputs)

                score, class_iou = running_metrics.get_scores()
                if update_raw:
                    p, r, thres = running_metrics.compute_break_even()
                    logger.info("P/R Break even at {}.".format((p + r) / 2))

                results[result_tags[loader_type]] = {}
                if loader_type == 0:
                    logger.info('validation set performance :')

                elif loader_type == 1:
                    logger.info('test set performance')

                for k, v in score.items():
                    logger.info(wrap_str(k, v))
                    results[result_tags[loader_type]][k] = v

                for i in range(n_classes - 1 if is_void_class else n_classes):
                    logger.info(wrap_str("iou class {}: \t".format(i), class_iou[i]))
                    results[result_tags[loader_type]]['iou{}'.format(i)] = class_iou[i]

                # running_metrics = None

    result_path = result_root(cfg, create=True) + '.yml'
    with open(result_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(model_path)
    # IPython.embed()
    clean_logger(logger)
    return results


if __name__ == "__main__":
    parser = validate_parser()
    args = parser.parse_args()

    args.config = 'configs/dataset/eythhand.yml'
    models = [
        # 'dru_eythhand-h128-1-r12-w-0.4-gate3-bs-8-fscale-4/76229',
        # 'dru_eythhand-h128-1-r9-w-0.4-gate3-bs-8-fscale-4/94411'
        # 'dru_eythhand-h128-1-r6-w-0.4-gate3-bs-8-fscale-4/51149'
        'dru_eythhand-h128-1-r3-w-0.4-gate3-bs-8-fscale-4/33301'
        # 'gruunet_eythhand-h32-1-r6-w-0.4-gate3-bs-1/36264',
        # 'gruunet_eythhand-h32-1-r9-w-0.4-gate3-bs-1/55375',
        # 'gruunet_eythhand-h32-1-r6-w-0.4-gate3-bs-1/43168',
        # 'gruunet_eythhand-h32-1-r12-w-0.4-gate3-bs-1/87319',
    ]
    args.model = models[0]

    subfolder = 'iccvablation'

    args.batch_size = 1
    args.feature_scale = 4

    # gruunet_eythhand-h32-1-r3-w-0.4-gate3/75721
    args.eval_flip = False

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    cfg['model']['arch'] = args.model.split("_")[0]
    cfg['model']['hidden_size'] = 128
    args.hidden_size = 128
    cfg['model']['steps'] = 6
    cfg['model']['feature_scale'] = 4

    cfg['training']['resume'] = cfg['model']['arch'] + '_' + cfg['data']['dataset'] + '_best_model.pkl'
    cfg['training']['batch_size'] = 1
    cfg['training']['prefix'] = 'iccvablation'
    # set the log DIR.
    logdir = os.path.join('runs', cfg['data']['dataset'], subfolder, args.model)
    cfg['logdir'] = logdir

    validate(cfg, args)
