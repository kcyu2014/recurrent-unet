import logging
import os

import IPython
import yaml
import time
import shutil
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter
from ptsemseg.models.utils import MergeParametric
from utils import train_parser, validate_parser, RNG_SEED
from validate import validate, wrap_str
from ptsemseg.models.sync_batchnorm.replicate import patch_replication_callback


def best_model_path(cfg):
    return "{}_{}_best_model.pkl".format(
        cfg['model']['arch'],
        cfg['data']['dataset'])


def overwrite(cfg, args):

    if args.scale_weight > 0.:
        cfg['training']['loss']['scale_weight'] = args.scale_weight
    if args.model != "":
        cfg['model']['arch'] = args.model
    # update the model parameters in this config.
    cfg['model']['initial'] = args.initial
    cfg['model']['steps'] = args.steps
    cfg['model']['gate'] = args.gate
    cfg['model']['hidden_size'] = args.hidden_size
    cfg['model']['feature_scale'] = args.feature_scale
    if args.batch_size != 0:
        cfg['training']['batch_size'] = args.batch_size
    if args.lr_n != 0:
        cfg['training']['optimizer']['lr'] = args.lr_n*1.0/(10**args.lr_exponent)

    # These two only operates on R-UNet.
    if cfg['model']['arch'] == "runet":
        cfg['model']['unet_level'] = args.unet_level
        cfg['model']['recurrent_level'] = args.recurrent_level
    cfg['training']['prefix'] = args.prefix

    """ Change the valid steps, if baseline model """
    if args.prefix == 'baseline':
        cfg['training']['val_interval'] = 6000
        cfg['training']['print_interval'] = 500

    if args.prefix == 'benchmark':
        cfg['training']['val_internal'] = 10
        cfg['training']['train_internal'] = 11
        cfg['training']['print_interval'] = 10

    cfg['training']['loss']['name'] = args.loss

    if args.loss == 'cross_entropy':
        del cfg['training']['loss']['scale_weight']

    # manipulate learning rate
    args.lr = float(args.lr)
    cfg['training']['optimizer']['lr'] = float(cfg['training']['optimizer']['lr'])
    print("args.lr is {}".format(args.lr))
    if 1 > float(args.lr) > 0:
        # Only override if lr is given positive value from (0,1)
        cfg['training']['optimizer']['lr'] = args.lr

    # Over write the void class
    if not cfg['data'].get('void_class'):
        cfg['data']['void_class'] = -1
    return cfg


def load_cfg_with_overwrite(args):
    # Overwrite the figure.
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    cfg = overwrite(cfg, args)

    cfg['run_id'] = run_id = random.randint(1, 100000)
    config_name = os.path.basename(args.config)[:-4]
    config_name = args.model + '_' + config_name if len(args.model) > 0 else config_name

    if use_grad_clip(cfg['model']['arch']) and cfg['model']['arch'] != 'unet':
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-h{}-{}-r{}-w-{}-gate{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.hidden_size,
                                  args.initial,
                                  args.steps,
                                  cfg['training']['loss']['scale_weight'],
                                  args.gate,
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    if cfg['model']['arch'] in ['vanillarnnunet', 'unethidden', 'vanillarnnunetr']:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-h{}-w-{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.initial,
                                  cfg['training']['loss']['scale_weight'],
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    elif cfg['model']['arch'] in ['unet', 'unet_expand', 'unet_expand_all', 'unetbnslim', 'unetgnslim',
                                  'unet_deep_as_dru', 'deeplabv3', 'icnet']:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    elif cfg['model']['arch'] in ['unetvgg11', 'unetvgg16', 'unetvgg16gn', 'unetresnet50', 'unetresnet50bn']:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-bs-{}'.format(
                                  config_name,
                                  args.batch_size,
                              ),
                              str(run_id))

    else:
        logdir = os.path.join('runs', cfg['data']['dataset'], cfg['training']['prefix'],
                              '{}-h{}-{}-r{}-gate{}-bs-{}-fscale-{}'.format(
                                  config_name,
                                  args.hidden_size,
                                  args.initial,
                                  args.steps,
                                  args.gate,
                                  args.batch_size,
                                  args.feature_scale,
                              ),
                              str(run_id))

    cfg['config'] = config_name
    cfg['logdir'] = logdir
    # update the resume accordingly
    cfg['training']['resume'] = best_model_path(cfg)

    # with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
    #     yaml.dump(cfg, fp, default_flow_style=False)
    return cfg


def use_grad_clip(name):
    if 'rcnn' in name:
        return True
    if 'runet' in name:
        return True
    # if "gruunetnew" == name:
    #     return True
    if "gruunet" in name:
        return True
    if "NoParamShare" in name:
        return True
    if "rnnunet" in name:
        return True
    if "rec" in name:
        return True
    if "dru" in name:
        return True
    if "sru" in name:
        return True
    return False


def weights_init(m):
    if isinstance(m, MergeParametric):
        logger.info('initializing merge layer ...')
        pass
    elif isinstance(m, nn.Conv2d):
        logger.warning(f'initializing {m}')
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        logger.warning(f'initializing {m}')
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_model(model):
    for m in model.paramGroup2.modules():
        if isinstance(m, nn.GroupNorm) or isinstance(m, _BatchNorm):
            logger.warning(f'initializing {m}')
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            weights_init(m)
    # for name, param in model.named_parameters():
    #     if 'encoder' in name:
    #         print('ignoring pre-trained layer ', name)
    #         pass
    #     else:
    #         print('initializing layer ', name)
    #         if 'weight' in name:
    #             nn.init.kaiming_normal_(param.data)
    #         elif 'bias' in name:
    #             nn.init.zeros_(param.data)
    #         else:
    #             print('error in init ... find this layer ', name)


def train(cfg, writer, logger, args):

    # Setup seeds
    torch.manual_seed(cfg.get('seed', RNG_SEED))
    torch.cuda.manual_seed(cfg.get('seed', RNG_SEED))
    np.random.seed(cfg.get('seed', RNG_SEED))
    random.seed(cfg.get('seed', RNG_SEED))

    # Setup device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)

    # Setup Augmentations
    # augmentations = cfg['training'].get('augmentations', None)
    if cfg['data']['dataset'] in ['cityscapes']:
        augmentations = cfg['training'].get('augmentations',
                                            {'brightness': 63. / 255.,
                                             'saturation': 0.5,
                                             'contrast': 0.8,
                                             'hflip': 0.5,
                                             'rotate': 10,
                                             'rscalecropsquare': 704,  # 640, # 672, # 704,
                                             })
    elif cfg['data']['dataset'] in ['drive']:
        augmentations = cfg['training'].get('augmentations',
                                            {'brightness': 63. / 255.,
                                             'saturation': 0.5,
                                             'contrast': 0.8,
                                             'hflip': 0.5,
                                             'rotate': 180,
                                             'rscalecropsquare': 576,
                                             })
        # augmentations = cfg['training'].get('augmentations',
        #                                     {'rotate': 10, 'hflip': 0.5, 'rscalecrop': 512, 'gaussian': 0.5})
    else:
        augmentations = cfg['training'].get('augmentations', {'rotate': 10, 'hflip': 0.5})
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes, cfg['data']['void_class'] > 0)

    # Setup Model
    print('trying device {}'.format(device))
    model = get_model(cfg['model'], n_classes, args)  # .to(device)

    if cfg['model']['arch'] not in ['unetvgg16', 'unetvgg16gn', 'druvgg16', 'unetresnet50', 'unetresnet50bn',
                                    'druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
        model.apply(weights_init)
    else:
        init_model(model)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # if cfg['model']['arch'] in ['druresnet50syncedbn']:
    #     print('using synchronized batch normalization')
    #     time.sleep(5)
    #     patch_replication_callback(model)

    model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=(3, 2))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}
    if cfg['model']['arch'] in ['unetvgg16', 'unetvgg16gn', 'druvgg16', 'druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
        optimizer = optimizer_cls([
            {'params': model.module.paramGroup1.parameters(), 'lr': optimizer_params['lr'] / 10},
            {'params': model.module.paramGroup2.parameters()}
        ], **optimizer_params)
    else:
        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.warning(f"Model parameters in total: {sum([p.numel() for p in model.parameters()])}")
    logger.warning(f"Trainable parameters in total: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    weight = torch.ones(n_classes)
    if cfg['data'].get('void_class'):
        if cfg['data'].get('void_class') >= 0:
            weight[cfg['data'].get('void_class')] = 0.
    weight = weight.to(device)

    logger.info("Set the prediction weights as {}".format(weight))

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if cfg['model']['arch'] in ['reclast']:
                h0 = torch.ones([images.shape[0], args.hidden_size, images.shape[2], images.shape[3]],
                                dtype=torch.float32)
                h0.to(device)
                outputs = model(images, h0)

            elif cfg['model']['arch'] in ['recmid']:
                W, H = images.shape[2], images.shape[3]
                w = int(np.floor(np.floor(np.floor(W/2)/2)/2)/2)
                h = int(np.floor(np.floor(np.floor(H/2)/2)/2)/2)
                h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                dtype=torch.float32)
                h0.to(device)
                outputs = model(images, h0)

            elif cfg['model']['arch'] in ['dru', 'sru']:
                W, H = images.shape[2], images.shape[3]
                w = int(np.floor(np.floor(np.floor(W/2)/2)/2)/2)
                h = int(np.floor(np.floor(np.floor(H/2)/2)/2)/2)
                h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                dtype=torch.float32)
                h0.to(device)
                s0 = torch.ones([images.shape[0], n_classes, W, H],
                                dtype=torch.float32)
                s0.to(device)
                outputs = model(images, h0, s0)

            elif cfg['model']['arch'] in ['druvgg16', 'druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
                W, H = images.shape[2], images.shape[3]
                w, h = int(W / 2 ** 4), int(H / 2 ** 4)
                if cfg['model']['arch'] in ['druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
                    w, h = int(W / 2 ** 5), int(H / 2 ** 5)
                h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                                dtype=torch.float32, device=device)
                s0 = torch.zeros([images.shape[0], n_classes, W, H],
                                 dtype=torch.float32, device=device)
                outputs = model(images, h0, s0)

            else:
                outputs = model(images)

            loss = loss_fn(input=outputs, target=labels, weight=weight, bkargs=args)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # if use_grad_clip(cfg['model']['arch']):  #
            # if cfg['model']['arch'] in ['rcnn', 'rcnn2', 'rcnn3']:  #
            if use_grad_clip(cfg['model']['arch']):
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                # print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                torch.backends.cudnn.benchmark = False
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        if args.benchmark:
                            if i_val > 10:
                                break
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        if cfg['model']['arch'] in ['reclast']:
                            h0 = torch.ones([images_val.shape[0], args.hidden_size, images_val.shape[2], images_val.shape[3]],
                                            dtype=torch.float32)
                            h0.to(device)
                            outputs = model(images_val, h0)

                        elif cfg['model']['arch'] in ['recmid']:
                            W, H = images_val.shape[2], images_val.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images_val.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32)
                            h0.to(device)
                            outputs = model(images_val, h0)

                        elif cfg['model']['arch'] in ['dru', 'sru']:
                            W, H = images_val.shape[2], images_val.shape[3]
                            w = int(np.floor(np.floor(np.floor(W / 2) / 2) / 2) / 2)
                            h = int(np.floor(np.floor(np.floor(H / 2) / 2) / 2) / 2)
                            h0 = torch.ones([images_val.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32)
                            h0.to(device)
                            s0 = torch.ones([images_val.shape[0], n_classes, W, H],
                                            dtype=torch.float32)
                            s0.to(device)
                            outputs = model(images_val, h0, s0)

                        elif cfg['model']['arch'] in ['druvgg16', 'druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
                            W, H = images_val.shape[2], images_val.shape[3]
                            w, h = int(W / 2**4), int(H / 2**4)
                            if cfg['model']['arch'] in ['druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
                                w, h = int(W / 2 ** 5), int(H / 2 ** 5)
                            h0 = torch.ones([images_val.shape[0], args.hidden_size, w, h],
                                            dtype=torch.float32)
                            h0.to(device)
                            s0 = torch.zeros([images_val.shape[0], n_classes, W, H],
                                             dtype=torch.float32)
                            s0.to(device)
                            outputs = model(images_val, h0, s0)

                        else:
                            outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val, bkargs=args)

                        if cfg['training']['loss']['name'] in ['multi_step_cross_entropy']:
                            pred = outputs[-1].data.max(1)[1].cpu().numpy()
                        else:
                            pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()
                        logger.debug('pred shape: ', pred.shape, '\t ground-truth shape:',gt.shape)
                        # IPython.embed()
                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())
                    # assert i_val > 0, "Validation dataset is empty for no reason."
                torch.backends.cudnn.benchmark = True
                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))
                # IPython.embed()
                score, class_iou, _ = running_metrics_val.get_scores()
                for k, v in score.items():
                    # print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i+1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             best_model_path(cfg))
                    torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                save_path = os.path.join(writer.file_writer.get_logdir(),
                                         "{}_{}_final_model.pkl".format(
                                             cfg['model']['arch'],
                                             cfg['data']['dataset']))
                torch.save(state, save_path)
                break


if __name__ == "__main__":

    parser = train_parser()
    args = parser.parse_args()

    cfg = load_cfg_with_overwrite(args)

    run_id = cfg['run_id']
    logdir = cfg['logdir']

    writer = SummaryWriter(log_dir=logdir)

    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(cfg, fp, default_flow_style=False)

    print('RUNDIR: {}'.format(logdir))

    # Write the config file to logdir
    # shutil.copy(args.config, logdir)

    logger = get_logger(logdir, level=logging.WARN if args.prefix == 'benchmark' else logging.INFO)
    logger.info('Let the games begin')

    try:
        train(cfg, writer, logger, args)
    # except (RuntimeError, KeyboardInterrupt) as e:
    except (KeyboardInterrupt) as e:
        logger.error(e)

    logger.info("\nValidate the training result...")
    valid_parser = validate_parser(parser)
    valid_args = valid_parser.parse_args()
    # set the model path.
    # valid_args.steps = 3
    if args.prefix == 'benchmark':
        valid_args.benchmark = True
    valid_args.model_path = os.path.join(cfg['logdir'], best_model_path(cfg))
    validate(cfg, valid_args)

# --config=configs/dataset/eythhand.yml --model=unetvgg16 --lr=1e-8 /
# --batch_size=8 --structure=unetvgg16 --loss=cross_entropy --prefix=iccvablation

# python train_hand.py --config=configs/dataset/eythhand.yml --model=unetvgg16gn --lr=1e-8 /
# --batch_size=8 --structure=unetvgg16gn --loss=cross_entropy --prefix=iccvablation

# --config=configs/dataset/eythhand.yml --model=unetresnet50 --lr=1e-8 --batch_size=8 --structure=unetresnet50 --loss=cross_entropy --prefix=iccvablation

