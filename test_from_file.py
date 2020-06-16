"""
Serve as a convenient wrapper for validate.py.

"""
import gc
import os
import timeit

import IPython
import torch
import numpy as np
import scipy.misc as misc

import yaml

from torch.utils import data
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger

from utils import test_parser
from validate import validate, load_model_and_preprocess
from validate_from_file import load_complete_info_from_dir, final_run_dirs

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


ROOT = '/cvlabdata2/home/kaicheng/pycharm/pytorch-semseg'
M_ROOT = "runs/"
os.chdir(ROOT)


def _save_output(img_path_target, outputs, resized_img, loader, cfg):
    """
    Better code reusing.
    :param outputs:
    :param resized_img:
    :return:
    """
    if args.dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = os.path.splitext(img_path_target)[0] + "_drf.png"
        misc.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    if isinstance(outputs, torch.Tensor):
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    else:
        pred = np.squeeze(outputs, axis=0)
    if cfg['model']['arch'] in ["pspnet", "icnet", "icnetBN"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        pred = misc.imresize(pred, cfg['data']['orig_size'], "nearest", mode="F")
    if cfg['data']['dataset'] == 'drive':
        pred = misc.imresize(pred, (584, 565), "nearest", mode="F")
    decoded = pred
    print("Classes found: ", np.unique(pred))
    misc.imsave(img_path_target, decoded)
    print("Segmentation Mask Saved at: {}".format(img_path_target))


def output_masks_to_files(outputs, loader, resized_img, img_name, output_dir, args, cfg):
    """
    Save the prediction in ORIGINAL image size. to output_dir

    :param outputs:
    :param loader:
    :param resized_img:
    :param img_name: img_name
    :param args:
    :param cfg:
    :return:
    """
    # img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_name = img_name.replace('/', '-')
    out_path = output_dir
    if args.is_recurrent:
        for step, output in enumerate(outputs):
            img_path_target = os.path.join(out_path, img_name + '_step{}.png'.format(step + 1))
            _save_output(img_path_target, output, resized_img, loader, cfg)
    else:
        img_path_target = os.path.join(out_path, img_name + '.png')
        _save_output(img_path_target, outputs, resized_img, loader, cfg)


def _evaluate_from_model(model, images, args, cfg, n_classes, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            outputs_flipped = model(flipped_images, h0, s0)

        elif cfg['model']['arch'] in ['druvgg16', 'druresnet50']:
            W, H = images.shape[2], images.shape[3]
            w, h = int(W / 2 ** 4), int(H / 2 ** 4)
            if cfg['model']['arch'] in ['druresnet50']:
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

        if type(outputs) is list:
            outputs_list = [output.data.cpu().numpy() for output in outputs]
            outputs_flipped_list = [output_flipped.data.cpu().numpy() for output_flipped in outputs_flipped]
            outputs_list = [(outputs + outputs_flipped[:, :, :, ::-1]) / 2.0 for
                            outputs, outputs_flipped in zip(outputs_list, outputs_flipped_list)]
            pred = [np.argmax(outputs, axis=1) for outputs in outputs_list]
        else:
            outputs = outputs.data.cpu().numpy()
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0
            pred = np.argmax(outputs, axis=1)

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

        elif cfg['model']['arch'] in ['druvgg16', 'druresnet50']:
            W, H = images.shape[2], images.shape[3]
            w, h = int(W / 2 ** 4), int(H / 2 ** 4)
            if cfg['model']['arch'] in ['druresnet50']:
                w, h = int(W / 2 ** 5), int(H / 2 ** 5)
            h0 = torch.ones([images.shape[0], args.hidden_size, w, h],
                            dtype=torch.float32, device=device)
            s0 = torch.zeros([images.shape[0], n_classes, W, H],
                             dtype=torch.float32, device=device)
            outputs = model(images, h0, s0)

        else:
            outputs = model(images)

        outputs_list = [output.data.cpu().numpy() for output in outputs]
        if len(outputs_list)>1:
            outputs_list = [output.data.cpu().numpy() for output in outputs]
            pred = [np.argmax(outputs, axis=1) for outputs in outputs_list]
        else:
            outputs = outputs.data.cpu().numpy()
            pred = np.argmax(outputs, axis=1)

    # if args.eval_flip:
    #     outputs = model(images)
    #     # Flip images in numpy (not support in tensor)
    #     flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
    #     flipped_images = torch.from_numpy(flipped_images).float().to(args.device)
    #     outputs_flipped = model(flipped_images)
    #
    #     if args.is_recurrent:
    #         outputs_list = [output.data.cpu().numpy() for output in outputs]
    #         outputs_flipped_list = [output_flipped.data.cpu().numpy() for output_flipped in outputs_flipped]
    #         outputs_list = [(outputs + outputs_flipped[:, :, :, ::-1]) / 2.0 for
    #                         outputs, outputs_flipped in zip(outputs_list, outputs_flipped_list)]
    #         pred = [np.argmax(outputs, axis=1) for outputs in outputs_list]
    #     else:
    #         outputs = outputs.data.cpu().numpy()
    #         outputs_flipped = outputs_flipped.data.cpu().numpy()
    #         outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0
    #         pred = np.argmax(outputs, axis=1)
    # else:
    #     outputs = model(images)
    #     if args.is_recurrent:
    #         pred = [output.data.max(1)[1].cpu().numpy() for output in outputs]
    #     else:
    #         pred = outputs.data.max(1)[1].cpu().numpy()

    return pred


def test_with_cfg(cfg, args):

    logger = get_logger(cfg['logdir'], 'test')
    device = torch.device(args.device)
    out_path = cfg['eval_out_path']

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Setup image
    valid_images = [".jpg", ".gif", ".png", ".tga", ".tif", ".tiff"]

    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    print("Read Input Image from : {}".format(data_path))
    print(f"Save the output to : {cfg['eval_out_path']}")

    loader = data_loader(
        data_path,
        is_transform=True, # Return the original image without any augmentation.
        split=cfg['data']['test_split'],
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
    )
    im_loader = data_loader(
        data_path,
        is_transform=False, # Return the original image without any augmentation.
        split=cfg['data']['test_split'],
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
    )

    testloader = data.DataLoader(
        loader,
        batch_size=1,
        num_workers=2
    )

    roi_only = 'roi' in cfg['data']['dataset']

    n_classes = loader.n_classes

    # Setup Model
    model, model_path = load_model_and_preprocess(cfg, args, n_classes, device)
    logger.info(f"Loading model {cfg['model']['arch']} from {model_path}")

    # model_file_name = os.path.split(args.model_path)[1]
    # model_name = cfg['model']['arch']
    # flag_subf = False

    # Replace the entire loader, like doing the validation.
    # IPython.embed()
    with torch.no_grad():
        # For all the images in this loader.
        for i, (images, labels) in enumerate(testloader):
            # TODO DEBUGING here.
            # if i > 2:
            #     break

            (orig_img, org_lab) = im_loader[i]
            img_name = loader.files[loader.split][i]
            if type(img_name) is list:
                img_name = img_name[0]
            start_time = timeit.default_timer()
            images = images.to(device)
            n_classes = loader.n_classes
            pred = _evaluate_from_model(model, images, args, cfg, n_classes, device)
            gt = labels.numpy()

            # CHeck the org_lab == labels
            # IPython.embed()

            if roi_only:
                """ Process for ROI, basically, mask the Pred based on GT"""
                # IPython.embed()
                # if args.is_recurrent:
                if type(pred) is list:
                    for k in range(len(pred)):
                        pred[k] = np.where(gt == loader.void_classes, loader.void_classes, pred[k])
                        if cfg['data']['dataset'] == 'drive':
                            pred[k] = pred[k] + 1
                            pred[k][gt == 250] = 2
                            pred[k][pred[k] == 2] = 0
                else:
                    pred = np.where(gt == loader.void_classes, loader.void_classes, pred)
                    if cfg['data']['dataset'] == 'drive':
                        pred = pred + 1
                        pred[gt == 250] = 2
                        pred[pred == 2] = 0

            if type(pred) is list:
                for k in range(len(pred)):
                    pred[k] = np.where(gt == loader.void_classes, loader.void_classes, pred[k])
                    if cfg['data']['dataset'] == 'drive':
                        pred[k] = pred[k] + 1
                        pred[k][gt == 250] = 2
                        pred[k][pred[k] == 2] = 0
            else:
                pred = np.where(gt == loader.void_classes, loader.void_classes, pred)
                if cfg['data']['dataset'] == 'drive':
                    pred = pred + 1
                    pred[gt == 250] = 2
                    pred[pred == 2] = 0

            output_masks_to_files(pred, loader, orig_img, img_name, out_path, args, cfg)

            # Other unrelated stuff
            if args.measure_time:
                elapsed_time = timeit.default_timer() - start_time
                if (i + 1) % 50 == 0:
                    print(
                        "Inference time \
                          (iter {0:5d}): {1:3.5f} fps".format(
                            i + 1, pred[-1].shape[0] / elapsed_time
                        )
                    )


def run_test(args, run_dirs):
    """
    run multiple validation here.
    """

    for i, r_dir in enumerate(run_dirs):
        cfg, args = load_complete_info_from_dir(r_dir, args)
        # IPython.embed()
        test_with_cfg(cfg, args)
        cfg = None
        gc.collect()


if __name__ == '__main__':
    parser = test_parser()
    args = parser.parse_args()
    run_dirs = final_run_dirs(args)
    del args.dataset
    run_test(args, run_dirs)

