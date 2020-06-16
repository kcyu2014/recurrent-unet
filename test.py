import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(args):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img_paths = []
    valid_images = [".jpg", ".gif", ".png", ".tga"]

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    # Setup Model
    model_name_dic = {'arch': model_name}
    model = get_model(model_name_dic, n_classes, version=args.dataset, args=args)
    state = convert_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    flag_subf = False
    for f in os.listdir(args.img_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            src_subpath = os.path.join(args.img_path, f)
            tgt_subpath = os.path.join(args.out_path, f)
            if os.path.exists(src_subpath):
                if not os.path.exists(tgt_subpath):
                    os.makedirs(tgt_subpath)
                for subf in os.listdir(src_subpath):
                    ext = os.path.splitext(subf)[1]
                    if ext.lower() not in valid_images:
                        continue
                    img_paths.append(os.path.join(src_subpath, subf))
                    flag_subf = True
            else:
                continue
        else:
            img_paths.append(os.path.join(args.img_path, f))

    for img_path in img_paths:
        img = misc.imread(img_path)

        resized_img = misc.imresize(
            img, (loader.img_size[0], loader.img_size[1]), interp="bicubic"
        )

        orig_size = img.shape[:-1]
        if model_name in ["pspnet", "icnet", "icnetBN"]:
            # uint8 with RGB mode, resize width and height which are odd numbers
            img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
        else:
            img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

        if loader.is_transform:
            img = loader.tf(img)
            img = img.unsqueeze(0)
        else:
            img = img[:, :, ::-1]
            img = img.astype(np.float64)
            img -= loader.mean
            if args.img_norm:
                img = img.astype(float) / 255.0
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

        with torch.no_grad():
            images = img.to(device)
            outputs = model(images)

        if model_name in ["vanillaRNNunet", 'rcnn']:
            for step, output in enumerate(outputs):
                if args.dcrf:
                    unary = output.data.cpu().numpy()
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
                    dcrf_path = args.out_path[:-4] + "_step{}_drf.png".format(step+1)
                    misc.imsave(dcrf_path, decoded_crf)
                    print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

                pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
                decoded = pred
                print("Classes found: ", np.unique(pred))
                img_name = os.path.basename(img_path)[:-4]
                out_path = args.out_path
                if flag_subf:
                    parent_path = img_path.split(os.sep)[-2]
                    out_path = os.path.join(out_path, parent_path)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                img_path_target = os.path.join(out_path, img_name + '_step{}.png'.format(step+1))
                misc.imsave(img_path_target, decoded)
                print("Segmentation Mask Saved at: {}".format(out_path))
        else:
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
                dcrf_path = args.out_path[:-4] + "_drf.png"
                misc.imsave(dcrf_path, decoded_crf)
                print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            if model_name in ["pspnet", "icnet", "icnetBN"]:
                pred = pred.astype(np.float32)
                # float32 with F mode, resize back to orig_size
                pred = misc.imresize(pred, orig_size, "nearest", mode="F")

            decoded = pred
            print("Classes found: ", np.unique(pred))
            img_name = os.path.basename(img_path)[:-4]
            out_path = args.out_path
            if flag_subf:
                parent_path = img_path.split(os.sep)[-2]
                out_path = os.path.join(out_path, parent_path)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            img_path_target = os.path.join(out_path, img_name + '.png')
            misc.imsave(img_path_target, decoded)
            print("Segmentation Mask Saved at: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument("--device", nargs="?", type=str, default="cuda:0", help="GPU or CPU to use")
    parser.add_argument("--steps", nargs="?", type=int, default=3, help="Recurrent Steps")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    test(args)

# python test.py --model_path=./runs/fcn8s_hand/71064/fcn8s_epfl_hand_best_model.pkl --dataset=epfl_hand
# --img_path=/cvlabdata2/home/user/data/epfl-logitech-CTI/test/img
# --out_path=/cvlabdata2/home/user/data/epfl-logitech-CTI/test/gt_p_fcn8s_71064

# python test.py --model_path=./runs/unet_hand/64653/unet_epfl_hand_best_model.pkl --dataset=epfl_hand
# --img_path=/cvlabdata2/home/user/data/epfl-logitech-CTI/test/img
# --out_path=/cvlabdata2/home/user/data/epfl-logitech-CTI/test/gt_p_unet_64653

# python test.py --model_path=./runs/fcn8s_egohand/23134/fcn8s_ego_hand_best_model.pkl --dataset=ego_hand --img_path=/cvlabdata2/home/user/data/egohands_data/test/img --out_path=/cvlabdata2/home/user/data/egohands_data/test/gt_p_fcn8s_23134
# python test.py --model_path=./runs/unet_egohand/51514/unet_ego_hand_best_model.pkl --dataset=ego_hand --img_path=/cvlabdata2/home/user/data/egohands_data/test/img --out_path=/cvlabdata2/home/user/data/egohands_data/test/gt_p_unet_51514

# python test.py --model_path=./runs/vanillaRNNunet_egohand/3028/vanillaRNNunet_ego_hand_best_model.pkl --dataset=ego_hand --img_path=/cvlabdata2/home/user/data/egohands_data/test/img --out_path=/cvlabdata2/home/user/data/egohands_data/test/gt_p_vanillaRNNunet_3028
# python test.py --model_path=./runs/rcnn_egohand/63888/rcnn_ego_hand_best_model.pkl --dataset=ego_hand --img_path=/cvlabdata2/home/user/data/egohands_data/test/img --out_path=/cvlabdata2/home/user/data/egohands_data/test/gt_p_rcnn_63888
