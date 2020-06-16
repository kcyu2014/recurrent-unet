import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from ptsemseg.augmentations import *


def check_and_save_only_valid_bbox(file_list, save_path, load_path):
    """
    check and save only valid BBox to the given location.
    :param file_list:
    :param save_path:
    :param load_path:
    :return:
    """

    def _corners_to_polygon(bbox, scale=2):
        poly = []
        if isinstance(bbox, dict):
            if 'annotated_corners' in bbox.keys():
                for i in range(4, 8):
                    try:
                        poly.append(
                            tuple([i / scale for i in bbox['annotated_corners'][str(i)][:2]]))
                    except KeyError as e:
                        pass
        return poly
    final_list = []
    for ind, l in enumerate(file_list):
        path = os.path.join(load_path, l + '.json')
        try:
            with open(path, 'r') as f:
                bbox = json.load(f)
            poly = _corners_to_polygon(bbox)
            if len(poly) > 3:
                final_list.append(l)
        except Exception as e:
            print(e)
    # print(f"{ind} / {len(file_list)} are valid data points. Saving them now. to {save_path}")
    print("{} / {} are valid data points. Saving them now. to {}".format(ind, len(file_list), save_path))
    with open(save_path, 'w') as f:
        f.writelines([l + '\n' for l in final_list])


def corners_to_polygon(bbox, mask, scale=2):
    """
    ONLY WORKS FOR THE EPFL HAND. Since the corners are hard-coded
    :param bbox:
    :param mask:
    :return:
    """
    poly = []
    if isinstance(bbox, dict):
        if 'annotated_corners' in bbox.keys():
            for i in range(4, 8):
                try:
                    poly.append(
                        tuple([i / scale for i in bbox['annotated_corners'][str(i)][:2]]))
                except KeyError as e:
                    pass
    if len(poly) < 3:
        logger.warning("Not valid bounding box. {}.".format(poly))
        return poly, mask, None

    roi = Image.new('RGB', mask.size)
    pdraw = ImageDraw.Draw(roi)
    pdraw.polygon(poly, fill=(255, 255, 255), outline=(255, 255, 255))
    roi.resize(mask.size)

    # Will force this to be 0, 1.
    np_mask = np.array(mask) / 255.
    if len(np_mask.shape) == 3:
        np_mask = np_mask[:,:,0]
    np_roi = np.array(roi)[:,:, 0] / 255.
    np_mask_new = np.where(np_roi == 1., np_mask, 2*np.ones_like(np_mask))
    assert len(np.unique(np_mask_new)) <= 3
    roi_mask = Image.fromarray(np_mask_new / 2. * 255)
    # poly.show()
#     mask.show()
    return poly, mask, roi_mask


class epflHandLoader(data.Dataset):
    """Data loader for the hand segmentation dataset.
    A total of 3 data splits are provided for working with the hand segmentation data:
        train: - *** images
        val: - *** images
        trainval: The combination of `train` and `val` - *** images
    """

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        # img_size=(230, 306),  # (460, 612),
        img_size=(306, 230),  # (460, 612),
        augmentations=None,
        img_norm=True,
        bounding_box=False
    ):
        self.root = os.path.expandwei(root)
        self.im_root = pjoin(self.root, "_LABELLED_SAMPLES")
        self.lbl_root = pjoin(self.root, "_LABELLE")

        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.bounding_box = bounding_box
        self.n_classes = 2
        self.void_classes = 2
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        for split in ["train", "val", "trainval", "test"]:
            path = pjoin(self.root, 'train_val_test_split', split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, 'data_split_by_seq', im_name + '_halfres.jpg')
        lbl_path = pjoin(self.root, 'data_split_by_seq', im_name + '_mask.png')
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        if self.bounding_box:
            js_path = pjoin(self.root, 'data_split_by_seq', im_name + '.json')
            try:
                with open(js_path, 'r') as f:
                    bbox = json.load(f)
                _, old_lbl, roi_lbl = corners_to_polygon(bbox, lbl)
                lbl = roi_lbl if roi_lbl is not None else lbl
            except FileNotFoundError as e:
                logger.error(e)

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ('same', 'same'):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        lbl = np.array(lbl)
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]
        if self.bounding_box:
            lbl[lbl < 100] = 0
            lbl[lbl >= 200] = 2
            lbl = np.where(np.logical_and(lbl < 200, lbl >=100), np.ones_like(lbl), lbl)
            # lbl[] = 1
            # lbl[lbl != 0 and lbl != 2] = 1
            assert len(np.unique(lbl)) <= 3, "Serious error in converting ROI."
            lbl = torch.from_numpy(lbl).long()
        else:
            lbl = torch.from_numpy(lbl).long()
            lbl[lbl == 255] = 1
        return img, lbl


class EPFLROILoader(epflHandLoader):
    
    def __init__(self, root, bounding_box=True, **kwargs):
        super(EPFLROILoader, self).__init__(root, bounding_box=True,**kwargs)
        self.root = '/cvlabdata1/cvlab/datasets_kyu/logitech/annotations'
        logger.warning("Not this is only supported in {}. ".format(self.root) +
                       "If you need to use this ROI, please consider copy " +
                       "./roi-split to your dataset location.")
        for split in ["train", "val", "trainval", "test"]:
            path = pjoin(os.path.dirname(self.root), 'roi-split', split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def create_roi_filter(self):
        files = {}
        save_root = pjoin(os.path.dirname(self.root), 'roi-split')
        for split in ["train", "val", "trainval", "test"]:
            path = pjoin(self.root, 'train_val_test_split', split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            check_and_save_only_valid_bbox(
                file_list, save_path=pjoin(save_root, split + '.txt'), load_path=self.root)
            files[split] = file_list




# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    local_path = '/cvlabdata1/cvlab/datasets_hugonot/Logitech-CVLab_dataset_annotations'
    # local_path = '/cvlabdata2/home/user/data/epfl-logitech-CTI'
    # local_path = '/home/user/Desktop/CVLab_Logitech/logitech_seg'
    # local_path = '/home/kyu/mount/cvlabdata3/.keras/datasets/Logitech-CVLab_dataset_annotations'
    # need to add more augmentations later !!!
    aug = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5)])
    dst = epflHandLoader(root=local_path, is_transform=True, augmentations=aug, bounding_box=True)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        lab = labels.numpy()
        lab = lab.astype(np.float32) / 2.
        # lab[imgs[:,:,:,0] == 1.] = 1.
        lab = np.concatenate([np.expand_dims(lab, axis=3)] * 3, axis=3)

        f, axarr = plt.subplots(bs, 2, figsize=(10, 20))
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(lab[j])

        plt.show()
        plt.close()
