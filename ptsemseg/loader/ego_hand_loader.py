import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
# import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from ptsemseg.augmentations import *


class egoHandLoader(data.Dataset):
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
        img_size=(720, 1280),  # ('same', 'same'),
        augmentations=None,
        img_norm=True,
    ):
        self.root = os.path.expandwei(root)
        self.im_root = pjoin(self.root, "_LABELLED_SAMPLES")
        self.lbl_root = pjoin(self.root, "_LABELLE")

        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        for split in ["train", "val", "trainval", "test"]:
            path = pjoin(self.root, split + ".txt")
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
        im_path = pjoin(self.root, "_LABELLED_SAMPLES", im_name + ".jpg")
        lbl_path = pjoin(self.root, "_LABELLE", im_name + ".png")
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
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
        lbl = torch.from_numpy(lbl).long()
        lbl[lbl == 255] = 1
        return img, lbl


# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    local_path = '/cvlabdata2/home/user/data/egohands_data'
    # local_path = '/home/user/Desktop/CVLab_Logitech/logitech_seg'
    # need to add more augmentations later !!!
    aug = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5)])
    dst = egoHandLoader(root=local_path, is_transform=True, augmentations=aug)
    bs = 2
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()  # [:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels.numpy()[j])
            plt.show()
            a = 'ex'
            if a == 'ex':
                break
            else:
                plt.close()
