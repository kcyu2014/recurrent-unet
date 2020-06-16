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

from PIL import Image
# from scipy.misc import imread
from scipy.ndimage import imread
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from ptsemseg.augmentations import *
# from utils import RNG_SEED
RNG_SEED = 1337


class RoadLoader(data.Dataset):
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
        img_size=(1500, 1500),
        crop_size=(450, 450),
        augmentations=None,
        img_norm=True,
    ):
        self.root = os.path.expandwei(root)
        split = 'valid' if split == 'val' else split # Wrap val to valid.
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 3

        # Important here
        self.void_classes = [2]
        self.valid_classes = [0, 1]
        self.class_names = ['background', 'road', 'void']

        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.tf = {}
        self.random_tf = {}
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        # TODO Fixed now, only for training. To be align with baseline.
        self.crop_size = 448
        # self.crop_size = 450

        for split in ["train", "valid", "test"]:
            path = pjoin(self.root, 'train_val_test_split', split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

            if split in ['train', 'valid']:
                self.random_tf[split] = transforms.Compose([
                    transforms.RandomResizedCrop(size=self.crop_size),
                ])
                self.tf[split] = transforms.Compose([
                    # transforms.ToTensor(),
                    transforms.Normalize([0.393, 0.396, 0.356],
                                         [0.119, 0.118, 0.124])])
            else:
                self.random_tf[split] = None
                self.tf[split] = transforms.Compose([
                    # transforms.ToTensor(),
                    transforms.Normalize([0.340, 0.342, 0.296],
                                         [0.226, 0.220, 0.202])])

        self.to_tensor = transforms.ToTensor()
        self.to_byte = transforms.Lambda(lambda x: np.array(x))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # Processed already, should not contain any non-valid image.
        im_name, lbl_name = self.files[self.split][index].split(',')
        im_path = pjoin(self.root, im_name )
        lbl_path = pjoin(self.root, lbl_name)
        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        # im = imread(im_path)
        # lbl = imread(lbl_path)
        # Add the non-valid label.

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)

        # if len(np.unique(lbl)) == 3 and self.split == 'train':
        #     logger.warning("Contain void value! after cropping!")
            # return self.__getitem__(index)
        return im, lbl

    def transform(self, img, lbl):
        if self.img_size == ('same', 'same'):
            pass
        else:
            pass # Crop happens in Transform
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))

        if self.random_tf[self.split]:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            img = self.random_tf[self.split](img)
            random.seed(seed)
            lbl = self.random_tf[self.split](lbl)
            random.seed(RNG_SEED)

        img = self.to_tensor(img)
        lbl = self.to_byte(lbl)
        if len(lbl.shape) == 3:
            lbl = lbl[0, :, :]

        # val = 128
        lbl[lbl <= 127] = 0
        lbl[lbl > 127] = 1
        # ignore labels
        lbl[img.numpy()[0, :, :] == 1] = 2

        # normalize only after setting the labels
        # img = self.tf[self.split](img)
        # if not np.all(np.unique(lbl[lbl != self.void_classes]))
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    # local_path = '/cvlabdata1/cvlab/datasets_agata/DeepTrace/toronto'
    # need to add more augmentations later !!!
    local_path = '/user/mount/cvdata/cvlab/datasets_kyu/toronto'
    aug = Compose([RandomRotate(10), RandomHorizontallyFlip(0.5)])
    dst = RoadLoader(split='train', root=local_path, is_transform=True, augmentations=aug, img_size=('same', 'same'))
    # dst = RoadLoader(split='test', root=local_path, is_transform=True, img_size=('same', 'same'))
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
