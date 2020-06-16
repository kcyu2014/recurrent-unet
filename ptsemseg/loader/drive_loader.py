import os
import torch
import numpy as np
import scipy.misc as m
from os.path import join as pjoin
from torch.utils import data
from torchvision import transforms
from ptsemseg.augmentations import *
import ast


class driveLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [255, 255, 128],
        [0, 0, 232],
    ]

    label_colours = dict(zip(range(2), colors))

    def __init__(
        self,
        root,
        split="train",
        is_transform=True,
        img_size=(584, 565),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        test_mode=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        for split in ["train", "val", "test"]:
            path = pjoin(self.root, split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            file_list = [ast.literal_eval(triple)[0] for triple in file_list]
            self.files[split] = file_list
            if not self.files[split]:
                raise Exception("No files for split=[%s] found in %s" % (split, self.root))
            print("Found %d %s images" % (len(self.files[split]), split))

        self.void_classes = [0]
        self.valid_classes = [1, 2]
        self.class_names = ["unlabelled", "vessel", "white"]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(2)))
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        im_name, lbl_name, mask_name = self.files[self.split][index]
        img_path = pjoin(self.root, im_name)
        lbl_path = pjoin(self.root, lbl_name)
        mask_path = pjoin(self.root, mask_name)

        img = m.imread(img_path)
        lbl = m.imread(lbl_path)
        mask = m.imread(mask_path)

        lbl = np.array(lbl)
        lbl[lbl == 255] = 1
        lbl[lbl == 0] = 2
        mask = np.array(mask)
        lbl[mask == 0] = 0

        img = Image.fromarray(np.array(img, dtype=np.uint8))
        lbl = Image.fromarray(self.encode_segmap(np.array(lbl, dtype=np.uint8)))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        classes = np.unique(lbl)
        # lbl = lbl.astype(float)

        if self.img_size == ('same', 'same'):
            pass
        else:
            img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")

        img = self.tf(img)
        lbl = np.array(lbl).astype(int)
        # lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    crop_size = 576
    fill = 250
    # augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5), RandomGaussianBlur])
    # augmentations = Compose([RandomScaleCrop(crop_size, fill), RandomRotate(10), RandomHorizontallyFlip(0.5)])
    augmentations = {'brightness': 63. / 255.,
                     'saturation': 0.5,
                     'contrast': 0.8,
                     'hflip': 0.5,
                     'rotate': 10,
                     'rscalecropsquare': 576,
                     }
    data_aug = get_composed_augmentations(augmentations)
    local_path = '/cvlabdata2/home/user/data/DRIVE/'
    dst = driveLoader(local_path, img_size=576, is_transform=True,
                      img_norm=False, augmentations=data_aug)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        # import pdb
        # pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1]).astype(np.uint8)
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
