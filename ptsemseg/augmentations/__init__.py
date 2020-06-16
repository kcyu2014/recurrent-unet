import logging
from ptsemseg.augmentations.augmentations import *

logger = logging.getLogger('ptsemseg')

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'rcrop': RandomCrop,
           'hflip': RandomHorizontallyFlip,
           'vflip': RandomVerticallyFlip,
           'scale': Scale,
           'rsize': RandomSized,
           'rsizecrop': RandomSizedCrop,
           'rotate': RandomRotate,
           'translate': RandomTranslate,
           'ccrop': CenterCrop,
           'rscalecrop': RandomScaleCrop,
           'rscalecropsquare': RandomScaleCropSquare,
           'gaussian': RandomGaussianBlur}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)


