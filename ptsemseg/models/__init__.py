import copy
import torchvision.models as models
from ptsemseg.models.rcnn import *
from ptsemseg.models.rcnn2 import *
from ptsemseg.models.vanillaRNNunet import *
from ptsemseg.models.refinenet import *
from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *
from ptsemseg.models.recurrent_unet import *
from ptsemseg.models.reclast import *
from ptsemseg.models.recmid import *
from ptsemseg.models.dru import *
from ptsemseg.models.unetvgg import *
from ptsemseg.models.druvgg import *
from ptsemseg.models.unetresnet import *
from ptsemseg.models.druresnet import *
from ptsemseg.models.druresnetsyncedbn import *
from ptsemseg.models.sru import *

from ptsemseg.models.deeplabv3 import *

def get_model(model_dict, n_classes, args, version=None):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["vanillarnnunet", "vanillarnnunetr", "vanillaRNNunet", "vanillaRNNunet_NoParamShare"]:
        model = model(args=args, n_classes=n_classes, **param_dict)

    elif name == "rcnn":
        model = model(args=args, n_classes=n_classes, **param_dict)

    elif name == "rcnn2":
        model = model(args=args, n_classes=n_classes, **param_dict)

    elif name == "rcnn3":
        model = model(args=args, n_classes=n_classes, **param_dict)

    elif name in ["runet", "unethidden", 'runethidden']:
        model = model(args=args, n_classes=n_classes, **param_dict)

    elif name in ["gruunet", "gruunetr", 'gruunetnew', 'gruunetold', 'reclast', 'recmid', 'dru', 'sru',
                  'druvgg16', 'druresnet50', 'druresnet50bn', 'druresnet50syncedbn']:
        model = model(args=args, n_classes=n_classes, **param_dict)

    elif name == "refinenet":
        model = model(n_classes=n_classes, imagenet=True)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name in ['unetgn', 'unetbn', 'unetbnslim', 'unetgnslim', 'unet_deep_as_dru',
                  "unet", "unet_expand_all", "unet_expand",
                  "unetvgg11", "unetvgg16", "unetvgg16gn",
                  "unetresnet50", "unetresnet50bn"]:
        model = model(n_classes=n_classes)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)
    elif name == 'deeplabv3':
        model = model(n_classes=n_classes, backbone='resnet')
    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "rcnn": rcnn,
            "rcnn2": rcnn2,
            "rcnn3": rcnn3,
            "runet": GeneralRecurrentUnet,  # Ours-DRU
            "runethidden": GeneralRecurrentUnet_hidden,  # Rec-Middle
            "unethidden": UNetOnlyHidden,
            "gruunet": UNetWithGRU,  # Rec-Last
            "reclast": reclast,  # reclast
            "recmid": recmid,  # recmid
            "dru": dru,  # dru
            "sru": sru,  # sru
            "druvgg16": druvgg16,
            "druresnet50": druresnet50,
            "druresnet50bn": druresnet50bn,
            "druresnet50syncedbn": druresnet50syncedbn,
            "gruunetnew": UNetWithGRU_R,
            "gruunetr": UNetWithGRU_R,
            "gruunetold": UNetWithGRU_old,
            "vanillaRNNunet": vanillaRNNunet_R,
            "vanillaRNNunet_NoParamShare": vanillaRNNunet_NoParamShare,
            "vanillarnnunet": vanillaRNNunet,
            "vanillarnnunetr": vanillaRNNunet_R,
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,  # U-Net-Group-Norm
            "unetvgg11": UNet11,
            "unetvgg16": UNet16,  # pre-trained vgg16 as encoder
            "unetresnet50": UNetResNet50,  # pre-trained res-net50 as encoder
            "unetresnet50bn": UNetResNet50bn,
            "unetvgg16gn": UNet16GN,
            "unet_expand_all": unet_expand_all,  # sguo_v1
            "unet_expand": unet_expand,  # sguo_v2
            "unetbn": UNetBN,
            "unetgn": UNetGN,
            "unetbnslim": unet_bn,
            "unetgnslim": unet,
            "unet_deep_as_dru": unet_deep_as_dru,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            'deeplabv3': DeepLab,
            "refinenet": rf101
        }[name]
    except:
        raise("Model {} not available".format(name))
