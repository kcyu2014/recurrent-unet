from ptsemseg.models.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, BatchNorm, output_stride):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    if backbone == 'resnet50':
        return resnet.ResNet50(BatchNorm, output_stride)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
