"""
Testing the new implementation of UNET vs the old version.

"""
import torch
from ptsemseg.models.unet import unet, GeneralUNet_v2, GeneralUNet, UNetBN, UNetGN
from ptsemseg.models.recurrent_unet import GeneralRecurrentUnet, RecurrentUNetCell, UNetWithGRU, UNetOnlyHidden
from ptsemseg.utils import get_argparser
from utils import train_parser

parser = train_parser()
args = parser.parse_args()

def create_models():
    unet_o = unet(feature_scale=4,
                  n_classes=2,
                  is_deconv=True,
                  in_channels=3,
                  is_batchnorm=True, )
    unet_1 = GeneralUNet(
        feature_scale=4,
        out_channels=2,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
        feature_level=4,
    )
    unet_2 = GeneralUNet_v2(
        feature_scale=4,
        out_channels=2,
        is_deconv=True,
        in_channels=3,
        is_norm=True,
        feature_level=4
    )

    unify_parameters(unet_o, unet_1)
    unify_parameters(unet_o, unet_2)

    return unet_o, unet_1, unet_2


def unify_parameters(source, target):
    # Unify the encoder part:
    if isinstance(target, GeneralUNet):
        encoders = target.convs
        center = target.center
        decoders = target.up_concat

    elif isinstance(target, GeneralUNet_v2):
        encoders = target.encoder.convs
        center = target.center
        decoders = target.decoder.up_concat

    def _unify_param(m1, m2):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if p1.size() == p2.size():
                p1.data = p2.data
            else:
                raise ValueError("WRONG here")

    for i in range(4):
        _unify_param(getattr(source, f'conv{i+1}'), encoders[i])
        _unify_param(getattr(source, f'up_concat{4-i}'), decoders[i])

    _unify_param(source.center, center)


def testing_input():
    inp = torch.rand(size=[3, 3, 32, 32])
    unets = create_models()
    outs = []
    for unet in unets:
        outs.append(unet(inp))
    final_out = torch.cat(outs, dim=0)

    assert ((outs[2] - outs[0]).mean() < 1e-6)
    print((outs[1] - outs[0]).mean())
    print((outs[1] - outs[2]).mean())


def test_recurrent_unet_cell_one_layer():
    parser = get_argparser()
    args = parser.parse_args()
    runet = RecurrentUNetCell(
        args,
        feature_level=4,
        n_classes=2,
    )
    print(runet)
    inp = torch.rand(size=[3, 3, 32, 32])
    out = runet(inp)


def test_recurrent_unet_():
    parser = train_parser()
    args = parser.parse_args()
    runet = GeneralRecurrentUnet(
        args,
        n_classes=2
    )

    inp = torch.rand(size=[3, 3, 32, 32])
    out = runet(inp)
    _test_recurrent_output(out)

def test_runet_with_gru():
    args.gate = 3
    runet = UNetWithGRU(
        args,
        n_classes=2,
    )

    inp = torch.rand(size=[3, 3, 32, 32])
    out = runet(inp)
    # They should be different.
    _test_recurrent_output(out)

def _test_recurrent_output(out):
    thr = 1e-1
    assert (out[1] - out[2]).sum().abs().item() > thr
    assert (out[0] - out[2]).sum().abs().item() > thr
    assert (out[1] - out[0]).sum().abs().item() > thr


def test_unet_bngn():
    unetbn = UNetBN(n_classes=2)
    unetgn = UNetGN(n_classes=2)

    inp = torch.rand(size=[3, 3, 32, 32])
    out1 = unetbn(inp)
    out2 = unetgn(inp)


def test_unet_only_hidden():
    parser = train_parser()
    args = parser.parse_args()
    unet = UNetOnlyHidden(args, n_classes=2)

    inp = torch.rand(size=[3,3,32,32])
    out = unet(inp)


def test_runet_with_different_level():
    parser = train_parser()
    args = parser.parse_args()
    inp = torch.rand(size=[3,3,64,64])
    for i in range(3, 5):
        args.recurrent_level = i
        args.unet_level = 5 - i
        unet = GeneralRecurrentUnet(args, 2, )
        print(unet)
        out = unet(inp)
        _test_recurrent_output(out)



if __name__ == '__main__':
    # create_models()
    # testing_input()
    # test_recurrent_unet_cell_one_layer()
    test_runet_with_gru()
    # test_recurrent_unet_()

    # test_unet_bngn()
    # test_unet_only_hidden()
    # test_runet_with_different_level()