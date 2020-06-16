from ptsemseg.models.utils import *
from torchvision import models
import torchvision
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def encoder_rgbs(m, n_classes):
    [out_, in_] = m.weight.shape[0:2]
    mean_c = torch.mean(m.weight, 1, True)
    list_c = [m.weight] + [mean_c] * n_classes
    conv_rgbs = nn.Conv2d(in_ + n_classes, out_, kernel_size=7, stride=2, padding=3,
                          bias=False)
    conv_rgbs.weight = torch.nn.Parameter(torch.cat(list_c, 1))
    conv_rgbs.bias = m.bias
    return conv_rgbs


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class druresnet50(nn.Module):
    def __init__(self,
                 args,
                 n_classes=19,
                 initial=1,
                 steps=3,
                 gate=3,
                 hidden_size=32*8,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 num_filters=32,
                 pretrained=True
                 ):
        super(druresnet50, self).__init__()
        self.args = args
        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv
        self.num_filters = num_filters

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)
        self.encoder_rgb_s = encoder_rgbs(self.encoder.conv1, self.n_classes)
        self.conv1 = nn.Sequential(self.encoder_rgb_s,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   )
        self.pool1 = self.encoder.maxpool

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        # self.center = ConvBlock(2048, num_filters * 8 * 2, num_filters * 8)
        assert (self.hidden_size == num_filters * 8)
        self.gru = ConvDRU(2048, self.hidden_size)

        self.dec5 = DecoderBlock(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = DecoderBlock(64 + num_filters, num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

        self.paramGroup1 = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.paramGroup2 = nn.Sequential(self.gru, self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.final)

    def forward(self, inputs, h, s):
        list_st = []
        for i in range(self.steps):
            stacked_inputs = torch.cat([inputs, s], dim=1)

            conv1 = self.conv1(stacked_inputs)
            conv2 = self.conv2(self.pool1(conv1))
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            center = self.gru(conv5, h)

            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            s = self.final(dec1)

            list_st += [s]

        return list_st


class ConvDRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvDRU, self).__init__()

        self.reset_gate = ConvBlock(input_size, input_size, input_size)
        self.update_gate = ConvBlock(input_size, hidden_size*2, hidden_size)
        self.out_gate = ConvBlock(input_size, hidden_size*2, hidden_size)

    def forward(self, input_, h=None):
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        update = torch.sigmoid(self.update_gate(input_))
        reset = torch.sigmoid(self.reset_gate(input_))
        # print('input_, update, reset, h, shape ', input_.shape, update.shape, reset.shape, h.shape)
        # stacked_inputs_ = torch.cat([input_, h * reset], dim=1)
        # out_inputs = torch.tanh(self.out_gate(stacked_inputs_))
        out_inputs = torch.tanh(self.out_gate(input_ * reset))
        h_new = h * (1 - update) + out_inputs * update
        return h_new

    def __repr__(self):
        return 'ConvDRU: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())


class ConvbnRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvbnRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvBlockbn(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ConvBlockbn, self).__init__()

        self.block = nn.Sequential(
            ConvbnRelu(in_channels, middle_channels),
            ConvbnRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlockbn(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockbn, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvbnRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvbnRelu(in_channels, middle_channels),
                ConvbnRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class druresnet50bn(nn.Module):
    def __init__(self,
                 args,
                 n_classes=19,
                 initial=1,
                 steps=3,
                 gate=3,
                 hidden_size=32*8,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 num_filters=32,
                 pretrained=True
                 ):
        super(druresnet50bn, self).__init__()
        self.args = args
        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv
        self.num_filters = num_filters

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)
        self.encoder_rgb_s = encoder_rgbs(self.encoder.conv1, self.n_classes)
        self.conv1 = nn.Sequential(self.encoder_rgb_s,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   )
        self.pool1 = self.encoder.maxpool

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        # self.center = ConvBlock(2048, num_filters * 8 * 2, num_filters * 8)
        assert (self.hidden_size == num_filters * 8)
        self.gru = ConvDRU(2048, self.hidden_size)

        self.dec5 = DecoderBlockbn(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlockbn(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlockbn(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlockbn(256 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = DecoderBlockbn(64 + num_filters, num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

        self.paramGroup1 = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.paramGroup2 = nn.Sequential(self.gru, self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.final)

    def forward(self, inputs, h, s):
        list_st = []
        for i in range(self.steps):
            stacked_inputs = torch.cat([inputs, s], dim=1)

            conv1 = self.conv1(stacked_inputs)
            conv2 = self.conv2(self.pool1(conv1))
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

            center = self.gru(conv5, h)

            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))

            s = self.final(dec1)

            list_st += [s]

        return list_st


class ConvDRUbn(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvDRUbn, self).__init__()

        self.reset_gate = ConvBlockbn(input_size, input_size, input_size)
        self.update_gate = ConvBlockbn(input_size, hidden_size*2, hidden_size)
        self.out_gate = ConvBlockbn(input_size, hidden_size*2, hidden_size)

    def forward(self, input_, h=None):
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        update = torch.sigmoid(self.update_gate(input_))
        reset = torch.sigmoid(self.reset_gate(input_))
        # print('input_, update, reset, h, shape ', input_.shape, update.shape, reset.shape, h.shape)
        # stacked_inputs_ = torch.cat([input_, h * reset], dim=1)
        # out_inputs = torch.tanh(self.out_gate(stacked_inputs_))
        out_inputs = torch.tanh(self.out_gate(input_ * reset))
        h_new = h * (1 - update) + out_inputs * update
        return h_new

    def __repr__(self):
        return 'ConvDRU: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())