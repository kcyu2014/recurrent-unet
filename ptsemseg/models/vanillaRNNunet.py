import torch.nn as nn
import argparse
from ptsemseg.models.utils import *


class one_step_unet(nn.Module):
    def __init__(
        self,
        args,
        feature_scale=4,
        n_classes=2,
        hidden_size=2,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(one_step_unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.hidden_size = n_classes
        self.device = args.device
        self.initial = args.initial

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels + self.hidden_size, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.final_gn = nn.GroupNorm(n_classes, n_classes)

    def forward(self, inputs, prev_state=None):
        '''
        Parameters
        ----------
        inputs : 4D input tensor. (batch, channels, height, width).
        prev_state : 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        # get batch and spatial sizes
        batch_size = inputs.data.size()[0]
        spatial_size = inputs.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                if self.initial == 1:
                    prev_state = Variable(torch.ones(state_size)).to(torch.device(self.device))
                elif self.initial == 0:
                    prev_state = Variable(torch.zeros(state_size)).to(torch.device(self.device))  # Variable(torch.zeros(state_size)).cuda()
            else:
                if self.initial == 1:
                    prev_state = Variable(torch.ones(state_size))
                elif self.initial == 0:
                    prev_state = Variable(torch.zeros(state_size))  # Variable(torch.zeros(state_size))

        stacked_inputs = torch.cat([inputs, prev_state], dim=1)
        conv1 = self.conv1(stacked_inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        new_state = self.final(up1)
        new_state = self.final_gn(new_state)

        return new_state


class _vanillaRNNunet(nn.Module):
    def __init__(self, args, n_classes=2, feature_scale=4):
        super(_vanillaRNNunet, self).__init__()
        self.n_classes = n_classes
        self.rnn_steps = args.steps
        # 3 input image channel, 2 output channels
        self.one_step = one_step_unet(args, feature_scale=feature_scale, n_classes=self.n_classes, in_channels=3, hidden_size=2)

    def forward(self, inputs):
        list_ht = []
        ht = None
        for i in range(self.rnn_steps):
            ht = self.one_step(inputs, ht)
            list_ht += [ht]

        return list_ht


class _vanillaRNNunet_NoParamShare(nn.Module):
    def __init__(self, args, n_classes=2, feature_scale=4):
        super(_vanillaRNNunet_NoParamShare, self).__init__()
        self.n_classes = n_classes
        self.rnn_steps = args.steps
        # 3 input image channel, 2 output channels
        self.unets = []

        for i in range(args.steps):
            _unet = one_step_unet(args, feature_scale=feature_scale, n_classes=self.n_classes, in_channels=3,
                                  hidden_size=2)
            self.unets.append(_unet)
            self.add_module('{}-unet',format(i), _unet)

    def forward(self, inputs):
        list_ht = []
        ht = None
        for i in range(self.rnn_steps):
            ht = self.unets[i](inputs, ht)
            list_ht += [ht]

        return list_ht


class vanillaRNNunet(_vanillaRNNunet):
    def __init__(self, args, n_classes=2, **kwargs):
        super(vanillaRNNunet, self).__init__(args, n_classes=n_classes, feature_scale=1)


class vanillaRNNunet_R(_vanillaRNNunet):

    def __init__(self, args, n_classes=2, **kwargs):
        super(vanillaRNNunet_R, self).__init__(args, n_classes=n_classes, feature_scale=4)


class vanillaRNNunet_NoParamShare(_vanillaRNNunet_NoParamShare):
    def __init__(self, args, n_classes=2, **kwargs):
        super(vanillaRNNunet_NoParamShare, self).__init__(args, n_classes=n_classes, feature_scale=4)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--device", nargs="?", type=str, default="cuda:0", help="GPU or CPU to use")
    parser.add_argument("--steps", nargs="?", type=int, default=3, help="Recurrent Steps")
    parser.add_argument("--initial", nargs="?", type=int, default=1, help="initial values")

    args = parser.parse_args()
    device = torch.device(args.device)
    # model = vanillaRNNunet(args).to(device)
    model = vanillaRNNunet_NoParamShare(args).to(device)
    if torch.cuda.is_available():
        x = Variable(torch.FloatTensor(1, 3, 64, 64)).to(device)
    else:
        x = Variable(torch.FloatTensor(1, 3, 64, 64))
    for i in range(3000):
        output = model(x)
        print(output[0].size())

