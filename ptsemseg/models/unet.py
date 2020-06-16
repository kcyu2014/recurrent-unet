import torch.nn as nn

from ptsemseg.models.utils import *


class UnetDecoder(nn.Module):
    def __init__(self, filters, feature_scale, feature_level, out_channels, is_deconv,
                 is_norm=True, is_groupnorm=True, **kwargs):
        super(UnetDecoder, self).__init__()
        self.filters = filters
        self.feature_scale = feature_scale
        self.feature_level = feature_level
        self.is_norm = is_norm
        self.is_groupnorm = is_groupnorm
        self.is_deconv = is_deconv
        self.out_channels = out_channels

        # upsampling
        self.up_concat = []
        for i in range(feature_level):
            b = feature_level - i - 1
            self.up_concat.append(unetUp(filters[b+2], filters[b+1], self.is_deconv))
            self.add_module('up_concat{}'.format(b), self.up_concat[i])

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[1], out_channels, 1)
        self.add_module('final', self.final)

        # if out_channels % 8 == 0:
        #     group = int(out_channels/8)
        # else:
        #     group = out_channels
        #
        # # print('out_channels is {}, group no is {}'.format(out_channels, group))
        #
        # print('Group No is ', group)
        # self.final_gn = nn.GroupNorm(group, out_channels) \
        #     if self.is_groupnorm else nn.BatchNorm2d(out_channels)
        # # self.final_gn = nn.GroupNorm(min(8, out_channels), out_channels) \
        # #     if self.is_groupnorm else nn.BatchNorm2d(out_channels)
        # self.add_module('final_gn', self.final_gn)

    def forward(self, inputs):
        """
        take the inputs from the decoder.
        :param inputs: conv_outputs, center from decoder.
        :return:
        """
        conv_outputs, center = inputs
        up_output = [center]
        for i in range(self.feature_level):
            b = self.feature_level - i - 1
            x = self.up_concat[i](conv_outputs[b], up_output[i])
            up_output.append(x)

        final = self.final(up_output[-1])

        # print(final.device)
        # print(self.final_gn.weight.device)
        # print(self.final_gn.bias.device)
        # final_gn = self.final_gn(final)
        # print(final_gn.device)
        # return final_gn
        return final

    @property
    def display_filter(self):
        return [self.filters[self.feature_level - i - 1] for i in range(self.feature_level)]

    def __repr__(self):
        a = super(UnetDecoder, self).__repr__()

        return "UNet-Decoder: \n" \
               "\t feature_level: {} \n \t aux_head_input: {}\n \t final_output: {} \n".format(
            self.feature_level,
            self.display_filter,
            self.out_channels)

        # return f"UNet-Decoder: \n" \
        #        f"\t feature_level: {self.feature_level} \n" \
        #        f"\t aux_head_input: {self.display_filter}\n" \
        #        f"\t final_output: {self.out_channels} \n"
        #        # f"\t model_arch: {a} \n" \


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, filters, feature_scale, feature_level,
                 is_norm=True, is_groupnorm=True,**kwargs):
        super(UnetEncoder, self).__init__(**kwargs)
        self.is_norm = is_norm
        self.is_groupnorm = is_groupnorm
        self.in_channels = in_channels
        self.filters = filters
        self.feature_scale = feature_scale
        self.feature_level = feature_level

        self.convs = []
        self.poolings = []

        assert self.in_channels == filters[0], \
            "UnetEncoder filter 1 {}must match the input_channels {} ".format(filters[0], in_channels)
            # f"UnetEncoder filter 1 {filters[0]}must match the input_channels {in_channels} "

        for i in range(feature_level):
            self.convs.append(unetConv2(filters[i], filters[i+1], self.is_norm, self.is_groupnorm))
            self.add_module('conv_{}'.format(i), self.convs[i])
            self.poolings.append(nn.MaxPool2d(kernel_size=2))
            self.add_module('maxpool_{}'.format(i), self.poolings[i])

    def forward(self, inputs):
        conv_outputs = []
        pools_outputs = [inputs, ]

        for i in range(self.feature_level):
            # Conv take the output of previous pool.
            # Pool takes the output of previous conv.
            x = self.convs[i](pools_outputs[i])
            p = self.poolings[i](x)
            conv_outputs.append(x)
            pools_outputs.append(p)

        return conv_outputs, pools_outputs

    def __repr__(self):
        """
        Override this representation.
        :return:
        """
        a = super(UnetEncoder, self).__repr__()
        return "UNet-Encoder: \n" \
               "\t feature_level: {} \n \t aux_head_output: {}\n".format(
            self.feature_level,
            self.filters[1: self.feature_level])
        # f"\t model_arch: {a} \n" \
        # return f"UNet-Encoder: \n" \
        #        f"\t feature_level: {self.feature_level} \n" \
        #        f"\t aux_head_output: {self.filters[1: self.feature_level]}\n"
        #        # f"\t model_arch: {a} \n" \


class unet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
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
        # self.final_gn = nn.GroupNorm(min(8, n_classes), n_classes)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
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

        final = self.final(up1)
        # final = self.final_gn(final)

        return final


class unet_expand(nn.Module):
    """
        Only Expand Unetconv2d
    """
    def __init__(
        self,
        feature_scale=1,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet_expand, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_expand(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2_expand(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2_expand(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2_expand(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2_expand(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_gn = nn.GroupNorm(min(8, n_classes), n_classes)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
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

        final = self.final(up1)
        # final = self.final_gn(final)

        return final


class unet_expand_all(nn.Module):
    """
    Expand Unetconv2d and UnetUP
    """
    def __init__(
        self,
        feature_scale=1,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet_expand_all, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_expand(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2_expand(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2_expand(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2_expand(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2_expand(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp_expand(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp_expand(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp_expand(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp_expand(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_gn = nn.GroupNorm(min(8, n_classes), n_classes)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
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

        final = self.final(up1)
        # final = self.final_gn(final)

        return final


class unet_deep_as_dru(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=21,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet_deep_as_dru, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        deep_filters = [128, 192, 128, 128]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.center_conv1 = nn.Sequential(
            nn.Conv2d(deep_filters[0], deep_filters[1], 3, 1, 1),
            nn.GroupNorm(min(8, deep_filters[1]), deep_filters[1]),
            nn.ReLU(),
        )
        self.center_conv2 = nn.Sequential(
            nn.Conv2d(deep_filters[1], deep_filters[2], 3, 1, 1),
            nn.GroupNorm(min(8, deep_filters[2]), deep_filters[2]),
            nn.ReLU(),
        )
        self.center_conv3 = nn.Sequential(
            nn.Conv2d(deep_filters[2], deep_filters[3], 3, 1, 1),
            nn.GroupNorm(min(8, deep_filters[3]), deep_filters[3]),
            nn.ReLU(),
        )

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_gn = nn.GroupNorm(min(8, n_classes), n_classes)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center1 = self.center_conv1(center)
        center2 = self.center_conv2(center1)
        center3 = self.center_conv3(center2)
        up4 = self.up_concat4(conv4, center3)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        # final = self.final_gn(final)

        return final


class unet_bn(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=21,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet_bn, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, is_groupnorm=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, is_groupnorm=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, is_groupnorm=False)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, is_groupnorm=False)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, is_groupnorm=False)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.final_gn = nn.BatchNorm2d(n_classes)
        # self.final_gn = nn.GroupNorm(min(8, n_classes), n_classes)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
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

        final = self.final(up1)
        final = self.final_gn(final)

        return final


class GeneralUNet(nn.Module):
    """
    Develop for the ConvGRU, those update gates.
    About the filters, needs quite a bit tuning?
    or just use the structure like other unet.

    Quite interesting to see, if start late, what it will look like.

    """
    def __init__(
        self,
        feature_scale=4,
        out_channels=4,
        is_deconv=True,
        in_channels=4,
        is_batchnorm=True,
        feature_level=4,
        filters=None
    ):
        """

        :param feature_scale:
        :param out_channels:
        :param is_deconv:
        :param in_channels:
        :param is_batchnorm:
        :param feature_level:
        :param filters: should be with length shorter than
        """
        super(GeneralUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.feature_level = feature_level

        if filters is None:
            filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
            filters = [int(x / self.feature_scale) for x in filters]

        # assert out_channels == filters[0], 'oh no ! out_channels should be {}'.format(filters[0])

        # process this feature maps.
        assert feature_level + 1 <= len(filters)
        filters = filters[:feature_level + 1]

        self.convs = []
        self.poolings = []
        filters = [self.in_channels, ] + filters
        for i in range(feature_level):
            self.convs.append(unetConv2(filters[i], filters[i+1], self.is_batchnorm))
            self.add_module('conv_{}'.format(i),self.convs[i])
            self.poolings.append(nn.MaxPool2d(kernel_size=2))
            self.add_module('maxpool_{}'.format(i), self.poolings[i])

        self.encoder = nn.ModuleList(self.convs + self.poolings)
        self.center = unetConv2(filters[-2], filters[-1], self.is_batchnorm)
        self.add_module('center', self.center)

        self.up_concat = []
        for i in range(feature_level):
            b = feature_level - i - 1
            self.up_concat.append(unetUp(filters[b+2], filters[b+1], self.is_deconv))
            self.add_module('up_concat{}', format(b), self.up_concat[i])

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[1], out_channels, 1)
        self.add_module('final', self.final)
        self.final_gn = nn.GroupNorm(min(8, out_channels), out_channels)
        self.add_module('final_gn', self.final_gn)
        self.decoder = nn.ModuleList(self.up_concat + [self.final, self.final_gn])

    def forward(self, inputs):
        conv_outputs = []
        pools_outputs = [inputs,]

        for i in range(self.feature_level):
            # Conv take the output of previous pool.
            # Pool takes the output of previous conv.
            x = self.convs[i](pools_outputs[i])
            p = self.poolings[i](x)
            conv_outputs.append(x)
            pools_outputs.append(p)

        center = self.center(pools_outputs[-1])
        up_output = [center]
        for i in range(self.feature_level):
            b = self.feature_level - i - 1
            x = self.up_concat[i](conv_outputs[b], up_output[i])
            up_output.append(x)

        final = self.final(up_output[-1])
        final_gn = self.final_gn(final)

        return final_gn
        # return up1


class GeneralUNet_v2(nn.Module):
    """
    Develop for the ConvGRU, those update gates.
    About the filters, needs quite a bit tuning?
    or just use the structure like other unet.

    Quite interesting to see, if start late, what it will look like.

    """
    def __init__(
        self,
        feature_scale=1,
        out_channels=4,
        is_deconv=True,
        in_channels=4,
        is_norm=True,
        is_groupnorm=True,
        feature_level=4,
        filters=None
    ):
        """

        :param feature_scale:
        :param out_channels:
        :param is_deconv:
        :param in_channels:
        :param is_norm:
        :param feature_level:
        :param filters: level + 1.
            [:feature_level] is for encoder/decoder
            [feature_level] is for center node.
        """
        super(GeneralUNet_v2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_norm = is_norm
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.feature_level = feature_level

        if filters is None:
            filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
            filters = [int(x / self.feature_scale) for x in filters]

        # assert out_channels == filters[0], 'oh no ! out_channels should be {}'.format(filters[0])

        # process this feature maps.
        if feature_level + 1 <= len(filters):
            pass
        elif feature_level == len(filters):
            filters += [filters[-1]*2]
            # filters = [filters[0]//2] + filters
        else:
            raise ValueError("GeneralUNet_v2: filter {} must match the following requirement:".format(filters) +
                             "length {} >= feature_level + 1 = {}.  ".format(len(filters), feature_level + 1) +
                             "The following is for Encoder and decoder {} ".format(filters[:feature_level]) +
                             "and this is for center {}".format(filters[feature_level]))

        filters = filters[:feature_level + 1]
        filters = [self.in_channels, ] + filters
        self.filters = filters
        self.encoder = UnetEncoder(
            in_channels=in_channels,
            filters=filters,
            feature_scale=feature_scale,
            feature_level=feature_level,
            is_norm=is_norm,
            is_groupnorm=is_groupnorm
        )

        self.decoder = UnetDecoder(
            filters=filters,
            feature_scale=feature_scale,
            feature_level=feature_level,
            out_channels=out_channels,
            is_deconv=is_deconv,
            is_norm=is_norm,
            is_groupnorm=is_groupnorm
        )

        self.center = unetConv2(filters[-2], filters[-1], is_norm, is_groupnorm)

    def forward(self, inputs):
        # print(inputs.device)
        # print(self.encoder)
        conv_outputs, pools_outputs = self.encoder(inputs)
        # print(len(pools_outputs))
        # print(pools_outputs[-1].device)
        # print(self.center)
        center = self.center(pools_outputs[-1])
        # print(center.device)
        # print(self.decoder)
        final = self.decoder([conv_outputs, center])
        # print('final', final.device)
        return final

    def __repr__(self):
        res = "GeneralUNet_v2: \n" + \
              "\t input_dim: {} \n".format(self.in_channels) + \
              "\t output_dim: {} \n".format(self.out_channels) + \
              "details:\n" + "\t Encoder filters: {} \n".format(self.filters[1:self.feature_level+1]) + \
              "\t Center filters: {} \n".format(self.filters[-2:]) + \
              "\t Decoder filters: {} \n".format(self.decoder.display_filter)
        # res = f"GeneralUNet_v2: \n" \
        #       f"\t input_dim: {self.in_channels} \n" \
        #       f"\t output_dim: {self.out_channels} \n" \
        #       f"details:\n"  \
        #       f"\t Encoder filters: {self.filters[1:self.feature_level+1]} \n" \
        #       f"\t Center filters: {self.filters[-2:]} \n" \
        #       f"\t Decoder filters: {self.decoder.display_filter} \n"
        return res


class UNetGN(GeneralUNet_v2):
    def __init__(self, n_classes, **kwargs):
        filters = [32, 64, 128, 256, 512]
        # filters = [4 * i for i in [32, 64, 128, 256, 512]]
        super(UNetGN, self).__init__(
            feature_scale=1,
            is_norm=True,
            is_groupnorm=True,
            filters=filters,
            in_channels=3,
            out_channels=n_classes,
        )


class UNetBN(GeneralUNet_v2):
    def __init__(self, n_classes, **kwargs):
        filters = [32, 64, 128, 256, 512]
        super(UNetBN, self).__init__(
            feature_scale=4,
            is_norm=True,
            is_groupnorm=False,
            filters=filters,
            in_channels=3,
            out_channels=n_classes,
        )


class UNetGN_R(GeneralUNet_v2):
    def __init__(self, n_classes, **kwargs):
        filters = [32, 64, 128, 256, 512]
        super(UNetGN_R, self).__init__(
            feature_scale=4,
            is_norm=True,
            is_groupnorm=True,
            filters=filters,
            in_channels=3,
            out_channels=n_classes,
        )


class UNetBN_R(GeneralUNet_v2):
    def __init__(self, n_classes, **kwargs):
        filters = [32, 64, 128, 256, 512]
        super(UNetBN_R, self).__init__(
            feature_scale=4,
            is_norm=True,
            is_groupnorm=False,
            filters=filters,
            in_channels=3,
            out_channels=n_classes,
        )


if __name__ == '__main__':
    # Test general Unet
    unet_test = GeneralUNet(feature_scale=4, out_channels=2, is_deconv=True,
                            in_channels=256, is_batchnorm=True, feature_level=1)
    unet_2 = GeneralUNet_v2(feature_scale=4, out_channels=2, is_deconv=True,
                            in_channels=256, is_norm=True, feature_level=1)
    inp = torch.ones([2, 256, 5, 5])
    out = unet_test(inp)
    out2 = unet_2(inp)

    out.size()