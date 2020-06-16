from ptsemseg.models.utils import *


class sru(nn.Module):
    def __init__(self,
                 args,
                 n_classes=19,
                 initial=1,
                 steps=3,
                 gate=3,
                 hidden_size=32,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 ):
        super(sru, self).__init__()
        self.args = args
        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels + self.n_classes, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # convgru
        assert(self.hidden_size == filters[4])
        self.gru = ConvSRU(filters[3], self.hidden_size)

        # final conv (without any concat)
        self.conv_down = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, h, s):
        list_st = []
        for i in range(self.steps):
            stacked_inputs = torch.cat([inputs, s], dim=1)

            conv1 = self.conv1(stacked_inputs)
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            h = self.gru(maxpool4, h)
            up4 = self.up_concat4(conv4, h)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            s = self.conv_down(up1)
            list_st += [s]

        return list_st


class ConvSRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvSRU, self).__init__()

        self.update_gate = unetConv2(input_size, hidden_size, True)
        self.out_gate = unetConv2(input_size, hidden_size, True)

    def forward(self, input_, h=None):
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        update = torch.sigmoid(self.update_gate(input_))
        # print('input_, reset, h, shape ', input_.shape, reset.shape, h.shape)
        # stacked_inputs_ = torch.cat([input_, h * reset], dim=1)

        out_inputs = torch.tanh(self.out_gate(input_))
        h_new = h * (1 - update) + out_inputs * update
        return h_new

    def __repr__(self):
        return 'ConvDRU: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())
