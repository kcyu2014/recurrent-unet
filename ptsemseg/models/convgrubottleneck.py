from torch.nn import init
from ptsemseg.models.utils import *
from ptsemseg.models.utils import get_upsampling_weight
from ptsemseg.loss import cross_entropy2d


class unet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        out_channels=4,
        is_deconv=True,
        in_channels=4,
        is_batchnorm=True,
        shared_unet=None
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        if isinstance(shared_unet, unet):
            self.conv1 = unet.conv1
            self.conv2 = unet.conv2
            self.conv3 = unet.conv3
            self.conv4 = unet.conv4
            self.maxpool1 = unet.maxpool1
            self.maxpool2 = unet.maxpool2
            self.maxpool3 = unet.maxpool3
            self.maxpool4 = unet.maxpool4
        else:
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
        self.final = nn.Conv2d(filters[0], out_channels, 1)

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

        return final


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, args, input_size, hidden_size, n_classes):
        super(ConvGRU, self).__init__()
        self.device = args.device
        self.initial = args.initial
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        self.reset_gate = unet(in_channels=input_size + n_classes, out_channels=hidden_size)
        self.update_gate = unet(in_channels=input_size + n_classes, out_channels=hidden_size, shared_unet=self.reset_gate)
        self.out_gate = unet(in_channels=hidden_size + hidden_size, out_channels=hidden_size)

        self.conv_down = nn.Conv2d(hidden_size, n_classes, 1)
        self.conv_up = nn.Conv2d(input_size, hidden_size, 1)

        # padding = kernel_size // 2
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # init.orthogonal_(self.reset_gate.weight)
        # init.orthogonal_(self.update_gate.weight)
        # init.orthogonal_(self.out_gate.weight)
        # init.constant_(self.reset_gate.bias, 0.)
        # init.constant_(self.update_gate.bias, 0.)
        # init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state, prev_h):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None or prev_h is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            h_size = [batch_size, self.n_classes] + list(spatial_size)
            if torch.cuda.is_available():
                if self.initial == 1:
                    prev_state = Variable(torch.ones(state_size)).to(torch.device(self.device))
                    prev_h = Variable(torch.ones(h_size)).to(torch.device(self.device))
                elif self.initial == 0:
                    prev_state = Variable(torch.zeros(state_size)).to(torch.device(self.device))
                    prev_h = Variable(torch.zeros(h_size)).to(torch.device(self.device))
            else:
                if self.initial == 1:
                    prev_state = Variable(torch.ones(state_size))
                    prev_h = Variable(torch.ones(h_size))
                elif self.initial == 0:
                    prev_state = Variable(torch.zeros(state_size))
                    prev_h = Variable(torch.zeros(h_size))
        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        stacked_inputs_reset = torch.cat([input_, prev_h], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs_reset))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs_reset))

        input_up = self.conv_up(input_)
        stacked_inputs_h = torch.cat([input_up, prev_state * reset], dim=1)
        out_inputs = torch.tanh(self.out_gate(stacked_inputs_h))
        new_state = prev_state * (1 - update) + out_inputs * update

        new_h = self.conv_down(new_state)
        return new_state, new_h
