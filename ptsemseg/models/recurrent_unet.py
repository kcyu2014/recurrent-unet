import IPython
import torch
import torch.nn as nn
from torch.autograd import Variable
from .utils import unetConv2, unetConv1
from .unet import UnetEncoder, UnetDecoder, GeneralUNet_v2


def init_hidden_state(layer, prev_h, prev_state, out_size, hidden_size, batch_size, spatial_size):
    """
    Init hidden state.
    :param layer:
    :param batch_size:
    :param spatial_size:
    :return: prev_h, prev_state
    """
    try:
        if hasattr(layer, "args"):
            initial = layer.args.initial
        else:
            initial = layer.initial
    except AttributeError as e:
        raise e

    if prev_h is None:
        h_size = [batch_size, out_size] + list(spatial_size)
        prev_h = Variable(torch.ones(h_size))
        if initial == 1:
            pass
        elif initial == 0:
            prev_h.fill_(0.)

    if prev_state is None:
        state_size = [batch_size, hidden_size] + list(spatial_size)
        prev_state = Variable(torch.ones(state_size))
        if initial == 1:
            pass
        elif initial == 0:
            prev_state.fill_(0.)

    if torch.cuda.is_available():
        prev_state = prev_state.to(torch.device(layer.device))
        prev_h = prev_h.to(torch.device(layer.device))
    return prev_h, prev_state


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional, this is only one cell.

    """

    def __init__(self, args,
                 filters,
                 input_size,
                 hidden_size,
                 output_size,
                 feature_level,
                 ):
        super(ConvGRUCell, self).__init__()
        self.args = args
        self.device = args.device
        self.initial = args.initial

        # Original filter.
        self.filters = filters

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if len(filters) == 1:
            # One layer ConvGRU baseline
            if args.structure is not 'ours':
                print("USING original GRU.")
                unet_conv = unetConv1
            else:
                print("USING conv ")
                unet_conv = unetConv2
            self.reset_gate = unet_conv(input_size, hidden_size, True)
            self.update_gate = unet_conv(input_size, hidden_size, True)
            if self.args.gate == 3:
                print("USING GRU with 3 gates")
                self.out_gate = unet_conv(input_size + hidden_size, hidden_size, True)
            else:
                print("USING GRU only with 2 gates. No reset gates")
                self.out_gate = unet_conv(input_size, hidden_size, True)
        else:

            self.reset_gate = GeneralUNet_v2(
                filters=filters,
                in_channels=input_size,
                out_channels=hidden_size,
                feature_level=feature_level,
                is_deconv=True,
            )
            self.update_gate = GeneralUNet_v2(
                filters=filters,
                in_channels=input_size,
                out_channels=hidden_size,
                feature_level=feature_level,
                is_deconv=True,
            )
            if self.args.gate == 3:
                print("Using GRU with 3 GATE, with UNet level {}".format(feature_level))
                out_gate_input_dim = input_size + hidden_size
            else:
                out_gate_input_dim = input_size

            self.out_gate = GeneralUNet_v2(
                filters=filters,
                in_channels=out_gate_input_dim,
                out_channels=hidden_size,
                feature_level=feature_level,
                is_deconv=True,)
            #
        self.conv_down = nn.Conv2d(hidden_size, output_size, 1)
        # self.conv_up = nn.Conv2d(input_size, hidden_size, 1)

    def forward(self, input_, prev_state, prev_h=None):
        """

        :param input_: tensor input, (H, W, input_size)
        :param prev_state:  Ht-1
        :param prev_h: should be None
        :return: Ht, output (H, W, output_size)

        """
        # prev_h should not matter
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        _, prev_state = init_hidden_state(self,
                                          prev_h=prev_h,
                                          prev_state=prev_state,
                                          out_size=self.output_size,
                                          hidden_size=self.hidden_size,
                                          batch_size=batch_size,
                                          spatial_size=spatial_size)

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())
        # TODO refine the logic here! to match the output. Hidden-size is just a single layer GRU.
        # Hidden-state is defined as h = H.
        # reset = torch.sigmoid(self.reset_gate(stacked_inputs_))
        # input_up = self.conv_up(input_)
        # stacked_inputs_h = torch.cat([input_up, prev_state * reset], dim=1)
        stacked_inputs_ = input_
        # IPython.embed()
        if self.args.gate == 2:
            update = torch.sigmoid(self.update_gate(stacked_inputs_))
        elif self.args.gate == 3:
            update = torch.sigmoid(self.update_gate(input_))
            reset = torch.sigmoid(self.reset_gate(input_))
            stacked_inputs_ = torch.cat([input_, prev_state * reset], dim=1)

        out_inputs = torch.tanh(self.out_gate(stacked_inputs_))
        new_state = prev_state * (1 - update) + out_inputs * update
        output = self.conv_down(new_state)
        return output, new_state

    def __repr__(self):
        return 'ConvGRUCell: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())

        # return f'ConvGRUCell: \n' \
        #        f'\t reset_gate: \n {self.reset_gate.__repr__()}\n' \
        #        f'\t update_gate: \n {self.update_gate.__repr__()}\n' \
        #        f'\t out_gate:  \n {self.out_gate.__repr__()}\n'


class RecurrentUNetCell(nn.Module):
    """
    This is the implementation of the recent idea, add a very basic Convolutional GRU to the bottleneck.

        Decoder
        Encoder
    Can be adapted into a general ConvGRU style with different Hidden state.

    """

    def __init__(self,
                 args,
                 feature_scale=4,
                 feature_level=4,
                 n_classes=21,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 is_input_stack=True,
                 **kwargs
                 ):
        super(RecurrentUNetCell, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.unet_level = feature_level
        self.gru_level = args.recurrent_level if args.recurrent_level > 1 else 5 - feature_level
        self.is_input_stack = is_input_stack

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        unet_filters = [self.in_channels,] + filters[:feature_level+1]
        recurrent_filters = filters[feature_level:]
        assert self.gru_level <= len(recurrent_filters)
        assert self.unet_level <= len(unet_filters)

        self.encoder = UnetEncoder(
            in_channels=self.in_channels,
            filters=unet_filters,
            feature_scale=feature_scale,
            feature_level=feature_level)

        self.decoder = UnetDecoder(
            unet_filters,
            feature_scale,
            feature_level,
            n_classes,
            is_deconv)
        # TODO further handle this logic.
        # support better unet representation?
        #  change the hidden state.
        self.gru = ConvGRUCell(
            args,
            filters=recurrent_filters,
            input_size=unet_filters[-2],
            hidden_size=args.hidden_size,
            output_size=unet_filters[-1],
            feature_level=self.gru_level,
        )

    def forward(self, inputs, prev_state=None):
        """
        This should implement the Recurrent inside.
        """
        # init

        conv_outputs, pools_outputs = self.encoder(inputs)
        center, next_state = self.gru(pools_outputs[-1], prev_state=prev_state)
        output = self.decoder([conv_outputs, center])

        return output, next_state

    def __repr__(self):
        a = 'RecurrentUNet CELL general \n'
        a += "\t" + self.encoder.__repr__()
        a += "\t" + self.decoder.__repr__()
        a += "\t" + self.gru.__repr__()
        return a


class _RecurrentUnet(nn.Module):

    def __init__(self, args, n_classes=2, **kwargs):
        super(_RecurrentUnet, self).__init__()
        self.rnn_steps = args.steps
        self.device = args.device
        self.args = args
        self.input_size = 3
        self.hidden_size = args.hidden_size
        self.n_classes = n_classes
        self.cell = None
        self.is_input_stack = True

    def forward(self, inputs):
        # get batch and spatial sizes
        list_ht = []

        batch_size = inputs.data.size()[0]
        spatial_size = inputs.data.size()[2:]

        ht, _ = init_hidden_state(self, None, None, self.n_classes, self.hidden_size, batch_size, spatial_size)
        Ht = None
        for i in range(self.rnn_steps):
            if self.is_input_stack:
                stack_inputs = torch.cat([inputs, ht], dim=1)
            else:
                stack_inputs = inputs
            ht, Ht = self.cell(stack_inputs, Ht)
            h = ht
            list_ht += [h]

        return list_ht


class GeneralRecurrentUnet(_RecurrentUnet):

    def __init__(self, args, n_classes=2, **kwargs):

        feature_level = args.unet_level if 4 >= args.unet_level > 0 else 4
        super(GeneralRecurrentUnet, self).__init__(args, n_classes=n_classes, **kwargs)
        self.cell = RecurrentUNetCell(
            args,
            feature_scale=args.feature_scale,
            feature_level=feature_level,
            n_classes=n_classes,
            is_deconv=True,
            in_channels=self.input_size + n_classes,
            is_batchnorm=True,
            is_input_stack=True,
        )

        self.is_input_stack = True

    def __repr__(self):
        return self.cell.__repr__()


class GeneralRecurrentUnet_hidden(_RecurrentUnet):

    def __init__(self, args, n_classes=2, **kwargs):
        feature_level = args.unet_level if 4 >= args.unet_level > 0 else 4
        super(GeneralRecurrentUnet_hidden, self).__init__(args, n_classes=n_classes, **kwargs)
        self.cell = RecurrentUNetCell(
            args,
            feature_scale=args.feature_scale,
            feature_level=feature_level,
            n_classes=n_classes,
            is_deconv=True,
            in_channels=self.input_size,
            is_batchnorm=True,
            is_input_stack=False,
        )

        self.is_input_stack = False

    def __repr__(self):
        return self.cell.__repr__()


class UNetOnlyHidden(_RecurrentUnet):

    def __init__(self, args, n_classes=2, **kwargs):
        # Override the default.
        args.hidden_size = 128
        args.structure = 'gru'

        super(UNetOnlyHidden, self).__init__(args, n_classes=2, **kwargs)
        self.cell = RecurrentUNetCell(
            args,
            feature_scale=4,
            feature_level=4,
            n_classes=n_classes,
            is_deconv=True,
            in_channels=self.input_size,
            is_batchnorm=True,
            is_input_stack=False,
        )
        self.is_input_stack = False


class _UNetWithGRU(_RecurrentUnet):
    """
    Implement the true baseline, that, Convolution GRU in the end.

    """
    def __init__(self, args, n_classes=2, feature_scale=4,  **kwargs):
        super(_UNetWithGRU, self).__init__(args, n_classes=2, **kwargs)

        self.unet = GeneralUNet_v2(
            feature_scale=feature_scale,
            feature_level=4,
            in_channels=3,
            out_channels=args.hidden_size
        )
        self.gru = ConvGRUCell(
            args,
            filters=[32],
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            output_size=n_classes,
            feature_level=1
        )
        self.is_input_stack = False

    def forward(self, inputs):
        # get batch and spatial sizes
        list_ht = []

        batch_size = inputs.data.size()[0]
        spatial_size = inputs.data.size()[2:]

        ht, _ = init_hidden_state(self, None, None, self.n_classes, self.hidden_size, batch_size, spatial_size)
        Ht = None
        x = self.unet(inputs)

        for i in range(self.rnn_steps):
            ht, Ht = self.gru(x, Ht)
            h = ht
            list_ht += [h]

        return list_ht


class UNetWithGRU_R(_UNetWithGRU):
    """
    Implement the true baseline, that, Convolution GRU in the end.

    """
    def __init__(self, args, n_classes=2, feature_scale=4, **kwargs):
        super(UNetWithGRU_R, self).__init__(args, n_classes=n_classes, feature_scale=4, **kwargs)


class UNetWithGRU(_UNetWithGRU):
    """
    Implement the true baseline, that, Convolution GRU in the end.

    """
    def __init__(self, args, n_classes=2, feature_scale=4, **kwargs):
        super(UNetWithGRU, self).__init__(args, n_classes=n_classes, feature_scale=feature_scale, **kwargs)



class UNetWithGRU_old(_RecurrentUnet):
    """
    Implement the true baseline, that, Convolution GRU in the end.

    """

    def __init__(self, args, n_classes=2, **kwargs):
        super(UNetWithGRU_old, self).__init__(args, n_classes=2, **kwargs)

        self.unet = GeneralUNet_v2(
            feature_level=4,
            in_channels=3,
            out_channels=args.hidden_size
        )
        self.gru = ConvGRUCell(
            args,
            filters=[32],
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            output_size=n_classes,
            feature_level=1
        )
        self.cell = self.cell_forward
        self.is_input_stack = False

    def cell_forward(self, inputs, prev_state):
        x = self.unet(inputs)
        x, next_state = self.gru(x, prev_state)
        return x, next_state

    def forward(self, inputs):
        # get batch and spatial sizes
        list_ht = []

        batch_size = inputs.data.size()[0]
        spatial_size = inputs.data.size()[2:]

        ht, _ = init_hidden_state(self, None, None, self.n_classes, self.hidden_size, batch_size, spatial_size)
        Ht = None
        for i in range(self.rnn_steps):
            if self.is_input_stack:
                stack_inputs = torch.cat([inputs, ht], dim=1)
            else:
                stack_inputs = inputs
            ht, Ht = self.cell(stack_inputs, Ht)
            h = ht
            list_ht += [h]

        return list_ht

