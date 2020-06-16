import torch.nn as nn
from ptsemseg.models.convgru import ConvGRU
import torch
from torch.autograd import Variable
import argparse


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class rcnn(nn.Module):
    def __init__(self, args, n_classes=2):
        super(rcnn, self).__init__()
        self.rnn_steps = args.steps
        # 3 input image channel, 32 output channels, 3x3 square convolution kernel
        self.convgru = ConvGRU(args, input_size=3, hidden_size=args.hidden_size, n_classes=n_classes)

    def forward(self, inputs):
        list_ht = []
        Ht = None
        ht = None
        for i in range(self.rnn_steps):
            # print('recurrence {}'.format(i))
            Ht, ht = self.convgru(inputs, Ht, ht)
            h = ht
            list_ht += [h]
        return list_ht


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--device", nargs="?", type=str, default="cuda:0", help="GPU or CPU to use")
    parser.add_argument("--steps", nargs="?", type=int, default=3, help="Recurrent Steps")
    parser.add_argument("--hidden_size", nargs="?", type=int, default=32, help="hidden size")
    parser.add_argument("--initial", nargs="?", type=int, default=0, help="initial value of hidden state")

    args = parser.parse_args()
    device = torch.device(args.device)
    model = rcnn(args).to(device)
    print(model)
    for i in range(100):
        if torch.cuda.is_available():
            x = Variable(torch.rand(1, 3, 32, 18)).to(device)
            # x = Variable(torch.rand(1, 3, 1280, 720)).to(device)
        else:
            x = Variable(torch.rand(1, 3, 1280, 720))
        output = model(x)
        print(output[0].size())
        print(output[0][0, 0, 0, 0:4])
