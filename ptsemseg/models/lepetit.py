import torch.nn as nn

from ptsemseg.models.utils import *


class build_network_lepetit(nn.Module):
    def __init__(self, args):
        super(build_network_lepetit, self).__init__()
        # 3 input image channel, 32 output channels, 3x3 square convolution kernel
        ch1, ch2, ch3 = 32, 32, 16
        fl1, fl2, fl3 = 3, 5, 7
        self.layer0_conv0 = nn.Conv2d(3, ch1, fl1, padding=1)
        self.layer0_conv1 = nn.Conv2d(ch1, ch2, fl2, padding=2)
        self.layer0_conv2 = nn.Conv2d(ch2, ch3, fl3, padding=3)
        self.layer2_conv0 = nn.Conv2d(3, ch1, fl1, padding=1)
        self.layer2_conv1 = nn.Conv2d(ch1, ch2, fl2, padding=2)
        self.layer2_conv2 = nn.Conv2d(ch2, ch3, fl3, padding=3)
        self.layer4_conv0 = nn.Conv2d(3, ch1, fl1, padding=1)
        self.layer4_conv1 = nn.Conv2d(ch1, ch2, fl2, padding=2)
        self.layer4_conv2 = nn.Conv2d(ch2, ch3, fl3, padding=3)
        # an affine operation: y = Wx + b
        self.upscale2 = nn.UpsamplingBilinear2d(size=(args.INPUT_HEIGHT, args.INPUT_WIDTH))
        self.upscale4 = nn.UpsamplingBilinear2d(size=(args.INPUT_HEIGHT, args.INPUT_WIDTH))
        self.fc = nn.Linear(3 * ch3, 2)

    def forward(self, x0, x2, x4):
        x0 = F.leaky_relu(self.layer0_conv0(x0), negative_slope=0.05)
        x0 = F.leaky_relu(self.layer0_conv1(x0), negative_slope=0.05)
        x0 = F.leaky_relu(self.layer0_conv2(x0), negative_slope=0.05)
        x2 = F.leaky_relu(self.layer2_conv0(x2), negative_slope=0.05)
        x2 = F.leaky_relu(self.layer2_conv1(x2), negative_slope=0.05)
        x2 = F.leaky_relu(self.layer2_conv2(x2), negative_slope=0.05)
        x2 = self.upscale2(x2)
        x4 = F.leaky_relu(self.layer4_conv0(x4), negative_slope=0.05)
        x4 = F.leaky_relu(self.layer4_conv1(x4), negative_slope=0.05)
        x4 = F.leaky_relu(self.layer4_conv2(x4), negative_slope=0.05)
        x4 = self.upscale4(x4)
        x = torch.cat((x0, x2, x4), 1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc(x)
        return x


class build_network_stage2(nn.Module):
    def __init__(self, args):
        super(build_network_stage2, self).__init__()
        # 3 input image channel, 32 output channels, 3x3 square convolution kernel
        ch1, ch2, ch3 = 8, 4, 1
        fl1, fl2, fl3 = 3, 3, 3
        self.upscale_prob0 = nn.UpsamplingBilinear2d(size=(args.INPUT_HEIGHT_STAGE2, args.INPUT_WIDTH_STAGE2))
        self.upscale_prob2 = nn.UpsamplingBilinear2d(size=(int(args.INPUT_HEIGHT_STAGE2/2), int(args.INPUT_WIDTH_STAGE2/2)))
        self.upscale_prob4 = nn.UpsamplingBilinear2d(size=(int(args.INPUT_HEIGHT_STAGE2/4), int(args.INPUT_WIDTH_STAGE2/4)))
        self.layer0_conv0 = nn.Conv2d(4, ch1, fl1, padding=1)
        self.layer0_conv1 = nn.Conv2d(ch1, ch2, fl2, padding=1)
        self.layer0_conv2 = nn.Conv2d(ch2, ch3, fl3, padding=1)
        self.layer2_conv0 = nn.Conv2d(4, ch1, fl1, padding=1)
        self.layer2_conv1 = nn.Conv2d(ch1, ch2, fl2, padding=1)
        self.layer2_conv2 = nn.Conv2d(ch2, ch3, fl3, padding=1)
        self.layer4_conv0 = nn.Conv2d(4, ch1, fl1, padding=1)
        self.layer4_conv1 = nn.Conv2d(ch1, ch2, fl2, padding=1)
        self.layer4_conv2 = nn.Conv2d(ch2, ch3, fl3, padding=1)
        # an affine operation: y = Wx + b
        self.upscale2 = nn.UpsamplingBilinear2d(size=(args.INPUT_HEIGHT_STAGE2, args.INPUT_WIDTH_STAGE2))
        self.upscale4 = nn.UpsamplingBilinear2d(size=(args.INPUT_HEIGHT_STAGE2, args.INPUT_WIDTH_STAGE2))
        self.fc = nn.Linear(3 * ch3, 2)

    def forward(self, prob, x0, x2, x4):
        prob0 = self.upscale_prob0(prob)
        prob2 = self.upscale_prob2(prob)
        prob4 = self.upscale_prob4(prob)
        x0 = torch.cat((x0, prob0), 1)
        x2 = torch.cat((x2, prob2), 1)
        x4 = torch.cat((x4, prob4), 1)
        x0 = F.leaky_relu(self.layer0_conv0(x0), negative_slope=0.05)
        x0 = F.leaky_relu(self.layer0_conv1(x0), negative_slope=0.05)
        x0 = F.leaky_relu(self.layer0_conv2(x0), negative_slope=0.05)
        x2 = F.leaky_relu(self.layer2_conv0(x2), negative_slope=0.05)
        x2 = F.leaky_relu(self.layer2_conv1(x2), negative_slope=0.05)
        x2 = F.leaky_relu(self.layer2_conv2(x2), negative_slope=0.05)
        x2 = self.upscale2(x2)
        x4 = F.leaky_relu(self.layer4_conv0(x4), negative_slope=0.05)
        x4 = F.leaky_relu(self.layer4_conv1(x4), negative_slope=0.05)
        x4 = F.leaky_relu(self.layer4_conv2(x4), negative_slope=0.05)
        x4 = self.upscale4(x4)
        x = torch.cat((x0, x2, x4), 1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc(x)
        return x

