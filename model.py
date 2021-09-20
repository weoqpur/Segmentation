import torch
import torch.nn as nn
from torchvision import transforms, datasets

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # convolution, batch normalization, RelU, 2D
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr