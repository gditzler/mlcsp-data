

import torch.nn as nn

class WaterRail(nn.Module): 
    def __init__(self, num_classes:int=9) -> None:
        super(WaterRail).__init__()
    
    def conv_sub_block(self, x, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x): 
        return x