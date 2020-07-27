import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision


class simple_block(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=1,  
                is_down=True,
                mode='CNN',
                ):
        super(simple_block, self).__init__()

        if is_down:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, 
                            out_channels//2, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=2),
                nn.InstanceNorm3d(out_channels//2, affine=True),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv3d(out_channels//2, 
                            out_channels, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=2),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
        
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, 
                            out_channels, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=2),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv3d(out_channels, 
                            out_channels, 
                            kernel_size, 
                            stride=1, 
                            padding=padding, 
                            groups=2),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.1, inplace=True)
            )


    def forward(self, x):
        x = self.block(x)

        return x

class Up_sample(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=2, 
                padding=1, 
                output_padding=0, 
                groups=2):
        super(Up_sample, self).__init__()

        self.block = nn.ConvTranspose3d(in_channels, 
                                        out_channels, 
                                        kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        output_padding=output_padding,
                                        groups=groups)
        
    def forward(self, x):
        return self.block(x)

class Down_sample(nn.Module):
    def __init__(self, kernel_size, stride=2, padding=1):
        super(Down_sample, self).__init__()
        self.block = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
   
    def forward(self, x):
        return self.block(x)

class BRDown_sample(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=2, padding=1, maxpool=True):
        super(BRDown_sample, self).__init__()

        if maxpool:
            self.block = nn.Sequential(
                nn.MaxPool3d(kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.BatchNorm3d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)

            
#------------------------------------------------------

class ResBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        stride,
        activation=True,
        se=False,
        **kwargs
    ):
        super(ResBasicBlock, self).__init__()

        middle_channels = min(in_channels, out_channels)

        self.activation = activation
        self.se = se 
        self.resblock = nn.Sequential(
            nn.BatchNorm3d(num_features=in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=middle_channels, kernel_size=3, stride=stride, padding=1, groups=2),

            nn.BatchNorm3d(num_features=middle_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(in_channels=middle_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=2)
        )
        if self.se:
            self.seblock = SELayer(out_channels)
        self.shortcut = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=2)

    def forward(self, x):
        # identity = x
        resblock = self.resblock(x)
        if self.se:
            resblock = self.seblock(resblock)
        shortcut = self.shortcut(x)

        return F.relu(resblock + shortcut) if self.activation else resblock + shortcut



class ResStem(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        activation=True,
    ):
        super(ResStem, self).__init__()

        self.activation = activation
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=2),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=2)
        )
        self.shortcut = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=2)

    def forward(self, x):
        stem = self.stem(x)
        shortcut = self.shortcut(x)
        
        return F.relu(stem + shortcut) if self.activation else stem + shortcut



class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)

        return x * y.expand_as(x)