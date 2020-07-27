'''
D-RESUNET: RESUNET AND DILATED CONVOLUTION FOR HIGH RESOLUTION
SATELLITE IMAGERY ROAD EXTRACTION
'''


import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from blocks import simple_block, Down_sample, Up_sample, ResBasicBlock, ResStem

class DResUNet(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        base_channel,
        pad_method='pad',
        softmax=True
    ):
        super(DResUNet, self).__init__()

        self.softmax = softmax
        self.pad_method = pad_method

        self.stem = ResStem(in_channels, base_channel * 2)
        self.down_conv1 = ResBasicBlock(base_channel * 2, base_channel * 4, stride=2)
        self.down_conv2 = ResBasicBlock(base_channel * 4, base_channel * 8, stride=2)

        self.bridge = center_part(base_channel * 8, base_channel * 16)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv3 = ResBasicBlock(base_channel * (16 + 8), base_channel * 8, stride=1)
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = ResBasicBlock(base_channel * (8 + 4), base_channel * 4, stride=1)
        self.up_sample_stem = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ResBasicBlock(base_channel * (4 + 2), base_channel * 2, stride=1)  

        self.out = nn.Conv3d(base_channel * 2, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):

        stem = self.stem(x)
        down_conv1 = self.down_conv1(stem)
        down_conv2 = self.down_conv2(down_conv1)

        bridge = self.bridge(down_conv2)
        
        up_sample_2 = self.up_sample_2(bridge)
        up_sample_2 = self.pad(down_conv2, up_sample_2, method=self.pad_method)
        concat2 = torch.cat([down_conv2, up_sample_2], dim=1)
        up_conv3 = self.up_conv3(concat2)

        up_sample_1 = self.up_sample_1(up_conv3)
        up_sample_1 = self.pad(down_conv1, up_sample_1, method=self.pad_method)
        concat1 = torch.cat([down_conv1, up_sample_1], dim=1)
        up_conv2 = self.up_conv2(concat1)

        up_sample_stem = self.up_sample_stem(up_conv2)
        up_sample_stem = self.pad(stem, up_sample_stem, method=self.pad_method)
        concat0 = torch.cat([stem, up_sample_stem], dim=1)
        up_conv1 = self.up_conv1(concat0)

        out = self.out(up_conv1)

        if self.softmax:
            out = F.softmax(out, dim=1)

        return out

    def pad(self, encoder, decoder, method='pad'):

        encoder_z, encoder_y, encoder_x = encoder.shape[-3], encoder.shape[-2], encoder.shape[-1]
        decoder_z, decoder_y, decoder_x = decoder.shape[-3], decoder.shape[-2], decoder.shape[-1]
        diff_z, diff_y, diff_x = encoder_z - decoder_z, encoder_y - decoder_y, encoder_x - decoder_x

        if method == 'pad':
            x = F.pad(decoder, (diff_x//2, diff_x - diff_x//2, 
                                diff_y//2, diff_y - diff_y//2, 
                                diff_z//2, diff_z - diff_z//2), 
                                mode='constant', value=0)
        elif method == 'interpolate':
            x = F.interpolate(decoder, size=(encoder_z, encoder_y, encoder_x), mode='nearest')
        else:
            raise NotImplementedError()

        return x  


class center_part(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        groups=2
    ):
        super(center_part, self).__init__()

        self.branch_1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=1, padding=1, groups=2)

        self.branch_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=1, padding=1, groups=groups),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, groups=groups)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=1, padding=1, groups=groups),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=2, padding=2, groups=groups),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, dilation=4, padding=4, groups=groups)
        )

        self.short_cut = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=groups)

    def forward(self, x):

        short_cut = self.short_cut(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        return F.relu(branch_1 + branch_2 + branch_3 + short_cut)


if __name__ == '__main__':

    from utils import load_config
    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    base_channel = 4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = DResUNet(input_modalites, output_channels, base_channel)
    net.to(device)
    # input = torch.randn(1, 4, 98, 98, 98).to(device)
    # y = net(input)
    
    # print(y.shape)

    def count_params(model):
        
        ''' print number of trainable parameters and its size of the model'''

        num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : params number {}, params size: {:4f}M'.format(model._get_name(), num_of_param, num_of_param*4/1000/1000))
    
    count_params(model=net)