'''
concatenate the original image at each layer directly
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from blocks import simple_block, Down_sample, Up_sample, BRDown_sample
from model import init_U_Net

class U_Net_direct_concat(init_U_Net):
    def __init__(
        self,
        input_modalites, 
        output_channels, 
        base_channel,
        pad_method='pad',
        softmax=True,
    ):
        super(U_Net_direct_concat, self).__init__(
            input_modalites, 
            output_channels, 
            base_channel,
            pad_method,
            softmax,
        )

        self.up_conv1  = simple_block(self.min_channel*24 + input_modalites, self.min_channel*8, 3, is_down=False)
        self.up_conv2  = simple_block(self.min_channel*12 + input_modalites, self.min_channel*4, 3, is_down=False)
        self.up_conv3  = simple_block(self.min_channel*6 + input_modalites, self.min_channel*2, 3, is_down=False)
                
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        self.x_layer1 = x
        self.x_layer2 = self.down_sample_2(self.x_layer1)
        self.x_layer3 = self.down_sample_3(self.x_layer2)

        self.block_1 = self.down_conv1(x)
        self.block_1_pool = self.down_sample_1(self.block_1)
        self.block_2 = self.down_conv2(self.block_1_pool)
        self.block_2_pool = self.down_sample_2(self.block_2)
        self.block_3 = self.down_conv3(self.block_2_pool)
        self.block_3_pool = self.down_sample_3(self.block_3)

        # bridge
        self.block_4 = self.bridge(self.block_3_pool)

        # decoder path
        self.block_5_upsample = self.up_sample_1(self.block_4)

        self.block_5_upsample = self.pad(self.block_3, self.block_5_upsample, self.pad_method)
        self.concat_1 = torch.cat([self.block_5_upsample, self.block_3, self.x_layer3], dim=1)
        self.block_5 = self.up_conv1(self.concat_1)
        self.block_6_upsample = self.up_sample_2(self.block_5)

        self.block_6_upsample = self.pad(self.block_2, self.block_6_upsample, self.pad_method)
        self.concat_2 = torch.cat([self.block_6_upsample, self.block_2, self.x_layer2], dim=1)
        self.block_6 = self.up_conv2(self.concat_2)
        self.block_7_upsample = self.up_sample_3(self.block_6)

        self.block_7_upsample = self.pad(self.block_1, self.block_7_upsample, self.pad_method)
        self.concat_3 = torch.cat([self.block_7_upsample, self.block_1, self.x_layer1], dim=1)
        self.block_7 = self.up_conv3(self.concat_3)
    
        res = self.out(self.block_7)

        if self.softmax:
            res = F.softmax(res, dim=1)

        return res

if __name__ == '__main__':

    from utils import load_config
    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    base_channel = 4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = U_Net_direct_concat(input_modalites, output_channels, base_channel)
    net.to(device)

    def count_params(model):
        
        ''' print number of trainable parameters and its size of the model'''

        num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : params number {}, params size: {:4f}M'.format(model._get_name(), num_of_param, num_of_param*4/1000/1000))
    
    count_params(model=net)

    # input = torch.randn(1, 4, 64, 64, 64).to(device)
    # print(net)
    # y = net(input)
    # print(y.shape)