import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from blocks import simple_block, Down_sample, Up_sample
from torchsummary import summary


# simplest U-Net
class init_U_Net(nn.Module):

    def __init__(
        self, 
        input_modalites, 
        output_channels, 
        base_channel,
        pad_method='pad',
        softmax=True,
    ):
        
        super(init_U_Net, self).__init__()
        self.softmax = softmax
        self.pad_method = pad_method
        
        self.min_channel = base_channel
        self.down_conv1  = simple_block(input_modalites, self.min_channel*2, 3)
        self.down_sample_1 = Down_sample(2)
        self.down_conv2  = simple_block(self.min_channel*2, self.min_channel*4, 3)
        self.down_sample_2 = Down_sample(2)
        self.down_conv3  = simple_block(self.min_channel*4, self.min_channel*8, 3)
        self.down_sample_3 = Down_sample(2)

        self.bridge = simple_block(self.min_channel*8, self.min_channel*16, 3)
        
        self.up_sample_1   = Up_sample(self.min_channel*16, self.min_channel*16, 2)
        self.up_conv1  = simple_block(self.min_channel*24, self.min_channel*8, 3, is_down=False)
        self.up_sample_2   = Up_sample(self.min_channel*8, self.min_channel*8, 2)
        self.up_conv2  = simple_block(self.min_channel*12, self.min_channel*4, 3, is_down=False)
        self.up_sample_3   = Up_sample(self.min_channel*4, self.min_channel*4, 2)
        self.up_conv3  = simple_block(self.min_channel*6, self.min_channel*2, 3, is_down=False)

        self.out = nn.Conv3d(self.min_channel*2, output_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # encoder path

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
        self.concat_1 = torch.cat([self.block_5_upsample, self.block_3], dim=1)

        self.block_5 = self.up_conv1(self.concat_1)
        self.block_6_upsample = self.up_sample_2(self.block_5)
        self.block_6_upsample = self.pad(self.block_2, self.block_6_upsample, self.pad_method)
        self.concat_2 = torch.cat([self.block_6_upsample, self.block_2], dim=1)

        self.block_6 = self.up_conv2(self.concat_2)
        self.block_7_upsample = self.up_sample_3(self.block_6)
        self.block_7_upsample = self.pad(self.block_1, self.block_7_upsample, self.pad_method)
        self.concat_3 = torch.cat([self.block_7_upsample, self.block_1], dim=1)

        self.block_7 = self.up_conv3(self.concat_3)

        res = self.out(self.block_7)
        
        if self.softmax:
            res = F.softmax(res, dim=1)

        return res

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


if __name__ == '__main__':

    from utils import load_config
    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    base_channel = 4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = init_U_Net(input_modalites, output_channels, base_channel)
    net.to(device)

    import tensorwatch as tw
    from tensorboardX import SummaryWriter

    # print(net)
    # params = list(net.parameters())
    # for i in range(len(params)):
    #     layer_shape = params[i].size()
        
    #     print(len(layer_shape))

    # print parameters infomation
    # count_params(net)
    input = torch.randn(1, 4, 64, 64, 64).to(device)
    # tw.draw_model(net, input)
    

    # input = torch.randn(1, 4, 130, 130, 130).to(device)
    # print(y.shape)
    # summary(net, input_size=(4, 64, 64, 64))
    # print(net)
    # print(net._modules.keys())
    # net.out = nn.Conv3d(16, 8, 3, padding=1)
    # net.to(device)
    # y = net(input)
    
    # print(y.data.shape)

    
    def count_params(model):
        
        ''' print number of trainable parameters and its size of the model'''

        num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : params number {}, params size: {:4f}M'.format(model._get_name(), num_of_param, num_of_param*4/1000/1000))
    
    count_params(model=net)



        


