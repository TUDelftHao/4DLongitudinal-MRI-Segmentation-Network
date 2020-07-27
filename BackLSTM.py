'''
BiLSTM is added at the end of UNet
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from blocks import simple_block, Down_sample, Up_sample
from model import init_U_Net
from model_try import U_Net_direct_concat
from ResUNet import ResUNet
from DResUNet import DResUNet
from RNN import ConvLSTMCell, ConvLSTM, DenseBiConvLSTM, BiConvLSTM
from data_prepara import distributed_is_initialized
from utils import WrappedModel

class BackLSTM(nn.Module):
    def __init__(
        self,  
        input_dim, 
        hidden_dim, 
        output_dim,
        kernel_size,
        num_layers,
        conv_type,
        lstm_backbone,
        unet_module,
        base_channel,
        return_sequence=True,
        is_pretrain=False
    ):
        super(BackLSTM, self).__init__()

        self.layer_dim = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.return_sequence = return_sequence
        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # load backbone
        if unet_module == 'UNet':
            self.unet_backbone = init_U_Net(input_dim, output_dim, base_channel, softmax=False)
            ckp_path = 'best_newdata/UNet-p64-b4-newdata-oriinput_best_model.pth.tar'            
        elif unet_module == 'ResUNet':
            self.unet_backbone = ResUNet(input_dim, output_dim, base_channel, softmax=False)
            ckp_path = 'best_newdata/ResUNet-p64-b4-newdata-oriinput_best_model.pth.tar'
        elif unet_module == 'DResUNet':
            self.unet_backbone = DResUNet(input_dim, output_dim, base_channel, softmax=False)
            ckp_path = 'best_newdata/DResUNet-p64-b4-newdata-oriinput_best_model.pth.tar'
        else:
            raise NotImplementedError()

        # load pretrain weights
        if is_pretrain:
            self.unet_backbone = WrappedModel(self.unet_backbone)
            checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
            self.unet_backbone.load_state_dict(checkpoint['model_state_dict'])
            for param in self.unet_backbone.parameters():
                param.requires_grad = False
            self.unet_backbone.module.out = nn.Conv3d(base_channel * 2, self.hidden_dim * 2, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(self.unet_backbone.module.out.weight, mode='fan_out', nonlinearity='leaky_relu')
            for param in self.unet_backbone.module.up_conv3.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.module.up_conv2.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.module.up_conv1.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.module.up_sample_1.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.module.up_sample_2.parameters():
                param.requires_grad = True
            for param in self.unet_backbone.module.up_sample_3.parameters():
                param.requires_grad = True

        else:
            self.unet_backbone.out = nn.Conv3d(base_channel * 2, self.hidden_dim * 2, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(self.unet_backbone.out.weight, mode='fan_out', nonlinearity='leaky_relu')

        # for name, param in self.unet_backbone.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # load LSTM
        if lstm_backbone == 'ConvLSTM':
            self.net = ConvLSTM(self.hidden_dim * 2, hidden_dim, kernel_size, num_layers, conv_type, return_sequence)
        elif lstm_backbone == 'BiConvLSTM':
            self.net = BiConvLSTM(self.hidden_dim * 2, hidden_dim, kernel_size, num_layers, conv_type, return_sequence)
        elif lstm_backbone == 'DenseBiLSTM':
            self.net = DenseBiConvLSTM(self.hidden_dim * 2, hidden_dim, kernel_size, num_layers, conv_type, return_sequence)
        else:
            raise NotImplementedError()

        self.conv1x1 = nn.Conv3d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, x.size(-3), x.size(-2), x.size(-1)).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, x.size(-3), x.size(-2), x.size(-1)).requires_grad_().to(x.device)

        self.unet_backbone.to(x.device)
        batch_size, seq_len = x.shape[0], x.shape[1]

        out_size = np.array(x.shape)
        out_size[2] = self.hidden_dim * 2

        out_of_unet = torch.zeros(tuple(out_size)).to(x.device)

        for tp in range(seq_len):
            sub_x = x[:, tp]
            out = self.unet_backbone(sub_x)
            out_of_unet[:, tp] = out

        del x 
        out = self.net(out_of_unet, (h0, c0))[0]

        del out_of_unet

        if self.return_sequence:
            res = torch.zeros(out.size(0), out.size(1), self.output_dim, out.size(-3), out.size(-2), out.size(-1)).to(out.device)
            for i in range(out.size(1)):
                res[:, i, ...] = F.softmax(self.conv1x1(out[:, i, ...]), dim=1)
        else:
            res = F.softmax(self.conv1x1(out), dim=1)

        return res

        
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    conv_type = 'cba'
    model = BackLSTM(input_dim=4, hidden_dim=4, output_dim=5, kernel_size=3, num_layers=1, conv_type=conv_type, lstm_backbone='ConvLSTM', unet_module='UNet', base_channel=4, return_sequence=True, is_pretrain=True).to(device)


    # print(next(model.parameters()).is_cuda)
    input = torch.randn(1, 2, 4, 64, 64, 64).to(device)
    y = model(input)
    print(y.shape)


    






