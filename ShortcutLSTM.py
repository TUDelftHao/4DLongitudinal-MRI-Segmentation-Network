'''
LSTM is inserted at the concatenation path
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from blocks import simple_block, Down_sample, Up_sample
from model import init_U_Net
from RNN import ConvLSTMCell, ConvLayer
from CenterLSTM import CenterLSTMEncoder, CenterLSTMDecoder
from utils import WrappedModel


class ShortcutLSTM(nn.Module):
    def __init__(
        self,
        input_modalites, 
        output_channels,
        base_channel,
        num_layers,
        num_connects,
        pad_method='pad',
        conv_type='plain',
        softmax=True,
        return_sequence=True,
        is_pretrain=True
    ):
        super(ShortcutLSTM, self).__init__()

        self.input_channel = input_modalites
        self.output_channel = output_channels
        self.base_channel = base_channel
        self.return_sequence = return_sequence 

        self.body = ShortcutLSTMBody(input_modalites, output_channels, base_channel, num_layers, num_connects, pad_method, conv_type, softmax, is_pretrain=is_pretrain)

    def forward(self, x):

        self.body.to(x.device)
        batch_size, time_step, modality, depth, height, width = x.shape
        output = torch.zeros(batch_size, time_step, self.output_channel, depth, height, width).to(x.device)

        for i in range(time_step):
            tp = x[:, i]
            if i == 0:
                hidden_state = None
            res, hidden_state = self.body(tp, hidden_state=hidden_state)
            output[:, i] = res 

        if self.return_sequence:
            return output
        else:
            return output[:, -1]


class ShortcutLSTMBody(init_U_Net, nn.Module):
    def __init__(
        self, 
        input_modalites, 
        output_channels,
        base_channel,
        num_layers,
        num_connects,
        pad_method='pad',
        conv_type='plain',
        softmax=True,
        is_pretrain=True
    ):
        super(ShortcutLSTMBody, self).__init__(input_modalites, output_channels, base_channel, pad_method, softmax)

        self.input_modalites = input_modalites
        self.output_channels = output_channels
        self.base_channel = base_channel
        self.pad_method = pad_method
        self.softmax = softmax
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.num_connects = num_connects

        if is_pretrain:
            backbone = init_U_Net(input_modalites, output_channels, base_channel, softmax=False)
            ckp_path = 'best_newdata/UNet-p64-b4-newdata-oriinput_best_model.pth.tar' 
            backbone = WrappedModel(backbone)
            checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.load_state_dict(checkpoint['model_state_dict'])

            self.down_conv1 = backbone.module.down_conv1
            self.down_conv2 = backbone.module.down_conv2
            self.down_conv3 = backbone.module.down_conv3
            self.down_sample_1 = backbone.module.down_sample_1
            self.down_sample_2 = backbone.module.down_sample_2
            self.down_sample_3 = backbone.module.down_sample_3
            self.bridge = backbone.module.bridge
            self.up_sample_1 = backbone.module.up_sample_1
            self.up_sample_2 = backbone.module.up_sample_2
            self.up_sample_3 = backbone.module.up_sample_3
            self.up_conv1 = backbone.module.up_conv1
            self.up_conv2 = backbone.module.up_conv2
            self.up_conv3 = backbone.module.up_conv3
            # self.out  = backbone.module.out

        self.up_conv1 = nn.Sequential(*list(self.up_conv1.block)[:3])
        self.up_conv2 = nn.Sequential(*list(self.up_conv2.block)[:3])
        self.up_conv3 = nn.Sequential(*list(self.up_conv3.block)[:3])

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        
    def forward(self, x, hidden_state=None):
        
        if not hidden_state:
            hidden_state = [None] * 3
        batch_size = x.shape[0]

        # encoder path
        block_1 = self.down_conv1(x)
        block_1_pool = self.down_sample_1(block_1)
        block_2 = self.down_conv2(block_1_pool)
        block_2_pool = self.down_sample_2(block_2)
        block_3 = self.down_conv3(block_2_pool)
        block_3_pool = self.down_sample_3(block_3)

        # bridge
        block_4 = self.bridge(block_3_pool)

        # decoder path
        block_5_upsample = self.up_sample_1(block_4)
        block_5_upsample = self.pad(block_3, block_5_upsample, self.pad_method)
        concat_1 = torch.cat([block_5_upsample, block_3], dim=1)

        if self.num_connects == 3:
            state_1 = self.LSTMcellprocess(concat_1, batch_size, self.num_layers, self.conv_type, hidden_state=hidden_state[0])
            block_5 = self.up_conv1(state_1[-1][0])
        else:
            state_1 = None
            block_5 = self.up_conv1(concat_1)

        block_6_upsample = self.up_sample_2(block_5)
        block_6_upsample = self.pad(block_2, block_6_upsample, self.pad_method)
        concat_2 = torch.cat([block_6_upsample, block_2], dim=1)

        if self.num_connects == 2:
            state_2 = self.LSTMcellprocess(concat_2, batch_size, self.num_layers, self.conv_type, hidden_state=hidden_state[1])
            block_6 = self.up_conv2(state_2[-1][0])
        else:
            state_2 = None
            block_6 = self.up_conv2(concat_2)

        block_7_upsample = self.up_sample_3(block_6)
        block_7_upsample = self.pad(block_1, block_7_upsample, self.pad_method)
        concat_3 = torch.cat([block_7_upsample, block_1], dim=1)

        state_3 = self.LSTMcellprocess(concat_3, batch_size, self.num_layers, self.conv_type, hidden_state=hidden_state[2])
        block_7 = self.up_conv3(state_3[-1][0])

        res = self.out(block_7)

        if self.softmax:
            res = F.softmax(res, dim=1)

        return res, [state_1, state_2, state_3]

    def LSTMcellprocess(self, input_fm, batch_size, num_layers, conv_type, hidden_state=None):
        depth, height, width = input_fm.size(-3), input_fm.size(-2), input_fm.size(-1)
        input_dim = input_fm.size(1)
        hidden_dim = [input_dim] * num_layers
        cur_layer_input = input_fm

        if not hidden_state: 
            hidden = torch.zeros(batch_size, input_dim, depth, height, width, device=input_fm.device).requires_grad_()
            cell = torch.zeros(batch_size, input_dim, depth, height, width, device=input_fm.device).requires_grad_()
            hidden_state = [[hidden, cell] for _ in range(num_layers)]

        outputs = []
        for i in range(num_layers):
            cell_input = input_dim if i == 0 else hidden_dim[i-1]
            LSTMCell = ConvLSTMCell(input_dim=cell_input, hidden_dim=hidden_dim[i], kernel_size=1, conv_type=conv_type).to(input_fm.device)

            h, c = LSTMCell(cur_layer_input, pre_state=hidden_state[i])
            cur_layer_input = h 
            outputs.append([h, c])

        return outputs

        
if __name__ == '__main__':

    from utils import load_config
    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    base_channel = 4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def count_params(model):
    
        ''' print number of trainable parameters and its size of the model'''

        num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : params number {}, params size: {:4f}M'.format(model._get_name(), num_of_param, num_of_param*4/1000/1000))


    model = ShortcutLSTM(input_modalites, 
        output_channels,
        base_channel,
        num_layers=1,
        num_connects=3,
        conv_type='cba'
        ).to(device)

    # print(model)
    count_params(model)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # print(next(model.parameters()).is_cuda)
    input = torch.randn(1, 3, 4, 64, 64, 64).to(device)
    y = model(input)
    print(y.shape)


        

        




