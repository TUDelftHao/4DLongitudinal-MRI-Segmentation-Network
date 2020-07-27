import torch 
import torch.nn as nn 
import torchvision 
import torch.nn.functional as F 
from RNN import ConvLSTMCell, ConvLayer
from model_try import U_Net_direct_concat
from utils import WrappedModel
import numpy as np 

class DirectCenterLSTMEncoder(U_Net_direct_concat, nn.Module):
    def __init__(self, input_modalites, output_channels, base_channel, is_pretrain=True):
        U_Net_direct_concat.__init__(self, input_modalites, output_channels, base_channel)

        if is_pretrain:
            backbone = U_Net_direct_concat(input_modalites, output_channels, base_channel, softmax=False)
            ckp_path = 'best_newdata/direct-UNet-p64-newdata-oriinput_best_model.pth.tar' 
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

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)

    def forward(self, x):
        x_layer1 = x
        x_layer2 = self.down_sample_2(x_layer1)
        x_layer3 = self.down_sample_3(x_layer2)

        block_1 = self.down_conv1(x)
        block_1_pool = self.down_sample_1(block_1)
        block_2 = self.down_conv2(block_1_pool)
        block_2_pool = self.down_sample_2(block_2)
        block_3 = self.down_conv3(block_2_pool)
        block_3_pool = self.down_sample_3(block_3)

        return block_3_pool, [block_1, block_2, block_3], [x_layer1, x_layer2, x_layer3]

class DirectCenterLSTMDecoder(U_Net_direct_concat, nn.Module):
    def __init__(self, input_modalites, output_channels, base_channel, is_pretrain=False):
        U_Net_direct_concat.__init__(self, input_modalites, output_channels, base_channel)

        if is_pretrain:
            backbone = U_Net_direct_concat(input_modalites, output_channels, base_channel, softmax=False)
            ckp_path = 'best_newdata/direct-UNet-p64-newdata-oriinput_best_model.pth.tar' 
            backbone = WrappedModel(backbone)
            checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.load_state_dict(checkpoint['model_state_dict'])

            self.up_sample_1 = backbone.module.up_sample_1
            self.up_sample_2 = backbone.module.up_sample_2
            self.up_sample_3 = backbone.module.up_sample_3
            self.up_conv1 = backbone.module.up_conv1
            self.up_conv2 = backbone.module.up_conv2
            self.up_conv3 = backbone.module.up_conv3

            self.out = backbone.module.out
            for param in self.out.parameters():
                param.requires_grad = True
            nn.init.kaiming_normal_(self.out.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x, shortcut, direct):
        block_1, block_2, block_3 = shortcut
        x_layer1, x_layer2, x_layer3 = direct

        block_5_upsample = self.up_sample_1(x)

        block_5_upsample = self.pad(block_3, block_5_upsample, self.pad_method)
        concat_1 = torch.cat([block_5_upsample, block_3, x_layer3], dim=1)
        block_5 = self.up_conv1(concat_1)
        block_6_upsample = self.up_sample_2(block_5)

        block_6_upsample = self.pad(block_2, block_6_upsample, self.pad_method)
        concat_2 = torch.cat([block_6_upsample, block_2, x_layer2], dim=1)
        block_6 = self.up_conv2(concat_2)
        block_7_upsample = self.up_sample_3(block_6)

        block_7_upsample = self.pad(block_1, block_7_upsample, self.pad_method)
        concat_3 = torch.cat([block_7_upsample, block_1, x_layer1], dim=1)
        block_7 = self.up_conv3(concat_3)
    
        res = self.out(block_7)

        if self.softmax:
            res = F.softmax(res, dim=1)

        return res

class DirectCenterLSTM(nn.Module):
    def __init__(
        self,
        input_modalites,
        output_channels,
        base_channel,
        num_layers,
        pad_method='pad',
        conv_type='plain',
        softmax=True,
        return_sequence=True,
        is_pretrain=False
    ):
        super(DirectCenterLSTM, self).__init__()

        self.num_layers = num_layers
        self.outdim = output_channels
        self.return_sequence = return_sequence

        self.encoder = DirectCenterLSTMEncoder(input_modalites, output_channels, base_channel, is_pretrain=is_pretrain)
        self.decoder = DirectCenterLSTMDecoder(input_modalites, output_channels, base_channel)

        self._LSTM_input_dim = base_channel * 8
        self._LSTM_hidden_dim = [base_channel * 16] * self.num_layers
        self._kernel_size = [3] * self.num_layers

        cell_list_per_layer = []
        for i in range(self.num_layers):
            cell_input = self._LSTM_input_dim if i==0 else self._LSTM_hidden_dim[i-1]
            cell_list_per_layer.append(ConvLSTMCell(input_dim=cell_input, hidden_dim=self._LSTM_hidden_dim[i], kernel_size=self._kernel_size[i], conv_type=conv_type))

        self.bridge = nn.ModuleList(cell_list_per_layer)

    def forward(self, x):
        batch_size, seq_dim = x.size(0), x.size(1)
        output = torch.zeros(batch_size, seq_dim, self.outdim, x.size(-3), x.size(-2), x.size(-1)).requires_grad_().to(x.device)

        hidden_state = []
        
        for t in range(seq_dim):
            cur_layer_input, down_blocks, direct = self.encoder(x[:, t])
            layer_h_hidden_state = []
            layer_c_hidden_state = []
            for i in range(self.num_layers):
                # initilize hidden state at each layer
                if t == 0:
                    h = torch.zeros(batch_size, self._LSTM_hidden_dim[i], cur_layer_input.size(-3), cur_layer_input.size(-2), cur_layer_input.size(-1)).requires_grad_().to(x.device)
                    c = torch.zeros(batch_size, self._LSTM_hidden_dim[i], cur_layer_input.size(-3), cur_layer_input.size(-2), cur_layer_input.size(-1)).requires_grad_().to(x.device)
                else:
                    h, c = hidden_state[t-1][0][:, i], hidden_state[t-1][1][:, i]

                h, c = self.bridge[i](cur_layer_input, pre_state=[h, c])
                cur_layer_input = h
                layer_h_hidden_state.append(h)
                layer_c_hidden_state.append(c)

            # hidden states at time t
            layer_h_hidden_state = torch.stack(layer_h_hidden_state, dim=1)
            layer_c_hidden_state = torch.stack(layer_c_hidden_state, dim=1)

            hidden_state.append([layer_h_hidden_state, layer_c_hidden_state])

            res = self.decoder(h, down_blocks, direct)
            output[:, t] = res 

        if self.return_sequence:
            return output
        else:
            return output[:, -1]


class BiDirectCenterLSTM(DirectCenterLSTM, nn.Module):
    def __init__(self, input_modalites, output_channels, base_channel, num_layers, connect='normal', pad_method='pad', conv_type='plain', softmax=True, return_sequence=True, is_pretrain=False):
        DirectCenterLSTM.__init__(self, input_modalites, output_channels, base_channel, num_layers, pad_method=pad_method, conv_type=conv_type, softmax=softmax, return_sequence=return_sequence, is_pretrain=is_pretrain)

        self.connect = connect
        compressConv_list = []
        for i in range(self.num_layers):
            compressConv_list.append(ConvLayer(in_channels=self._LSTM_hidden_dim[i]*2, out_channels=self._LSTM_hidden_dim[i], kernel_size=1, conv_type=conv_type))
        
        self.compressConv_list = nn.ModuleList(compressConv_list)

    def forward(self, x):
        
        batch_size, seq_dim = x.size(0), x.size(1)
        output = torch.zeros(batch_size, seq_dim, self.outdim, x.size(-3), x.size(-2), x.size(-1)).requires_grad_().to(x.device)

        seq_input = []
        down_blocks = []
        directs = []
        for t in range(seq_dim):
            to_bridge, down_block, direct = self.encoder(x[:, t])
            seq_input.append(to_bridge)
            down_blocks.append(down_block)
            directs.append(direct)

        seq_input = torch.stack(seq_input, dim=1)

        for i in range(self.num_layers):
            h_f = torch.zeros(batch_size, self._LSTM_hidden_dim[i], seq_input[:, 0].size(-3), seq_input[:, 0].size(-2), seq_input[:, 0].size(-1)).requires_grad_().to(x.device)
            c_f = h_f.clone()
            h_b = h_f.clone()
            c_b = h_f.clone()

            layer_output = []

            for t in range(seq_dim):

                h_f, c_f = self.bridge[i](seq_input[:, t], pre_state=[h_f, c_f])
                h_b, c_b = self.bridge[i](seq_input[:, seq_dim-t-1], pre_state=[h_b, c_b])

                if self.connect == 'dense':
                    h_add = torch.add(h_f, h_b)
                elif self.connect == 'normal':
                    h_combined = torch.cat((h_f, h_b), dim=1)
                    h_add = self.compressConv_list[i](h_combined)
                else:
                    raise NotImplementedError()
                    
                layer_output.append(h_add)

            seq_input = torch.stack(layer_output, dim=1)
            
        bridge_out = seq_input

        for t in range(seq_dim):
            output[:, t] = self.decoder(bridge_out[:, t], down_blocks[t], directs[t])

        if self.return_sequence:
            return output
        else:
            return output[:, -1]


if __name__ == '__main__':
    from utils import load_config
    from loss import DiceLoss
    from tqdm import tqdm
    import torch.optim as optim

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

    model = BiDirectCenterLSTM(
        input_modalites, 
        output_channels,
        base_channel,
        num_layers=1,
        connect='dense',
        is_pretrain=True,
        conv_type='cba'
    ).to(device)

    input = torch.randn(1, 1, 4, 64, 64, 64).to(device)
    
    count_params(model)
    # print(model)

    out = model(input)
    # print(out.shape)
        