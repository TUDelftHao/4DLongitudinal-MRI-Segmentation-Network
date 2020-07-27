import torch 
import torch.nn as nn 
import torchvision 
import torch.nn.functional as F 
from RNN import ConvLSTMCell, ConvLayer
from ResUNet import ResUNet
from utils import WrappedModel
import numpy as np 

class ResCenterLSTMEncoder(ResUNet, nn.Module):
    def __init__(self, in_channels, out_channels, base_channel, is_pretrain=True):
        ResUNet.__init__(self, in_channels, out_channels, base_channel)

        if is_pretrain:
            backbone = ResUNet(in_channels, out_channels, base_channel, softmax=False)
            ckp_path = 'best_newdata/ResUNet-p64-b4-newdata-oriinput_best_model.pth.tar'
            backbone = WrappedModel(backbone)
            checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.load_state_dict(checkpoint['model_state_dict'])

            self.stem = backbone.module.stem
            self.down_conv1 = backbone.module.down_conv1
            self.down_conv2 = backbone.module.down_conv2


        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # 
    def forward(self, x):
        stem = self.stem(x)
        down_conv1 = self.down_conv1(stem)
        down_conv2 = self.down_conv2(down_conv1)

        return down_conv2, [stem, down_conv1, down_conv2]

class ResCenterLSTMDecoder(ResUNet, nn.Module):
    def __init__(self, in_channels, out_channels, base_channel, is_pretrain=False):
        ResUNet.__init__(self, in_channels, out_channels, base_channel)

        if is_pretrain:
            backbone = ResUNet(in_channels, out_channels, base_channel, softmax=False)
            ckp_path = 'best_newdata/ResUNet-p64-b4-newdata-oriinput_best_model.pth.tar'
            backbone = WrappedModel(backbone)
            checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.load_state_dict(checkpoint['model_state_dict'])

            self.up_sample_2 = backbone.module.up_sample_2
            self.up_sample_1 = backbone.module.up_sample_1
            self.up_sample_stem = backbone.module.up_sample_stem
            self.up_conv3 = backbone.module.up_conv3
            self.up_conv2 = backbone.module.up_conv2
            self.up_conv1 = backbone.module.up_conv1
            self.out = backbone.module.out
            for param in self.out.parameters():
                param.requires_grad = True
            nn.init.kaiming_normal_(self.out.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x, shortcut):
        stem, down_conv1, down_conv2 = shortcut

        up_sample_2 = self.up_sample_2(x)
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

class ResCenterLSTM(nn.Module):
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
        super(ResCenterLSTM, self).__init__()

        self.num_layers = num_layers
        self.outdim = output_channels
        self.return_sequence = return_sequence

        self.encoder = ResCenterLSTMEncoder(input_modalites, output_channels, base_channel, is_pretrain=is_pretrain)
        self.decoder = ResCenterLSTMDecoder(input_modalites, output_channels, base_channel)

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
            cur_layer_input, down_blocks = self.encoder(x[:, t])
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

            res = self.decoder(h, down_blocks)
            output[:, t] = res 

        if self.return_sequence:
            return output
        else:
            return output[:, -1]


class BiResCenterLSTM(ResCenterLSTM, nn.Module):
    def __init__(
        self,
        input_modalites,
        output_channels,
        base_channel,
        num_layers,
        connect='normal',
        pad_method='pad',
        conv_type='plain',
        softmax=True,
        return_sequence=True,
        is_pretrain=False
    ):
        ResCenterLSTM.__init__(
            self,
            input_modalites,
            output_channels,
            base_channel,
            num_layers,
            pad_method=pad_method,
            conv_type=conv_type,
            softmax=softmax,
            return_sequence=return_sequence,
            is_pretrain=is_pretrain
        )
        '''
        @params connect: 'dense' or 'normal', connect method between forward and backward output
        '''

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
        for t in range(seq_dim):
            to_bridge, down_block = self.encoder(x[:, t])
            seq_input.append(to_bridge)
            down_blocks.append(down_block)

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
            output[:, t] = self.decoder(bridge_out[:, t], down_blocks[t])

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

    model = ResCenterLSTM(input_modalites, 
        output_channels,
        base_channel,
        num_layers=1,
        is_pretrain=True,
        conv_type='cba'
        ).to(device)

    model = BiResCenterLSTM(
        input_modalites, 
        output_channels,
        base_channel,
        num_layers=1,
        connect='dense',
        is_pretrain=True,
        conv_type='cba'
    ).to(device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    input = torch.randn(1, 1, 4, 64, 64, 64).to(device)
    
    count_params(model)
    # print(model)

    out = model(input)
    print(out.shape)