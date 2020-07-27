'''
adapted from:
https://github.com/SreenivasVRao/ConvGRU-ConvLSTM-PyTorch/blob/master/convlstm.py
'''

import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, conv_type='plain'):
        '''
        @param conv_type: the order of BN, Activation and convlution layer. 'bac': BN + ReLU + Conv; 'cba': Conv + BN + ReLU; 'plain': Conv
        '''

        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.conv_type = conv_type
        
        if self.conv_type == 'plain':
            self.conv3D = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.input_dim+self.hidden_dim,
                    out_channels=self.hidden_dim*4,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=True
                )
            )
            
        elif self.conv_type == 'bac':
            self.conv3D = nn.Sequential(
                nn.InstanceNorm3d(self.input_dim + self.hidden_dim, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    in_channels=self.input_dim+self.hidden_dim,
                    out_channels=self.hidden_dim*4,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=True
                )
            )
        elif self.conv_type == 'cba':
            self.U = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.input_dim,
                    out_channels=self.hidden_dim*4,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=True
                ),
                nn.InstanceNorm3d(self.hidden_dim * 4, affine=True),
                nn.ReLU(inplace=True),
            )
            self.V = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim*4,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=True
                ),
                nn.InstanceNorm3d(self.hidden_dim * 4, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x, pre_state):
        
        hidden_state, cell_state = pre_state
        cell_input = self.U(x) + self.V(hidden_state)
        
        f_in, i_in, o_in, g_in = torch.split(cell_input, self.hidden_dim, dim=1)
        
        f = torch.sigmoid(f_in)
        i = torch.sigmoid(i_in)
        o = torch.sigmoid(o_in)
        g = torch.tanh(g_in)
        
        cell_state_cur = f*cell_state + i*g
        hidden_state_cur = o*torch.tanh(cell_state_cur)
        
        return hidden_state_cur, cell_state_cur
    
    def init_state(self, batch_size, image_size):
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv3D[0].weight.device).requires_grad_(),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv3D[0].weight.device).requires_grad_())


class ConvLSTM(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim_list, 
                kernel_size_list,
                num_layers, 
                conv_type,
                return_sequence=True,
                *args, **kwargs):
        
        super(ConvLSTM, self).__init__()

        if isinstance(hidden_dim_list, int):
            hidden_dim_list = [hidden_dim_list] * num_layers
        if isinstance(kernel_size_list, int):
            kernel_size_list = [kernel_size_list] * num_layers
        if not len(kernel_size_list) == len(hidden_dim_list) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list
        self.kernel_size_list = kernel_size_list
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.return_sequence = return_sequence
        
        cell_list_per_layer = []
        
        for i in range(self.num_layers):
            cell_input = self.input_dim if i==0 else self.hidden_dim_list[i-1]
            
            cell_list_per_layer.append(ConvLSTMCell(input_dim=cell_input,
                                                   hidden_dim=self.hidden_dim_list[i],
                                                   kernel_size=self.kernel_size_list[i],
                                                   conv_type=self.conv_type))
        
        self.cell_list_per_layer = nn.ModuleList(cell_list_per_layer)
        
    def forward(self, x, hidden_state):
        
        seq_dim = x.shape[1]
        h0, c0 = hidden_state

        layer_output_list = []
        last_state_list   = []
        cur_layer_input = x 
        
        for i in range(self.num_layers):
            h, c = h0[i], c0[i]
            output_layer_i = []
            
            for t in range(seq_dim):
                h, c = self.cell_list_per_layer[i](cur_layer_input[:, t, ...], pre_state=[h, c])
                # collect output of cells in layer i
                output_layer_i.append(h)
            
            # outputs (hidden state) of layer i
            cur_layer_input = torch.stack(output_layer_i, dim=1)
            # collect output of each layer
            layer_output_list.append(cur_layer_input)
            # collect the final hidden state and cell state of each layer
            last_state_list.append([h, c])

        out = torch.stack(layer_output_list, dim=1)
        
        # retur the final layer
        out = out[:, -1]
        last_state_list = last_state_list[-1]

        if not self.return_sequence:
            out = out[:, -1]
            last_state_list = last_state_list[-1]
        
        return out, last_state_list

            

class DenseBiConvLSTM(ConvLSTM):

    ''' Dense connection between two layers '''

    def __init__(self, 
                input_dim, 
                hidden_dim_list, 
                kernel_size_list, 
                num_layers, 
                conv_type,
                return_sequence=True,
                *args, **kwargs):

        super(DenseBiConvLSTM, self).__init__(input_dim, hidden_dim_list, kernel_size_list, num_layers, conv_type, return_sequence, *args, **kwargs)

    def forward(self, x, hidden_state):

        seg_dim = x.shape[1]
        h0, c0 = hidden_state
        layer_output_list = []
        last_state_list   = []
        cur_layer_input = x 

        for i in range(self.num_layers):
    
            output_inner = []
            h_f, c_f = h0[i], c0[i]
            h_b, c_b = h0[i], c0[i]

            for t in range(seg_dim):

                h_f, c_f = self.cell_list_per_layer[i](cur_layer_input[:, t, ...], pre_state=(h_f, c_f))
                h_b, c_b = self.cell_list_per_layer[i](cur_layer_input[:, seg_dim-t-1, ...], pre_state=(h_b, c_b))

                h_add = torch.add(h_f, h_b)
                output_inner.append(h_add)
                c_add = torch.add(c_f, c_b)
            
            cur_layer_input = torch.stack(output_inner, dim=1)
            layer_output_list.append(cur_layer_input)
            last_state_list.append((h_add, c_add))

        out = torch.stack(layer_output_list, dim=1)

        out = out[:, -1]
        last_state_list = last_state_list[-1]

        if not self.return_sequence:
            out = out[:, -1]
            last_state_list = last_state_list[-1]

        return out, last_state_list


class ConvLayer(nn.Module):

    ''' used for compressing the outputs from forward and backward direction '''

    def __init__(self, in_channels, out_channels, kernel_size, conv_type='plain'):
        '''
        @param conv_type: the order of BN, Activation and convlution layer. 'bac': BN + ReLU + Conv; 'cba': Conv + BN + ReLU; 'plain': Conv
        '''

        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        if self.conv_type == 'plain':
            self.conv3D = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2
            )
        elif self.conv_type == 'bac':
            self.conv3D = nn.Sequential(
                nn.InstanceNorm3d(self.in_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size//2
                )
            )
        elif self.conv_type == 'cba':
            self.conv3D = nn.Sequential(
                nn.Conv3d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size//2
                ),
                nn.InstanceNorm3d(self.out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError()

    
    def forward(self, x):
        out = self.conv3D(x)

        return out

class BiConvLSTM(ConvLSTM):

    ''' 1x1 convolution used between two layers to compress channels '''

    def __init__(self, 
                input_dim, 
                hidden_dim_list, 
                kernel_size_list, 
                num_layers, 
                conv_type,
                return_sequence=True,
                *args, **kwargs):
        super(BiConvLSTM, self).__init__(input_dim, hidden_dim_list, kernel_size_list, num_layers, conv_type, return_sequence, *args, **kwargs)

        compressConv_list = []
        for i in range(self.num_layers):
            compressConv_list.append(ConvLayer(in_channels=self.hidden_dim_list[i]*2,
                        out_channels=self.hidden_dim_list[i],
                        kernel_size=1,
                        conv_type=conv_type
                        ))
        
        self.compressConv_list = nn.ModuleList(compressConv_list)

    def forward(self, x, hidden_state):

        h0, c0 = hidden_state   
        seg_dim = x.shape[1]     
        layer_output_list = []
        last_state_list   = []
        # cur_layer_input_forward = x 
        # cur_layer_input_backward = x
        cur_layer_input = x

        for i in range(self.num_layers):

            # backward_states = []
            # forward_states = []
            output_inner = []

            h_f, c_f = h0[i], c0[i]
            h_b, c_b = h0[i], c0[i]

            for t in range(seg_dim):

                h_f, c_f = self.cell_list_per_layer[i](cur_layer_input[:, t, ...], pre_state=(h_f, c_f))
                # forward_states.append(h_f)

                h_b, c_b = self.cell_list_per_layer[i](cur_layer_input[:, seg_dim-t-1, ...], pre_state=(h_b, c_b))
                # backward_states.append(h_b)

                h_combined = torch.cat((h_f, h_b), dim=1)
                h_combined = self.compressConv_list[i](h_combined)
                output_inner.append(h_combined)

                c_combined = torch.cat((c_f, c_b), dim=1)
                c_combined = self.compressConv_list[i](c_combined)
            
            cur_layer_input = torch.stack(output_inner, dim=1)
            
            layer_output_list.append(cur_layer_input)
            last_state_list.append((h_combined, c_combined))

            # cur_layer_output_forward = torch.stack(forward_states, dim=1)
            # cur_layer_output_backward = torch.stack(backward_states, dim=1)
            # cur_layer_input_forward = cur_layer_output_forward
            # cur_layer_input_backward = cur_layer_output_backward

            # output_layer = torch.cat((cur_layer_output_forward, cur_layer_output_backward), dim=2) # caoncate two direction outputs
            # layer_output_list.append(output_layer)

            # last_state_h = torch.cat((h_f, h_b), dim=2)
            # last_state_c = torch.cat((c_f, c_b), dim=2)
            # last_state_list.append([last_state_h, last_state_c])

        out = torch.stack(layer_output_list, dim=1)

        out = out[:, -1]
        last_state_list = last_state_list[-1]

        if not self.return_sequence:
            out = out[:, -1]
            last_state_list = last_state_list[-1]

        return out, last_state_list


class LSTMSegNet(nn.Module):
    def __init__(
        self,
        lstm_backbone,
        input_dim,
        output_dim, 
        hidden_dim, 
        kernel_size, 
        num_layers, 
        conv_type,
        return_sequence=True,
        *args, **kwargs
    ):
        super(LSTMSegNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        self.output_dim = output_dim
        self.return_sequence = return_sequence
        
        if lstm_backbone == 'ConvLSTM':
            self.net = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, conv_type, return_sequence)
        elif lstm_backbone == 'BiConvLSTM':
            self.net = BiConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, conv_type, return_sequence)
        elif lstm_backbone == 'DenseBiLSTM':
            self.net = DenseBiConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, conv_type, return_sequence)
        else:
            raise NotImplementedError()
        
        self.conv1x1 = nn.Conv3d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1)

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, x.size(-3), x.size(-2), x.size(-1)).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, x.size(-3), x.size(-2), x.size(-1)).requires_grad_().to(x.device)

        out = self.net(x, (h0, c0))[0]
        
        if self.return_sequence:
            res = torch.zeros(out.size(0), out.size(1), self.output_dim, out.size(-3), out.size(-2), out.size(-1)).to(x.device)
            for i in range(out.size(1)):
                res[:, i, ...] = F.softmax(self.conv1x1(out[:, i, ...]), dim=1)
        else:
            res = F.softmax(self.conv1x1(out), dim=1)

        return res



if __name__ == '__main__':

    # input 
    x1 = torch.randn([2, 4, 20, 20, 20]) #(batch, modalities, d, h, w)
    x2 = torch.rand_like(x1)
    x3 = torch.rand_like(x1)

    x_fwd = torch.stack([x1, x2, x3], dim=1)
    print('input shape: ', x_fwd.shape) # (batch, timestep, modalities, d, h, w)

    net = LSTMSegNet(lstm_backbone='ConvLSTM', input_dim=4, output_dim=5, hidden_dim=8, kernel_size=3, num_layers=2, conv_type='cba', return_sequence=False)

    out = net(x_fwd)
    print(out.shape)
    # network initilization
    # convLSTM = ConvLSTM(input_dim=4, hidden_dim_list=[8, 8], kernel_size_list=[3, 3], num_layers=2, conv_type='plain')
    # biconvLSTM = BiConvLSTM(input_dim=4, hidden_dim_list=[8, 8], kernel_size_list=[3, 3], num_layers=2, conv_type='plain')
    # densebiConvLSTM = DenseBiConvLSTM(input_dim=4, hidden_dim_list=[8,8], kernel_size_list=[3,3], num_layers=2, conv_type='plain')

    # out = convLSTM(x_fwd)[0][-1]
    # print('out of convLSTM shape: ', out.shape)

    # out = biconvLSTM(x_fwd)[0][-1]
    # print('out of BiConvLSTM shape: ', out.shape)

    # out = densebiConvLSTM(x_fwd)[0][-1]
    # print('out of densebiconvLSTM shape: ', out.shape)

    # print(out[0][-1].view(2,3,2,8,20,20,20).shape) # split the bidirection

    # tp = torch.split(out, 1, dim=1)
    # print(len(tp))
    # print(tp[0].squeeze().shape) # split into each time point
    


        
    