import torch
import torch.nn as nn 
import torchvision 
import torch.nn.functional as F 

class Simplified_Inception_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, softmax=True):
        super(Simplified_Inception_UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.softmax = softmax

        self.encoder_block_1 = Inception_Res_Module(self.in_channels, self.base_channels * 4)
        self.encoder_down_1 = Simple_Downsample(2)
        self.encoder_block_2 = Inception_Res_Module(self.base_channels * 4, self.base_channels * 8)
        self.encoder_down_2 = Simple_Downsample(2)
        self.encoder_block_3 = Inception_Res_Module(self.base_channels * 8, self.base_channels * 16)
        self.encoder_down_3 = Simple_Downsample(2)

        self.encoder_bridge_1 = BasicConv3d(self.base_channels * 16, self.base_channels * 32, kernel_size=3, padding=1)
        self.encoder_bridge_2 = BasicConv3d(self.base_channels * 32, self.base_channels * 32, kernel_size=3, padding=1)

        self.decoder_up_3 = Simple_Upsample(self.base_channels * 32, self.base_channels * 32)
        self.decoder_block_3 = Inception_Res_Module(self.base_channels * (32 + 16) + 4, self.base_channels * 16)
        self.decoder_up_2 = Simple_Upsample(self.base_channels * 16, self.base_channels * 16)
        self.decoder_block_2 = Inception_Res_Module(self.base_channels * (16 + 8) + 4, self.base_channels * 8)
        self.decoder_up_1 = Simple_Upsample(self.base_channels * 8, self.base_channels * 8)
        self.decoder_block_1 = Inception_Res_Module(self.base_channels * (8 + 4) + 4, self.base_channels * 4)

        self.out = nn.Conv3d(self.base_channels * 4, self.out_channels, kernel_size=1)

        self.down_sample_input = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x_layer1 = x
        x_layer2 = self.down_sample_input(x_layer1)
        x_layer3 = self.down_sample_input(x_layer2)

        encoder_block_1 = self.encoder_block_1(x)
        encoder_down_1 = self.encoder_down_1(encoder_block_1)
        encoder_block_2 = self.encoder_block_2(encoder_down_1)
        encoder_down_2 = self.encoder_down_2(encoder_block_2)
        encoder_block_3 = self.encoder_block_3(encoder_down_2)
        encoder_down_3 = self.encoder_down_3(encoder_block_3)

        encoder_bridge_1 = self.encoder_bridge_1(encoder_down_3)
        encoder_bridge_2 = self.encoder_bridge_2(encoder_bridge_1)
        # encoder_bridge_2 = encoder_bridge_2 + encoder_bridge_1

        decoder_up_3 = self.decoder_up_3(encoder_bridge_2)
        encoder_block_3 = pad(decoder_up_3, encoder_block_3)
        cat_layer_3 = torch.cat([decoder_up_3, encoder_block_3, x_layer3], 1)
        decoder_block_3 = self.decoder_block_3(cat_layer_3)

        decoder_up_2 = self.decoder_up_2(decoder_block_3)
        decoder_up_2 = pad(encoder_block_2, decoder_up_2)
        x_layer2 = pad(encoder_block_2, x_layer2)
        cat_layer_2 = torch.cat([decoder_up_2, encoder_block_2, x_layer2], 1)
        decoder_block_2 = self.decoder_block_2(cat_layer_2)
        decoder_up_1 = self.decoder_up_1(decoder_block_2)

        encoder_block_1 = pad(decoder_up_1, encoder_block_1)
        cat_layer_1 = torch.cat([decoder_up_1, encoder_block_1, x_layer1], 1)
        decoder_block_1 = self.decoder_block_1(cat_layer_1)

        out = self.out(decoder_block_1)

        if self.softmax:
            out = F.softmax(out, dim=1)

        return out


class Inception_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, softmax=True):
        super(Inception_UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.softmax = softmax

        self.encoder_block_1 = Inception_Res_Module(self.in_channels, self.base_channels * 4)
        self.encoder_down_1 = Down_Sampling_Module(self.base_channels * 4)
        self.encoder_block_2 = Inception_Res_Module(self.base_channels * 4, self.base_channels * 8)
        self.encoder_down_2 = Down_Sampling_Module(self.base_channels * 8)
        self.encoder_block_3 = Inception_Res_Module(self.base_channels * 8, self.base_channels * 16)

        self.encoder_down_3 = Down_Sampling_Module(self.base_channels * 16)
        self.encoder_bridge_1 = Inception_Dense_Module(self.base_channels * 16, self.base_channels * 32)
        self.encoder_bridge_2 = Inception_Dense_Module(self.base_channels * 32, self.base_channels * 32)

        self.decoder_up_3 = Up_Sampling_Module(self.base_channels * 32)
        self.decoder_block_3 = Inception_Res_Module(self.base_channels * (32 + 16) + 4, self.base_channels * 16)
        self.decoder_up_2 = Up_Sampling_Module(self.base_channels * 16)
        self.decoder_block_2 = Inception_Res_Module(self.base_channels * (16 + 8) + 4, self.base_channels * 8)
        self.decoder_up_1 = Up_Sampling_Module(self.base_channels * 8)
        self.decoder_block_1 = Inception_Res_Module(self.base_channels * (8 + 4) + 4, self.base_channels * 4)

        self.out = nn.Conv3d(self.base_channels * 4, self.out_channels, kernel_size=1)

        self.down_sample_input = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x_layer1 = x
        x_layer2 = self.down_sample_input(x_layer1)
        x_layer3 = self.down_sample_input(x_layer2)

        encoder_block_1 = self.encoder_block_1(x)
        encoder_down_1 = self.encoder_down_1(encoder_block_1)
        encoder_block_2 = self.encoder_block_2(encoder_down_1)
        encoder_down_2 = self.encoder_down_2(encoder_block_2)
        encoder_block_3 = self.encoder_block_3(encoder_down_2)
        encoder_down_3 = self.encoder_down_3(encoder_block_3)

        encoder_bridge_1 = self.encoder_bridge_1(encoder_down_3)
        encoder_bridge_2 = self.encoder_bridge_2(encoder_bridge_1)
        encoder_bridge_2 = F.relu(encoder_bridge_2 + encoder_bridge_1, inplace=True)

        decoder_up_3 = self.decoder_up_3(encoder_bridge_2)
        cat_layer_3 = torch.cat([decoder_up_3, encoder_block_3, x_layer3], 1)
        decoder_block_3 = self.decoder_block_3(cat_layer_3)
        decoder_up_2 = self.decoder_up_2(decoder_block_3)
        cat_layer_2 = torch.cat([decoder_up_2, encoder_block_2, x_layer2], 1)
        decoder_block_2 = self.decoder_block_2(cat_layer_2)
        decoder_up_1 = self.decoder_up_1(decoder_block_2)
        cat_layer_1 = torch.cat([decoder_up_1, encoder_block_1, x_layer1], 1)
        decoder_block_1 = self.decoder_block_1(cat_layer_1)

        out = self.out(decoder_block_1)

        if self.softmax:
            out = F.softmax(out, dim=1)

        return out


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)

class TransposeConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(TransposeConv3d, self).__init__()
        self.transconv = nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)

class Inception_Res_Module(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Inception_Res_Module, self).__init__()

        self.branch1x1_1 = BasicConv3d(in_channels, out_channels, kernel_size=1)

        self.branch3x3_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            BasicConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch3x3_2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            BasicConv3d(out_channels, out_channels, kernel_size=3, padding=1),
            BasicConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.conv1x1_cat = BasicConv3d(out_channels * 3, out_channels, kernel_size=1)
        self.conv1x1_res = BasicConv3d(in_channels, out_channels, kernel_size=1)
        

    def forward(self, x):

        branch1x1_1 = self.branch1x1_1(x)
        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2 = self.branch3x3_2(x)
        cat = torch.cat([branch1x1_1, branch3x3_1, branch3x3_2], 1)
        conv1x1_cat = self.conv1x1_cat(cat)
        conv1x1_res = self.conv1x1_res(x)
       
        return F.relu(conv1x1_cat + conv1x1_res, inplace=True)

class Inception_Dense_Module(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Inception_Dense_Module, self).__init__()

        self.branch1x1_1 = BasicConv3d(in_channels, out_channels, kernel_size=1)

        self.branch3x3_1 = BasicConv3d(in_channels, out_channels, kernel_size=1)
        self.branch3x3_2a = BasicConv3d(out_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.branch3x3_2b = BasicConv3d(out_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        self.branch3x3stack_1 = BasicConv3d(in_channels, 2 * out_channels, kernel_size=1)
        self.branch3x3stack_2 = BasicConv3d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv3d(out_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.branch3x3stack_3b = BasicConv3d(out_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        self.conv1x1_cat = BasicConv3d(5 * out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        cat = torch.cat([branch1x1, branch3x3, branch3x3stack], 1)
        conv1x1_cat = self.conv1x1_cat(cat)

        return conv1x1_cat

class Down_Sampling_Module(nn.Module):
    def __init__(self, in_channels):
        super(Down_Sampling_Module, self).__init__()

        self.branch_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.branch3x3_1 = nn.Sequential(
            BasicConv3d(in_channels, in_channels, kernel_size=1),
            BasicConv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        )
        self.branch3x3_2 = nn.Sequential(
            BasicConv3d(in_channels, in_channels, kernel_size=1),
            BasicConv3d(in_channels, in_channels, kernel_size=3, padding=1),
            BasicConv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conv1x1_cat = BasicConv3d(3 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        branch_pool = self.branch_pool(x)
        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2 = self.branch3x3_2(x)
        cat = torch.cat([branch_pool, branch3x3_1, branch3x3_2], 1)
        conv1x1_cat = self.conv1x1_cat(cat)

        return conv1x1_cat

class Up_Sampling_Module(nn.Module):
    def __init__(self, in_channels):
        super(Up_Sampling_Module, self).__init__()

        self.branch_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.branch3x3_1 = nn.Sequential(
            TransposeConv3d(in_channels, in_channels, kernel_size=1),
            TransposeConv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        )
        self.branch3x3_2 = nn.Sequential(
            TransposeConv3d(in_channels, in_channels, kernel_size=1),
            TransposeConv3d(in_channels, in_channels, kernel_size=3, padding=1),
            TransposeConv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        )
        self.conv_cat = TransposeConv3d(3 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        branch_upsample = self.branch_upsample(x)
        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2 = self.branch3x3_2(x)
        branch3x3_1 = pad(branch_upsample, branch3x3_1)
        branch3x3_2 = pad(branch_upsample, branch3x3_2)

        cat = torch.cat([branch_upsample, branch3x3_1, branch3x3_2], dim=1)
        conv_cat = self.conv_cat(cat)

        return conv_cat

class Simple_Downsample(nn.Module):
    def __init__(self, kernel_size=2):
        super(Simple_Downsample, self).__init__()

        self.downsample = nn.MaxPool3d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)

class Simple_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Simple_Upsample, self).__init__()

        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        return self.upsample(x)


def pad(encoder, decoder, method='pad'):
    
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
    base_channel = int(config['PARAMETERS']['base_channels'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def count_params(model):
        
        ''' print number of trainable parameters and its size of the model'''

        num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Model {} : params number {}, params size: {:4f}M'.format(model._get_name(), num_of_param, num_of_param*4/1000/1000))
    
    net = Inception_UNet(input_modalites, output_channels, 4)
    # net = Simplified_Inception_UNet(input_modalites, output_channels, 8)
    net.to(device)
    count_params(net)

    input = torch.randn(1, 4, 64, 64, 64).to(device)
    # print(net)
    y = net(input)
    print(y.shape)


