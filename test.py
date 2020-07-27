import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
import torch 
import os
from tqdm import tqdm 
import numpy as np
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import argparse
import time
import seaborn as sns
from metrics import dice_coe, dice_coe_infernce, dice, Hausdorf, AverageSurfaceDist
from data_prepara import data_split, data_construction, time_parser, distributed_is_initialized
from UNet import init_U_Net
from ResUNet import ResUNet
from DilationResUNet import DResUNet
from DirectConnectUNet import U_Net_direct_concat
from BackLSTM import BackLSTM
from CenterLSTM import CenterLSTM, BiCenterLSTM
from DirectCenterLSTM import DirectCenterLSTM, BiDirectCenterLSTM
from ResCenterLSTM import ResCenterLSTM, BiResCenterLSTM
from ShortcutLSTM import ShortcutLSTM
from utils import load_config, load_ckp, image_rebuild, inference_output, crop_index_gen, image_crop, logfile, box_plot, print_size_of_model
from evaluation import volumn_ratio, plot_volumn_dev
import json

def Longitudinal_predict(args):
    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    conv_type = config['PARAMETERS']['lstm_convtype']
    connect = config['PARAMETERS']['connect']    
    root_path = config['PATH']['model_root']
    best_dir = config['PATH']['save_best_model']
    best_path = os.path.join(root_path, best_dir)

    model_name = args.model_name
    crop_size = args.crop_size
    overlap_size = args.overlap_size
    base_channels = args.base_channels
    lstm_backbone = args.lstmbase
    unet_backbone = args.unetbase
    layer_num = args.layer_num
    return_sequence = args.return_sequence
    nb_shortcut = args.nb_shortcut
    is_pretrain = args.is_pretrain
    inference_step = 3

    if model_name.startswith('Back'):
        net = BackLSTM(input_dim=input_modalites, hidden_dim=base_channels, output_dim=output_channels, kernel_size=3, num_layers=layer_num, conv_type=conv_type, lstm_backbone=lstm_backbone, unet_module=unet_backbone, base_channel=base_channels, return_sequence=return_sequence, is_pretrain=is_pretrain)
    elif model_name.startswith('CenterLSTM'):
        net = CenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channels, num_layers=layer_num, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
    elif model_name.startswith('CenterDenseBiLSTM') or model_name.startswith('CenterNormalBiLSTM'):
        net = BiCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channels, num_layers=layer_num, connect=connect, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
    elif model_name.startswith('BiDirectCenterNormal') or model_name.startswith('BiDirectCenterDense'):
        net = BiDirectCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channels, num_layers=layer_num, connect=connect, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
    elif model_name.startswith('BiResCenterNormal') or model_name.startswith('BiResCenterDense'):
        net = BiResCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channels, num_layers=layer_num, connect=connect, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
    elif model_name.startswith('Shortcut'):
        net = ShortcutLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channels, num_layers=layer_num, num_connects=nb_shortcut, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
    else:
        raise NotImplementedError()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
    if distributed_is_initialized():
        net = nn.DataParallel.DistributedDataParallel(net)
    else:
        net = nn.DataParallel(net)
    net.to(device)

    ckp_path = os.path.join(best_path, model_name +'_best_model.pth.tar')
    checkpoint = torch.load(ckp_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    print('{} size is: '.format(model_name))
    print_size_of_model(net)

    # predict
    data_class = data_split()
    train, val, test = data_construction(data_class)
    test_dict = time_parser(test)
    patient_id = [key for key in test_dict.keys()]
    # patient_id = ['EGD-0505']
    modalities = ['flair', 't1', 't1gd', 't2']

    Dice = {}
    CSF_Dice = []
    GM_Dice = []
    WM_Dice = []
    TM_Dice = []

    HD95 = {}
    CSF_HD = []
    GM_HD = []
    WM_HD = []
    TM_HD = []

    ASD = {}
    CSF_ASD = []
    GM_ASD = []
    WM_ASD = []
    TM_ASD = []

    for i in range(len(patient_id)):

        predicted_masks = {}

        time_dict = test_dict[patient_id[i]]
        time_dict = sorted(time_dict.items(), key=lambda item:item[0])
        
        tot_timesteps = len(time_dict)
        fold = int(np.ceil(tot_timesteps / inference_step))

        patient_inference_folder = os.path.join('inference_result', patient_id[i])
        if not os.path.exists(patient_inference_folder):
            os.makedirs(patient_inference_folder)

        for j in range(fold):
            selected_time_points = time_dict[j * inference_step : (j+1) * inference_step]
            image_stack = []
            seg = []
            time_list = []

            # predict per group
            for time_point in selected_time_points:
                time_list.append(time_point[0])
                brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(time_point[1]['brainmask']))
                images = [sitk.GetArrayFromImage(sitk.ReadImage(time_point[1][modality]))*brain_mask for modality in modalities]
                images = np.stack(images)
                image_stack.append(images)
                mask = sitk.ReadImage(time_point[1]['combined_fast'])
                seg.append(mask)
            image_shape = images.shape[-3:]

            inferenced_mask = predict(trained_net=net, image=np.stack(image_stack), image_shape=image_shape, crop_size=crop_size, overlap_size=overlap_size, model_type='RNN')
                
                
            for k in range(inferenced_mask.shape[0]):
                Dice_score, HD95_score, ASD_score = plot_save(image_stack[k][0], inferenced_mask[k], segmentation=seg[k], inference_folder=patient_inference_folder, model_name=args.model_name, inference_name=args.model_name + '_' + patient_id[i] + '_' + time_list[k], save_mask=True)

                predicted_masks[time_list[k]] = inferenced_mask[k]

                CSF_Dice.append(Dice_score['csf'])
                GM_Dice.append(Dice_score['gm'])
                WM_Dice.append(Dice_score['wm'])
                TM_Dice.append(Dice_score['tm'])

                CSF_HD.append(HD95_score['csf'])
                GM_HD.append(HD95_score['gm'])
                WM_HD.append(HD95_score['wm'])
                TM_HD.append(HD95_score['tm'])

                CSF_ASD.append(ASD_score['csf'])
                GM_ASD.append(ASD_score['gm'])
                WM_ASD.append(ASD_score['wm'])
                TM_ASD.append(ASD_score['tm'])

        plot_volumn_dev(test_dict, patient_id[i], model_name=model_name, predicted_labels=predicted_masks)
    
    Dice['csf'] = CSF_Dice
    Dice['gm'] = GM_Dice 
    Dice['wm'] = WM_Dice
    Dice['tm'] = TM_Dice

    HD95['csf'] = CSF_HD
    HD95['gm'] = GM_HD
    HD95['wm'] = WM_HD
    HD95['tm'] = TM_HD

    ASD['csf'] = CSF_ASD
    ASD['gm'] = GM_ASD
    ASD['wm'] = WM_ASD
    ASD['tm'] = TM_ASD

    dice_dir = os.path.join('inference_result', 'dice_' + args.model_name)
    HD_dir = os.path.join('inference_result', 'HD95_' + args.model_name)
    ASD_dir = os.path.join('inference_result', 'ASD_' + args.model_name)
    if not os.path.exists(dice_dir):
        os.mkdir(dice_dir)
    if not os.path.exists(HD_dir):
        os.mkdir(HD_dir)
    if not os.path.exists(ASD_dir):
        os.mkdir(ASD_dir)
    dice_file = os.path.join(dice_dir, 'dice.json')
    HD_file = os.path.join(HD_dir, 'HD95.json')
    ASD_file = os.path.join(ASD_dir, 'ASD.json')

    with open(dice_file, 'w') as f:
        json.dump(Dice, f)
    with open(HD_file, 'w') as f:
        json.dump(HD95, f)
    with open(ASD_file, 'w') as f:
        json.dump(ASD, f)

    box_plot(Dice, dice_dir, metric='Dice')
    box_plot(HD95, HD_dir, metric='HD')
    box_plot(ASD, ASD_dir, metric='ASD')



def predict(trained_net, 
            image,
            image_shape,
            crop_size, 
            overlap_size, 
            model_type='CNN'):

    ''' 
    used for inferencing image segmentation with trained model
    image shape: (C, D, W, H)
    '''

    crop_info = crop_index_gen(image_shape=image_shape, crop_size=crop_size, overlap_size=overlap_size)
    image_patches = image_crop(image, crop_info, ToTensor=True)
    if model_type == 'CNN':
        patch_num, channel, z, y, x = image_patches.cpu().numpy().shape
        CNN_cropped_image_list = np.zeros((patch_num, 5, z, y, x))
    else:
        patch_num, step, channel, z, y, x = image_patches.cpu().numpy().shape
        RNN_cropped_image_list = np.zeros((patch_num, step, 5, z, y, x))

    with tqdm(total=patch_num, desc='inference test image', unit='patch') as pbar:
        with torch.no_grad():
            trained_net.eval()
            for i, image in enumerate(image_patches):
                image = image.unsqueeze(dim=0)

                # deal with different type of model
                if model_type == 'RNN':
                    preds = trained_net(image) 
                    RNN_cropped_image_list[i, ...] = preds.squeeze(0).detach().cpu().numpy()
                else:
                    preds = trained_net(image)  
                    CNN_cropped_image_list[i, ...] = preds.squeeze(0).detach().cpu().numpy()

                pbar.update(1)

    crop_index = crop_info['index_array']
    if model_type == 'RNN':
        inferenced_mask = []
        for i in range(step):
            inferenced = image_rebuild(crop_index, RNN_cropped_image_list[:, i])
            inferenced_mask.append(inferenced)
        inferenced_mask = np.stack(inferenced_mask)
    else:
        inferenced_mask = image_rebuild(crop_index, CNN_cropped_image_list)

    return inferenced_mask


def plot_save(input_image, pred, segmentation, inference_folder=None, model_name=None, inference_name=None, save_mask=True):

    # plot inferenced heatmap and ground truth heatmap
    if inference_folder and not os.path.exists(inference_folder):
        os.mkdir(inference_folder)

    plt.figure(figsize=(20, 10))
    target = sitk.GetArrayFromImage(segmentation)
    ground_truth = target[target.shape[-3]//2]
    predicted = pred[target.shape[-3]//2]
    input_image = input_image[target.shape[-3]//2]
    image_list = [ground_truth, predicted]

    subtitles = ['ground truth', 'predicted']
    plt.subplots_adjust(wspace=0.3)
    ax = plt.subplot(1,3,1)
    ax.axis('off')
    ax.imshow(input_image)
    ax.set_title('input image')
    for i in range(2,4):
        ax = plt.subplot(1,3,i)
        ax.set_title(subtitles[i-2], fontsize=15)
        sns.heatmap(image_list[i-2], vmin=0, vmax=4, xticklabels=False, yticklabels=False, square=True, cmap='coolwarm', cbar=False)
        ax.axis('off')

    save_dir = os.path.join(inference_folder, model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not inference_folder or not inference_name:
        plt.savefig('result.png')
    else:
        plt.savefig(os.path.join(save_dir, inference_name +'.png'))

    mask = sitk.GetImageFromArray(pred.astype(np.int16))
    if save_mask:
        mask.CopyInformation(segmentation)
        sitk.WriteImage(mask, os.path.join(save_dir, inference_name + '.nii.gz'))

    # calculate dice score
    dice_score = dice(pred, target)
    HD95_score = Hausdorf(pred, target)
    ASD_score = AverageSurfaceDist(pred, target)

    return dice_score, HD95_score, ASD_score

def predict_CNN(args):

    config_file = 'config.yaml'
    config = load_config(config_file)
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])   
    root_path = config['PATH']['model_root']
    best_dir = config['PATH']['save_best_model']
    best_path = os.path.join(root_path, best_dir)

    model_type = args.net
    model_name = args.model_name
    crop_size = args.crop_size
    overlap_size = args.overlap_size
    base_channels = args.base_channels

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load best trained model
    
    if model_name.startswith('UNet'):
        net = init_U_Net(input_modalites, output_channels, base_channels)
    elif model_name.startswith('direct'):
        net = U_Net_direct_concat(input_modalites, output_channels, base_channels)
    elif model_name.startswith('ResUNet'):
        net = ResUNet(input_modalites, output_channels, base_channels)
    elif model_name.startswith('DResUNet'):
        net = DResUNet(input_modalites, output_channels, base_channels)
    else:
        raise NotImplementedError()


    if distributed_is_initialized():
        net = nn.DataParallel.DistributedDataParallel(net)
    else:
        net = nn.DataParallel(net)
    net.to(device)

    ckp_path = os.path.join(best_path, model_name +'_best_model.pth.tar')
    checkpoint = torch.load(ckp_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    print_size_of_model(net)

    # predict
    data_class = data_split()
    train, val, test = data_construction(data_class)
    test_dict = time_parser(test)
    patient_id = [key for key in test_dict.keys()]
    modalities = ['flair', 't1', 't1gd', 't2']

    Dice = {}
    CSF_Dice = []
    GM_Dice = []
    WM_Dice = []
    TM_Dice = []

    HD95 = {}
    CSF_HD = []
    GM_HD = []
    WM_HD = []
    TM_HD = []

    ASD = {}
    CSF_ASD = []
    GM_ASD = []
    WM_ASD = []
    TM_ASD = []

    for i in range(len(patient_id)):

        time_dict = test_dict[patient_id[i]]
        time_dict = sorted(time_dict.items(), key=lambda item:item[0])
        predicted_masks = {}

        patient_inference_folder = os.path.join('inference_result', patient_id[i])
        if not os.path.exists(patient_inference_folder):
            os.makedirs(patient_inference_folder)

        for time_point in time_dict:
            
            print('Predicting patient {} at time {}'.format(patient_id[i], time_point[0]))

            brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(time_point[1]['brainmask']))
            images = [sitk.GetArrayFromImage(sitk.ReadImage(time_point[1][modality]))*brain_mask for modality in modalities]
            images = np.stack(images)

            mask = sitk.ReadImage(time_point[1]['combined_fast'])
            image_shape = images.shape[-3:]

            inferenced_mask = predict(trained_net=net, image=images, image_shape=image_shape, crop_size=crop_size, overlap_size=overlap_size, model_type=model_type)
            
            Dice_score, HD95_score, ASD_score = plot_save(images[0], inferenced_mask, segmentation=mask, inference_folder=patient_inference_folder, model_name=args.model_name, inference_name=args.model_name + '_' + patient_id[i] + '_' + time_point[0], save_mask=True)

            CSF_Dice.append(Dice_score['csf'])
            GM_Dice.append(Dice_score['gm'])
            WM_Dice.append(Dice_score['wm'])
            TM_Dice.append(Dice_score['tm'])

            CSF_HD.append(HD95_score['csf'])
            GM_HD.append(HD95_score['gm'])
            WM_HD.append(HD95_score['wm'])
            TM_HD.append(HD95_score['tm'])

            CSF_ASD.append(ASD_score['csf'])
            GM_ASD.append(ASD_score['gm'])
            WM_ASD.append(ASD_score['wm'])
            TM_ASD.append(ASD_score['tm'])

            predicted_masks[time_point[0]] = inferenced_mask
        
        plot_volumn_dev(test_dict, patient_id[i], model_name=model_name, predicted_labels=predicted_masks)
        break
        

    Dice['csf'] = CSF_Dice
    Dice['gm'] = GM_Dice 
    Dice['wm'] = WM_Dice
    Dice['tm'] = TM_Dice

    HD95['csf'] = CSF_HD
    HD95['gm'] = GM_HD
    HD95['wm'] = WM_HD
    HD95['tm'] = TM_HD

    ASD['csf'] = CSF_ASD
    ASD['gm'] = GM_ASD
    ASD['wm'] = WM_ASD
    ASD['tm'] = TM_ASD

    dice_dir = os.path.join('inference_result', 'dice_' + args.model_name)
    HD_dir = os.path.join('inference_result', 'HD95_' + args.model_name)
    ASD_dir = os.path.join('inference_result', 'ASD_' + args.model_name)
    if not os.path.exists(dice_dir):
        os.mkdir(dice_dir)
    if not os.path.exists(HD_dir):
        os.mkdir(HD_dir)
    if not os.path.exists(ASD_dir):
        os.mkdir(ASD_dir)
    dice_file = os.path.join(dice_dir, 'dice.json')
    HD_file = os.path.join(HD_dir, 'HD95.json')
    ASD_file = os.path.join(ASD_dir, 'ASD.json')

    with open(dice_file, 'w') as f:
        json.dump(Dice, f)
    with open(HD_file, 'w') as f:
        json.dump(HD95, f)
    with open(ASD_file, 'w') as f:
        json.dump(ASD, f)

    box_plot(Dice, dice_dir, metric='Dice')
    box_plot(HD95, HD_dir, metric='Hausdorf')
    box_plot(ASD, ASD_dir, metric='ASD')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--model_name', default='baseline_withmask_130', type=str)
    parser.add_argument('-p', '--crop_size', default=-1, type=int)
    parser.add_argument('-overlap', '--overlap_size', default=0, type=int)
    parser.add_argument('-b', '--base_channels', default=4, type=int, help='U-Net base channel number')
    parser.add_argument('-lstmbase', '--lstmbase', default='ConvLSTM', type=str, help='RNN model type, "BiConvLSTM", "ConvLSTM", "DenseBiLSTM"')
    parser.add_argument('-unetbase', '--unetbase', default='UNet', type=str, help='backbone type, UNet, direct-UNet, ResUNet, DResUNet')
    parser.add_argument('-layer', '--layer_num', default=1, type=int, help='stack numebr of RNN')
    parser.add_argument('-return_sequence', '--return_sequence', default=True, type=str, help='LSTM return the whole sequence or only last time step')
    parser.add_argument('-nb_shortcut', '--nb_shortcut', default=1, type=int, help='The number of shortcuts to be connected')
    parser.add_argument('-net', '--net', default='CNN', type=str, help='inference of longitudinal inference')
    parser.add_argument('-pretrain', '--is_pretrain', default=True, type=str, help='use pretrained backbone or not')
    args = parser.parse_args()

    if args.net == 'CNN':
        predict_CNN(args)
    else:
        Longitudinal_predict(args)

if __name__ == '__main__':

    main()
    
    
    