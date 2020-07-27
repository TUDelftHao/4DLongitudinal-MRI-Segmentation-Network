import torch 
import torch.nn as nn
import torch.optim as optim
from RNN import ConvLSTM, BiConvLSTM, DenseBiConvLSTM, LSTMSegNet
from BackLSTM import BackLSTM
from CenterLSTM import CenterLSTM, BiCenterLSTM
from DirectCenterLSTM import DirectCenterLSTM, BiDirectCenterLSTM
from ResCenterLSTM import ResCenterLSTM, BiResCenterLSTM
from ShortcutLSTM import ShortcutLSTM
from data_prepara import data_split, data_construction, time_parser, data_loader, distributed_is_initialized
from utils import load_config, save_ckp, load_ckp, logfile, loss_plot, heatmap_plot
from metrics import dice_coe, dice, binary_dice
from loss import DiceLoss, GneralizedDiceLoss, WeightedCrossEntropyLoss, DiceLossLSTM
import json
import numpy as np 
import shutil
import warnings
import time
import argparse
from tqdm import tqdm 
import os
from apex import amp 
from tensorboardX import SummaryWriter
from torchsummary import summary


def train(args):

    torch.cuda.manual_seed(1)
    torch.manual_seed(1)

    # user defined parameters
    model_name = args.model_name  
    model_type = args.model_type
    lstm_backbone = args.lstmbase
    unet_backbone = args.unetbase
    layer_num = args.layer_num
    nb_shortcut = args.nb_shortcut
    loss_fn = args.loss_fn
    world_size = args.world_size
    rank = args.rank
    base_channel = args.base_channels
    crop_size = args.crop_size
    ignore_idx = args.ignore_idx
    return_sequence = args.return_sequence
    variant = args.LSTM_variant
    epochs = args.epoch
    is_pretrain = args.is_pretrain

    # system setup parameters
    config_file = 'config.yaml'
    config = load_config(config_file)
    labels = config['PARAMETERS']['labels']
    root_path = config['PATH']['model_root']
    model_dir = config['PATH']['save_ckp']
    best_dir = config['PATH']['save_best_model']

    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    batch_size = int(config['PARAMETERS']['batch_size'])
    is_best = bool(config['PARAMETERS']['is_best'])
    is_resume = bool(config['PARAMETERS']['resume'])
    patience = int(config['PARAMETERS']['patience'])
    time_step = int(config['PARAMETERS']['time_step'])
    num_workers = int(config['PARAMETERS']['num_workers'])
    early_stop_patience = int(config['PARAMETERS']['early_stop_patience'])
    lr = int(config['PARAMETERS']['lr'])
    optimizer = config['PARAMETERS']['optimizer']
    connect = config['PARAMETERS']['connect']
    conv_type = config['PARAMETERS']['lstm_convtype']


    # build up dirs
    model_path = os.path.join(root_path, model_dir)
    best_path = os.path.join(root_path, best_dir)
    intermidiate_data_save = os.path.join(root_path, 'train_newdata',model_name)
    train_info_file = os.path.join(intermidiate_data_save, '{}_train_info.json'.format(model_name))
    log_path = os.path.join(root_path, 'logfiles')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(best_path):
        os.mkdir(best_path)
    if not os.path.exists(intermidiate_data_save):
        os.makedirs(intermidiate_data_save)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_name = model_name + '_' + config['PATH']['log_file'] 
    logger = logfile(os.path.join(log_path, log_name))
    logger.info('labels {} are ignored'.format(ignore_idx))
    logger.info('Dataset is loading ...')
    writer = SummaryWriter('ProcessVisu/%s' %model_name)

    
    # load training set and validation set
    data_class = data_split()
    train, val, test = data_construction(data_class)
    train_dict = time_parser(train, time_patch=time_step)
    val_dict = time_parser(val, time_patch=time_step)


    # LSTM initilization

    if model_type == 'LSTM':
        net = LSTMSegNet(lstm_backbone=lstm_backbone, input_dim=input_modalites, output_dim=output_channels, hidden_dim=base_channel, kernel_size=3, num_layers=layer_num, conv_type=conv_type, return_sequence=return_sequence)
    elif model_type == 'UNet_LSTM':
        if variant == 'back':
            net = BackLSTM(input_dim=input_modalites, hidden_dim=base_channel, output_dim=output_channels, kernel_size=3, num_layers=layer_num, conv_type=conv_type, lstm_backbone=lstm_backbone, unet_module=unet_backbone, base_channel=base_channel, return_sequence=return_sequence, is_pretrain=is_pretrain)
            logger.info('the pretrained status of backbone is {}'.format(is_pretrain))
        elif variant == 'center':
            net = CenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
        elif variant == 'bicenter':
            net = BiCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, connect=connect, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
        elif variant == 'directcenter':
            net = DirectCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
        elif variant == 'bidirectcenter':
            net = BiDirectCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, connect=connect, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
        elif variant == 'rescenter':
            net = ResCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
        elif variant == 'birescenter':
            net = BiResCenterLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, connect=connect, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
        elif variant == 'shortcut':
            net = ShortcutLSTM(input_modalites=input_modalites, output_channels=output_channels, base_channel=base_channel, num_layers=layer_num, num_connects=nb_shortcut, conv_type=conv_type, return_sequence=return_sequence, is_pretrain=is_pretrain)
    else:
        raise NotImplementedError()

    # loss and optimizer setup
    if loss_fn == 'Dice':
        criterion = DiceLoss(labels=labels, ignore_idx=ignore_idx)
    elif loss_fn == 'GDice':
        criterion = GneralizedDiceLoss(labels=labels)
    elif loss_fn == 'WCE':
        criterion = WeightedCrossEntropyLoss(labels=labels)
    else:
        raise NotImplementedError()

    if optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # optimizer = optim.Adam(net.parameters())
    elif optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=patience)

    # device setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:38366', rank=rank, world_size=world_size)
    if distributed_is_initialized():
        print('distributed is initialized')
        net.to(device)
        net = nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
    else:
        print('data parallel')
        net = nn.DataParallel(net)
        net.to(device)


    min_loss = float('Inf')
    early_stop_count = 0
    global_step = 0
    start_epoch = 0
    start_loss = 0
    train_info = {'train_loss':[], 
                'val_loss':[],
                'label_0_acc':[],
                'label_1_acc':[],
                'label_2_acc':[],
                'label_3_acc':[],
                'label_4_acc':[]}

    if is_resume:
        try: 
            # open previous check points
            ckp_path = os.path.join(model_path, '{}_model_ckp.pth.tar'.format(model_name))
            net, optimizer, scheduler, start_epoch, min_loss, start_loss = load_ckp(ckp_path, net, optimizer, scheduler)

            # open previous training records
            with open(train_info_file) as f:
                train_info = json.load(f)
            

            logger.info('Training loss from last time is {}'.format(start_loss) + '\n' + 'Mininum training loss from last time is {}'.format(min_loss))
            logger.info('Training accuracies from last time are: label 0: {}, label 1: {}, label 2: {}, label 3: {}, label 4: {}'.format(train_info['label_0_acc'][-1], train_info['label_1_acc'][-1], train_info['label_2_acc'][-1], train_info['label_3_acc'][-1], train_info['label_4_acc'][-1]))

        except:
            logger.warning('No checkpoint available, strat training from scratch')

    for epoch in range(start_epoch, epochs):

        train_set = data_loader(train_dict, 
                            batch_size=batch_size, 
                            key='train',
                            num_works=num_workers,
                            time_step=time_step,
                            patch=crop_size,
                            model_type='RNN'
                            )
        n_train = len(train_set)

        val_set = data_loader(val_dict, 
                                batch_size=batch_size, 
                                key='val',
                                num_works=num_workers,
                                time_step=time_step,
                                patch=crop_size,
                                model_type='CNN'
                                )
        n_val = len(val_set)

        logger.info('Dataset loading finished!')
        
        nb_batches = np.ceil(n_train/batch_size)
        n_total = n_train + n_val
        logger.info('{} images will be used in total, {} for trainning and {} for validation'.format(n_total, n_train, n_val))


        train_loader = train_set.load()

        # setup to train mode
        net.train()
        running_loss = 0
        dice_score_label_0 = 0
        dice_score_label_1 = 0
        dice_score_label_2 = 0
        dice_score_label_3 = 0
        dice_score_label_4 = 0

        logger.info('Training epoch {} will begin'.format(epoch+1))
        
        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='patch') as pbar:

            for i, data in enumerate(train_loader, 0):
           
                # i : patient
                images, segs = data['image'].to(device), data['seg'].to(device)
                
                outputs = net(images)
                loss = criterion(outputs, segs)
                loss.backward()
                optimizer.step()

                # if i == 0:
                #     in_images = images.detach().cpu().numpy()[0]
                #     in_segs = segs.detach().cpu().numpy()[0]
                #     in_pred = outputs.detach().cpu().numpy()[0]
                #     heatmap_plot(image=in_images, mask=in_segs, pred=in_pred, name=model_name, epoch=epoch+1, is_train=True)

                running_loss += loss.detach().item()

                outputs = outputs.view(-1, outputs.shape[-4], outputs.shape[-3], outputs.shape[-2], outputs.shape[-1])
                segs = segs.view(-1, segs.shape[-3], segs.shape[-2], segs.shape[-1])
                _, preds = torch.max(outputs.data, 1)
                dice_score = dice(preds.data.cpu(), segs.data.cpu(), ignore_idx=None)

                dice_score_label_0 += dice_score['bg']
                dice_score_label_1 += dice_score['csf']
                dice_score_label_2 += dice_score['gm']
                dice_score_label_3 += dice_score['wm']
                dice_score_label_4 += dice_score['tm']

                # show progress bar
                pbar.set_postfix(**{'training loss': loss.detach().item(), 'Training accuracy': dice_score['avg']})
                pbar.update(images.shape[0])

                global_step += 1
                if global_step % nb_batches == 0:
                    net.eval()
                    val_loss, val_acc, val_info = validation(net, val_set, criterion, device, batch_size, ignore_idx=None, name=model_name, epoch=epoch+1)
                    net.train()

        
        train_info['train_loss'].append(running_loss / nb_batches)
        train_info['val_loss'].append(val_loss)
        train_info['label_0_acc'].append(dice_score_label_0 / nb_batches)
        train_info['label_1_acc'].append(dice_score_label_1 / nb_batches)
        train_info['label_2_acc'].append(dice_score_label_2 / nb_batches)
        train_info['label_3_acc'].append(dice_score_label_3 / nb_batches)
        train_info['label_4_acc'].append(dice_score_label_4 / nb_batches)

        # save bast trained model
        scheduler.step(running_loss / nb_batches)
        logger.info('Epoch: {}, LR: {}'.format(epoch+1, optimizer.param_groups[0]['lr']))

        if min_loss > running_loss / nb_batches:
            min_loss = running_loss / nb_batches
            is_best = True
            early_stop_count = 0
        else:
            is_best = False
            early_stop_count += 1

        state = {
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':running_loss / nb_batches,
            'min_loss': min_loss
        }
        verbose = save_ckp(state, is_best, early_stop_count=early_stop_count, early_stop_patience=early_stop_patience, save_model_dir=model_path, best_dir=best_path, name=model_name)

        # summarize the training results of this epoch 
        logger.info('The average training loss for this epoch is {}'.format(running_loss / nb_batches))
        logger.info('The best training loss till now is {}'.format(min_loss))
        logger.info('Validation dice loss: {}; Validation (avg) accuracy of the last timestep: {}'.format(val_loss, val_acc))
                    
        # save the training info every epoch
        logger.info('Writing the training info into file ...')
        val_info_file = os.path.join(intermidiate_data_save, '{}_val_info.json'.format(model_name))
        with open(train_info_file, 'w') as fp:
            json.dump(train_info, fp)
        with open(val_info_file, 'w') as fp:
            json.dump(val_info, fp)

        for name, layer in net.named_parameters():
            if layer.requires_grad:
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
        if verbose:
            logger.info('The validation loss has not improved for {} epochs, training will stop here.'.format(early_stop_patience))
            break
        
    loss_plot(train_info_file, name=model_name)
    logger.info('finish training!')
    
    return 

def validation(trained_net, val_set, criterion, device, batch_size, ignore_idx, name=None, epoch=None):
    n_val = len(val_set)
    val_loader = val_set.load()

    tot = 0 
    acc = 0
    dice_score_bg = 0
    dice_score_wm = 0
    dice_score_gm = 0
    dice_score_csf = 0
    dice_score_tm = 0

    val_info = {
                'bg':[],
                'wm':[],
                'gm':[],
                'csf':[],
                'tm':[]}

    with tqdm(total=n_val, desc='Validation round', unit='patch', leave=False) as pbar:
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                images, segs = sample['image'].to(device=device), sample['seg'].to(device=device)

                outputs = trained_net(images)
                val_loss = criterion(outputs, segs)

                if i == 0:
                    in_images = images.detach().cpu().numpy()[0]
                    in_segs = segs.detach().cpu().numpy()[0]
                    in_pred = outputs.detach().cpu().numpy()[0]
                    heatmap_plot(image=in_images, mask=in_segs, pred=in_pred, name=name, epoch=epoch, is_train=False)

                outputs = outputs.view(-1, outputs.shape[-4], outputs.shape[-3], outputs.shape[-2], outputs.shape[-1])
                segs = segs.view(-1, segs.shape[-3], segs.shape[-2], segs.shape[-1])
                
                _, preds = torch.max(outputs.data, 1)
                dice_score = dice(preds.data.cpu(), segs.data.cpu(), ignore_idx=ignore_idx)

                dice_score_bg += dice_score['bg']
                dice_score_wm += dice_score['wm']
                dice_score_gm += dice_score['gm']
                dice_score_csf += dice_score['csf']
                dice_score_tm += dice_score['tm']

                tot += val_loss.detach().item() 
                acc += dice_score['avg']

                pbar.set_postfix(**{'validation loss (images)': val_loss.detach().item(), 'val_acc_avg':dice_score['avg']})
                pbar.update(images.shape[0])

        val_info['bg'] = dice_score_bg / (np.ceil(n_val/batch_size))
        val_info['wm'] = dice_score_wm / (np.ceil(n_val/batch_size))
        val_info['gm'] = dice_score_gm / (np.ceil(n_val/batch_size)) 
        val_info['csf'] = dice_score_csf / (np.ceil(n_val/batch_size))
        val_info['tm'] = dice_score_tm / (np.ceil(n_val/batch_size))

    return tot/(np.ceil(n_val/batch_size)), acc/(np.ceil(n_val/batch_size)), val_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--model_name', default='bclstm', type=str, help='model name')
    parser.add_argument('-type', '--model_type', default='LSTM', type=str, help='model type to be implemented: LSTM, UNet_LSTM')
    parser.add_argument('-lstmbase', '--lstmbase', default='ConvLSTM', type=str, help='RNN model type, "BiConvLSTM", "ConvLSTM", "DenseBiLSTM"')
    parser.add_argument('-unetbase', '--unetbase', default='UNet', type=str, help='backbone type, UNet, direct-UNet, ResUNet, DResUNet')
    parser.add_argument('-layer', '--layer_num', default=1, type=int, help='stack numebr of RNN')
    parser.add_argument('-nb_shortcut', '--nb_shortcut', default=1, type=int, help='The number of shortcuts to be connected')
    parser.add_argument('-l', '--loss_fn', default='Dice', type=str, help='loss function: GDice, Dice, WCE')
    parser.add_argument('-s', '--world_size', default=1, type=int, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', default=0, type=int, help='Rank of the current process.')
    parser.add_argument('-p', '--crop_size', default=64, type=int, help='image crop patch size')
    parser.add_argument('-b', '--base_channels', default=4, type=int, help='U-Net base channel number')
    parser.add_argument('-i', '--ignore_idx', default=None, nargs='+', type=int, help='ignore certain label')
    parser.add_argument('-return_sequence', '--return_sequence', default=True, type=str, help='LSTM return the whole sequence or only last time step')
    parser.add_argument('-variant', '--LSTM_variant', default='back', type=str, help='LSTM combined with UNet, back, center, bicenter, shortcut')
    parser.add_argument('-ep', '--epoch', default=300, type=int, help='training epoches')
    parser.add_argument('-pretrain', '--is_pretrain', default=True, type=str, help='use pretrained backbone or not')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()