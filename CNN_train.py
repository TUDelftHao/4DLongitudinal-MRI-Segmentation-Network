from loss import DiceLoss, GneralizedDiceLoss, WeightedCrossEntropyLoss
from UNet import init_U_Net
from DirectConnectUNet import U_Net_direct_concat
from Inception_UNet import Inception_UNet, Simplified_Inception_UNet
from ResUNet import ResUNet, SEResUNet
from DilationResUNet import DResUNet
from data_prepara import data_split, data_construction, time_parser, data_loader, distributed_is_initialized
from utils import load_config, save_ckp, load_ckp, logfile, loss_plot, heatmap_plot
from metrics import dice_coe, dice
import torch.optim as optim
import torch 
import torch.nn.functional as F 
import os
import torch.nn as nn
from tqdm import tqdm 
import json
import numpy as np 
import shutil
import warnings
import time
import argparse
from apex import amp 
from torchsummary import summary
from tensorboardX import SummaryWriter


def train(args):

    torch.cuda.manual_seed(1)
    torch.manual_seed(1)

    # user defined
    model_name = args.model_name  
    model_type = args.model_type  
    loss_func = args.loss
    world_size = args.world_size
    rank = args.rank
    base_channel = args.base_channels
    crop_size = args.crop_size
    ignore_idx = args.ignore_idx
    epochs = args.epoch


    # system setup
    config_file = 'config.yaml'
    config = load_config(config_file)
    labels = config['PARAMETERS']['labels']
    root_path = config['PATH']['model_root']
    model_dir = config['PATH']['save_ckp']
    best_dir = config['PATH']['save_best_model']

    output_channels = int(config['PARAMETERS']['output_channels'])
    batch_size = int(config['PARAMETERS']['batch_size'])
    is_best = bool(config['PARAMETERS']['is_best'])
    is_resume = bool(config['PARAMETERS']['resume'])
    patience = int(config['PARAMETERS']['patience'])
    time_step = int(config['PARAMETERS']['time_step'])
    num_workers = int(config['PARAMETERS']['num_workers'])
    early_stop_patience = int(config['PARAMETERS']['early_stop_patience'])
    pad_method = config['PARAMETERS']['pad_method']
    lr = int(config['PARAMETERS']['lr'])
    optimizer = config['PARAMETERS']['optimizer']
    softmax = True
    modalities = ['flair', 't1', 't1gd', 't2']
    input_modalites = len(modalities)
    
    # build up dirs
    model_path = os.path.join(root_path, model_dir)
    best_path = os.path.join(root_path, best_dir)
    intermidiate_data_save = os.path.join(root_path, 'train_newdata', model_name)
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
    logger.info('Dataset is loading ...')
    writer = SummaryWriter('ProcessVisu/%s' %model_name)

    logger.info('patch size: {}'.format(crop_size))
    
    # load training set and validation set
    data_class = data_split()
    train, val, test = data_construction(data_class)
    train_dict = time_parser(train, time_patch=time_step)
    val_dict = time_parser(val, time_patch=time_step)

    # groups = 4
    if model_type == 'UNet':
        net = init_U_Net(input_modalites, output_channels, base_channel, pad_method, softmax)
    elif model_type == 'ResUNet':
        net = ResUNet(input_modalites, output_channels, base_channel, pad_method, softmax)
    elif model_type == 'DResUNet':
        net = DResUNet(input_modalites, output_channels, base_channel, pad_method, softmax)
    elif model_type == 'direct_concat':
        net = U_Net_direct_concat(input_modalites, output_channels, base_channel, pad_method, softmax)
    elif model_type == 'Inception':
        net = Inception_UNet(input_modalites, output_channels, base_channel, softmax)
    elif model_type == 'Simple_Inception':
        net = Simplified_Inception_UNet(input_modalites, output_channels, base_channel, softmax)

    # device setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net.to(device)  

    # print model structure
    summary(net, input_size=(input_modalites, crop_size, crop_size, crop_size))
    dummy_input = torch.rand(1, input_modalites, crop_size, crop_size, crop_size).to(device)
    writer.add_graph(net, (dummy_input,))

    # loss and optimizer setup
    if loss_func == 'Dice' and softmax:
        criterion = DiceLoss(labels=labels, ignore_idx=ignore_idx)
    elif loss_func == 'GDice' and softmax:
        criterion = GneralizedDiceLoss(labels=labels)
    elif loss_func == 'CrossEntropy':
        criterion = WeightedCrossEntropyLoss(labels=labels)
        if not softmax:
            criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise NotImplementedError()
    
    if optimizer == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=patience)

    # net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    if torch.cuda.device_count() > 1:
        logger.info('{} GPUs avaliable'.format(torch.cuda.device_count()))
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:38366', rank=rank, world_size=world_size)
    if distributed_is_initialized():
        logger.info('distributed is initialized')
        net.to(device)
        net = nn.parallel.DistributedDataParallel(net)
    else:
        logger.info('data parallel')
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
            # min_loss = float('Inf')

        except:
            logger.warning('No checkpoint available, strat training from scratch')
    

    # start training
    for epoch in range(start_epoch, epochs):
    
        # every epoch generate a new set of images
        train_set = data_loader(train_dict, 
                            batch_size=batch_size, 
                            key='train',
                            num_works=num_workers,
                            time_step=time_step,
                            patch=crop_size,
                            modalities=modalities,
                            model_type='CNN'
                            )
        n_train = len(train_set)
        train_loader = train_set.load()

        val_set = data_loader(val_dict, 
                                batch_size=batch_size, 
                                key='val',
                                num_works=num_workers,
                                time_step=time_step,
                                patch=crop_size,
                                modalities=modalities,
                                model_type='CNN'
                                )
        n_val = len(val_set)
        
        nb_batches = np.ceil(n_train/batch_size)
        n_total = n_train + n_val
        logger.info('{} images will be used in total, {} for trainning and {} for validation'.format(n_total, n_train, n_val))
        logger.info('Dataset loading finished!')

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
                images, segs = data['image'].to(device), data['seg'].to(device)

                if model_type == 'SkipDenseSeg' and not softmax:
                    segs = segs.long()

                # combine the batch and time step
                batch, time, channel, z, y, x = images.shape
                images = images.view(-1, channel, z, y, x)
                segs = segs.view(-1, z, y, x)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = net(images)

                loss = criterion(outputs, segs)
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                optimizer.step()
    
                running_loss += loss.detach().item()
                _, preds = torch.max(outputs.data, 1)
                dice_score = dice(preds.data.cpu(), segs.data.cpu(), ignore_idx=ignore_idx)

                dice_score_label_0 += dice_score['bg']
                dice_score_label_1 += dice_score['csf']
                dice_score_label_2 += dice_score['gm']
                dice_score_label_3 += dice_score['wm']
                dice_score_label_4 += dice_score['tm']

                # show progress bar
                pbar.set_postfix(**{'Training loss': loss.detach().item(), 'Training accuracy': dice_score['avg']})
                pbar.update(images.shape[0])

                del images, segs

                global_step += 1
                if global_step % nb_batches == 0:
                    net.eval()
                    val_loss, val_acc, val_info = validation(net, val_set, criterion, device, batch_size, model_type=model_type, softmax=softmax, ignore_idx=ignore_idx)
        
        train_info['train_loss'].append(running_loss / nb_batches)
        train_info['val_loss'].append(val_loss)
        train_info['label_0_acc'].append(dice_score_label_0 / nb_batches)
        train_info['label_1_acc'].append(dice_score_label_1 / nb_batches)
        train_info['label_2_acc'].append(dice_score_label_2 / nb_batches)
        train_info['label_3_acc'].append(dice_score_label_3 / nb_batches)
        train_info['label_4_acc'].append(dice_score_label_4 / nb_batches)

        # save bast trained model
        if model_type == 'SkipDenseSeg':
            scheduler.step()
        else:
            scheduler.step(val_loss)
        # debug
        for param_group in optimizer.param_groups:
            logger.info('%0.6f | %6d ' % (param_group['lr'], epoch))

        if min_loss > running_loss / nb_batches + 1e-2:
            min_loss = running_loss / nb_batches
            is_best = True
            early_stop_count = 0
        else:
            is_best = False
            early_stop_count += 1

        # save the check point
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
        logger.info('Average training loss of this epoch is {}'.format(running_loss / nb_batches))
        logger.info('Best training loss till now is {}'.format(min_loss))
        logger.info('Validation dice loss: {}; Validation accuracy: {}'.format(val_loss, val_acc))
                    
        # save the training info every epoch
        logger.info('Writing the training info into file ...')
        val_info_file = os.path.join(intermidiate_data_save, '{}_val_info.json'.format(model_name))
        with open(train_info_file, 'w') as fp:
            json.dump(train_info, fp)
        with open(val_info_file, 'w') as fp:
            json.dump(val_info, fp)

        loss_plot(train_info_file, name=model_name)
        for name, layer in net.named_parameters():
            writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        if verbose:
            logger.info('The validation loss has not improved for {} epochs, training will stop here.'.format(early_stop_patience))
            break
    
    writer.close()
    logger.info('finish training!')


def validation(trained_net, 
                val_set, 
                criterion, 
                device, 
                batch_size,
                model_type=None,
                softmax=True,
                ignore_idx=None):

    '''
    used for evaluation during training phase

    params trained_net: trained U-net
    params val_set: validation dataset 
    params criterion: loss function
    params device: cpu or gpu
    '''

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
                
                if model_type == 'SkipDenseSeg' and not softmax:
                    segs = segs.long()

                batch, time, channel, z, y, x = images.shape
                images = images.view(-1, channel, z, y, x)
                segs = segs.view(-1, z, y, x)
                preds = trained_net(images)
                val_loss = criterion(preds, segs)

                _, preds = torch.max(preds, 1)
                dice_score = dice(preds.data.cpu(), segs.data.cpu(), ignore_idx)

                dice_score_bg += dice_score['bg']
                dice_score_wm += dice_score['wm']
                dice_score_gm += dice_score['gm']
                dice_score_csf += dice_score['csf']
                dice_score_tm += dice_score['tm']

                tot += val_loss.detach().item() 
                acc += dice_score['avg']

                pbar.set_postfix(**{'validation loss': val_loss.detach().item(), 'val_acc_avg':dice_score['avg']})
                pbar.update(images.shape[0])

        val_info['bg'] = dice_score_bg / (np.ceil(n_val/batch_size))
        val_info['wm'] = dice_score_wm / (np.ceil(n_val/batch_size))
        val_info['gm'] = dice_score_gm / (np.ceil(n_val/batch_size)) 
        val_info['csf'] = dice_score_csf / (np.ceil(n_val/batch_size))
        val_info['tm'] = dice_score_tm / (np.ceil(n_val/batch_size)) 

    return tot/(np.ceil(n_val/batch_size)), acc/(np.ceil(n_val/batch_size)), val_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', '--model_type', default='UNet', type=str, help='which model to use: UNet, ResUNet, DResUNet, SEResUNet, direct_concat, Inception, Simple_Inception, SkipDenseSeg')
    parser.add_argument('-name', '--model_name', default='init', type=str, help='model name')
    parser.add_argument('-l', '--loss', default='Dice', type=str, help='loss function, Dice, GDice, CrossEntropy')
    parser.add_argument('-s', '--world_size', default=1, type=int, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', default=0, type=int, help='Rank of the current process.')
    parser.add_argument('-p', '--crop_size', default=64, type=int, help='image crop patch size')
    parser.add_argument('-b', '--base_channels', default=4, type=int, help='U-Net base channel number')
    parser.add_argument('-i', '--ignore_idx', default=None, type=int, help='ignore certain label')
    parser.add_argument('-ep', '--epoch', default=260, type=int, help='training epoches')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
