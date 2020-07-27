import yaml 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import os
import shutil
import numpy as np 
import warnings
import time
import seaborn as sns
import logging

def load_config(file_path):
    return yaml.safe_load(open(file_path, 'r'))

def logfile(path, level='debug'):
    
    if os.path.exists(path):
        os.remove(path)
        
    # set up log file

    logger = logging.getLogger(__name__)
    if level == 'debug':
        logger.setLevel(level=logging.DEBUG)   
    elif level == 'info':
        logger.setLevel(level=logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # FileHandler
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def image_show(image, seg, slice_pos=2, title=None):
    
    z = image.shape[-3]
    slice_idx = z//slice_pos

    fig, axes = plt.subplots(1,5,figsize=(10,10))
    ax = axes.ravel()
    
    for i in range(4):
        ax[i].set_axis_off()
        ax[i].imshow(image[i, slice_idx])
        
    ax[4].set_axis_off()
    ax[4].imshow(seg[slice_idx])
    if title:
        plt.title(title)

    plt.show()


def save_ckp(state, is_best, early_stop_count, early_stop_patience, save_model_dir, best_dir, name):
    f_path = os.path.join(save_model_dir, '{}_model_ckp.pth.tar'.format(name))
    torch.save(state, f_path)
    verbose = False

    if is_best:
        best_path = os.path.join(best_dir, '{}_best_model.pth.tar'.format(name))
        shutil.copyfile(f_path, best_path)
    if early_stop_count == early_stop_patience:
        verbose = True
    
    return verbose


class WrappedModel(nn.Module):
	def __init__(self, model):
		super(WrappedModel, self).__init__()
		self.module = model
	def forward(self, x):
		return self.module(x)


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['min_loss'], checkpoint['loss']

def loss_plot(train_info_file, name, nb_epoch=None):
    
    #plt.cla()
    train_info = load_config(train_info_file)
    train_loss = train_info['train_loss']
    val_loss = train_info['val_loss']
    label_0_acc = train_info['label_0_acc']
    label_1_acc = train_info['label_1_acc']
    label_2_acc = train_info['label_2_acc']
    label_3_acc = train_info['label_3_acc']
    label_4_acc = train_info['label_4_acc']

    if nb_epoch:
        train_loss = train_loss[:nb_epoch]
        val_loss = val_loss[:nb_epoch]
        label_0_acc = label_0_acc[:nb_epoch]
        label_1_acc = label_1_acc[:nb_epoch]
        label_2_acc = label_2_acc[:nb_epoch]
        label_3_acc = label_3_acc[:nb_epoch]
        label_4_acc = label_4_acc[:nb_epoch]

    epoch = len(train_loss)
    x_axis = np.arange(epoch)

    figure = plt.figure(1, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(121)
    plt.title('loss')
    plt.plot(x_axis, train_loss, lw=3, color='black', label='training loss')
    plt.plot(x_axis, val_loss, lw=3, color='green', label='validation loss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.title('accuracy')
    plt.plot(x_axis, label_0_acc, color='red', label='BG accuracy')
    plt.plot(x_axis, label_1_acc,  color='skyblue', label='CSF accuracy')
    plt.plot(x_axis, label_2_acc, color='blue', label='GM accuracy')
    plt.plot(x_axis, label_3_acc, color='yellow', label='WM accuracy')
    plt.plot(x_axis, label_4_acc, color='green', label='TM accuracy')
    plt.legend(loc='best') 

    plt.xlabel('epochs')
    plt.ylabel('acc')

    plt.savefig(os.path.join(os.path.dirname(train_info_file), '{}_loss_acc_plot.png'.format(name)))
    # plt.show()

def heatmap_plot(image, mask, pred, epoch, name, is_train, save=True):
    # image, mask, pred should be numpy.array()
    warnings.filterwarnings("ignore")
    # plt.cla()

    current_path = os.getcwd()
    plot_dir = os.path.join(current_path, 'temp_plot', name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if is_train:
        _dir = os.path.join(plot_dir, 'train')
    else:
        _dir = os.path.join(plot_dir, 'validation')
    if not os.path.exists(_dir):
        os.mkdir(_dir)

    size = image.shape[2]
    fig = plt.figure(figsize=(20, 20))

    for i in range(pred.shape[0]):
        # original image
        ax = fig.add_subplot(pred.shape[0], (pred.shape[1]+2), 1+i*(pred.shape[1]+2))
        ax.axis('off')
        ax.imshow(image[i, 0, size//2])
        # ground truth map
        ax = fig.add_subplot(pred.shape[0], (pred.shape[1]+2), 2+i*(pred.shape[1]+2))
        ax.axis('off')
        sns.heatmap(mask[i, size//2], vmin=0, vmax=4, xticklabels=False, yticklabels=False, square=True, cmap='coolwarm', cbar=False)
        for j in range(pred.shape[1]):
            ax = fig.add_subplot(pred.shape[0], (pred.shape[1]+2), j+3+i*(pred.shape[1]+2))
            ax.axis('off')
            sns.heatmap(pred[i, j, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    if save:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        plot_name = os.path.join(_dir, '{}-epoch-{}_{}.png'.format(name, epoch, current_time))
    
        plt.savefig(plot_name)


def box_plot(dice_score, save_path, metric='Dice'):

    CSF_score = dice_score['csf']
    GM_score = dice_score['gm'] 
    WM_score = dice_score['wm']
    TM_score = dice_score['tm']
    labels = ['CSF', 'GM', 'WM', 'TM']

    plt.figure()  
    ax = plt.subplot() 
    bplot = ax.boxplot([CSF_score, GM_score, WM_score, TM_score], whis=[5, 95], patch_artist=True, labels=labels) 
    plt.title('Box plot of {} score'.format(metric))
    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.gca().yaxis.grid(True)
    plt.xlabel('Brain tissues and tumour')
    plt.ylabel('Score')
    plt.title('{} score'.format(metric))

    if save_path:
        plt.savefig(os.path.join(save_path, 'boxplot.png'))

    

##############################################################
### inference use
##############################################################
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def image_rebuild(crop_index_array, cropped_image_list):
    
    '''
    concatenate all the pateches and rebuild a new image with the same size as orignal one, the final image has four channels in one-hot format 
    
    params crop_index_array: crop start corner index, saved in order
    params cropped_image_list: predict patches with the same order of crop index
    
    '''

    if isinstance(cropped_image_list, list):
        cropped_image_list = np.asarray(cropped_image_list)

    assert crop_index_array.shape[0] == cropped_image_list.shape[0], 'The number of index array should equal image number'

    cropped_image_shape = cropped_image_list[0].shape
    target_imagesize_z = max(np.unique(crop_index_array[:,0])) + cropped_image_shape[1]
    target_imagesize_y = max(np.unique(crop_index_array[:,1])) + cropped_image_shape[2]
    target_imagesize_x = max(np.unique(crop_index_array[:,2])) + cropped_image_shape[3]

    target_imagechannel = cropped_image_shape[0]
    total_mask = np.zeros((target_imagechannel, target_imagesize_z, target_imagesize_y, target_imagesize_x), dtype='float_')


    for i in range(target_imagechannel):
        overlap_mask = np.zeros((target_imagesize_z, target_imagesize_y, target_imagesize_x), dtype='float_') # used to count the overlap 
        for (crop_index, cropped_image) in zip(crop_index_array, cropped_image_list):
            image_channel = cropped_image[i] # (D,H,W)
            
            total_mask[i, crop_index[0]:crop_index[0]+cropped_image_shape[1], crop_index[1]:crop_index[1]+cropped_image_shape[2], crop_index[2]:crop_index[2]+cropped_image_shape[3]] += image_channel
            overlap_mask[crop_index[0]:crop_index[0]+cropped_image_shape[1], crop_index[1]:crop_index[1]+cropped_image_shape[2], crop_index[2]:crop_index[2]+cropped_image_shape[3]] += 1

        total_mask_channel = total_mask[i, ...]
        total_mask_channel /= overlap_mask

        # total_mask_channel = np.where(total_mask_channel>=0.5, 1, 0)
        total_mask[i, ...] = total_mask_channel

    inferenced_image = np.argmax(total_mask, axis=0)

    # return total_mask
    return inferenced_image

def inference_output(output_image):
    
    shape = output_image.shape
    channel = shape[0]
    inferenced_image = np.zeros(shape[-3:])

    for i in range(1, channel):
    
        inferenced_image[output_image[i]==1] = i
        
    return inferenced_image

def crop_index_gen(image_shape, crop_size=64, overlap_size=None):
    
    ''' return a dict containing the sorted cropping start index, crop size and patch number '''

    crop_info = {}

    if crop_size == -1:
        # predict the whole image
        crop_info['index_array'] = np.array([[0, 0, 0]])
        crop_info['crop_size'] = image_shape[-3:]
        crop_info['crop_number'] = 1

        return crop_info

    if overlap_size is None:
        overlap_size = 0

    if isinstance(crop_size, int):
        crop_size = np.asarray([crop_size]*len(image_shape))
    if isinstance(overlap_size, int):
        overlap_size = np.asarray([overlap_size]*len(image_shape))

    num_block_per_dim = (image_shape - overlap_size) // (crop_size - overlap_size)

    index_per_axis_dict = {}
    for j, num in enumerate(num_block_per_dim):
        
        initial_point_dim = [i*(crop_size[j]-overlap_size[j]) for i in range(num)]
        initial_point_dim.append(image_shape[j]-crop_size[j])
        index_per_axis_dict[j] = initial_point_dim

    index_axis_z = index_per_axis_dict[0]
    index_axis_y = index_per_axis_dict[1]
    index_axis_x = index_per_axis_dict[2]
    index_array = []

    for val_z in index_axis_z:
        for val_y in index_axis_y:
            for val_x in index_axis_x:
                index_array.append([val_z, val_y, val_x])
    
    index_array = np.asarray(index_array).reshape(-1,3)

    crop_info['index_array'] = index_array
    crop_info['crop_size'] = crop_size
    crop_info['crop_number'] = index_array.shape[0]

    return crop_info

def image_crop(image, crop_info, ToTensor=False):
    
    '''return a list of cropped image patches according to crop index'''
    '''return: patches * channels * D * H * W''' 

    # assert image.ndim == 4 # C*W*H*D

    crop_index, crop_size, crop_num = crop_info['index_array'], crop_info['crop_size'], crop_info['crop_number']
    if image.ndim == 4:
        cropped_images = np.zeros((crop_num, image.shape[0], crop_size[0], crop_size[1], crop_size[2]))
    elif image.ndim == 5:
        cropped_images = np.zeros((crop_num, image.shape[0], image.shape[1], crop_size[0], crop_size[1], crop_size[2]))

    for i, index in enumerate(crop_index):
        cp_img = image.copy()
        img = cp_img[..., index[0]:index[0]+crop_size[0], index[1]:index[1]+crop_size[1], index[2]:index[2]+crop_size[2]]

        if image.ndim == 4:
            norm_img = normalize(img)
        elif image.ndim == 5:
            norm_img = []
            for j in range(image.shape[0]):
                norm = normalize(img[j])
                norm_img.append(norm)
            norm_img = np.stack(norm_img)
            
        cropped_images[i, ...] = norm_img

    if ToTensor:
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        cropped_images = torch.from_numpy(cropped_images).type(dtype)
    return cropped_images

def normalize(image):

    for i in range(image.shape[0]):
        img = np.asarray(image[i], dtype='float_')      
        img = (img - img.mean())/(img.std()+1e-5)
        image[i, ...] = img

    return image

if __name__ == '__main__':
   
    # box_plot(dice, save_path=None)
    # train_file = 'train_newdata/BackBiLSTM1layer-unet-p64-4x3-pretrained/BackBiLSTM1layer-unet-p64-4x3-pretrained_train_info.json'
    # train_file = 'train_newdata/BiConvLSTM1layer/BiConvLSTM1layer_train_info.json'
    # train_file = 'train_newdata/BiDirectCenterNormalLSTM1layer-dcunet-p64-4x3/BiDirectCenterNormalLSTM1layer-dcunet-p64-4x3_train_info.json'
    train_file = 'train_newdata/BiResCenterNormalLSTM1layer-resunet-p64-4x3/BiResCenterNormalLSTM1layer-resunet-p64-4x3_train_info.json'

    # train_file = 'train_newdata/CenterDenseBiLSTM1layer-unet-p64-4x3-halfpretrain/CenterDenseBiLSTM1layer-unet-p64-4x3-halfpretrain_train_info.json'
    # train_file = 'train_newdata/ShortcutLSTM1layer-1-shortcut-p64-4x3-pretrained/ShortcutLSTM1layer-1-shortcut-p64-4x3-pretrained_train_info.json'
   
    model_name = train_file.split('/')[-2]
    loss_plot(train_file, model_name, nb_epoch=None)
    


    
