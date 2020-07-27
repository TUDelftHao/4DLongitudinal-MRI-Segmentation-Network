import SimpleITK as sitk 
import torch
import numpy as np 
import random
import skimage
from scipy import ndimage


        
class RandomRotation:
    
    ''' Randomly rotate the images within 10 degrees '''

    def __init__(self, max_angle=30, prob=0.1):
    
        # 90% probability to happen
        self.max_angle = max_angle
        self.axes = [(0,1), (1,2), (0,2)] # rotate along x, z, y, repsectively
        self.prob = prob

    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']
        time_points, channels = image.shape[0], image.shape[1]

        rotated_images = []
        rotated_segs = []

        rotate_angle = np.random.uniform(-self.max_angle, self.max_angle)
        rotate_axes  = self.axes[np.random.randint(len(self.axes))]

        for i in range(time_points):

            instant_img = image[i]
            instant_seg = seg[i]

            rotated_img = [ndimage.rotate(instant_img[j, ...], rotate_angle, rotate_axes, reshape=False, order=0, mode='constant', cval=0) for j in range(channels)]
            rotated_img = np.stack(rotated_img, axis=0)
            rotated_images.append(rotated_img)

            rotated_sg  = ndimage.rotate(instant_seg, rotate_angle, rotate_axes, reshape=False, order=0, mode='constant', cval=0)
            rotated_segs.append(rotated_sg)

        rotated_images = np.stack(rotated_images, axis=0)
        rotated_segs = np.stack(rotated_segs, axis=0)

        return {'image':rotated_images, 'seg':rotated_segs}
        

class RandomGaussianNoise:

    ''' Randomly add gaussian noise to images '''

    def __init__(self, prob=0.5):

        # 50% probability to happen
        self.prob = prob

    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']
        time_points = image.shape[0]

        noised_img = []

        for i in range(time_points):

            instant_img = np.asarray(image[i], dtype='float_')
            std = np.random.uniform(0, 0.5)
            noise = np.random.normal(0, std, size=instant_img.shape)
            instant_img += noise
            noised_img.append(instant_img)

        noised_img = np.stack(noised_img)
        return {'image':noised_img, 'seg':seg}


class ElasticDeformation:

    def __init__(self, spline_order=3, alpha=2000, sigma=50):

        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']
        time_points, channel, z, y, x = image.shape

        coordinates = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
        xi = np.meshgrid(np.linspace(0, 2, z), np.linspace(0, 2, y), np.linspace(0, 2, x), indexing='ij') 
        grid = [3]*3

        for i in range(3):
            yi = np.random.randn(*grid)*self.sigma
            y = ndimage.map_coordinates(yi, xi, order=self.spline_order).reshape(image.shape[-3:])
            coordinates[i] = np.add(coordinates[i], y)

        for i in range(time_points):
            for j in range(channel):
                ndimage.map_coordinates(image[i, j], coordinates, order=self.spline_order).reshape(image.shape[-3:])
            
            ndimage.map_coordinates(seg[i], coordinates, order=0).reshape(seg[i].shape)

        return {'image':image, 'seg': seg}

class RandomFlip:
    
    ''' Randomly flip the axes '''

    def __init__(self, prob=0.1):

        # 90% probability to happen
        self.axes = [0, 1, 2]
        self.prob = prob

    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']

        # if self.random_state.uniform() > self.prob:

        for axis in self.axes:
            flipped_image = [np.flip(image[:, i, ...], axis) for i in range(image.shape[1])]
            flipped_image = np.stack(flipped_image, axis=1)
            flipped_seg = np.flip(seg, axis)

        return {'image':flipped_image, 'seg':flipped_seg}


class RandomCrop:
    
    ''' randomly crop the patch to (crop_size, crop_size, crop_size) in xy plane'''

    def __init__(self, crop_size=64):

        if isinstance(crop_size, int):
            self.crop_size = [crop_size]*3
        else:
            self.crop_size = crop_size
    
    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']
        time_points, channel, z, y, x = image.shape
        new_x, new_y, new_z = self.crop_size[0], self.crop_size[1], self.crop_size[2]
       
        size = [(x, new_x), (y, new_y), (z, new_z)]
        pads = [abs(val-new_val) for (val, new_val) in size]
        starts = [np.random.randint(0, pad) for pad in pads]

        image = image[..., starts[-1]:starts[-1]+new_z, starts[-2]:starts[-2]+new_y, starts[-3]:starts[-3]+new_x]
        seg = seg[..., starts[-1]:starts[-1]+new_z, starts[-2]:starts[-2]+new_y, starts[-3]:starts[-3]+new_x]

        return {'image':image, 'seg':seg}

class Crop:
    def __init__(self, crop_size=64, model_type='CNN'):

        if isinstance(crop_size, int):
            self.crop_size = [crop_size]*3
        else:
            self.crop_size = crop_size

        if model_type == 'CNN':
            self.tumor_ratio = 0
        else:
            self.tumor_ratio = 0.4
    
    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']
        time_points, channel, z, y, x = image.shape
        new_z, new_y, new_x = self.crop_size[0], self.crop_size[1], self.crop_size[2]

        loc = np.where(seg[0]==4)
        num_tumor = len(loc[0])

        if num_tumor > 0 and np.random.random(1)[0] < self.tumor_ratio:
            idx = np.random.choice(range(num_tumor), 1, replace=False)
            c_z, c_y, c_x = int(loc[0][idx]), int(loc[1][idx]), int(loc[2][idx])

            new_x_start = c_x - new_x//2
            pad_x_start = new_x_start if new_x_start < 0 else 0
            new_x_end = c_x + new_x//2
            pad_x_end = new_x_end - x if new_x_end > x else 0 
            new_y_start = c_y - new_y//2
            pad_y_start = new_y_start if new_y_start < 0 else 0
            new_y_end = c_y + new_y//2
            pad_y_end = new_y_end - y if new_y_end > y else 0
            new_z_start = c_z - new_z//2
            pad_z_start = new_z_start if new_z_start < 0 else 0
            new_z_end = c_z + new_z//2
            pad_z_end = new_z_end - z if new_z_end > z else 0  

            image = image[..., new_z_start-pad_z_start-pad_z_end : new_z_end-pad_z_start-pad_z_end, new_y_start-pad_y_start-pad_y_end : new_y_end-pad_y_start-pad_y_end, new_x_start-pad_x_start-pad_x_end : new_x_end-pad_x_start-pad_x_end]
            seg = seg[..., new_z_start-pad_z_start-pad_z_end : new_z_end-pad_z_start-pad_z_end, new_y_start-pad_y_start-pad_y_end : new_y_end-pad_y_start-pad_y_end, new_x_start-pad_x_start-pad_x_end : new_x_end-pad_x_start-pad_x_end]
        else:
            size = [(x, new_x), (y, new_y), (z, new_z)]
            pads = [abs(val-new_val) for (val, new_val) in size]
            starts = [np.random.randint(0, pad) for pad in pads]

            image = image[..., starts[-1]:starts[-1]+new_z, starts[-2]:starts[-2]+new_y, starts[-3]:starts[-3]+new_x]
            seg = seg[..., starts[-1]:starts[-1]+new_z, starts[-2]:starts[-2]+new_y, starts[-3]:starts[-3]+new_x]

        return {'image':image, 'seg':seg}        
       
        
class ToTensor:
    
    """ Convert ndarrays in sample to Tensors """
    def __init__(self):
        pass

    def __call__(self, sample):
        dtype = torch.FloatTensor
        image, seg = sample['image'], sample['seg']
        
        return {'image':torch.from_numpy(image).type(dtype), 'seg':torch.from_numpy(seg).type(dtype)}

class Normalize:
    
    '''Z-scoring Normalize the images '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, seg, brainmask = sample['image'], sample['seg'], sample['mask']
        time_points, channel, z, y, x = image.shape

        for i in range(time_points):
            for j in range(channel):
                img = np.asarray(image[i,j], dtype='float_')
                img = img*brainmask[i]
                img = (img - img.mean())/(img.std()+1e-5)
                image[i, j, ...] = img

        return {'image':image, 'seg':seg}


def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)


if __name__ == '__main__':
    pass







                





            

            
