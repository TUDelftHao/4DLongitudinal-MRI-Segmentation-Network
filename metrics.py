import numpy as np 
import torch
from loss import to_one_hot
from collections import OrderedDict
from scipy.ndimage import morphology
from medpy.metric.binary import hd, asd


def indiv_dice(im1, im2, tid):
    im1 = im1 == tid
    im2 = im2 == tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / ((im1.sum() + im2.sum()) + 1e-5)

    return dsc

def dice(im1, im2, ignore_idx=None):

    label_dict = OrderedDict()
    label_dict['bg'] = 0
    label_dict['csf'] = 0
    label_dict['gm'] = 0
    label_dict['wm'] = 0
    label_dict['tm'] = 0
    tot = 0
    count = 0

    if isinstance(ignore_idx, int):
        ignore_idx = [ignore_idx]

    for i, label in enumerate(label_dict):
        if ignore_idx and i not in ignore_idx:
            dsc = indiv_dice(im1, im2, i)
            label_dict[label] += dsc
            if i != 0:
                tot += dsc 
                count += 1
        elif not ignore_idx:
            dsc = indiv_dice(im1, im2, i)
            label_dict[label] += dsc
            if i != 0:
                tot += dsc
                count += 1 

    label_dict['avg'] = tot / count

    return label_dict


def dice_coe(output, target, eps=1e-5):

    target = to_one_hot(target)

    output = output.contiguous().view(output.shape[0], output.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1).type_as(output)
    
    num = 2*torch.sum(output*target, dim=-1)
    den = torch.sum(output + target, dim=-1) + eps

    BG_dice_coe = torch.mean(num[:, 0]/den[:, 0]).numpy()
    flair_dice_coe = torch.mean(num[:, 1]/den[:, 1]).numpy()
    t1_dice_coe = torch.mean(num[:, 2]/den[:, 2]).numpy()
    t1gd_dice_coe = torch.mean(num[:, 3]/den[:, 3]).numpy()
    # if there is no tumor in patch, the dice score should be 1
    if torch.sum(target[:, 4]) == 0 and torch.sum(output[:, 4]) < 1:
        t2_dice_coe = np.ones(t1gd_dice_coe.shape)
    else:
        t2_dice_coe = torch.mean(num[:, 4]/den[:, 4]).numpy()

    # average dice score only consider positive data
    avg_dice_coe = (flair_dice_coe + t1_dice_coe + t1gd_dice_coe + t2_dice_coe)/4

    dice_coe = {}
    dice_coe['avg'] = avg_dice_coe
    dice_coe['bg'] = BG_dice_coe
    dice_coe['csf'] = flair_dice_coe
    dice_coe['gm'] = t1_dice_coe
    dice_coe['wm'] = t1gd_dice_coe
    dice_coe['tm'] = t2_dice_coe
    
    return dice_coe

def one_hot_numpy(array):
    labels = np.unique(array)
    size = np.array(array.shape)
    one_hot_target = np.zeros(np.insert(size, 0, len(labels), axis=0))

    for i in range(len(labels)):
        channel = np.where(array == i, 1, 0)
        one_hot_target[i, ...] = channel
    
    return one_hot_target

def dice_coe_infernce(pred, target):

    target = one_hot_numpy(target)

    pred = np.reshape(pred, (pred.shape[0], -1))
    target = np.reshape(target, (pred.shape[0], -1))
    num = 2 * np.sum(pred * target, axis=-1)
    den = np.sum(pred + target, axis=-1) + 1e-5

    BG = num[0] / den[0]
    CSF = num[1] / den[1]
    GM = num[2] / den[2]
    WM = num[3] / den[3]
    if np.sum(target[4]) == 0 and np.sum(pred[4]) == 0:
        TM = 1.
    else:
        TM = num[4] / den[4]

    avg = (CSF + GM + WM + TM) / 4
    score = {}
    score['avg'] = np.round(avg, 3)
    score['bg'] = np.round(BG, 3)
    score['csf'] = np.round(CSF, 3)
    score['gm'] = np.round(GM, 3)
    score['wm'] = np.round(WM, 3)
    score['tm'] = np.round(TM, 3) 

    return score

def Hausdorf(pred, gt, replace_NaN=100):
    HD95_dict = {}

    bg_hd = hd(pred==0, gt==0)
    if 1 in np.unique(pred):
        csf_hd = hd(pred==1, gt==1)
    else:
        csf_hd = replace_NaN
    if 2 in np.unique(pred):
        gm_hd = hd(pred==2, gt==2)
    else:
        gm_hd = replace_NaN
    if 3 in np.unique(pred):
        wm_hd = hd(pred==3, gt==3)
    else:
        wm_hd = replace_NaN
    if 4 in np.unique(pred):
        tm_hd = hd(pred==4, gt==4)
    else:
        tm_hd = replace_NaN
    
    HD95_dict['avg'] = (csf_hd + gm_hd + wm_hd + tm_hd) / 4
    HD95_dict['bg'] = bg_hd
    HD95_dict['csf'] = csf_hd
    HD95_dict['gm'] = gm_hd
    HD95_dict['wm'] = wm_hd
    HD95_dict['tm'] = tm_hd

    return HD95_dict
    
def AverageSurfaceDist(pred, gt, replace_NaN=100):
    ASD = {}
    bg_asd = asd(pred==0, gt==0)
    if 1 in np.unique(pred):
        csf_asd = asd(pred==1, gt==1)
    else:
        csf_asd = replace_NaN
    if 2 in np.unique(pred):
        gm_asd = asd(pred==2, gt==2)
    else:
        gm_asd = replace_NaN
    if 3 in np.unique(pred):
        wm_asd = asd(pred==3, gt==3)
    else:
        wm_asd = replace_NaN
    if 4 in np.unique(pred):
        tm_asd = asd(pred==4, gt==4)
    else:
        tm_asd = replace_NaN

    ASD['avg'] = (csf_asd + gm_asd + wm_asd + tm_asd) / 4 
    ASD['bg'] = bg_asd
    ASD['csf'] = csf_asd
    ASD['gm'] = gm_asd
    ASD['wm'] = wm_asd
    ASD['tm'] = tm_asd

    return ASD


if __name__ == '__main__':

    # yp = np.random.random(size=(2, 5, 3, 3, 3))

    # yp = torch.from_numpy(yp)
    # yt = np.zeros(shape=(2, 3, 3, 3))
    # yt = yt + 1
    # yt = torch.from_numpy(yt)
    # coe = dice_coe(yp, yt)
    # print(coe)
    
    # print(end)
    im1 = np.random.random((4,3,3,3))
    im2 = np.random.random((2,3,3,3))
    gt = np.argmax(im1, axis=0)
    im = gt ==1
    print(gt ==1)
    # sds = surfd(im1[0], gt, sampling=[1,1,1], HD95=True)
    # print(sds)
    
    # dict = dice(im1, im2, ignore_idx=4)
    # print(dict)
