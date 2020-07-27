import SimpleITK as sitk 
import os 
import torch
import shutil 
import json
import yaml 
import numpy as np 
import glob
import random
from utils import load_config, image_show
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_transform import RandomRotation, RandomGaussianNoise, ToTensor, Normalize, ElasticDeformation, RandomFlip, Crop

config_file = 'config.yaml'

def data_split(test_ratio=0.1, save=True):
    '''
    params test_ratio: how many data are left to be test
    '''

    cfg = load_config(config_file)
    data_root = cfg['PATH']['data_root']
    IDs = os.listdir(data_root)
    ID_num = len(IDs)

    data_content = {
        'train': [],
        'val': [],
        'test': []
    }


    test_IDs = IDs[-int(ID_num*test_ratio)-1:] # the last 0.1 patients are left to be test set 
    for ID in test_IDs:
        test_patient = os.path.join(data_root, ID)
        data_content['test'].append(test_patient)
    
    train_val_IDs = IDs[:int(ID_num*(1-test_ratio))]
    permuted_idx = np.random.permutation(train_val_IDs)
    train_idx = permuted_idx[:int(len(train_val_IDs)*0.9)]
    val_idx = permuted_idx[int(len(train_val_IDs)*0.9):]
    
    for ID in train_idx:
        train_patient = os.path.join(data_root, ID)
        data_content['train'].append(train_patient)
    for ID in val_idx:
        val_patient = os.path.join(data_root, ID)
        data_content['val'].append(val_patient)

    if save:
        with open('data_split.json', 'w') as fp:
            json.dump(data_content, fp)

    return data_content


def data_construction(data_content):

    train_patient = data_content['train']
    val_patient = data_content['val']
    test_patient = data_content['test']

    train = {}
    val = {}
    test = {}

    train = patient_parser(train_patient, train)
    val = patient_parser(val_patient, val)
    test = patient_parser(test_patient, test)

    return train, val, test


def patient_parser(dataset, collection_data):

    '''
    params dataset: [list of patient folder]
    params collection_data: {}
    return: dictionary contains different formats of images(raw, diff, long, ...) for each patient
    '''

    for patient in dataset:

        patient_content = os.listdir(patient)

        images_time_step = [os.path.join(patient, member) for member in patient_content if member.startswith('EGD')]
        longitudinal_images = [os.path.join(patient, member) for member in patient_content if member.startswith('long')]
        mean_images = [os.path.join(patient, member) for member in patient_content if member.startswith('mean')]
        diff_images = [os.path.join(patient, member) for member in patient_content if member.startswith('diff')]
        sum_images = [os.path.join(patient, member) for member in patient_content if member.startswith('sum')]

        patient_id = os.path.basename(patient)
        collection_data[patient_id] = {}
        collection_data[patient_id].update({'raw': images_time_step})
        collection_data[patient_id].update({'long': longitudinal_images})
        collection_data[patient_id].update({'mean': mean_images})
        collection_data[patient_id].update({'diff': diff_images})
        collection_data[patient_id].update({'sum': sum_images})

    return collection_data

def time_parser(collection_data, time_patch=3):

    '''
    params collection_data: collection_data[patient ids]['raw', 'long', 'mean', ...]
    return: dictionary contains different raw modalities at each time point per patient. Example: {'patient id': {'date': {'flair': dir, 't1': dir}}}
    '''

    patient_dict = {}
    for patient in collection_data:

        time_folder = collection_data[patient]['raw']
        if time_folder:
            time_dict = {}

            # if any patient has time points less than 3, just skip it
            if len(time_folder) < time_patch:
                print('Time points are less than {} in patient {}'.format(time_patch, patient))
                continue

            for tri in time_folder:
                image_dict = {}
                contents = os.listdir(tri)

                flair_l = [os.path.join(tri, x) for x in contents if x.startswith('flair') and x.endswith('gz')]
                t2_l = [os.path.join(tri, x) for x in contents if x.startswith('t2') and x.endswith('gz')]
                t1gd_l = [os.path.join(tri, x) for x in contents if x.startswith('t1gd') and x.endswith('gz')]
                t1_l = [os.path.join(tri, x) for x in contents if x.startswith('t1') and x.endswith('gz')]
                brainmask_l = [os.path.join(tri, x) for x in contents if x.startswith('brain') and x.endswith('gz')]
                combined_fast_l = [os.path.join(tri, x) for x in contents if x.startswith('combine') and x.endswith('gz')]
                label_l = [os.path.join(tri, x) for x in contents if x.startswith('hd') and x.endswith('gz')]

                # if any modality lost, just skip it
                if not flair_l or not t2_l or not t1_l or not t1gd_l or not brainmask_l or not combined_fast_l or not label_l:
                    continue

                image_dict['flair'] = flair_l[0] 
                image_dict['t2'] = t2_l[0] 
                image_dict['t1'] = t1_l[0] 
                image_dict['t1gd'] = t1gd_l[0] 
                image_dict['brainmask'] = brainmask_l[0] 
                image_dict['combined_fast'] = combined_fast_l[0] 
                image_dict['label'] = label_l[0] 

                time_step = os.path.basename(tri).split('_')[-1]
                time_dict[time_step] = image_dict
                
            if time_dict:
                patient_id = os.path.basename(patient)
                patient_dict[patient_id] = time_dict
    
    return patient_dict

class EMCdata(Dataset):

    '''
    return sample: sample['image'].shape = (time_step, channel, d, h, w)
                   sample['seg'].shape = (time_step, d, h, w)
    '''
    def __init__(self, 
                patient_dict, 
                transform=None, 
                time_step=3,
                modalities=['flair', 't1', 't1gd', 't2']
                ):
            
        self.patient_dict = patient_dict
        self.transform = transform
        self.time_step = time_step
        self.modalities = modalities
    

    def __len__(self):
        return len(self.patient_dict)

    def __getitem__(self, index):

        ''' index used for select patient, self.time_step used for seletc time steps'''

        patient_idx = np.arange(len(self.patient_dict))
        zip_patient = list(zip(patient_idx, self.patient_dict))
        individual_patient = self.patient_dict[zip_patient[index][1]]
        sorted_individual_patient = sorted(individual_patient.items(), key=lambda item:item[0])
        # print patient name
        # print(zip_patient[index]) 
        selected_start_idx = np.random.randint(len(sorted_individual_patient)-self.time_step+1)
        selected_time_points = sorted_individual_patient[selected_start_idx:selected_start_idx+self.time_step]

        out_image = []
        out_seg = []
        out_brainmask = []
        for time_point in selected_time_points:
            image = [sitk.GetArrayFromImage(sitk.ReadImage(time_point[1][modality])) for modality in self.modalities]
            image = np.stack(image)
            out_image.append(image)

            out_seg.append(sitk.GetArrayFromImage(sitk.ReadImage(time_point[1]['combined_fast'])))
            out_brainmask.append(sitk.GetArrayFromImage(sitk.ReadImage(time_point[1]['brainmask'])))

        out_image = np.stack(out_image)
        out_seg = np.stack(out_seg)
        out_brainmask = np.stack(out_brainmask)

        sample = {'image': out_image, 'seg': out_seg, 'mask': out_brainmask}

        if self.transform:
            sample = self.transform(sample)

        return sample

class data_loader:

    def __init__(self, 
                patient_dict, 
                batch_size, 
                key='train', 
                num_works=0, 
                time_step=3,
                patch=64,
                modalities=['flair', 't1', 't1gd', 't2'],
                model_type='CNN',
                dataset=EMCdata):

        '''
        params batch_size: the number of patients to be loaded
        params key: ('train', 'val', 'test')
        '''

        self.patient_dict = patient_dict
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_works = num_works
        self.key = key
        self.time_step = time_step
        self.modalities = modalities
        
        
        if self.key == 'train':
            self.data = self.dataset(
                self.patient_dict,
                transform=transforms.Compose([
                    Normalize(),
                    RandomGaussianNoise(),
                    RandomFlip(),
                    RandomRotation(),
                    Crop(model_type=model_type, crop_size=patch),
                    ElasticDeformation(),
                    ToTensor()
                ]),
                time_step=self.time_step,
                modalities=self.modalities
            )
        else:
            self.data = self.dataset(
                self.patient_dict,
                transform=transforms.Compose([
                    Normalize(),
                    Crop(model_type=model_type, crop_size=patch),
                    ToTensor()                    
                ]),
                time_step=self.time_step,
                modalities=self.modalities
            )


    def __len__(self):
        return len(self.data)

    def load(self):

        '''
        loaded data shape:
        image: (batch size, time step, channel, d, w, h)
        seg: (batch size, time step, d, w, h)
        '''
        datasampler = None
        if self.key == 'train' and distributed_is_initialized():
            datasampler = DistributedSampler(self.data)

        dataloader = DataLoader(self.data, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_works, 
                                shuffle=(datasampler is None), 
                                sampler=datasampler,
                                worker_init_fn=lambda x: np.random.seed())

        return dataloader

def distributed_is_initialized():
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            return True
    return False

if __name__ == '__main__':

    data_class = data_split()
    train, val, test = data_construction(data_class)
    train_dict = time_parser(val, time_patch=4)
    data = data_loader(val, time_step=1, batch_size=1)
    patient_id = [key for key in train_dict.keys()]
    print(patient_id)

    data = EMCdata(train_dict, time_step=4)
    # for i in range(len(data)):
    #     sample = data[i]

    #     print(i, sample['image'].shape, sample['seg'].shape, sample['mask'].shape)
        

    # train_set = data_loader(train_dict, 
    #                         batch_size=1, 
    #                         key='train',
    #                         num_works=1,
    #                         time_step=2,
    #                         patch=64)
    # n_train = len(train_set)
    # train_loader = train_set.load()

    # for i, sample in enumerate(train_loader, 0):
    #     image, seg = sample['image'], sample['seg']
    #     print(i, image.shape)
        # print(i, torch.TensorType(image), type(seg))




    



    



