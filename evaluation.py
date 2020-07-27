''' Evaluate the segmentation consistancy '''

import SimpleITK as sitk 
import numpy as np
import torch 
import os 
import sys 
import pandas as pd
from data_prepara import data_split, data_construction, time_parser
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.pyplot import MultipleLocator


LONGITUDINAL = 'longitudinal_plot'
PATIENT = 'EGD-0265'
MODEL_1 = 'CenterNormalBiLSTM1layer-unet-p64-4x3-halfpretrain'
MODEL_2 = 'UNet-p64-b4-newdata-oriinput'
DATA = 'longitudinal.csv'
ANALYSIS_DIR = 'analysis'
if not os.path.exists(ANALYSIS_DIR):
    os.mkdir(ANALYSIS_DIR)

def volumn_ratio(mask):

    ''' count the ratio of each label'''

    labels = map(int, np.unique(mask))
    label_volume = {}

    for label in labels:
        if label != 0:
            label_volume[str(label)] = np.sum(mask==label)

    return label_volume


def plot_volumn_dev(patient_dict, patient_id, model_name, save_dir='longitudinal_plot', predicted_labels=None):
    
    ''' plot volumn development curve along time dim per patient '''

    save_path = os.path.join(save_dir, patient_id, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    time_dict = patient_dict[patient_id]
    time_dict = sorted(time_dict.items(), key=lambda item:item[0]) # sort according to time

    plt.clf()
    plt.figure(figsize=(10, 10))

    period = []

    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    predicted_label_1 = []
    predicted_label_2 = []
    predicted_label_3 = []
    predicted_label_4 = []

    for i, time_point in enumerate(time_dict):
        
        period.append(time_point[0])

        mask = sitk.GetArrayFromImage(sitk.ReadImage(time_point[1]['combined_fast']))
        labelwise_ratio = volumn_ratio(mask)
        if predicted_labels:
            predicted_mask = predicted_labels[time_point[0]]
            predicted_labelwise_ratio = volumn_ratio(predicted_mask)

        label_1.append(labelwise_ratio['1'])
        label_2.append(labelwise_ratio['2'])
        label_3.append(labelwise_ratio['3'])
        label_4.append(labelwise_ratio['4'])
        if predicted_labels:
            if '1' in predicted_labelwise_ratio:
                predicted_label_1.append(predicted_labelwise_ratio['1'])
            else:
                predicted_label_1.append(0)
            if '2' in predicted_labelwise_ratio:    
                predicted_label_2.append(predicted_labelwise_ratio['2'])
            else:
                predicted_label_2.append(0)
            if '3' in predicted_labelwise_ratio:
                predicted_label_3.append(predicted_labelwise_ratio['3'])
            else:
                predicted_label_3.append(0)
            if '4' in predicted_labelwise_ratio:                
                predicted_label_4.append(predicted_labelwise_ratio['4'])
            else:
                predicted_label_4.append(0)

    df = pd.DataFrame(list(zip(period, label_1, label_2, label_3, label_4, predicted_label_1, predicted_label_2, predicted_label_3, predicted_label_4)), 
    columns=['date', 'GT CSF', 'GT GM', 'GT WM', 'GT TM', 'predicted CSF', 'predicted GM', 'predicted WM', 'predicted TM'])
    df.to_csv(os.path.join(save_path, 'longitudinal.csv'))

    plt.plot(period, label_1, 'r--', label='GT CSF')
    plt.plot(period, label_2, 'g--', label='GT GM')
    plt.plot(period, label_3, 'b--', label='GT WM')
    plt.plot(period, label_4, 'y--', label='GT TM')
    if predicted_labels:
        plt.plot(period, predicted_label_1, c='r', label='predicted CSF')
        plt.plot(period, predicted_label_2, c='g', label='predicted GM')
        plt.plot(period, predicted_label_3, c='b', label='predicted WM')
        plt.plot(period, predicted_label_4, c='y', label='predicted TM')
    
    plt.xlabel('time', fontsize=15)
    plt.ylabel('volumn (cm3)', fontsize=15)
    plt.xticks(period, rotation=45)
    plt.legend(loc='best')
    plt.title('Comparison of brain tissues and tumor development in ground truth and predicted')
    plt.savefig(os.path.join(save_path, 'longitudinal.png'))


# normalization
def norm(model):
    csv_dir = os.path.join(LONGITUDINAL, PATIENT, model, DATA)
    df = pd.read_csv(csv_dir, index_col=0)
    date = df['date']
    pure_data = df.drop('date', axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(pure_data)
    scaled_data = pd.DataFrame(X, columns=pure_data.columns, index=pure_data.index)
    data = scaled_data
    scaled_data.loc['mean'] = data.apply(np.mean)
    scaled_data.loc['std'] = data.apply(np.std)
    scaled_data['date'] = date.astype('object')

    return scaled_data

def longitudinal_plot(model_1, model_2):
    data_1 = norm(model_1)
    data_2 = norm(model_2)

    modalities = ['CSF', 'GM', 'WM', 'TM']
    date = data_1['date'].dropna().to_list()

    x = list(str(d) for d in date)
    plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(4, 3, wspace=0.5, hspace=0.5)

    for i, modality in enumerate(modalities):
        GT = data_1['GT {}'.format(modality)].to_list()[:-2]
        pred_1 = data_1['predicted {}'.format(modality)].to_list()[:-2]
        pred_2 = data_2['predicted {}'.format(modality)].to_list()[:-2]

        upper_1 = [max(a, b) for (a, b) in zip(GT, pred_1)]
        lower_1 = [min(a, b) for (a, b) in zip(GT, pred_1)]

        ax1 = plt.subplot(grid[i, 0:2])

        ax1.plot(x, GT, 'k', lw=2)
        ax1.plot(x, pred_1, 'y', lw=1)
        ax1.plot(x, pred_2, 'r', lw=1)

        ax1.plot(x, upper_1, 'y', lw=1, alpha=0.1)
        ax1.plot(x, lower_1, 'y', lw=1, alpha=0.1)
        ax1.fill_between(x, upper_1, lower_1, facecolor='yellow', edgecolor='yellow', alpha=0.2, label='{}'.format(model_1.split('-')[0]))

        upper_2 = [max(a, b) for (a, b) in zip(GT, pred_2)]
        lower_2 = [min(a, b) for (a, b) in zip(GT, pred_2)]
        ax1.plot(x, upper_2, 'r', lw=1, alpha=0.1)
        ax1.plot(x, lower_2, 'r', lw=1, alpha=0.1)
        ax1.fill_between(x, upper_2, lower_2, facecolor='red', edgecolor='red', alpha=0.1, label='{}'.format(model_2.split('-')[0]))

        ax1.set_ylim(-0.5, 1.5)
        ax1.set_title('{}'.format(modality))
        ax1.set_xticklabels([])
        ax1.legend(loc='upper left', fontsize=10)

        ax2 = plt.subplot(grid[i, 2])
        
        ax2.errorbar(0, data_1.loc['mean', 'GT CSF'], yerr=data_1.loc['std', 'GT CSF'], fmt="o",color="black", elinewidth=2, capsize=4)
        ax2.errorbar(1, data_1.loc['mean', 'predicted CSF'], yerr=data_1.loc['std', 'predicted CSF'], fmt="o",color="yellow", elinewidth=2, capsize=4)
        ax2.errorbar(2, data_2.loc['mean', 'predicted CSF'], yerr=data_2.loc['std', 'predicted CSF'], fmt="o",color="red", elinewidth=2, capsize=4)

        ax2.set_ylim(0, 1)
        ax2.set_xlim(-0.5, 2.5)
        ax2.set_xticklabels([])
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.set_title('{}'.format(modality))

    ax1.set_xticklabels(x, rotation=30)
    ax1.set_xlabel('date', fontsize=15)
    ax2.set_xticklabels(['', 'GT', 'longitudinal', 'UNet'])
    ax2.set_xlabel('model', fontsize=15)
    plt.suptitle(PATIENT, fontsize=20)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'logitudinal_{}.png'.format(PATIENT)), dpi=150)


def transition_matrix(patient_dict, patient_id, model_name_1, model_name_2, save_dir='inference_result'):
    pred_1_path = os.path.join(save_dir, patient_id, model_name_1)
    pred_1_list = [pred_mask for pred_mask in os.listdir(pred_1_path) if pred_mask.endswith('.gz')]
    pred_1_dir_list  = [os.path.join(pred_1_path, pred) for pred in pred_1_list]
    pred_2_path = os.path.join(save_dir, patient_id, model_name_2)
    pred_2_list = [pred_mask for pred_mask in os.listdir(pred_2_path) if pred_mask.endswith('.gz')]
    pred_2_dir_list  = [os.path.join(pred_2_path, pred) for pred in pred_2_list]
    time_dict = patient_dict[patient_id]
    time_dict = sorted(time_dict.items(), key=lambda item:item[0]) # sort according to time
    time_points = len(time_dict)
    labels = [0, 1, 2, 3, 4]
    label_str = ['BG', 'CSF', 'GM', 'WM', 'TM']

    fig = plt.figure(figsize=(15, 5 * time_points))
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    grid = plt.GridSpec(time_points, 3, wspace=0.3, hspace=0.3)
    
    for k in range(time_points-1):
        mask_pre = sitk.GetArrayFromImage(sitk.ReadImage(time_dict[k][1]['combined_fast']))
        predicted_1_mask_pre = sitk.GetArrayFromImage(sitk.ReadImage(pred_1_dir_list[k]))
        predicted_2_mask_pre = sitk.GetArrayFromImage(sitk.ReadImage(pred_2_dir_list[k]))
        mask_later = sitk.GetArrayFromImage(sitk.ReadImage(time_dict[k+1][1]['combined_fast']))
        predicted_1_mask_later = sitk.GetArrayFromImage(sitk.ReadImage(pred_1_dir_list[k+1]))
        predicted_2_mask_later = sitk.GetArrayFromImage(sitk.ReadImage(pred_2_dir_list[k+1]))

        corr_mask = np.zeros((len(labels), len(labels)))
        corr_pred_1 = np.zeros((len(labels), len(labels)))
        corr_pred_2 = np.zeros((len(labels), len(labels)))

        for i, pre in enumerate(labels):
            for j, later in enumerate(labels):
                corr_mask[i, j] = np.sum(mask_later[mask_pre == pre] == later)
                corr_pred_1[i, j] = np.sum(predicted_1_mask_later[predicted_1_mask_pre == pre] == later)
                corr_pred_2[i, j] = np.sum(predicted_2_mask_later[predicted_2_mask_pre == pre] == later)


        df_mask = pd.DataFrame(corr_mask, columns=label_str, index=label_str)
        df_pred_1 = pd.DataFrame(corr_pred_1, columns=label_str, index=label_str)
        df_pred_2 = pd.DataFrame(corr_pred_2, columns=label_str, index=label_str)
        norm_mask = df_mask.div(df_mask.sum(axis=1), axis=0)
        norm_pred_1 = df_pred_1.div(df_pred_1.sum(axis=1), axis=0)
        norm_pred_2 = df_pred_2.div(df_pred_2.sum(axis=1), axis=0)

        ax1 = plt.subplot(grid[k, 0])
        ax1.imshow(norm_mask, aspect='equal', cmap="YlGn")
        ax1.set_xticks(np.arange(len(label_str)))
        ax1.set_yticks(np.arange(len(label_str)))
        ax1.set_xticklabels(label_str)
        ax1.set_yticklabels(label_str)
        ax1.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        for edge, spine in ax1.spines.items():
            spine.set_visible(False)

        ax1.set_xticks(np.arange(len(label_str)+1) - 0.5, minor=True)
        ax1.set_yticks(np.arange(len(label_str)+1) - 0.5, minor=True)
        ax1.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax1.tick_params(which="minor", bottom=False, left=False)
        ax1.set_title('Mask labels transition from t{} to t{}'.format(k, k+1), fontsize=12)


        ax2 = plt.subplot(grid[k, 1])
        ax2.imshow(norm_pred_1, aspect='equal', cmap="YlGn")
        ax2.set_xticks(np.arange(len(label_str)))
        ax2.set_yticks(np.arange(len(label_str)))
        ax2.set_xticklabels(label_str)
        ax2.set_yticklabels(label_str)
        ax2.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        for edge, spine in ax2.spines.items():
            spine.set_visible(False)

        ax2.set_xticks(np.arange(len(label_str)+1) - 0.5, minor=True)
        ax2.set_yticks(np.arange(len(label_str)+1) - 0.5, minor=True)
        ax2.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax2.tick_params(which="minor", bottom=False, left=False)
        ax2.set_title('Longitudinal predicted labels transition from t{} to t{}'.format(k, k+1), fontsize=12)


        ax3 = plt.subplot(grid[k, 2])
        ax3.imshow(norm_pred_2, aspect='equal', cmap="YlGn")
        ax3.set_xticks(np.arange(len(label_str)))
        ax3.set_yticks(np.arange(len(label_str)))
        ax3.set_xticklabels(label_str)
        ax3.set_yticklabels(label_str)
        ax3.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        for edge, spine in ax3.spines.items():
            spine.set_visible(False)

        ax3.set_xticks(np.arange(len(label_str)+1) - 0.5, minor=True)
        ax3.set_yticks(np.arange(len(label_str)+1) - 0.5, minor=True)
        ax3.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax3.tick_params(which="minor", bottom=False, left=False)
        ax3.set_title('U-Net predicted labels transition from t{} to t{}'.format(k, k+1), fontsize=12)
    
    plt.suptitle('Transition matrix of patient {}'.format(patient_id), fontsize=20)
    fig.savefig(os.path.join(ANALYSIS_DIR, 'transition_matrix_{}.png'.format(patient_id)), dpi=150)        


def transition_heatmap(patient_dict, patient_id, model_name_1, model_name_2, save_dir='inference_result'):
    pred_1_path = os.path.join(save_dir, patient_id, model_name_1)
    pred_1_list = [pred_mask for pred_mask in os.listdir(pred_1_path) if pred_mask.endswith('.gz')]
    pred_1_dir_list  = [os.path.join(pred_1_path, pred) for pred in pred_1_list]
    pred_2_path = os.path.join(save_dir, patient_id, model_name_2)
    pred_2_list = [pred_mask for pred_mask in os.listdir(pred_2_path) if pred_mask.endswith('.gz')]
    pred_2_dir_list  = [os.path.join(pred_2_path, pred) for pred in pred_2_list]
    time_dict = patient_dict[patient_id]
    time_dict = sorted(time_dict.items(), key=lambda item:item[0]) # sort according to time
    time_points = len(time_dict)
   

    fig = plt.figure(figsize=(15, 5 * time_points))
    grid = plt.GridSpec(time_points, 3, wspace=0.4, hspace=0.3)
    
    for k in range(time_points-1):
        mask_pre = sitk.GetArrayFromImage(sitk.ReadImage(time_dict[k][1]['combined_fast']))
        predicted_mask_1_pre = sitk.GetArrayFromImage(sitk.ReadImage(pred_1_dir_list[k]))
        predicted_mask_2_pre = sitk.GetArrayFromImage(sitk.ReadImage(pred_2_dir_list[k]))
        mask_later = sitk.GetArrayFromImage(sitk.ReadImage(time_dict[k+1][1]['combined_fast']))
        predicted_mask_1_later = sitk.GetArrayFromImage(sitk.ReadImage(pred_1_dir_list[k+1]))
        predicted_mask_2_later = sitk.GetArrayFromImage(sitk.ReadImage(pred_2_dir_list[k+1]))

        if k == 0:
            loc = np.where(mask_pre[0]==4)
            num_tumor = len(loc[0])
            if num_tumor > 0:
                idx = np.random.choice(range(num_tumor), 1, replace=False)
                c_z = int(loc[0][idx])
            else:
                c_z = mask_pre.shape[-3]//2

        mask_pre = mask_pre[c_z]
        predicted_mask_1_pre = predicted_mask_1_pre[c_z]
        predicted_mask_2_pre = predicted_mask_2_pre[c_z]
        mask_later = mask_later[c_z]
        predicted_mask_1_later = predicted_mask_1_later[c_z]
        predicted_mask_2_later = predicted_mask_2_later[c_z]

        mask_transition_heatmap = np.zeros_like(mask_pre)
        pred_mask_1_transition_heatmap = np.zeros_like(predicted_mask_1_pre)
        pred_mask_2_transition_heatmap = np.zeros_like(predicted_mask_2_pre)
        mask_transition_heatmap[mask_pre != mask_later] = 1
        pred_mask_1_transition_heatmap[predicted_mask_1_pre != predicted_mask_1_later] = 1 
        pred_mask_2_transition_heatmap[predicted_mask_2_pre != predicted_mask_2_later] = 1 

        ax1 = plt.subplot(grid[k, 0])
        ax1.imshow(mask_transition_heatmap)
        ax1.set_title('Mask labels transition from t{} to t{}'.format(k, k+1), fontsize=12)
        ax1.axis('off')

        ax2 = plt.subplot(grid[k, 1])
        ax2.imshow(pred_mask_1_transition_heatmap)
        ax2.set_title('longitudinal Predicted labels transition from t{} to t{}'.format(k, k+1), fontsize=12)
        ax2.axis('off')

        ax3 = plt.subplot(grid[k, 2])
        ax3.imshow(pred_mask_2_transition_heatmap)
        ax3.set_title('UNet Predicted labels transition from t{} to t{}'.format(k, k+1), fontsize=12)
        ax3.axis('off')

    plt.suptitle('Transition heatmap of patient {}'.format(patient_id), fontsize=20)
    fig.savefig(os.path.join(ANALYSIS_DIR, 'transition_heatmap_{}.png'.format(patient_id)), dpi=150) 

    
if __name__ == '__main__':
    
    labels = [0, 1, 2, 3, 4]
    data_class = data_split()
    train, val, test = data_construction(data_class)
    test_dict = time_parser(test)
    model_name_1 = 'CenterNormalBiLSTM1layer-unet-p64-4x3-halfpretrain'
    model_name_2 = 'UNet-p64-b4-newdata-oriinput'

    longitudinal_plot(MODEL_1, MODEL_2)
    transition_matrix(test_dict, PATIENT, MODEL_1, MODEL_2)
    transition_heatmap(test_dict, PATIENT, MODEL_1, MODEL_2)
  
    
    





