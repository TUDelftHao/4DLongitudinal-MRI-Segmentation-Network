import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
from data_prepara import data_split, data_construction, data_loader, time_parser
import SimpleITK as sitk
import json
from utils import load_config
from metrics import AverageSurfaceDist, Hausdorf


LONGITUDINAL = 'longitudinal_plot'
PATIENT = 'EGD-0505'
DATA = 'longitudinal.csv'
ANALYSIS_DIR = 'analysis'
INFERENCE_DIR = 'inference_result'

# columns=['date', 'GT CSF', 'GT GM', 'GT WM', 'GT TM', 'predicted CSF', 'predicted GM', 'predicted WM', 'predicted TM'])


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

def plot(model_1, model_2):
    data_1 = norm(model_1)
    data_2 = norm(model_2)

    modalities = ['GM', 'WM']
    date = data_1['date'].dropna().to_list()

    # x = [datetime.strptime(str(d), '%Y%m%d').date() for d in date]
    x = list(str(d) for d in date)
    # x = np.arange(len(date))
    plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)

    for i, modality in enumerate(modalities):
        GT = data_1['GT {}'.format(modality)].to_list()[:-2]
        pred_1 = data_1['predicted {}'.format(modality)].to_list()[:-2]
        pred_2 = data_2['predicted {}'.format(modality)].to_list()[:-2]

        ax1 = plt.subplot(grid[i, 0:2])

        ax1.plot(x, GT, 'k', lw=2, label='Mask')
        ax1.plot(x, pred_1, 'b', lw=1, label='{}'.format('Longitudinal'))
        ax1.plot(x, pred_2, 'r', lw=1, label='{}'.format('3D DC U-Net'))

        ax1.set_ylim(-0.5, 1.5)
        ax1.set_title('{}'.format(modality))
        ax1.set_xticklabels([])
        ax1.set_ylabel('Normalized volume')
        ax1.legend(loc='upper right', fontsize=10)

        ax2 = plt.subplot(grid[i, 2])
        
        ax2.errorbar(0, data_1.loc['mean', 'GT {}'.format(modality)], yerr=data_1.loc['std', 'GT {}'.format(modality)], fmt="o", color="black", elinewidth=2, capsize=4)
        ax2.text(len(x)+2.4, 0.9, np.round(data_1.loc['std', 'GT {}'.format(modality)], 3), transform=ax1.get_xaxis_transform(),horizontalalignment='center', size='x-small', color="black", fontsize=12)
        ax2.errorbar(1, data_1.loc['mean', 'predicted {}'.format(modality)], yerr=data_1.loc['std', 'predicted {}'.format(modality)], fmt="o", color="blue", elinewidth=2, capsize=4)
        ax2.text(len(x)+3.9, 0.9, np.round(data_1.loc['std', 'predicted {}'.format(modality)], 3), transform=ax1.get_xaxis_transform(),horizontalalignment='center', size='x-small', color="blue", fontsize=12)
        ax2.errorbar(2, data_2.loc['mean', 'predicted {}'.format(modality)], yerr=data_2.loc['std', 'predicted {}'.format(modality)], fmt="o", color="red", elinewidth=2, capsize=4)
        ax2.text(len(x)+5.4, 0.9, np.round(data_2.loc['std', 'predicted {}'.format(modality)], 3), transform=ax1.get_xaxis_transform(),horizontalalignment='center', size='x-small', color="red", fontsize=12)


        ax2.set_ylim(-0.3, 1.3)
        ax2.set_xlim(-0.5, 2.5)
        ax2.set_xticklabels([])
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.set_title('{}'.format(modality))
        ax2.set_ylabel('Value')

    ax1.set_xticklabels(x, rotation=30)
    ax1.set_xlabel('Date', fontsize=15)
    ax2.set_xticklabels(['', 'Mask', 'Longitudinal', '3D DC U-Net'], fontsize=10)
    ax2.set_xlabel('Models', fontsize=15)
    plt.suptitle('Patient' + PATIENT, fontsize=20)

    plt.savefig(os.path.join(ANALYSIS_DIR, 'DC_logitudinal_{}.png'.format(PATIENT)), dpi=150)



def longitudinal_HD(patient_dict, patient_id, model_name_1, model_name_2, save_dir=INFERENCE_DIR):
    pred_1_path = os.path.join(save_dir, patient_id, model_name_1)
    pred_1_list = [pred_mask for pred_mask in os.listdir(pred_1_path) if pred_mask.endswith('.gz')]
    pred_1_dir_list  = [os.path.join(pred_1_path, pred) for pred in pred_1_list]
    pred_2_path = os.path.join(save_dir, patient_id, model_name_2)
    pred_2_list = [pred_mask for pred_mask in os.listdir(pred_2_path) if pred_mask.endswith('.gz')]
    pred_2_dir_list  = [os.path.join(pred_2_path, pred) for pred in pred_2_list]
    time_dict = patient_dict[patient_id]
    time_dict = sorted(time_dict.items(), key=lambda item:item[0]) # sort according to time
    date = [time[0] for time in time_dict]
    time_points = len(time_dict)

    GM_VARY_MASK = []
    GM_VARY_PRED1 = []
    GM_VARY_PRED2 = []
    WM_VARY_MASK = []
    WM_VARY_PRED1 = []
    WM_VARY_PRED2 = []
    CSF_VARY_MASK = []
    CSF_VARY_PRED1 = []
    CSF_VARY_PRED2 = []
    

    for k in range(time_points-1):
        mask_pre = sitk.GetArrayFromImage(sitk.ReadImage(time_dict[k][1]['combined_fast']))
        predicted_mask_1_pre = sitk.GetArrayFromImage(sitk.ReadImage(pred_1_dir_list[k]))
        predicted_mask_2_pre = sitk.GetArrayFromImage(sitk.ReadImage(pred_2_dir_list[k]))
        mask_later = sitk.GetArrayFromImage(sitk.ReadImage(time_dict[k+1][1]['combined_fast']))
        predicted_mask_1_later = sitk.GetArrayFromImage(sitk.ReadImage(pred_1_dir_list[k+1]))
        predicted_mask_2_later = sitk.GetArrayFromImage(sitk.ReadImage(pred_2_dir_list[k+1]))

        ASD_mask = Hausdorf(mask_pre, mask_later)
        ASD_pred1 = Hausdorf(predicted_mask_1_pre, predicted_mask_1_later)
        ASD_pred2 = Hausdorf(predicted_mask_2_pre, predicted_mask_2_later)

        GM_VARY_MASK.append(ASD_mask['gm'])
        GM_VARY_PRED1.append(ASD_pred1['gm'])
        GM_VARY_PRED2.append(ASD_pred2['gm'])
        WM_VARY_MASK.append(ASD_mask['wm'])
        WM_VARY_PRED1.append(ASD_pred1['wm'])
        WM_VARY_PRED2.append(ASD_pred2['wm'])
        CSF_VARY_MASK.append(ASD_mask['csf'])
        CSF_VARY_PRED1.append(ASD_pred1['csf'])
        CSF_VARY_PRED2.append(ASD_pred2['csf'])

    x = np.arange(len(date)-1)
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(311)
    ax1.plot(x, GM_VARY_MASK, 'r-', label='mask')
    ax1.plot(x, GM_VARY_PRED1, 'y-', label='{}'.format(model_name_1))
    ax1.plot(x, GM_VARY_PRED2, 'g-', label='{}'.format(model_name_2))
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_title('GM')
    # ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(312)
    ax2.plot(x, WM_VARY_MASK, 'r-', label='mask')
    ax2.plot(x, WM_VARY_PRED1, 'y-', label='{}'.format(model_name_1))
    ax2.plot(x, WM_VARY_PRED2, 'g-', label='{}'.format(model_name_2))
    ax2.legend()
    ax2.set_xticks([])
    ax2.set_title('WM')
    # ax2.set_ylim(0, 1)

    ax3 = fig.add_subplot(313)
    ax3.plot(x, CSF_VARY_MASK, 'r-', label='mask')
    ax3.plot(x, CSF_VARY_PRED1, 'y-', label='{}'.format(model_name_1))
    ax3.plot(x, CSF_VARY_PRED2, 'g-', label='{}'.format(model_name_2))
    ax3.legend()
    ax3.set_title('CSF')
    # ax3.set_ylim(0, 1)

    plt.savefig(os.path.join(ANALYSIS_DIR, 'HD_longitudinal_{}.png'.format(patient_id)))

def avg_transition_matrix(patient_dict, model_name_1, model_name_2, save_dir=INFERENCE_DIR):
    # id_list = [key for key in test_dict.keys()]
    id_list = ['EGD-0125', 'EGD-0265', 'EGD-0505']
    labels = [0, 1, 2, 3, 4]
    label_str = ['BG', 'CSF', 'GM', 'WM', 'TM']
    fig = plt.figure(figsize=(15, 5 * len(id_list)))
    grid = plt.GridSpec(len(id_list), 3, wspace=0.5, hspace=0.3)

    for m, patient_id in enumerate(id_list):
        pred_1_path = os.path.join(save_dir, patient_id, model_name_1)
        pred_1_list = [pred_mask for pred_mask in os.listdir(pred_1_path) if pred_mask.endswith('.gz')]
        pred_1_dir_list  = [os.path.join(pred_1_path, pred) for pred in pred_1_list]
        pred_2_path = os.path.join(save_dir, patient_id, model_name_2)
        pred_2_list = [pred_mask for pred_mask in os.listdir(pred_2_path) if pred_mask.endswith('.gz')]
        pred_2_dir_list  = [os.path.join(pred_2_path, pred) for pred in pred_2_list]
        time_dict = patient_dict[patient_id]
        time_dict = sorted(time_dict.items(), key=lambda item:item[0]) # sort according to time
        time_points = len(time_dict)

        corr_mask_sum = np.zeros((len(labels), len(labels)))
        corr_pred_1_sum = np.zeros((len(labels), len(labels)))
        corr_pred_2_sum = np.zeros((len(labels), len(labels)))

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

            corr_mask_sum += np.round(norm_mask.to_numpy(), 3)
            corr_pred_1_sum += np.round(norm_pred_1.to_numpy(), 3)
            corr_pred_2_sum += np.round(norm_pred_2.to_numpy(), 3)

        df_mask_sum = pd.DataFrame(corr_mask_sum / (time_points-1), columns=label_str, index=label_str)
        df_pred_1_sum = pd.DataFrame(corr_pred_1_sum / (time_points-1), columns=label_str, index=label_str)
        df_pred_2_sum = pd.DataFrame(corr_pred_2_sum / (time_points-1), columns=label_str, index=label_str)

        ax1 = plt.subplot(grid[m, 0])
        sns.heatmap(df_mask_sum, vmin=0, vmax=1, annot=True, square=True, cmap="YlGn", fmt ='.0%', cbar=False)
        ax1.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        ax2 = plt.subplot(grid[m, 1])
        sns.heatmap(df_pred_1_sum, vmin=0, vmax=1, annot=True, square=True, cmap="YlGn", fmt ='.0%', cbar=False)
        ax2.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax2.set_title('Patient {}'.format(patient_id), fontsize=15)
        
        ax3 = plt.subplot(grid[m, 2])
        sns.heatmap(df_pred_2_sum, vmin=0, vmax=1, annot=True, square=True, cmap="YlGn", fmt ='.0%', cbar=False)
        ax3.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        
    ax1.set_xlabel('Mask', fontsize=20)
    ax2.set_xlabel('longitudinal model', fontsize=20)
    ax3.set_xlabel('CNN model', fontsize=20)

    fig.savefig(os.path.join(ANALYSIS_DIR, 'avg_transition_matrix_DC_2.png'), dpi=150)   
    # 
    # return df_mask_sum, df_pred_1_sum, df_pred_2_sum  
    

def transition_matrix(patient_dict, patient_id, model_name_1, model_name_2, save_dir=INFERENCE_DIR):
    # df_mask_sum, df_pred_1_sum, df_pred_2_sum = avg_transition_matrix(patient_dict, patient_id, model_name_1, model_name_2)

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

    fig = plt.figure(figsize=(15, 9))
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    grid = plt.GridSpec(3, 3, wspace=0.3, hspace=0.3)

    mask_trans = np.zeros((3, time_points-1))
    corr_pred_1_trans = np.zeros((3, time_points-1))
    corr_pred_2_trans = np.zeros((3, time_points-1))
    
    
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

        mask_trans[0, k] = norm_mask.to_numpy()[1,1]
        mask_trans[1, k] = norm_mask.to_numpy()[2,2]
        mask_trans[2, k] = norm_mask.to_numpy()[3,3]
        corr_pred_1_trans[0, k] = norm_pred_1.to_numpy()[1,1]
        corr_pred_1_trans[1, k] = norm_pred_1.to_numpy()[2,2]
        corr_pred_1_trans[2, k] = norm_pred_1.to_numpy()[3,3]
        corr_pred_2_trans[0, k] = norm_pred_2.to_numpy()[1,1]
        corr_pred_2_trans[1, k] = norm_pred_2.to_numpy()[2,2]
        corr_pred_2_trans[2, k] = norm_pred_2.to_numpy()[3,3]

        '''
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
        ax2.set_title('Predicted labels transition from t{} to t{}'.format(k, k+1), fontsize=12)

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
        ax3.set_title('Predicted labels transition from t{} to t{}'.format(k, k+1), fontsize=12)
        '''
   
    ax1 = plt.subplot(grid[0, 0:2])
    ax1.plot(np.arange(time_points-1), mask_trans[0], 'k-', label='mask')
    ax1.plot(np.arange(time_points-1), corr_pred_1_trans[0], 'b-', label='longitudinal')
    ax1.plot(np.arange(time_points-1), corr_pred_2_trans[0], 'r-', label='CNN')
    ax1.legend()
    ax1.set_ylim(0.3, 1)
    ax1.set_title('CSF')
 
    ax2 = plt.subplot(grid[1, 0:2])
    ax2.plot(np.arange(time_points-1), mask_trans[1], 'k-', label='mask')
    ax2.plot(np.arange(time_points-1), corr_pred_1_trans[1], 'b-', label='longitudinal')
    ax2.plot(np.arange(time_points-1), corr_pred_2_trans[1], 'r-', label='CNN')
    ax2.legend()
    ax2.set_ylim(0.3, 1)
    ax2.set_title('GM')
   
    ax3 = plt.subplot(grid[2, 0:2])
    ax3.plot(np.arange(time_points-1), mask_trans[2], 'k-', label='mask')
    ax3.plot(np.arange(time_points-1), corr_pred_1_trans[2], 'b-', label='longitudinal')
    ax3.plot(np.arange(time_points-1), corr_pred_2_trans[2], 'r-', label='CNN')
    ax3.legend()
    ax3.set_ylim(0.3, 1)
    ax3.set_title('WM')

    ax4 = plt.subplot(grid[0, 2])
    ax4.errorbar(0, np.mean(mask_trans[0]), yerr=np.std(mask_trans[0]), fmt="o", color="black", elinewidth=2, capsize=4)
    ax4.errorbar(1, np.mean(corr_pred_1_trans[0]), yerr=np.std(corr_pred_1_trans[0]), fmt="o", color="blue", elinewidth=2, capsize=4)
    ax4.errorbar(2, np.mean(corr_pred_2_trans[0]), yerr=np.std(corr_pred_2_trans[0]), fmt="o", color="red", elinewidth=2, capsize=4)
    ax4.text(0, 0.9, np.round(np.mean(mask_trans[0]), 3), transform=ax4.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="black", fontsize=12)
    ax4.text(0, 0.1, np.round(np.std(mask_trans[0]), 3), transform=ax4.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="black", fontsize=12)
    ax4.text(1, 0.9, np.round(np.mean(corr_pred_1_trans[0]), 3), transform=ax4.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="blue", fontsize=12)
    ax4.text(1, 0.1, np.round(np.std(corr_pred_1_trans[0]), 3), transform=ax4.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="blue", fontsize=12)
    ax4.text(2, 0.9, np.round(np.mean(corr_pred_2_trans[0]), 3), transform=ax4.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="red", fontsize=12)
    ax4.text(2, 0.1, np.round(np.std(corr_pred_2_trans[0]), 3), transform=ax4.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="red", fontsize=12)

    ax4.set_ylim(-0.3, 1.3)
    ax4.set_xlim(-0.5, 2.5)
    ax4.set_xticklabels([])
    ax4.xaxis.set_major_locator(MultipleLocator(1))
    ax4.set_xticklabels(['', 'Mask', 'Longitudinal', 'CNN'], fontsize=10)

    ax5 = plt.subplot(grid[1, 2])
    ax5.errorbar(0, np.mean(mask_trans[1]), yerr=np.std(mask_trans[1]), fmt="o", color="black", elinewidth=2, capsize=4)
    ax5.errorbar(1, np.mean(corr_pred_1_trans[1]), yerr=np.std(corr_pred_1_trans[1]), fmt="o", color="blue", elinewidth=2, capsize=4)
    ax5.errorbar(2, np.mean(corr_pred_2_trans[1]), yerr=np.std(corr_pred_2_trans[1]), fmt="o", color="red", elinewidth=2, capsize=4)
    ax5.text(0, 0.9, np.round(np.mean(mask_trans[1]), 3), transform=ax5.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="black", fontsize=12)
    ax5.text(0, 0.1, np.round(np.std(mask_trans[1]), 3), transform=ax5.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="black", fontsize=12)
    ax5.text(1, 0.9, np.round(np.mean(corr_pred_1_trans[1]), 3), transform=ax5.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="blue", fontsize=12)
    ax5.text(1, 0.1, np.round(np.std(corr_pred_1_trans[1]), 3), transform=ax5.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="blue", fontsize=12)
    ax5.text(2, 0.9, np.round(np.mean(corr_pred_2_trans[1]), 3), transform=ax5.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="red", fontsize=12)
    ax5.text(2, 0.1, np.round(np.std(corr_pred_2_trans[1]), 3), transform=ax5.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="red", fontsize=12)
    ax5.set_ylim(-0.3, 1.3)
    ax5.set_xlim(-0.5, 2.5)
    ax5.set_xticklabels([])
    ax5.xaxis.set_major_locator(MultipleLocator(1))
    ax5.set_xticklabels(['', 'Mask', 'Longitudinal', 'CNN'], fontsize=10)

    ax6 = plt.subplot(grid[2, 2])
    ax6.errorbar(0, np.mean(mask_trans[2]), yerr=np.std(mask_trans[2]), fmt="o", color="black", elinewidth=2, capsize=4)
    ax6.errorbar(1, np.mean(corr_pred_1_trans[2]), yerr=np.std(corr_pred_1_trans[2]), fmt="o", color="blue", elinewidth=2, capsize=4)
    ax6.errorbar(2, np.mean(corr_pred_2_trans[2]), yerr=np.std(corr_pred_2_trans[2]), fmt="o", color="red", elinewidth=2, capsize=4)
    ax6.text(0, 0.9, np.round(np.mean(mask_trans[2]), 3), transform=ax6.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="black", fontsize=12)
    ax6.text(0, 0.1, np.round(np.std(mask_trans[2]), 3), transform=ax6.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="black", fontsize=12)
    ax6.text(1, 0.9, np.round(np.mean(corr_pred_1_trans[2]), 3), transform=ax6.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="blue", fontsize=12)
    ax6.text(1, 0.1, np.round(np.std(corr_pred_1_trans[2]), 3), transform=ax6.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="blue", fontsize=12)
    ax6.text(2, 0.9, np.round(np.mean(corr_pred_2_trans[2]), 3), transform=ax6.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="red", fontsize=12)
    ax6.text(2, 0.1, np.round(np.std(corr_pred_2_trans[2]), 3), transform=ax6.get_xaxis_transform(), horizontalalignment='center', size='x-small', color="red", fontsize=12)
    ax6.set_ylim(-0.3, 1.3)
    ax6.set_xlim(-0.5, 2.5)
    ax6.set_xticklabels([])
    ax6.xaxis.set_major_locator(MultipleLocator(1))
    ax6.set_xticklabels(['', 'Mask', 'Longitudinal', 'CNN'], fontsize=10)


    # plt.suptitle('Transition matrix of patient {}'.format(patient_id), fontsize=20)
    fig.savefig(os.path.join(ANALYSIS_DIR, 'transition_dev_{}.png'.format(patient_id)), dpi=150)        


def transition_heatmap(patient_dict, patient_id, model_name_1, model_name_2, save_dir=INFERENCE_DIR):
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
        # sns.heatmap(mask_transition_heatmap, vmin=0, vmax=4, xticklabels=False, yticklabels=False, square=True, cmap='YlGnBu_r', cbar=False)
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
    

def dice_boxplot(*args):
    labels = ['CSF', 'GM', 'WM', 'TM']

    CSF = []
    WM = []
    GM = []
    TM = []
    model_name = []
    fig = plt.figure(figsize=(20, 15))

    for i in args:
        path = os.path.join(INFERENCE_DIR, i, 'dice.json')
        dice_score = load_config(path)
        _CSF = dice_score['csf']
        _WM = dice_score['wm']
        _GM = dice_score['gm']
        _TM = dice_score['tm']
        
        model_name.extend([i for _ in range(len(_TM))]) 
        CSF.extend(_CSF)
        WM.extend(_WM)
        GM.extend(_GM)
        TM.extend(_TM)

    df = pd.DataFrame({'MODEL':model_name, 'CSF':CSF, 'GM':GM, 'WM':WM, 'TM':TM})

    dd = pd.melt(df, id_vars=['MODEL'], value_vars=labels, var_name='Targets')
    sns.boxplot(x='MODEL', y='value', data=dd, hue='Targets', whis=0.9, palette='Set3', showmeans=True)
    plt.ylim(0, 1)
    plt.legend(fontsize=15)
    # plt.xticks(np.arange(len(args)), ['3D U-Net', '3D Res U-Net', '3D DRes U-Net', '3D DC U-Net'], fontsize=20)
    plt.xticks(np.arange(len(args)), ['U-Net backbone', 'Res U-Net backbone', 'DC U-Net backbone'], fontsize=20)
    plt.xlabel('')
    plt.ylabel('Score', fontsize=20)
    # plt.title('Dice scores of CNN models', fontsize=25)
    plt.savefig('longitudinalcomp.png')


def distance_boxplot(dist, *args):

    CSF = []
    WM = []
    GM = []
    TM = []
    model_name = []
    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(1, 3, wspace=0.3, hspace=0.3)

    for i in args:
        path = os.path.join(INFERENCE_DIR, i, dist)
        dice_score = load_config(path)
        _CSF = dice_score['csf']
        _WM = dice_score['wm']
        _GM = dice_score['gm']
        _TM = dice_score['tm']
        
        model_name.extend([i for _ in range(len(_TM))]) 
        CSF.extend(_CSF)
        WM.extend(_WM)
        GM.extend(_GM)
        TM.extend(_TM)

    df = pd.DataFrame({'MODEL':model_name, 'CSF':CSF, 'GM':GM, 'WM':WM, 'TM':TM})

    ax1 = plt.subplot(grid[0, 0:2])
    var = ['CSF', 'GM', 'WM']
    dd1 = pd.melt(df, id_vars=['MODEL'], value_vars=var, var_name='Targets')
    sns.boxplot(x='MODEL', y='value', data=dd1, hue='Targets', whis=0.9, palette='Set3', showmeans=True, meanprops={"markerfacecolor":"green", "markeredgecolor":"green"})
    ax1.legend(fontsize=15)
    ax1.xaxis.label.set_visible(False)
    # ax1.set_xticklabels(['3D U-Net', '3D Res U-Net', '3D DRes U-Net', '3D DC U-Net'], fontsize=15, rotation=30)
    ax1.set_xticklabels(['U-Net backbone', 'Res U-Net backbone', 'DC U-Net backbone'], fontsize=15, rotation=15)
    ax1.set_ylabel('Distance/voxel', fontsize=15)
    ax1.set_title('Normal tissues', fontsize=20)

    ax2 = plt.subplot(grid[0, 2])
    dd2 = pd.melt(df, id_vars=['MODEL'], value_vars=['TM'], var_name='Targets')
    #fb9a99
    flatui = ["#fb9a99"]
    sns.set_palette(sns.color_palette(flatui))
    sns.boxplot(x='MODEL', y='value', data=dd2, hue='Targets', whis=0.9, showmeans=True, meanprops={"markerfacecolor":"green", "markeredgecolor":"green"})
    ax2.legend(fontsize=15)
    ax2.xaxis.label.set_visible(False)
    formatter = ticker.FormatStrFormatter('$%1.2f')
    ax2.xaxis.set_major_formatter(formatter)
    # ax2.set_xticklabels(['3D U-Net', '3D Res U-Net', '3D DRes U-Net', '3D DC U-Net'], fontsize=15, rotation=30)
    ax2.set_xticklabels(['U-Net backbone', 'Res U-Net backbone', 'DC U-Net backbone'], fontsize=15, rotation=15)
    ax2.set_ylabel('Distance/voxel', fontsize=15)
    ax2.set_title('Glioma', fontsize=20)

    # plt.suptitle('{} of CNN models'.format(dist.split('.')[0][:2]), fontsize=25)
    plt.savefig('4Dcomp{}_boxplot.png'.format(dist.split('.')[0][:2]))

    
if __name__ == '__main__':

    labels = [0, 1, 2, 3, 4]
    data_class = data_split()
    train, val, test = data_construction(data_class)
    test_dict = time_parser(test)
    patient_id = [key for key in test_dict.keys()]
    model_name_1 = 'BiDirectCenterNormalLSTM1layer-dcunet-p64-4x3'
    model_name_2 = 'direct-UNet-p64-newdata-oriinput'

    # plot(model_name_1, model_name_2)
    avg_transition_matrix(test_dict, model_name_1, model_name_2)
    # transition_matrix(test_dict, PATIENT, model_name_1, model_name_2)
    # longitudinal_HD(test_dict, PATIENT, model_name_1, model_name_2)
    # transition_heatmap(test_dict, PATIENT, model_name_1, model_name_2)

    # dice1 = 'dice_UNet-p64-b4-newdata-oriinput'
    # dice1 = 'dice_ResUNet-p64-b4-newdata-oriinput'
    # dice3 = 'dice_DResUNet-p64-b4-newdata-oriinput'
    # dice1 = 'dice_direct-UNet-p64-newdata-oriinput'
    # dice1 = 'dice_BackBiLSTM1layer-p64-4x3-pretrained-newLSTM'
    # dice1 = 'dice_ShortcutLSTM-1-layer-1-shortcut-p64-b4-2x3-IN-decay'
    dice1 = 'dice_CenterNormalBiLSTM1layer-unet-p64-4x3-halfpretrain'
    dice2 = 'dice_BiResCenterNormalLSTM1layer-resunet-p64-4x3'
    dice3 = 'dice_BiDirectCenterNormalLSTM1layer-dcunet-p64-4x3'
    # dice_boxplot(dice1, dice2, dice3)
    
    # ASD1 = 'HD95_UNet-p64-b4-newdata-oriinput'
    # ASD1 = 'ASD_ResUNet-p64-b4-newdata-oriinput'
    # ASD3 = 'HD95_DResUNet-p64-b4-newdata-oriinput'
    # ASD1 = 'ASD_direct-UNet-p64-newdata-oriinput'
    # ASD1 = 'HD95_BackBiLSTM1layer-p64-4x3-pretrained-newLSTM'
    # ASD2 = 'ASD_BiDirectCenterNormalLSTM1layer-dcunet-p64-4x3'
    # ASD3 = 'HD95_BackBiConvLSTM1layer-p64-b4-2x3-IN'
    # ASD2 = 'ASD_BiResCenterNormalLSTM1layer-resunet-p64-4x3'
    ASD1 = 'HD95_CenterNormalBiLSTM1layer-unet-p64-4x3-halfpretrain'
    ASD2 = 'HD95_BiResCenterNormalLSTM1layer-resunet-p64-4x3'
    ASD3 = 'HD95_BiDirectCenterNormalLSTM1layer-dcunet-p64-4x3'

    # distance_boxplot('HD95.json', ASD1, ASD2, ASD3)
    
  







