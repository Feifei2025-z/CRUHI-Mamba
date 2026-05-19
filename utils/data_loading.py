# Data Loading and Processing
import os
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
from sklearn import preprocessing
import xarray as xr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_set_name, data_path='.data'):    # Loading data
    if data_set_name == 'cr_005':
        # Training-DN350
        labels = xr.open_dataset(os.path.join(data_path, 'cr_005', 'Ground_truth_005.nc'))['labels'].values
        data = xr.open_dataset(os.path.join(data_path, 'cr_005', 'transect_005.nc'))['cube'].values.astype(np.float32)

        # # Training-PCA
        # labels = xr.open_dataset(os.path.join(data_path, 'cr_005', 'Ground_truth_005.nc'))['labels'].values
        # data = xr.open_dataset(os.path.join(data_path, 'cr_005', 'transect_005_PCA100.nc'))['cube'].values

    elif data_set_name == 'cr_006':
        # Training-DN350
        labels = xr.open_dataset(os.path.join(data_path, 'cr_006', 'Ground_truth_006.nc'))['labels'].values
        data = xr.open_dataset(os.path.join(data_path, 'cr_006', 'transect_006.nc'))['cube'].values.astype(np.float32)

        # # Training-PCA
        # labels = xr.open_dataset(os.path.join(data_path, 'cr_006', 'Ground_truth_006.nc'))['labels'].values
        # data = xr.open_dataset(os.path.join(data_path, 'cr_006', 'transect_006_PCA100.nc'))['cube'].values

    elif data_set_name == 'cr_019':
        # Training-DN350
        labels = xr.open_dataset(os.path.join(data_path, 'cr_019', 'Ground_truth_019.nc'))['labels'].values
        data = xr.open_dataset(os.path.join(data_path, 'cr_019', 'transect_019.nc'))['cube'].values.astype(np.float32)

        # # Training-PCA
        # labels = xr.open_dataset(os.path.join(data_path, 'cr_019', 'Ground_truth_019.nc'))['labels'].values
        # data = xr.open_dataset(os.path.join(data_path, 'cr_019', 'transect_019_PCA100.nc'))['cube'].values

    return data, labels


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, val_label_index_dict, test_label_index_dict = {}, {}, {}, {}
    all_label_index_list, train_label_index_list, val_label_index_list, test_label_index_list = [], [], [], []

    for cls in range(class_count):
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)     # at least 3 samples per class
            val_index_flag = max(int(ratio_list[1] * len(cls_index)), 1)
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15
            val_index_flag = num_list[1]

        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:][val_index_flag:])
        val_label_index_dict[cls] = list(cls_index[train_index_flag:][:val_index_flag])

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, val_label_index_list, test_label_index_list, all_label_index_list


def generate_image_iter(data, uhi_h, uhi_w, label_reshape, index):
    def generate_label_map(num, uhi_w):
        num = np.array(num)
        idx_2d = np.zeros([num.shape[0], 2]).astype(int)
        idx_2d[:, 0] = num // uhi_w
        idx_2d[:, 1] = num % uhi_w
        label_map = np.zeros((uhi_h, uhi_w))
        for i in range(num.shape[0]):
            label_map[idx_2d[i, 0], idx_2d[i, 1]] = label_reshape[num[i]]
        return label_map.astype(int)

    # for data label
    train_labels = generate_label_map(index[0], uhi_w) - 1
    val_labels = generate_label_map(index[1], uhi_w) - 1
    test_labels = generate_label_map(index[2], uhi_w) - 1


    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_val = torch.from_numpy(val_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    return y_tensor_train, y_tensor_val, y_tensor_test
