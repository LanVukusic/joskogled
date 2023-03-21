# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2


class BreastCancerDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = np.genfromtxt(data_path, delimiter=",", dtype=str)

    def __getitem__(self, index):
        print("getdata monaa")
        patient_id = self.data[index, 0]
        l_cc = self.data[index, 1].replace("rakave", "").replace("zdrave", "")
        l_mlo = self.data[index, 2].replace("rakave", "").replace("zdrave", "")
        r_cc = self.data[index, 3].replace("rakave", "").replace("zdrave", "")
        r_mlo = self.data[index, 4].replace("rakave", "").replace("zdrave", "")
        years_to_cancer = self.data[index, 5]

        print(index)
        # read data to torch tensors
        l_cc_image = cv2.imread(
            "/home/lan/Desktop/joskogled/data/processed_data" + l_cc,
            cv2.IMREAD_UNCHANGED,
        )
        l_mlo_image = cv2.imread(
            "/home/lan/Desktop/joskogled/data/processed_data" + l_mlo,
            cv2.IMREAD_UNCHANGED,
        )
        r_cc_image = cv2.imread(
            "/home/lan/Desktop/joskogled/data/processed_data" + r_cc,
            cv2.IMREAD_UNCHANGED,
        )
        r_mlo_image = cv2.imread(
            "/home/lan/Desktop/joskogled/data/processed_data" + r_mlo,
            cv2.IMREAD_UNCHANGED,
        )

        print(l_cc_image.shape)
        print(l_cc)

        # convert to torch tensors
        # view as 1 channel image
        # expand dims to add channel dimension
        l_cc_image = torch.from_numpy(l_cc_image.astype(np.float16))
        l_mlo_image = torch.from_numpy(l_mlo_image.astype(np.float16))
        r_cc_image = torch.from_numpy(r_cc_image.astype(np.float16))
        r_mlo_image = torch.from_numpy(r_mlo_image.astype(np.float16))

        l_cc_image = np.expand_dims(l_cc_image, axis=0)
        l_mlo_image = np.expand_dims(l_mlo_image, axis=0)
        r_cc_image = np.expand_dims(r_cc_image, axis=0)
        r_mlo_image = np.expand_dims(r_mlo_image, axis=0)

        return (
            patient_id,
            l_cc_image,
            l_mlo_image,
            r_cc_image,
            r_mlo_image,
            years_to_cancer,
        )

    def __len__(self):
        return len(self.data)


def get_dataloader(data_path, batch_size=16, shuffle=False):
    dataset = BreastCancerDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
