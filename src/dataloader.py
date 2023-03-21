# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2


class BreastCancerDataset(Dataset):
    def __init__(self, data, img_path):
        self.data = data
        self.img_path = img_path

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
            self.img_path + l_cc,
            cv2.IMREAD_UNCHANGED,
        )
        l_mlo_image = cv2.imread(
            self.img_path + l_mlo,
            cv2.IMREAD_UNCHANGED,
        )
        r_cc_image = cv2.imread(
            self.img_path + r_cc,
            cv2.IMREAD_UNCHANGED,
        )
        r_mlo_image = cv2.imread(
            self.img_path + r_mlo,
            cv2.IMREAD_UNCHANGED,
        )

        print(l_cc_image.shape)
        print(l_cc)

        # convert to torch tensors
        l_cc_image = torch.from_numpy(l_cc_image.astype(np.float32))
        l_mlo_image = torch.from_numpy(l_mlo_image.astype(np.float32))
        r_cc_image = torch.from_numpy(r_cc_image.astype(np.float32))
        r_mlo_image = torch.from_numpy(r_mlo_image.astype(np.float32))

        # view as 1 channel image
        # expand dims to add channel dimension
        l_cc_image = np.expand_dims(l_cc_image, axis=0)
        l_mlo_image = np.expand_dims(l_mlo_image, axis=0)
        r_cc_image = np.expand_dims(r_cc_image, axis=0)
        r_mlo_image = np.expand_dims(r_mlo_image, axis=0)


        # change data type from str to tensor
        years_to_cancer = int(years_to_cancer)

        # for classification, map index 100 to 0, so that all possible values become (0, 1, 2, 3, 4)
        if years_to_cancer == 100:
            years_to_cancer = 0

        # convert to tensor
        years_to_cancer = torch.tensor(years_to_cancer)

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


def get_dataloader(data_path, img_path, batch_size=16, shuffle=False, p=0.7):
    # read data
    data = np.genfromtxt(data_path, delimiter=",", dtype=str)

    # shuffle data in place
    rng = np.random.default_rng(seed=3)
    rng.shuffle(data, axis=0)

    # clac dividing index and divide the dataset
    div_index = int(p*data.shape[0])
    dataset_train = BreastCancerDataset(data[:div_index], img_path)
    dataset_val = BreastCancerDataset(data[div_index:], img_path)

    # create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle)
    return dataloader_train, dataloader_val
