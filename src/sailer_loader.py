# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
import glob

# set device to GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert to torch tensors
    image = torch.from_numpy(image.astype(np.float32))

    return image


class BreastCancerDataset(Dataset):
    def __init__(self, data, image_root, transformation=None):
        self.transformation = transformation
        self.data = data
        self.image_root = image_root

        # # shuffle data in place
        rng = np.random.default_rng(seed=3)
        rng.shuffle(self.data, axis=0)

        print("data", len(self.data), self.data[0], self.data[1].shape)

        # split data into healthy and cancerous
        self.rakave = self.data[self.data[:, 5] == "1"]
        self.zdrave = self.data[self.data[:, 5] == "0"]

    def __getitem__(self, index):
        sample = None
        # patient_id, L_CC, L_MLO, R_CC, R_MLO, cancer

        # get sample type
        if index % 2 == 0:
            sample = self.rakave[index % len(self.rakave)]
        else:
            sample = self.zdrave[index % len(self.zdrave)]

        (
            patient_id,
            l_cc_image,
            l_mlo_image,
            r_cc_image,
            r_mlo_image,
            cancer,
        ) = sample

        # load images
        l_cc_image = load_image(self.image_root + l_cc_image)
        l_mlo_image = load_image(self.image_root + l_mlo_image)
        r_cc_image = load_image(self.image_root + r_cc_image)
        r_mlo_image = load_image(self.image_root + r_mlo_image)

        # add batch dimension and push to device
        l_cc_image = l_cc_image.unsqueeze(0).to(DEVICE)
        l_mlo_image = l_mlo_image.unsqueeze(0).to(DEVICE)
        r_cc_image = r_cc_image.unsqueeze(0).to(DEVICE)
        r_mlo_image = r_mlo_image.unsqueeze(0).to(DEVICE)

        # Do transformations ...
        if self.transformation is not None:
            l_cc_image = self.transformation(l_cc_image)
            l_mlo_image = self.transformation(l_mlo_image)
            r_cc_image = self.transformation(r_cc_image)
            r_mlo_image = self.transformation(r_mlo_image)

        # cancer to one hot tensor on device
        cancer = torch.tensor(cancer, dtype=torch.float32).to(DEVICE)

        return (patient_id, l_cc_image, l_mlo_image, r_cc_image, r_mlo_image, cancer)

    def __len__(self):
        return len(self.data)


def get_datasets(image_root, data_file, split=0.8, transform=None):
    # read data definition file
    data = np.genfromtxt(data_file, delimiter=",", dtype=str)
    data_len = len(data)
    split_index = int(data_len * split)

    # split data into train and test
    train_data = data[:split_index]
    test_data = data[split_index:]

    # create datasets
    train_dataset = BreastCancerDataset(train_data, image_root, transform)
    test_dataset = BreastCancerDataset(test_data, image_root, transform)

    # get first image to determine input dimensions
    sample = train_dataset[0]
    in_dims = sample[1].shape

    return train_dataset, test_dataset, in_dims
