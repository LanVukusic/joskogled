# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
import glob

DEVICE=torch.device('cuda:0')

def load_images(img_path):
    images = {}
    for path in glob.glob(img_path + "/*.png"):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32))

        # view as 1 channel image
        # expand dims to add channel dimension
        image = torch.unsqueeze(image, 0).to(DEVICE)
        images[path.split("/")[-1]] = image
    return images

class BreastCancerDataset(Dataset):
    def __init__(self, data, images, upsample=0, transformation=None):
        self.data = data
        self.images = images
        self.shape = ()
        self.samples = []
        self.transformation = transformation

        path_map = lambda path: path.replace("rakave/", "").replace("zdrave/", "")

        for sample in data:
            patient_id = sample[0]
            years_to_cancer = int(sample    [5])

            # for classification, map index 100 to 0, so that all possible values become (0, 1, 2, 3, 4)
            if years_to_cancer == 100:
                years_to_cancer = 0
            #elif years_to_cancer != 2:
            #    continue
            #else:
            #    years_to_cancer = 1
            if 1 <= years_to_cancer <= 4:
                years_to_cancer = 1


            # convert to tensor
            years_to_cancer = torch.tensor(years_to_cancer).to(DEVICE)

            l_cc_image = self.images[path_map(sample[1])]
            l_mlo_image = self.images[path_map(sample[2])]
            r_cc_image = self.images[path_map(sample[3])]
            r_mlo_image = self.images[path_map(sample[4])]

            s = (patient_id, l_cc_image, l_mlo_image, r_cc_image, r_mlo_image, years_to_cancer)
            self.samples.append(s)
            if years_to_cancer > 0:
                for i in range(upsample):
                    self.samples.append(s)

        if len(self.samples) > 0:
            self.shape = self.samples[0][1].shape[1:]


    def __getitem__(self, index):
        # print("getdata monaa")
        
        s = self.samples[index]
        (patient_id, l_cc_image, l_mlo_image, r_cc_image, r_mlo_image, years_to_cancer) = s

        # Do transformations ...
        if self.transformation is not None:
            l_cc_image = self.transformation(l_cc_image)
            l_mlo_image = self.transformation(l_mlo_image)
            r_cc_image = self.transformation(r_cc_image)
            r_mlo_image = self.transformation(r_mlo_image)

        s = (patient_id, l_cc_image, l_mlo_image, r_cc_image, r_mlo_image, years_to_cancer)
        return s

    def __len__(self):
        return len(self.samples)


def get_dataloader(data_path, img_path, batch_size=16, shuffle=False, p=0.8, upsample=0, transformation=None):
    # read data
    data = np.genfromtxt(data_path, delimiter=",", dtype=str)

    # shuffle data in place
    rng = np.random.default_rng(seed=3)
    rng.shuffle(data, axis=0)

    # load images
    images = load_images(img_path)

    # clac dividing index and divide the dataset
    div_index = int(p*data.shape[0])
    dataset_train = BreastCancerDataset(data[:div_index], images, upsample, transformation)
    dataset_val = BreastCancerDataset(data[div_index:], images)

    # create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=shuffle)
    return dataloader_train, dataloader_val, dataset_train.shape


class FinalDataset(Dataset):
    def __init__(self, data, images):
        self.data = data
        self.images = images
        self.shape = ()
        self.samples = []

        image_map = lambda id, side, view, : id + "_" + side + "_" + view

        for patient_id in data:

            l_cc_image = self.images[image_map(patient_id, 'L', 'CC')]
            l_mlo_image = self.images[image_map(patient_id, 'L', 'MLO')]
            r_cc_image = self.images[image_map(patient_id, 'R', 'CC')]
            r_mlo_image = self.images[image_map(patient_id, 'R', 'MLO')]

            s = (patient_id, l_cc_image, l_mlo_image, r_cc_image, r_mlo_image)
            self.samples.append(s)

    def __getitem__(self, index):
        s = self.samples[index]
        return s

    def __len__(self):
        return len(self.samples)


def get_final_dataloader(data_path, img_path, batch_size=16):
    # read data
    data = np.genfromtxt(data_path, delimiter="\n", dtype=str)

    # load images
    images = {}
    for path in glob.glob(img_path + "/*.png"):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32))

        # view as 1 channel image
        # expand dims to add channel dimension
        image = torch.unsqueeze(image, 0).to(DEVICE)
        key = path.split("/")[-1].split('.')[0].split('_')
        key = '_'.join([key[0], key[2], key[3]])
        images[key] = image

    dataset = FinalDataset(data, images)

    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
