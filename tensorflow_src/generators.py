from tensorflow import keras
import glob
import cv2
import numpy as np


def load_data(data_path, p, shuffle):
    data = np.genfromtxt(data_path, delimiter=",", dtype=str)
    if shuffle:
        np.random.shuffle(data)
    split_index = int(len(data) * p)
    return data[:split_index], data[split_index:]


def load_images(img_paths):
    images = {}
    image = np.zeros((0))
    for img_path in img_paths:
        for path in glob.glob(img_path + "/*.png"):
            key = path.split("/")[-1].split('.')[0].split('_')
            key = '_'.join([key[0], key[2], key[3]])
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32)
            image /= 255
            images[key] = image
    return images, image.shape


class Generator(keras.utils.Sequence):
    def __init__(self, data, images, batch_size, shuffle=True, transformation=None, map_labels=False):
        self.data = data
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformation = transformation
        self.map_labels = map_labels

        self.l_cc_images = []
        self.l_mlo_images = []
        self.r_cc_images = []
        self.r_mlo_images = []
        self.years_to_cancer = []
        
        self.ratio = 0.5

        self.__make_samples()
        self.on_epoch_end()

    def __make_samples(self):
        for d in self.data:
            patient_id, years_to_cancer = d[0], d[-1]
            l_cc_image = self.images['_'.join([patient_id, 'L', 'CC'])]
            l_mlo_image = self.images['_'.join([patient_id, 'L', 'MLO'])]
            r_cc_image = self.images['_'.join([patient_id, 'R', 'CC'])]
            r_mlo_image = self.images['_'.join([patient_id, 'R', 'MLO'])]
            years_to_cancer = int(years_to_cancer)
            if self.map_labels:
                if years_to_cancer == 100:
                    years_to_cancer = 0
                else:
                    years_to_cancer = 1
            self.l_cc_images.append(l_cc_image)
            self.l_mlo_images.append(l_mlo_image)
            self.r_cc_images.append(r_cc_image)
            self.r_mlo_images.append(r_mlo_image)
            self.years_to_cancer.append(years_to_cancer)

        self.l_cc_images = np.array(self.l_cc_images, dtype=np.float32)
        self.l_mlo_images = np.array(self.l_mlo_images, dtype=np.float32)
        self.r_cc_images = np.array(self.r_cc_images, dtype=np.float32)
        self.r_mlo_images = np.array(self.r_mlo_images, dtype=np.float32)
        self.years_to_cancer = np.array(self.years_to_cancer, dtype=np.float32)

        self.ratio = self.years_to_cancer.mean()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = (
            self.l_cc_images[indices],
            self.l_mlo_images[indices],
            self.r_cc_images[indices],
            self.r_mlo_images[indices]
        )
        if self.transformation is not None:
            images = [self.transformation(image) for image in images]
        y = self.years_to_cancer[indices]
        return images, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_class_weight(self):
        class_weight = {
            0: self.ratio,
            1: 1 - self.ratio
        }
        return class_weight
