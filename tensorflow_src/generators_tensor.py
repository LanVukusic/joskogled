import tensorflow as tf
from tensorflow import keras
import glob
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from PIL import Image


def load_data(data_path, p, shuffle):
    data = np.genfromtxt(data_path, delimiter=",", dtype=str)
    if shuffle:
        np.random.shuffle(data)
    split_index = int(len(data) * p)
    return data[:split_index], data[split_index:]


def load_images(img_paths):
    images = {}
    image = tf.zeros((0))
    for img_path in img_paths:
        paths = glob.glob(img_path + "/*.png")
        for path in tqdm(paths, total=len(paths)):
            key = path.split("/")[-1].split('.')[0]
            image = Image.open(path)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            image = tf.expand_dims(image, axis=-1)
            image /= 255
            images[key] = image
    return images, image.shape


class Generator(keras.utils.Sequence):
    def __init__(
            self,
            data,
            images,
            batch_size,
            shuffle=True,
            transformation=None,
            map_labels=False,
            label_smoothing=0.0
            ):
        self.data = data
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformation = transformation
        self.map_labels = map_labels
        self.label_smoothing = label_smoothing

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

            # retrieve images for curent patient_id
            l_cc_image = self.images['_'.join([patient_id, 'L', 'CC'])]
            l_mlo_image = self.images['_'.join([patient_id, 'L', 'MLO'])]
            r_cc_image = self.images['_'.join([patient_id, 'R', 'CC'])]
            r_mlo_image = self.images['_'.join([patient_id, 'R', 'MLO'])]
            years_to_cancer = int(years_to_cancer)

            # label mapping
            if self.map_labels:
                if years_to_cancer == 100:
                    years_to_cancer = 0
                else:
                    years_to_cancer = 1
            
            # labels smoothing
            if self.label_smoothing > 0 and years_to_cancer == 1:
                years_to_cancer = 1 - self.label_smoothing
            
            # save images in array
            self.l_cc_images.append(l_cc_image)
            self.l_mlo_images.append(l_mlo_image)
            self.r_cc_images.append(r_cc_image)
            self.r_mlo_images.append(r_mlo_image)
            self.years_to_cancer.append(years_to_cancer)
        
        # create tensors from python arrays
        dtype=tf.float32
        self.l_cc_images = tf.convert_to_tensor(self.l_cc_images, dtype=dtype)
        self.l_mlo_images = tf.convert_to_tensor(self.l_mlo_images, dtype=dtype)
        self.r_cc_images = tf.convert_to_tensor(self.r_cc_images, dtype=dtype)
        self.r_mlo_images = tf.convert_to_tensor(self.r_mlo_images, dtype=dtype)
        self.years_to_cancer = tf.convert_to_tensor(self.years_to_cancer, dtype=dtype)

        self.ratio = tf.reduce_mean(self.years_to_cancer)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = (
            tf.gather(self.l_cc_images, indices=indices),
            tf.gather(self.l_mlo_images, indices=indices),
            tf.gather(self.r_cc_images, indices=indices),
            tf.gather(self.r_mlo_images, indices=indices)
        )
        if self.transformation is not None:
            images = [self.transformation(image) for image in images]
        y = tf.gather(self.years_to_cancer, indices=indices)
        return images, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = tf.range(len(self.data))
        if self.shuffle:
            self.indices = tf.random.shuffle(self.indices)

    def get_class_weight(self):
        class_weight = {
            0: self.ratio,
            1: 1 - self.ratio
        }
        return class_weight


class GeneratorKaggle(keras.utils.Sequence):
    def __init__(
            self,
            data,
            images,
            batch_size,
            shuffle=True,
            transformation=None,
            label_smoothing=0.0
            ):
        self.data = data
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformation = transformation
        self.label_smoothing = label_smoothing

        self.images_data = []

        self.ratio = 0.5

        self.__make_samples()
        self.on_epoch_end()

    def __make_samples(self):
        for d in self.data:
            patient_id, image_id, years_to_cancer = d

            # retrieve images for curent patient_id
            image = self.images_data['_'.join([patient_id, image_id])]
            years_to_cancer = int(years_to_cancer)
            
            # labels smoothing
            if self.label_smoothing > 0 and years_to_cancer == 1:
                years_to_cancer = 1 - self.label_smoothing
            
            # save images in array
            self.images_data.append(image)
            self.years_to_cancer.append(years_to_cancer)
        
        # create tensors from python arrays
        dtype=tf.float32
        self.images_data = tf.convert_to_tensor(self.images_data, dtype=dtype)
        self.years_to_cancer = tf.convert_to_tensor(self.years_to_cancer, dtype=dtype)

        self.ratio = tf.reduce_mean(self.years_to_cancer)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = tf.gather(self.l_cc_images, indices=indices)
        if self.transformation is not None:
            images = self.transformation(images)
        y = tf.gather(self.years_to_cancer, indices=indices)
        return images, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = tf.range(len(self.data))
        if self.shuffle:
            self.indices = tf.random.shuffle(self.indices)

    def get_class_weight(self):
        class_weight = {
            0: self.ratio,
            1: 1 - self.ratio
        }
        return class_weight


class GeneratorKaggleDynamic(keras.utils.Sequence):
    def __init__(
            self,
            data,
            images_dir,
            batch_size,
            shuffle=True,
            transformation=None,
            label_smoothing=0.0
            ):
        self.data = data
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformation = transformation
        self.label_smoothing = label_smoothing

        self.years_to_cancer = []

        self.ratio = 0.5

        self.__make_samples()
        self.on_epoch_end()

    def __make_samples(self):
        for d in self.data:
            patient_id, image_id, years_to_cancer = d

            years_to_cancer = int(years_to_cancer)
            
            # labels smoothing
            if self.label_smoothing > 0 and years_to_cancer == 1:
                years_to_cancer = 1 - self.label_smoothing
            
            # save images in array
            self.years_to_cancer.append(years_to_cancer)
        
        # create tensors from python arrays
        dtype=tf.float32
        self.years_to_cancer = tf.convert_to_tensor(self.years_to_cancer, dtype=dtype)

        self.ratio = tf.reduce_mean(self.years_to_cancer)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        data = self.data[indices]
        images = []
        for patient_id, image_id, _ in data:
            # retrieve images for curent patient_id
            key = '_'.join([patient_id, image_id])
            image = Image.open(f'{self.images_dir}/{key}.png')
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            image = tf.expand_dims(image, axis=-1)
            image /= 255
            images.append(image)
        images = tf.convert_to_tensor(images)

        if self.transformation is not None:
            images = self.transformation(images)
        y = tf.gather(self.years_to_cancer, indices=indices)
        return images, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = tf.range(len(self.data))
        if self.shuffle:
            self.indices = tf.random.shuffle(self.indices)

    def get_class_weight(self):
        class_weight = {
            0: self.ratio,
            1: 1 - self.ratio
        }
        return class_weight