import os

# disable warnings blabla
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# disable cuda
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import generators_tensor as gen
import models as md
from tensorflow import keras
import numpy as np
import cv2

DATA_DIR = "../py-projects/data"
TXT_PATH = "/processed_data.txt"
IMAGE_PATH = "/processed_data_quterk"
FINAL_TXT_PATH = "/resitve.txt"
FINAL_IMAGE_PATH = "/final_data_quterk"
KAGGLE_TXT_PATH = "/kaggle_processed_data.txt"

BATCH_SIZE = 16
EPOCHS = 120

import keras.backend as K

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    


def main():
    data_train, data_val = gen.load_data(
        data_path=DATA_DIR + TXT_PATH,
        p=1.0,
        shuffle=True
    )
    data_val, _ = gen.load_data(
        data_path=DATA_DIR + FINAL_TXT_PATH,
        p=1.0,
        shuffle=False
    )
    images, image_shape = gen.load_images(
        img_paths=[
        DATA_DIR + IMAGE_PATH,
        DATA_DIR + FINAL_IMAGE_PATH
        ]
    )

    transformation = keras.Sequential([
        #keras.layers.RandomFlip(),
        #keras.layers.ZeroPadding2D(padding=20),
        keras.layers.RandomRotation(0.05, fill_mode='constant')
    ])

    gen_train = gen.Generator(
        data=data_train,
        images=images,
        batch_size=BATCH_SIZE,
        transformation=transformation,
        map_labels=True
    )
    gen_val = gen.Generator(
        data=data_val,
        images=images,
        batch_size=BATCH_SIZE,
        #map_labels=True         ###
    )

    """counter = 0
    for j, (images, y) in enumerate(gen_train):
        [cv2.imwrite("im" + str(i+4*j) + ".png", (image[0].numpy()*255).astype(np.uint8))for i, image in enumerate(images)]
        if counter == 2:
            break
        counter += 1"""
        



    model = md.model1(image_shape=image_shape)
    print(model.summary())
    #model.save('model-res.h5')
    
    model.compile(
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        metrics=[keras.metrics.AUC(), f1_metric]
    )
    model.fit(
        gen_train,
        validation_data=gen_val,
        epochs=EPOCHS,
        verbose=2,
        class_weight=gen_train.get_class_weight()
    )


if __name__ == "__main__":
    main()
