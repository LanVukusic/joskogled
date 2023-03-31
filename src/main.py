from dataloader import get_dataloader

# from model import Model
from models import Model as Model
from trainer import Trainer
from train import train
import torch.nn as nn
import torch
import math
import datetime
import torchmetrics
import torchvision.transforms as T

# fix absolute path problem
import os
import sys


PATH_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH_PREFIX)

# set device to GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PARAMS
NUM_EPOCHS = 50
BATCH_SIZE = 16
LR = 3e-4
MODEL_BASE_NAME = "model_dropout_yolo"
NUM_CLASSES = 2

model_name = "{}_lr-{}_bs-{}_ne-{}_{}".format(
    MODEL_BASE_NAME,
    LR,
    BATCH_SIZE,
    NUM_EPOCHS,
    datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
)


# METRICS
metrics = {
    "auroc": torchmetrics.AUROC(
        task="multiclass", num_classes=NUM_CLASSES, average="macro"
    ).to(DEVICE),
    "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE),
    "precision": torchmetrics.Precision(
        task="multiclass", num_classes=NUM_CLASSES, average="macro"
    ).to(DEVICE)
}


print("runnig model: {}".format(model_name), flush=True)


def main():
    print("zacetek", flush=True)
    # set random seed for torch
    torch.manual_seed(3)

    # define transformation to be aplied on train data images
    transformation = torch.nn.Sequential(T.RandomRotation(degrees=(-25, 25)))

    # get dataloaders
    dtl_train, dtl_val, shape = get_dataloader(
        data_path="../data/processed_data.txt",
        img_path="../data/processed_data_1k",
        batch_size=BATCH_SIZE,
        shuffle=True,
        p=0.8,
        upsample=3,
        transformation=transformation,
    )
    print("mogoce dela", flush=True)

    # define classification model
    model = Model(
        in_dims=shape,
        out_classes=NUM_CLASSES,  # 0, 1, 2, 3, 4
        channels=[1, 32, 64, 128, 256],
        strides=[2, 2, 1, 1],
        fc_sizes=[128, 64, 32],
        dropouts=[0.3, 0.3, 0.3]
    ).to(DEVICE)

    # define trainer with loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    trainer = Trainer(model, criterion, optimizer)

    # train for number of epochs
    train(
        dtl_train=dtl_train,
        dtl_val=dtl_val,
        trainer=trainer,
        epochs=NUM_EPOCHS,
        metrics=metrics
        )


if __name__ == "__main__":
    main()
    print("cau")
