from sailer_loader import get_datasets

# from final_pred import make_final_pred, save_pred

# from model import Model
from models import Model
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
MODEL_BASE_NAME = "model_dropout"
NUM_CLASSES = 2

model_name = "{}_lr-{}_bs-{}_ne-{}_{}".format(
    MODEL_BASE_NAME,
    LR,
    BATCH_SIZE,
    NUM_EPOCHS,
    datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
)


print("runnig model: {}".format(model_name), flush=True)


def main():
    print("zacetek", flush=True)
    # set random seed for torch
    torch.manual_seed(3)

    # define transformation to be aplied on train data images
    transformation = torch.nn.Sequential(T.RandomRotation(degrees=(-12, 12)))

    # get dataloaders

    # primer kako naloziti koncne podatke
    """
    dtl_final = get_final_dataloader(
        data_path="../data/test.txt",
        img_path="../data/final_data_halfk"
    )
    """
    dtl_train, dtl_val, shape = get_datasets(
        data_file="/d/hpc/home/ris002/joskogled/data/combined_data.csv",
        image_root="/d/hpc/home/ris002/joskogled/data/",
        split=0.8,
        transform=transformation,
    )
    print("mogoce dela", flush=True)

    # define classification model
    model = Model(
        in_dims=shape,
        out_classes=NUM_CLASSES,  # 0, 1
        channels=[1, 16, 32, 64, 128],
        strides=[2, 2, 2, 1, 1, 1],
        fc_sizes=[128, 64, 8],
        dropouts=[0.3, 0.1, 0.0],
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
    )

    # primer kako nardit predikcije na koncnih podatkih
    """
    final = make_final_pred(dtl_final, model)
    save_pred(final, "pred.txt")
    """


if __name__ == "__main__":
    main()
    print("cau")
