from dataloader import get_dataloader
from model import Model
from trainer import Trainer
import torch.nn as nn
import torch

# fix absolute path problem
import os
import sys

PATH_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH_PREFIX)


def main():
    dtl_train, dtl_val = get_dataloader(
        data_path="../data/processed_data.txt",
        img_path="../data/processed_data",
        batch_size=8,
        shuffle=True,
        p=0.7
    )
    print("mogoce dela")
    model = Model(
        in_dims=(2048, 1576),
        out_classes=5,              # 0, 1, 2, 3, 4
        channels=[1, 16, 32, 32, 64, 128],
        strides=[2, 2, 2, 1, 1],
        fc_size=128,
    )
    print("ne res ja")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, criterion, optimizer)

    # Train one epoch
    NUM_EPOCHS = 1
    for epoch in range(NUM_EPOCHS):

        # Train
        # model.train()
        for batch_idx, (
                patient_id,
                l_cc,
                l_mlo,
                r_cc,
                r_mlo,
                years_to_cancer,
        ) in enumerate(dtl_train):

            print(patient_id)
            print(l_cc.shape)

            loss = trainer.train((l_cc, l_mlo, r_cc, r_mlo), years_to_cancer)

            if (batch_idx + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, NUM_EPOCHS, batch_idx + 1, len(dtl_train), loss.item()
                    )
                )
        # after every epoch calculate val loss
        val_loss = 0
        batch_idx = 0
        for batch_idx, (
                patient_id,
                l_cc,
                l_mlo,
                r_cc,
                r_mlo,
                years_to_cancer,
        ) in enumerate(dtl_val):
            val_loss += trainer.eval((l_cc, l_mlo, r_cc, r_mlo), years_to_cancer)
        val_loss /= (batch_idx + 1)


if __name__ == "__main__":
    main()
