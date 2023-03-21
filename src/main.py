# import from relative path


from dataloader import get_dataloader
from model import Model
import torch.nn as nn
import torch

# fix absolute path problem
import os
import sys

PATH_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PATH_PREFIX)


def main():
    dtl = get_dataloader("/home/lan/Desktop/joskogled/data/processed_data.txt")
    print("mogoce dela")
    criterion = nn.CrossEntropyLoss()
    model = Model(
        in_dims_cc=(2048, 1664),
        in_dimms_mlo=(2048, 1664),
        channels=[1, 4],
        fc_size=2,
    )
    print("ne res ja")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        ) in enumerate(dtl):

            print(patient_id)
            print(l_cc.shape)
            break
            # Forward pass
            output = model(l_cc, l_mlo, r_cc, r_mlo)
            loss = criterion(output, years_to_cancer)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, NUM_EPOCHS, batch_idx + 1, len(dtl), loss.item()
                    )
                )


if __name__ == "__main__":
    main()
