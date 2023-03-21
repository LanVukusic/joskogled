import torch.nn as nn
from block_cnn import get_block_cnn
import torch

FC_SIZE = 256


class Model(nn.Module):
    def __init__(
        self, in_dims_cc, in_dimms_mlo, channels=[1, 32, 64, 128], fc_size=FC_SIZE
    ):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.l_cc_cnn = get_block_cnn(in_dims_cc, channels, fc_size)
        self.l_mlo_cnn = get_block_cnn(in_dimms_mlo, channels, fc_size)
        self.r_cc_cnn = get_block_cnn(in_dims_cc, channels, fc_size)
        self.r_mlo_cnn = get_block_cnn(in_dimms_mlo, channels, fc_size)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = nn.Sequential(
            nn.Linear(4 * fc_size, fc_size),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size, fc_size),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size, fc_size // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.l_cc_cnn(l_cc)
        l_mlo = self.l_mlo_cnn(l_mlo)
        r_cc = self.r_cc_cnn(r_cc)
        r_mlo = self.r_mlo_cnn(r_mlo)

        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_classifier(cat)
        return x
