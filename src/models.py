import torch.nn as nn
from block_cnn import get_block_cnn, get_block_cnn_pool
from block_classifier import get_block_classifier
import torch


class Model(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_sizes):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.block_cnn = get_block_cnn_pool(in_dims, channels, strides)

        # reduce from 4 cnn block
        self.cat_reduce_dim = nn.Sequential(
            nn.Linear(4 * channels[-1], channels[-1])
            nn.LeakyReLU(0.01)
        )

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            channels[-1], fc_sizes, out_classes
        )
        self.to_classes = nn.Linear(fc_sizes[-1], out_classes)

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn(l_cc)
        l_mlo = self.block_cnn(l_mlo)
        r_cc = self.block_cnn(r_cc)
        r_mlo = self.block_cnn(r_mlo)

        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.cat_reduce_dim(cat)
        x = self.block_classifier(x)
        return x


class Model4(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_size):
        super(Model4, self).__init__()
        # convolutional blocks for each image
        self.block_cnn_lcc = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_lmlo = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_rcc = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_rmlo = get_block_cnn_pool(in_dims, channels, strides)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = nn.Sequential(
            nn.Linear(4 * fc_size, fc_size),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size, fc_size // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size // 2, out_classes),
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn_lcc(l_cc)
        l_mlo = self.block_cnn_lmlo(l_mlo)
        r_cc = self.block_cnn_rcc(r_cc)
        r_mlo = self.block_cnn_rmlo(r_mlo)

        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_classifier(cat)
        return x
