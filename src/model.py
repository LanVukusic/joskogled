import torch.nn as nn
from block_cnn import get_block_cnn
import torch

FC_SIZE = 256


class Model(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_size=FC_SIZE):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.l_cc_cnn = get_block_cnn(in_dims, channels, strides, fc_size)
        self.l_mlo_cnn = get_block_cnn(in_dims, channels, strides, fc_size)
        self.r_cc_cnn = get_block_cnn(in_dims, channels, strides, fc_size)
        self.r_mlo_cnn = get_block_cnn(in_dims, channels, strides, fc_size)

        # downsample the image by a factor of 2
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding="same")

        # concat layer for the 4 images
        self.block_embedder = nn.Sequential(
            nn.Linear(4 * fc_size, fc_size // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size // 2, fc_size // 2),
            nn.LeakyReLU(0.01),
        )

        # classification layer
        self.block_classifier = nn.Sequential(
            nn.Linear(fc_size // 2, fc_size // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size // 2, fc_size // 4),
            nn.LeakyReLU(0.01),
            nn.Linear(fc_size // 4, out_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, l_cc_in, l_mlo_in, r_cc_in, r_mlo_in):
        l_cc = self.l_cc_cnn(l_cc_in)
        l_mlo = self.l_mlo_cnn(l_mlo_in)
        r_cc = self.r_cc_cnn(r_cc_in)
        r_mlo = self.r_mlo_cnn(r_mlo_in)
        first_size_embedding = torch.cat((l_cc, l_mlo, r_cc, r_mlo), dim=1)
        first_size_embedding = self.block_embedder(first_size_embedding)

        # downsample the image by a factor of 2
        # pool
        l_cc_ds = self.downsample(l_cc_in)
        l_mlo_ds = self.downsample(l_mlo_in)
        r_cc_ds = self.downsample(r_cc_in)
        r_mlo_ds = self.downsample(r_mlo_in)
        # push the images through the convolutional blocks
        l_cc_ds = self.l_cc_cnn(l_cc_ds)
        l_mlo_ds = self.l_mlo_cnn(l_mlo_ds)
        r_cc_ds = self.r_cc_cnn(r_cc_ds)
        r_mlo_ds = self.r_mlo_cnn(r_mlo_ds)
        second_size_embedding = torch.cat((l_cc_ds, l_mlo_ds, r_cc_ds, r_mlo_ds), dim=1)
        second_size_embedding = self.block_embedder(second_size_embedding)

        # second downsample
        # pool
        l_cc_ds2 = self.downsample(l_cc_ds)
        l_mlo_ds2 = self.downsample(l_mlo_ds)
        r_cc_ds2 = self.downsample(r_cc_ds)
        r_mlo_ds2 = self.downsample(r_mlo_ds)
        # push the images through the convolutional blocks
        l_cc_ds2 = self.l_cc_cnn(l_cc_ds2)
        l_mlo_ds2 = self.l_mlo_cnn(l_mlo_ds2)
        r_cc_ds2 = self.r_cc_cnn(r_cc_ds2)
        r_mlo_ds2 = self.r_mlo_cnn(r_mlo_ds2)
        third_size_embedding = torch.cat(
            (l_cc_ds2, l_mlo_ds2, r_cc_ds2, r_mlo_ds2), dim=1
        )
        third_size_embedding = self.block_embedder(third_size_embedding)

        # sum the embeddings
        cat = torch.sum(
            torch.stack(
                (first_size_embedding, second_size_embedding, third_size_embedding)
            ),
        )

        x = self.block_classifier(cat)
        return x
