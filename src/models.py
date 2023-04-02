import torch.nn as nn
from block_cnn import get_block_cnn_pool
from block_classifier import get_block_classifier
import torch


class Model(nn.Module):
    def __init__(
        self,
        in_dims,
        out_classes,
        channels,
        strides,
        fc_sizes,
        dropouts,
        embedder_sizes,
        embedder_dropouts,
    ):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.block_cnn = get_block_cnn_pool(in_dims, channels, strides)

        # downsample the image by a factor of 2
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding="same")

        # fully connected feature extractor for the 4 images
        # takes the output of the convolutional blocks and outputs a feature vector
        self.block_embedder = get_block_classifier(
            4 * channels[-1], embedder_sizes, embedder_dropouts, embedder_sizes[-1]
        )

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            embedder_sizes[-1], fc_sizes, dropouts, out_classes
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        # original size
        l_cc = self.block_cnn(l_cc)
        l_mlo = self.block_cnn(l_mlo)
        r_cc = self.block_cnn(r_cc)
        r_mlo = self.block_cnn(r_mlo)
        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_embedder(cat)

        # downsample the image by a factor of 2 and push through the convolutional blocks
        l_cc_ds = self.block_cnn(self.downsample(l_cc))
        l_mlo_ds = self.block_cnn(self.downsample(l_mlo))
        r_cc_ds = self.block_cnn(self.downsample(r_cc))
        r_mlo_ds = self.block_cnn(self.downsample(r_mlo))
        cat_ds = torch.cat([l_cc_ds, l_mlo_ds, r_cc_ds, r_mlo_ds], dim=1)
        x_ds = self.block_embedder(cat_ds)

        # sum the outputs of the two feature extractors
        x = x + x_ds

        # push the concatenated output through the classifier
        x = self.block_classifier(cat)
        return x
