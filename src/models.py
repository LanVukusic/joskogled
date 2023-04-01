import torch.nn as nn
from block_cnn import get_block_deconv, get_block_cnn_pool
from block_classifier import get_block_classifier
import torch


# class ModelYolo(nn.Module):
#     def __init__(self, in_dims, out_classes, channels, strides, fc_sizes):
#         super(ModelYolo, self).__init__()
#         # convolutional blocks for each image
#         self.block_yolo = get_block_yolo5(fc_sizes[0])

#         # concatenate the outputs of the convolutional blocks
#         self.block_classifier = get_block_classifier(
#             4 * channels[-1], fc_sizes, out_classes
#         )

#     def forward(self, l_cc, l_mlo, r_cc, r_mlo):
#         l_cc = self.block_yolo(l_cc)
#         l_mlo = self.block_yolo(l_mlo)
#         r_cc = self.block_yolo(r_cc)
#         r_mlo = self.block_yolo(r_mlo)

#         cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
#         x = self.block_classifier(cat)
#         return x

class Model(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_sizes, dropouts):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.block_cnn = get_block_cnn_pool(in_dims, channels, strides)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            4 * channels[-1], fc_sizes, dropouts, out_classes
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn(l_cc)
        l_mlo = self.block_cnn(l_mlo)
        r_cc = self.block_cnn(r_cc)
        r_mlo = self.block_cnn(r_mlo)

        # first prediction head
        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_classifier(cat)
        return x

class ModelNou(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_sizes):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.block_cnn = get_block_cnn_pool(in_dims, channels, strides)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            4 * channels[-1], fc_sizes, out_classes
        )

        self.block.deccoder = get_block_deconv(
            channels, strides,
        )

        self.gap = GlobalAveragePooling()

        self.l1 = nn.Linear()
        layers.append(nn.LeakyReLU(0.1))

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn(l_cc)
        l_mlo = self.block_cnn(l_mlo)
        r_cc = self.block_cnn(r_cc)
        r_mlo = self.block_cnn(r_mlo)
        # decouple

        # first prediction head
        cat = torch.cat([self.gap(l_cc), self.gap(l_mlo), self.gap(r_cc),self.gap(r_mlo)], dim=1)
        x = self.block_classifier(cat)

        # second deccoder head
        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)

class Model2(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_sizes, dropouts):
        super(Model4, self).__init__()
        # convolutional blocks for each image
        self.block_cnn_cc = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_mlo = get_block_cnn_pool(in_dims, channels, strides)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            4 * channels[-1], fc_sizes, dropouts, out_classes
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn_cc(l_cc)
        l_mlo = self.block_cnn_mlo(l_mlo)
        r_cc = self.block_cnn_cc(r_cc)
        r_mlo = self.block_cnn_mlo(r_mlo)

        # first prediction head
        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_classifier(cat)
        return x

class Model4(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_sizes, dropouts):
        super(Model4, self).__init__()
        # convolutional blocks for each image
        self.block_cnn_lcc = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_lmlo = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_rcc = get_block_cnn_pool(in_dims, channels, strides)
        self.block_cnn_rmlo = get_block_cnn_pool(in_dims, channels, strides)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            4 * channels[-1], fc_sizes, dropouts, out_classes
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn_lcc(l_cc)
        l_mlo = self.block_cnn_lmlo(l_mlo)
        r_cc = self.block_cnn_rcc(r_cc)
        r_mlo = self.block_cnn_rmlo(r_mlo)

        # first prediction head
        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_classifier(cat)
        return x
