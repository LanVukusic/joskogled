import torch.nn as nn
import torch


def get_block_cnn(in_dims, channels, strides, fc_size=256):
    layers = []
    kernel_size = 3
    for i in range(len(channels) - 1):
        layers.append(
            nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=strides[i],
                padding="valid",
            )
        )
        layers.append(nn.LeakyReLU(0.01))
        layers.append(
            nn.Conv2d(
                channels[i + 1],
                channels[i + 1],
                kernel_size=kernel_size,
                padding="same",
            )
        )
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    layers.append(nn.Flatten())

    # calculate the number of features after the last convolution
    with torch.no_grad():
        x = torch.zeros((1, 1, *in_dims))       # batch_size, channels, h, w
        for layer in layers:
            x = layer(x)
        n_features = x.shape[1]
        layers.append(nn.Linear(n_features, fc_size))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(fc_size, fc_size))
        layers.append(nn.LeakyReLU(0.01))

    return nn.Sequential(*layers)
