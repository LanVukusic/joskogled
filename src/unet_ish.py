import torch.nn as nn
import torch


# model = Model(
#     in_dims=shape,
#     out_classes=NUM_CLASSES,  # 0, 1, 2, 3, 4
#     channels=[1, 16, 32, 64, 128],
#     strides=[2, 2, 2, 1, 1, 1],
#     fc_sizes=[128, 64, 8],
#     dropouts=[0.3, 0.1, 0.0],
# ).to(DEVICE)


def get_block(in_c, out_c, kernel_size, stride):
    layers = []
    layers.append(
        nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid",
        )
    )
    layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.BatchNorm2d(out_c))
    layers.append(
        nn.Conv2d(
            out_c,
            out_c,
            kernel_size=kernel_size,
            padding="same",
        )
    )
    layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


def get_block_unet(in_dims, channels, strides):
    """
    Input a fulll size image and extract features by convolutions from it.
    Downsample the image by a factor of 2 and repeat.
    """

    inbetween_layer_outputs = []
    layers = []
    kernel_size = 3
    for i in range(len(channels) - 1):
        # create a convoloutional block
        block = get_block(channels[i], channels[i + 1], kernel_size, strides[i])
        layers = layers + block

        # create a pooling layer to downsample the image by a factor of 2
        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding="same")
        layers = layers + pool

    with torch.no_grad():
        x = torch.zeros((1, 1, *in_dims))  # batch_size, channels, h, w
        for layer in layers:
            x = layer(x)
        print(x.shape)

    # fully connected layers from the conv outputs of each image size concatenated together
    # layers.append(nn.Flatten())
