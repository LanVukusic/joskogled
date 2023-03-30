import torch.nn as nn

def get_block_classifier(in_size, sizes):
    sizes = in_size + sizes
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        layers.append(nn.LeakyReLU(0.01))
    return nn.Sequential(*layers)