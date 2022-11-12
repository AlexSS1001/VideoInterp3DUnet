from torchvision.transforms import ToPILImage, ToTensor
import kornia.filters as K_filters
import kornia.color as K_color
from tkinter import TOP
import torch.nn as nn
from pytorch_msssim import SSIM


class l1_ssim_loss(nn.Module):
    """
    Mixed loss between L1 and SSIM
    """

    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super(l1_ssim_loss, self).__init__()
        self.ssim_module = SSIM(
            data_range=data_range, size_average=size_average, channel=channel
        )
        self.l1_loss = nn.L1Loss()

    def forward(self, inputs, targets):
        return 0.3 * self.l1_loss(inputs, targets) + 0.7 * (
            1.0 - self.ssim_module(inputs, targets)
        )


class LossL2ColorEdges(nn.Module):
    """
    Appply L2 on both image and edges
    """

    def __init__(self):
        super(LossL2ColorEdges, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        # compute edges

        edge_loss = 0
        for i in range(inputs.shape[2]):
            input = inputs[:, :, i, :, :]
            target = targets[:, :, i, :, :]
            input = K_color.rgb_to_grayscale(input)
            target = K_color.rgb_to_grayscale(target)
            input_edges = K_filters.sobel(input)
            target_edges = K_filters.sobel(target)
            edge_loss += self.l2_loss(input_edges, target_edges)

        return 0.5 * self.l2_loss(inputs, targets) + 0.5 * edge_loss
