import torch
from PIL import Image


class PILResize(torch.nn.Module):
    def __init__(self, size, resample, reducing_gap):
        super().__init__()
        self.size = size
        self.resample = resample
        self.reducing_gap = reducing_gap

    def forward(self, x: Image.Image):
        return x.resize(self.size, self.resample, self.reducing_gap)


class Rescale(torch.nn.Module):
    def __init__(self, rescale_factor):
        super().__init__()
        self.rescale_factor = rescale_factor

    def forward(self, x: torch.tensor):
        return x * self.rescale_factor
