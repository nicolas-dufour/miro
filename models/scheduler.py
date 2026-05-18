import torch


class LinearScheduler:
    def __init__(self, start=1, end=0, shift=1.0, clip_min=1e-9):
        self.start = start
        self.end = end
        self.shift = shift
        self.clip_min = clip_min

    def __call__(self, t):
        output = (self.end - self.start) * t + self.start
        if self.shift > 1:
            output = self.shift * output / (1 + (self.shift - 1) * output)
        return torch.clamp(output, min=self.clip_min)
