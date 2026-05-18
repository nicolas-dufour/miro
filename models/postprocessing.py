import numpy as np
import torch
import torch.nn as nn


VAE_STATS = {
    "sdxl": {
        "scale": torch.tensor([0.13025, 0.13025, 0.13025, 0.13025]),
        "bias": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "scale_channel_wise": torch.tensor([7.8518, 5.8465, 6.8716, 5.3140]),
        "bias_channel_wise": torch.tensor([-0.3480, 0.3754, 0.0419, 2.4331]),
    },
}


def remap_image_torch(x, mean=None, std=None):
    if mean is not None and std is not None:
        mean = mean.to(device=x.device, dtype=x.dtype)
        std = std.to(device=x.device, dtype=x.dtype)
        x = x * std + mean
    x = torch.clamp(x * 0.5 + 0.5, 0, 1)
    x = (x * 255).to(torch.uint8)
    return x


class VAEPostProcessing(nn.Module):
    def __init__(self, channel_wise_normalisation=False, model_type="sdxl"):
        super().__init__()

        if model_type not in VAE_STATS:
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {list(VAE_STATS.keys())}"
            )

        stats = VAE_STATS[model_type]

        if channel_wise_normalisation:
            scale = 0.5 / stats["scale_channel_wise"]
            bias = -stats["bias_channel_wise"] * scale
        else:
            scale = 1.0 / stats["scale"]
            bias = -stats["bias"] * scale

        scale = scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, x):
        x = (x - self.bias) / self.scale
        return x


class VAEDecoderPostProcessing(VAEPostProcessing):
    def __init__(
        self,
        vae,
        channel_wise_normalisation=False,
        model_type="sdxl",
        max_batch_size=64,
        device=None,
    ):
        super().__init__(
            channel_wise_normalisation=channel_wise_normalisation, model_type=model_type
        )
        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.max_batch_size = max_batch_size

    def _decode(self, z):
        with torch.no_grad():
            images = remap_image_torch(self.vae.decode(z).sample.detach())
            images = images.permute(0, 2, 3, 1).cpu().numpy()
        return images

    def forward(self, x):
        x = super().forward(x)
        if x.shape[0] <= self.max_batch_size:
            return self._decode(x)
        else:
            output = []
            for i in range(0, x.shape[0], self.max_batch_size):
                output.append(self._decode(x[i : i + self.max_batch_size]))
            return np.concatenate(output, axis=0)
