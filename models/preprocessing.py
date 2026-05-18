import torch
from torch import nn
from miro.utils.misc import print0

VAE_STATS = {
    "sdxl": {
        "scale": torch.tensor([0.13025, 0.13025, 0.13025, 0.13025]),
        "bias": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "scale_channel_wise": torch.tensor([7.8518, 5.8465, 6.8716, 5.3140]),
        "bias_channel_wise": torch.tensor([-0.3480, 0.3754, 0.0419, 2.4331]),
    },
}


class PrecomputedSDLatentPreconditioning(nn.Module):
    def __init__(
        self,
        input_key_mean="vae_embeddings_mean",
        input_key_std="vae_embeddings_std",
        output_key_root="x_0",
        vae_sample=False,
        channel_wise_normalisation=False,
        model_type="sdxl",
    ):
        super().__init__()
        self.input_key_mean = input_key_mean
        self.input_key_std = input_key_std
        self.output_key_root = output_key_root
        self.vae_sample = vae_sample

        if model_type not in VAE_STATS:
            raise ValueError(
                f"Unknown model type: {model_type}. Available types: {list(VAE_STATS.keys())}"
            )

        stats = VAE_STATS[model_type]

        if channel_wise_normalisation:
            scale_val = 0.5 / stats["scale_channel_wise"]
            bias_val = -stats["bias_channel_wise"] * scale_val
            print0(f"Using channel-wise normalization for {model_type}")
        else:
            scale_val = 1.0 / stats["scale"]
            bias_val = -stats["bias"] * scale_val
            print0(f"Using standard normalization for {model_type}")

        scale = scale_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = bias_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("scale", nn.Parameter(scale))
        self.register_buffer("bias", nn.Parameter(bias))

    def forward(self, batch):
        if self.vae_sample:
            mean = batch[self.input_key_mean]
            std = batch[self.input_key_std]
            latents = torch.randn_like(mean) * std + mean
        else:
            latents = batch[self.input_key_mean]
        latents = latents * self.scale + self.bias
        batch[self.output_key_root] = latents
        return batch


class PrecomputedTextConditioning:
    def __init__(
        self,
        input_key="flan_t5_xl",
        output_key_root="text_tokens",
        drop_labels=False,
    ):
        self.input_key = input_key
        self.output_key_root = output_key_root
        self.drop_labels = drop_labels

    def __call__(self, batch, device=None):
        if self.drop_labels:
            batch[f"{self.output_key_root}_embeddings"] = None
            batch[f"{self.output_key_root}_mask"] = None
            return batch
        batch[f"{self.output_key_root}_embeddings"] = batch[
            f"{self.input_key}_embeddings"
        ]
        batch[f"{self.output_key_root}_mask"] = batch[f"{self.input_key}_mask"]
        return batch
