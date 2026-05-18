import math

import torch
import torch.nn as nn
from einops import rearrange

# einops compile bug is back in 2.7
import einops._torch_specific
from torch import Tensor
from miro.models.networks.positional_embeddings import FourierEmbedding

torch.fx.wrap("rearrange")
from typing import Optional, Tuple

from miro.models.networks.transformers import (
    CrossAttentionBlock,
    FusedMLP,
    RMSNorm,
    SelfAttentionBlock,
)

from functools import partial


class TimeEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        linear_layer: nn.Module = nn.Linear,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.encode_time = FourierEmbedding(num_channels=dim)
        self.map_time_input = linear_layer(dim, dim * expansion)  # mup_type="input")
        self.map_time_output = linear_layer(dim * expansion, dim * expansion)
        self.activation = activation()

    def forward(self, t: Tensor) -> Tensor:
        time = self.encode_time(t * 1.0)
        time_mean = time.mean(dim=-1, keepdim=True)
        time_std = time.std(dim=-1, keepdim=True)
        time = (time - time_mean) / time_std
        return self.map_time_output(self.activation(self.map_time_input(time)))


class RINBlock(nn.Module):
    def __init__(
        self,
        data_dim: int,
        latents_dim: int,
        num_processing_layers: int,
        read_write_heads: int = 16,
        compute_heads: int = 16,
        latent_mlp_multiplier: int = 4,
        data_mlp_multiplier: int = 4,
        compute_dropout: float = 0.0,
        rw_stochastic_depth: float = 0.0,
        compute_stochastic_depth: float = 0.0,
    ):
        super().__init__()

        self.retriever_ca = CrossAttentionBlock(
            dim_q=latents_dim,
            dim_kv=data_dim,
            num_heads=read_write_heads,
            mlp_multiplier=latent_mlp_multiplier,
            dropout=0.0,
            stochastic_depth=rw_stochastic_depth,
            use_biases=True,
        )
        self.processer_sa = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latents_dim,
                    num_heads=compute_heads,
                    mlp_multiplier=latent_mlp_multiplier,
                    dropout=compute_dropout,
                    stochastic_depth=compute_stochastic_depth,
                    use_biases=True,
                )
                for _ in range(num_processing_layers)
            ]
        )
        self.writer_ca = CrossAttentionBlock(
            dim_q=data_dim,
            dim_kv=latents_dim,
            num_heads=read_write_heads,
            mlp_multiplier=data_mlp_multiplier,
            dropout=0.0,
            stochastic_depth=rw_stochastic_depth,
            use_biases=True,
        )

    def forward(
        self,
        data: torch.Tensor,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_attention_mask: Optional[torch.Tensor] = None,
        cond_adaln: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retrieve the latent representations from the data
        latents = self.retriever_ca(latents, data)
        # Process the latent representations
        for sa in self.processer_sa:
            latents = sa(latents, conditioning=cond_adaln)
        # Write the latent representations into the data
        data = self.writer_ca(data, latents)
        return data, latents


class RINBlockCond(RINBlock):
    """RIN block with an explicit cross-attention over the conditioning tokens
    before the standard retrieve/process/write sequence."""

    def __init__(
        self,
        data_dim: int,
        latents_dim: int,
        num_processing_layers: int,
        read_write_heads: int = 16,
        compute_heads: int = 16,
        latent_mlp_multiplier: int = 4,
        data_mlp_multiplier: int = 4,
        compute_dropout: float = 0.0,
        rw_stochastic_depth: float = 0.0,
        compute_stochastic_depth: float = 0.0,
    ):
        super().__init__(
            data_dim=data_dim,
            latents_dim=latents_dim,
            num_processing_layers=num_processing_layers,
            read_write_heads=read_write_heads,
            compute_heads=compute_heads,
            latent_mlp_multiplier=latent_mlp_multiplier,
            data_mlp_multiplier=data_mlp_multiplier,
            compute_dropout=compute_dropout,
            rw_stochastic_depth=rw_stochastic_depth,
            compute_stochastic_depth=compute_stochastic_depth,
        )
        self.retrieve_cond = CrossAttentionBlock(
            dim_q=latents_dim,
            dim_kv=latents_dim,
            num_heads=read_write_heads,
            mlp_multiplier=latent_mlp_multiplier,
            dropout=0.0,
            stochastic_depth=rw_stochastic_depth,
            use_biases=True,
        )

    def forward(
        self,
        data: torch.Tensor,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_attention_mask: Optional[torch.Tensor] = None,
        cond_adaln: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.retrieve_cond(latents, cond, from_token_mask=cond_attention_mask)
        return super().forward(data, latents, cond, cond_attention_mask, cond_adaln)


class RINBackbone(nn.Module):
    def __init__(
        self,
        data_size: int,
        data_dim: int,
        num_input_channels: int,
        num_latents: int,
        latents_dim: int,
        label_dim: int,
        num_processing_layers: int,
        num_blocks: int,
        patch_size: int,
        num_cond_tokens: int,
        read_write_heads: int = 16,
        compute_heads: int = 16,
        latent_mlp_multiplier: int = 4,
        data_mlp_multiplier: int = 4,
        compute_dropout: float = 0.0,
        rw_stochastic_depth: float = 0.0,
        compute_stochastic_depth: float = 0.0,
        use_cond_rin_block: bool = False,
        concat_cond_token_to_latents: bool = True,
        use_self_conditioning: bool = False,
    ):
        super().__init__()
        self.latents_dim = latents_dim
        self.num_latents = num_latents
        self.embedding_initialization = "trunc_normal"
        self.linear_layer = nn.Linear
        self.time_embedder_activation = nn.SiLU
        self.use_cond_rin_block = use_cond_rin_block
        self.concat_cond_token_to_latents = concat_cond_token_to_latents
        self.use_self_conditioning = use_self_conditioning

        if concat_cond_token_to_latents:
            self.num_learned_latents = num_latents - num_cond_tokens - 1
        else:
            self.num_learned_latents = num_latents

        # Patch encoding
        self.patch_size = patch_size
        self.patch_extractor = nn.Linear(
            num_input_channels * patch_size * patch_size,
            data_dim,
            bias=True,
        )
        self.data_pos_embedding = nn.Parameter(
            torch.randn((data_size // patch_size) ** 2, data_dim),
            requires_grad=True,
        )
        self.data_pos_embedding.mup_type = "input"
        nn.init.trunc_normal_(
            self.data_pos_embedding, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )

        self.data_ln = RMSNorm(data_dim, eps=1e-6)
        # Latents

        self.latents = nn.Parameter(
            torch.randn(self.num_learned_latents, latents_dim),
            requires_grad=True,
        )
        self.latents.mup_type = "input"

        nn.init.trunc_normal_(self.latents, std=0.02, a=-2 * 0.02, b=2 * 0.02)

        self.time_embedder = TimeEmbedder(
            latents_dim // 4,
            expansion=4,
            linear_layer=partial(nn.Linear, bias=True),
            activation=nn.SiLU,
        )
        self.init_cond_mapping(label_dim, latents_dim, num_cond_tokens)

        if use_self_conditioning:
            self.ln_previous_1 = nn.LayerNorm(latents_dim, eps=1e-6)
            self.linear_previous = FusedMLP(
                dim_model=latents_dim,
                dropout=0.0,
                activation=nn.GELU,
                hidden_layer_multiplier=latent_mlp_multiplier,
            )
            self.ln_previous_2 = nn.LayerNorm(latents_dim, eps=1e-6)
            # Init the second LN to zero so the residual is a no-op at init time.
            if hasattr(self.ln_previous_2, "weight"):
                self.ln_previous_2.weight.data.fill_(0.0)
            if hasattr(self.ln_previous_2, "bias"):
                self.ln_previous_2.bias.data.fill_(0.0)

        # RIN blocks
        rin_block_kwargs = {
            "data_dim": data_dim,
            "latents_dim": latents_dim,
            "num_processing_layers": num_processing_layers,
            "read_write_heads": read_write_heads,
            "compute_heads": compute_heads,
            "latent_mlp_multiplier": latent_mlp_multiplier,
            "data_mlp_multiplier": data_mlp_multiplier,
            "compute_dropout": compute_dropout,
            "rw_stochastic_depth": rw_stochastic_depth,
            "compute_stochastic_depth": compute_stochastic_depth,
        }
        block_cls = RINBlockCond if use_cond_rin_block else RINBlock
        self.rin_blocks = nn.ModuleList([
            block_cls(**rin_block_kwargs)
            for i in range(num_blocks)
        ])

        self.map_tokens_to_patches = nn.Sequential(
            RMSNorm(data_dim, eps=1e-6),
            nn.Linear(
                data_dim,
                num_input_channels * patch_size * patch_size,
                bias=True,
            ),
        )
        self.init_weights()

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create patches
        x = batch["y"]
        gamma = batch["gamma"]
        previous_latents = batch.get("previous_latents", None)

        b, _, h, w = x.shape
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.patch_extractor(x)
        # Add positional embeddings
        x = self.data_ln(x) + self.data_pos_embedding.unsqueeze(0)

        # Cat latent tokens, conditioning tokens and timestep token
        encoded_noise = self.time_embedder(gamma)
        mapped_conditioning = self.cond_mapping(batch)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        cond = [
            encoded_noise.unsqueeze(1),
        ]
        if mapped_conditioning is not None:
            cond.append(mapped_conditioning)
        if self.concat_cond_token_to_latents:
            token_list = [
                latents,
                *cond,
            ]
            z = torch.cat(token_list, dim=1)
        else:
            z = latents + encoded_noise.unsqueeze(1)

        cond = torch.cat(cond, dim=1)

        cond_adaln = None

        if self.use_self_conditioning and previous_latents is not None:
            z = z + self.ln_previous_2(
                previous_latents
                + self.linear_previous(self.ln_previous_1(previous_latents))
            )

        for rin_block in self.rin_blocks:
            x, z = rin_block(
                x,
                z,
                cond=cond,
                cond_attention_mask=self.return_cond_mask(cond.shape[1], batch),
                cond_adaln=cond_adaln,
            )

        # Map tokens to patches
        x = self.map_tokens_to_patches(x)

        # Reshape to image
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            h=h // self.patch_size,
            w=w // self.patch_size,
        )

        return x, z

    def init_weights_(self, m):
        if (
            (
                hasattr(self.linear_layer, "func")
                and isinstance(m, self.linear_layer.func)
            )
            or (
                not hasattr(self.linear_layer, "func")
                and isinstance(m, self.linear_layer)
            )
            or isinstance(m, nn.Conv2d)
        ):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_weights(self):
        self.apply(self.init_weights_)

    def init_cond_mapping(self, label_dim: int, latents_dim: int, num_cond_tokens: int):
        raise NotImplementedError

    def cond_mapping(self, batch) -> Tensor:
        raise NotImplementedError

    def return_cond_mask(self, num_cond, batch) -> Tensor:
        raise NotImplementedError


class CADRINTextCond(RINBackbone):
    def __init__(
        self,
        *args,
        num_text_registers=16,
        coherence_keys=[],
        coherence_dropout=0.0,
        dropout_strategy="binomial",
        **kwargs,
    ):
        self.num_text_registers = num_text_registers
        self.dropout_strategy = dropout_strategy
        self.coherence_dropout = coherence_dropout
        self.coherence_keys = [f"{key}_coherence" for key in coherence_keys]
        if len(coherence_keys) == 0:
            raise ValueError("coherence_keys must be a list of keys")
        self.num_coherence_keys = len(coherence_keys)
        super().__init__(*args, **kwargs)

        #### This is necessary to make it work with torch.fx
        self.register_parameter(
            "registers_coherence_mask",
            nn.Parameter(
                torch.ones(
                    self.num_text_registers + self.num_coherence_keys, dtype=torch.bool
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "cond_mask",
            nn.Parameter(
                torch.ones(
                    self.num_text_registers + self.num_coherence_keys + 1,
                    dtype=torch.bool,
                ),
                requires_grad=False,
            ),
        )

    def init_cond_mapping(self, label_dim: int, latents_dim: int, num_cond_tokens: int):
        self.transformers_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=latents_dim,
                    num_heads=16,
                    mlp_multiplier=4,
                    dropout=0.0,
                    stochastic_depth=0.0,
                    use_biases=True,
                    use_layer_scale=True,
                    layer_scale_value=0.1,
                    use_qkv_biases=True,
                    use_qk_norm=False,
                    use_swiglu=False,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(2)
            ]
        )
        self.coherence_embedder = nn.ModuleDict(
            {
                key: TimeEmbedder(
                    latents_dim // 4,
                    expansion=4,
                    linear_layer=nn.Linear,
                    activation=self.time_embedder_activation,
                )
                for key in self.coherence_keys
            }
        )
        self.coherence_positional_embedding = nn.Parameter(
            torch.randn(self.num_coherence_keys, latents_dim),
            requires_grad=True,
        )
        self.coherence_positional_embedding.mup_type = "input"
        if self.embedding_initialization == "trunc_normal":
            nn.init.trunc_normal_(
                self.coherence_positional_embedding, std=0.02, a=-2 * 0.02, b=2 * 0.02
            )
        elif self.embedding_initialization == "normal":
            nn.init.normal_(self.coherence_positional_embedding)
        else:
            raise ValueError(
                f"Invalid embedding initialization {self.embedding_initialization}"
            )

        self.map_text_tokens = nn.Linear(
            label_dim,
            latents_dim,  # mup_type="input"
        )
        self.text_registers = nn.Parameter(
            torch.randn(self.num_text_registers, latents_dim),
            requires_grad=True,
        )
        self.text_registers.mup_type = "input"
        if self.embedding_initialization == "trunc_normal":
            nn.init.trunc_normal_(
                self.text_registers, std=0.02, a=-2 * 0.02, b=2 * 0.02
            )
        elif self.embedding_initialization == "normal":
            nn.init.normal_(self.text_registers)
        else:
            raise ValueError(
                f"Invalid embedding initialization {self.embedding_initialization}"
            )

    def init_weights(self):
        super().init_weights()

    def create_coherence_dropout_mask(
        self, batch_size: int, device: torch.device
    ) -> Tensor:
        if self.dropout_strategy == "binomial":
            mask = torch.bernoulli(
                torch.ones(batch_size, self.num_coherence_keys, device=device)
                * (1 - self.coherence_dropout)
            )
            # Ensure at least one entry is 1 per sample (no data-dependent branching)
            zero_rows = (mask.sum(dim=1) == 0).unsqueeze(1)
            random_indices = torch.randint(
                0, self.num_coherence_keys, (batch_size,), device=device
            )
            fallback = torch.zeros_like(mask)
            fallback.scatter_(1, random_indices.unsqueeze(1), 1.0)
            mask = torch.where(zero_rows, fallback, mask)
        elif self.dropout_strategy == "uniform":
            num_coherence_keys_to_drop = torch.randint(
                0, self.num_coherence_keys, (batch_size,), device=device
            )
            random_scores = torch.rand(
                batch_size, self.num_coherence_keys, device=device
            )
            _, indices = torch.sort(random_scores, dim=1)
            # Mask: 1 where rank >= num_to_drop, 0 otherwise
            ranks = torch.zeros_like(random_scores)
            ranks.scatter_(1, indices, torch.arange(
                self.num_coherence_keys, device=device, dtype=random_scores.dtype
            ).unsqueeze(0).expand(batch_size, -1))
            mask = (ranks >= num_coherence_keys_to_drop.unsqueeze(1)).float()
        elif self.dropout_strategy == "none":
            mask = torch.ones(
                batch_size, self.num_coherence_keys, device=device
            )
        else:
            raise ValueError(f"Invalid dropout strategy {self.dropout_strategy}")
        return mask

    def cond_mapping(self, batch) -> Tensor:
        embeddings = batch["text_tokens_embeddings"].float()
        mask = batch["text_tokens_mask"]

        batch_size, _, _ = embeddings.shape
        embeddings = self.map_text_tokens(embeddings)
        if self.training:
            coherence = torch.stack(
                [
                    self.coherence_embedder[key](
                        torch.nan_to_num(batch[key], nan=0.0).to(embeddings.dtype)
                    )
                    for key in self.coherence_keys
                ],
                dim=1,
            )
            coherence_mask = self.create_coherence_dropout_mask(
                batch_size, embeddings.device
            )
            coherence = coherence * coherence_mask.unsqueeze(-1)
        else:
            coherence = torch.stack(
                [
                    (
                        self.coherence_embedder[key](batch[key].to(embeddings.dtype))
                        if key in batch
                        else torch.full(
                            (batch_size, self.latents_dim),
                            float("nan"),
                            dtype=embeddings.dtype,
                            device=embeddings.device,
                        )
                    )
                    for key in self.coherence_keys
                ],
                dim=1,
            )
            coherence = torch.nan_to_num(coherence, nan=0.0)

        embeddings = torch.cat(
            [
                coherence
                + self.coherence_positional_embedding.unsqueeze(0).expand(
                    batch_size, -1, -1
                ),
                self.text_registers.unsqueeze(0).expand(batch_size, -1, -1),
                embeddings,
            ],
            dim=1,
        )
        registers_mask = self.registers_coherence_mask.unsqueeze(0).expand(
            batch_size, -1
        )
        mask = torch.cat(
            [
                registers_mask,
                mask,
            ],
            dim=1,
        )
        for block in self.transformers_blocks:
            embeddings = block(embeddings, token_mask=mask)
        return embeddings

    def return_cond_mask(self, num_cond, batch) -> Tensor:
        text_mask = batch["text_tokens_mask"]
        batch_size = text_mask.shape[0]
        mask = self.cond_mask.unsqueeze(0).expand(batch_size, -1)
        mask = torch.cat(
            [
                mask,
                text_mask,
            ],
            dim=1,
        )
        return mask
