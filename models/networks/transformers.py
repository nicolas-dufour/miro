import math

import torch
import torch.nn as nn
from einops import rearrange

# einops compile bug is back in 2.7
import einops._torch_specific
from torch import Tensor

torch.fx.wrap("rearrange")
from typing import Optional


class RMSNorm(nn.Module):
    """Float32-cast RMSNorm matching the legacy ``cad.models.networks.transformers.RMSNorm``.

    Why: ``torch.nn.RMSNorm`` computes the variance in the input dtype, which
    loses precision under bf16/fp16 autocast. The legacy module casts to fp32
    for the norm and back to input dtype — required to reproduce the
    ``miro/checkpoints_ground_truth/CC12M_256_RIN_small_flow_multi_cad_*``
    checkpoints' behaviour.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.elementwise_affine:
            return output * self.weight
        return output


class StochatichDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.survival_prob = 1.0 - p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.survival_prob < 1:
            mask = (
                torch.empty(x.shape[0], 1, 1, device=x.device).uniform_()
                + self.survival_prob
            )
            mask = mask.floor()
            if self.survival_prob > 0:
                mask = mask / self.survival_prob
            return x * mask
        else:
            return x


class FusedMLP(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_model * hidden_layer_multiplier, bias=bias)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_model * hidden_layer_multiplier, dim_model, bias=bias)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        hidden_layer = hidden_layer_multiplier * dim_model
        hidden_layer = int(hidden_layer * 2 / 3)
        hidden_layer = (
            hidden_layer
            if hidden_layer % 256 == 0
            else hidden_layer + 256 - hidden_layer % 256
        )
        self.linear_packed = nn.Linear(dim_model, 2 * hidden_layer, bias=bias)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_layer, dim_model, bias=bias)

    def forward(self, x):
        x = self.linear_packed(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * self.activation(x2)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


class CrossAttentionOp(nn.Module):
    def __init__(
        self,
        attention_dim,
        num_heads,
        dim_q,
        dim_kv,
        use_biases=True,
        is_sa=False,
        use_qkv_biases: bool = False,
        use_qk_norm: bool = True,
        qk_norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.is_sa = is_sa
        self.use_qk_norm = use_qk_norm
        if self.is_sa:
            self.qkv = nn.Linear(dim_q, attention_dim * 3, bias=use_qkv_biases)
        else:
            self.q = nn.Linear(dim_q, attention_dim, bias=use_qkv_biases)
            self.kv = nn.Linear(dim_kv, attention_dim * 2, bias=use_qkv_biases)
        self.out = nn.Linear(attention_dim, dim_q, bias=use_biases)
        if use_qk_norm:
            head_dim = attention_dim // num_heads
            self.q_norm = qk_norm_layer(head_dim, eps=1e-6)
            self.k_norm = qk_norm_layer(head_dim, eps=1e-6)

    def forward(self, x_to, x_from=None, attention_mask=None):
        if x_from is None:
            x_from = x_to
        if self.is_sa:
            q, k, v = self.qkv(x_to).chunk(3, dim=-1)
        else:
            q = self.q(x_to)
            k, v = self.kv(x_from).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        scale = 1.0 / math.sqrt(q.shape[-1])
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, scale=scale
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_qkv_biases: bool = False,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        norm_layer: nn.Module = RMSNorm,
    ):
        super().__init__()
        self.initial_to_ln = norm_layer(dim_q, eps=1e-6)
        if attention_dim == 0:
            attention_dim = min(dim_q, dim_kv)
        self.ca = CrossAttentionOp(
            attention_dim, num_heads, dim_q, dim_kv,
            is_sa=False, use_biases=use_biases,
            use_qkv_biases=use_qkv_biases, use_qk_norm=use_qk_norm,
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = norm_layer(dim_q, eps=1e-6)
        ffn_class = SwiGLU if use_swiglu else FusedMLP
        ffn_activation = nn.SiLU if use_swiglu else nn.GELU
        self.ffn = ffn_class(
            dim_model=dim_q, dropout=dropout, activation=ffn_activation,
            hidden_layer_multiplier=mlp_multiplier, bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)

        self.register_parameter(
            "attention_mask_dummy",
            nn.Parameter(torch.ones(1, 1, dtype=torch.bool), requires_grad=False),
        )

    def forward(
        self,
        to_tokens: Tensor,
        from_tokens: Tensor,
        to_token_mask: Optional[Tensor] = None,
        from_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if to_token_mask is None and from_token_mask is None:
            attention_mask = None
        else:
            if to_token_mask is None:
                to_token_mask = self.attention_mask_dummy.expand(
                    to_tokens.shape[0], to_tokens.shape[1],
                )
            if from_token_mask is None:
                from_token_mask = self.attention_mask_dummy.expand(
                    from_tokens.shape[0], from_tokens.shape[1],
                )
            attention_mask = from_token_mask.unsqueeze(1) * to_token_mask.unsqueeze(2)
        normed_to = self.initial_to_ln(to_tokens)
        attention_output = self.ca(normed_to, from_tokens, attention_mask=attention_mask)
        to_tokens = to_tokens + self.ca_stochastic_depth(attention_output)
        normed_middle = self.middle_ln(to_tokens)
        to_tokens = to_tokens + self.ffn_stochastic_depth(self.ffn(normed_middle))
        return to_tokens


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.1,
        use_qkv_biases: bool = False,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        norm_layer: nn.Module = RMSNorm,
    ):
        super().__init__()
        self.initial_ln = norm_layer(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.sa = CrossAttentionOp(
            attention_dim, num_heads, dim_qkv, dim_qkv,
            is_sa=True, use_biases=use_biases,
            use_qkv_biases=use_qkv_biases, use_qk_norm=use_qk_norm,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = norm_layer(dim_qkv, eps=1e-6)
        ffn_class = SwiGLU if use_swiglu else FusedMLP
        ffn_activation = nn.SiLU if use_swiglu else nn.GELU
        self.ffn = ffn_class(
            dim_model=dim_qkv, dropout=dropout, activation=ffn_activation,
            hidden_layer_multiplier=mlp_multiplier, bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

        self.register_parameter(
            "attention_mask_dummy",
            nn.Parameter(torch.ones(1, 1, dtype=torch.bool), requires_grad=False),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
    ):
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * self.attention_mask_dummy.expand(
                tokens.shape[0], tokens.shape[1],
            ).unsqueeze(2)
        attn_tokens = self.initial_ln(tokens)
        attention_output = self.sa(attn_tokens, attention_mask=attention_mask)
        if self.use_layer_scale:
            layer_scale_1 = self.layer_scale_1
            layer_scale_2 = self.layer_scale_2
        else:
            layer_scale_1 = 1.0
            layer_scale_2 = 1.0
        tokens = tokens + self.sa_stochastic_depth(layer_scale_1 * attention_output)
        tokens_ffn = self.middle_ln(tokens)
        tokens = tokens + self.ffn_stochastic_depth(layer_scale_2 * self.ffn(tokens_ffn))
        return tokens
