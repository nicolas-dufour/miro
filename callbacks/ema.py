import contextlib
import copy
import itertools
from typing import List, Union

import torch
from pytorch_lightning import Callback
from torch.distributed.fsdp import FullyShardedDataParallel


def _decay_to_attr_suffix(decay: float) -> str:
    """Convert decay rate to a valid attribute name suffix (e.g., 0.9999 -> '_0_9999')."""
    return "_" + str(decay).replace(".", "_")


class EMACallback(Callback):
    """
    EMA (Exponential Moving Average) callback that supports multiple decay rates.

    When multiple decay rates are provided:
    - The first decay rate's EMA is stored in `ema_module_attr_name` (e.g., `ema_network`)
    - Other EMAs are stored with suffixed names (e.g., `ema_network_0_999`)

    Module naming is always consistent (first = base name, others = suffixed).
    The default_ema_index only affects which EMA the DiffusionModule uses for inference.

    Args:
        module_attr_name: Name of the source module attribute (e.g., "network")
        ema_module_attr_name: Base name for EMA modules (e.g., "ema_network")
        decay: Single decay rate or list of decay rates
        start_ema_step: Step at which to start EMA updates
        init_ema_random: Whether to initialize EMA with random weights
        default_ema_index: Index of the decay rate to use for inference (doesn't affect naming)
    """

    def __init__(
        self,
        module_attr_name: str,
        ema_module_attr_name: str,
        decay: Union[float, List[float]] = 0.999,
        start_ema_step: int = 0,
        init_ema_random: bool = True,
        default_ema_index: int = 0,
    ):
        super().__init__()
        # Normalize decay to a list
        if isinstance(decay, (int, float)):
            self.decays = [float(decay)]
        else:
            self.decays = [float(d) for d in decay]

        if default_ema_index < 0 or default_ema_index >= len(self.decays):
            raise ValueError(
                f"default_ema_index ({default_ema_index}) must be in range [0, {len(self.decays)-1}]"
            )

        self.module_attr_name = module_attr_name
        self.ema_module_attr_name = ema_module_attr_name
        self.start_ema_step = start_ema_step
        self.init_ema_random = init_ema_random
        self.default_ema_index = default_ema_index

        # Build list of (decay, attr_name) pairs
        # Naming is consistent: first decay uses base name, others get suffixed
        # (default_ema_index does NOT affect naming, only inference selection)
        self.ema_configs = []
        for i, d in enumerate(self.decays):
            if i == 0:
                attr_name = self.ema_module_attr_name
            else:
                attr_name = self.ema_module_attr_name + _decay_to_attr_suffix(d)
            self.ema_configs.append((d, attr_name))

    @property
    def decay(self) -> float:
        """Return the primary (first) decay rate for backward compatibility."""
        return self.decays[0]

    def get_ema_module_names(self) -> List[str]:
        """Return list of all EMA module attribute names."""
        return [attr_name for _, attr_name in self.ema_configs]

    def on_train_start(self, trainer, pl_module):
        if pl_module.global_step == 0:
            if not hasattr(pl_module, self.module_attr_name):
                raise ValueError(
                    f"Module {pl_module} does not have attribute {self.module_attr_name}"
                )
            # Create EMA modules for all decay rates
            for decay, attr_name in self.ema_configs:
                if not hasattr(pl_module, attr_name):
                    pl_module.add_module(
                        attr_name,
                        copy.deepcopy(getattr(pl_module, self.module_attr_name))
                        .eval()
                        .requires_grad_(False),
                    )
            self.reset_ema(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step == self.start_ema_step:
            self.reset_ema(pl_module)
        elif (
            pl_module.global_step < self.start_ema_step
            and pl_module.global_step % 100 == 0
        ):
            # Slow EMA updates for visualization (same rate for all)
            self._update_all_emas(pl_module, override_decay=0.9)
        elif pl_module.global_step > self.start_ema_step:
            self._update_all_emas(pl_module)

    def _update_all_emas(self, pl_module, override_decay: float = None):
        """Update all EMA modules."""
        for decay, attr_name in self.ema_configs:
            effective_decay = override_decay if override_decay is not None else decay
            self._update_single_ema(pl_module, attr_name, effective_decay)

    def _update_single_ema(self, pl_module, ema_attr_name: str, decay: float):
        """Update a single EMA module with the given decay rate."""
        ema_module = getattr(pl_module, ema_attr_name)
        module = getattr(pl_module, self.module_attr_name)
        context_manager = self.get_model_context_manager(module)
        with context_manager:
            with torch.no_grad():
                ema_params = ema_module.state_dict()
                for name, param in itertools.chain(
                    module.named_parameters(), module.named_buffers()
                ):
                    if name in ema_params:
                        if param.requires_grad:
                            # Standard EMA: ema <- decay*ema + (1-decay)*param.
                            ema_params[name].mul_(decay).add_(
                                param.detach(), alpha=1 - decay
                            )

    def update_ema(self, pl_module, decay: float = None):
        """
        Update EMA for backward compatibility.
        If decay is None, updates all EMAs with their configured rates.
        If decay is provided, updates only the primary EMA with that rate.
        """
        if decay is None:
            self._update_all_emas(pl_module)
        else:
            # Update only the primary EMA with the given decay
            self._update_single_ema(pl_module, self.ema_module_attr_name, decay)

    def get_model_context_manager(self, module):
        fsdp_enabled = is_model_fsdp(module)
        model_context_manager = contextlib.nullcontext()
        if fsdp_enabled:
            model_context_manager = module.summon_full_params(module)
        return model_context_manager

    def reset_ema(self, pl_module):
        """Reset all EMA modules."""
        for _, attr_name in self.ema_configs:
            self._reset_single_ema(pl_module, attr_name)

    def _reset_single_ema(self, pl_module, ema_attr_name: str):
        """Reset a single EMA module."""
        ema_module = getattr(pl_module, ema_attr_name)
        if self.init_ema_random:
            ema_module.init_weights()
        else:
            module = getattr(pl_module, self.module_attr_name)
            context_manager = self.get_model_context_manager(module)
            with context_manager:
                ema_params = ema_module.state_dict()
                for name, param in itertools.chain(
                    module.named_parameters(), module.named_buffers()
                ):
                    if name in ema_params:
                        ema_params[name].copy_(param.detach())


def is_model_fsdp(model: torch.nn.Module) -> bool:
    try:
        if isinstance(model, FullyShardedDataParallel):
            return True

        # Check if model is wrapped with FSDP
        for _, obj in model.named_children():
            if isinstance(obj, FullyShardedDataParallel):
                return True
        return False
    except ImportError:
        return False
