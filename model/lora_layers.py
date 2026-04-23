import math

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be a positive integer")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_A._skip_vit_init = True
        self.lora_B._skip_vit_init = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class TaskLoRABank(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.experts = nn.ModuleDict()

    def _task_key(self, task_id):
        return str(int(task_id))

    def add_task(self, task_id, device=None, dtype=None):
        key = self._task_key(task_id)
        if key not in self.experts:
            expert = LoRAAdapter(
                self.in_features,
                self.out_features,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            )
            if device is not None or dtype is not None:
                expert = expert.to(device=device, dtype=dtype)
            self.experts[key] = expert

    def forward(self, x, task_id=None):
        if task_id is None:
            return x.new_zeros(*x.shape[:-1], self.out_features)

        key = self._task_key(task_id)
        if key not in self.experts:
            raise KeyError("task expert {} is not initialized".format(key))
        return self.experts[key](x)


class LoRALinearWithBranches(nn.Module):
    def __init__(
        self,
        base_linear,
        use_k1=True,
        use_k2=False,
        use_k3=False,
        rank=4,
        alpha=8,
        dropout=0.0,
        init_scale=1e-3,
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an instance of nn.Linear")

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.use_k1 = use_k1
        self.use_k2 = use_k2
        self.use_k3 = use_k3

        self.shared_adapter = (
            LoRAAdapter(self.in_features, self.out_features, rank=rank, alpha=alpha, dropout=dropout)
            if use_k1
            else None
        )
        self.rgb_adapter = (
            LoRAAdapter(self.in_features, self.out_features, rank=rank, alpha=alpha, dropout=dropout)
            if use_k2
            else None
        )
        self.ir_adapter = (
            LoRAAdapter(self.in_features, self.out_features, rank=rank, alpha=alpha, dropout=dropout)
            if use_k2
            else None
        )
        self.task_bank = (
            TaskLoRABank(self.in_features, self.out_features, rank=rank, alpha=alpha, dropout=dropout)
            if use_k3
            else None
        )

        self.gamma_k1 = nn.Parameter(torch.tensor(init_scale)) if use_k1 else None
        self.gamma_rgb = nn.Parameter(torch.tensor(init_scale)) if use_k2 else None
        self.gamma_ir = nn.Parameter(torch.tensor(init_scale)) if use_k2 else None
        self.gamma_k3 = nn.Parameter(torch.tensor(init_scale)) if use_k3 else None

    def add_task(self, task_id):
        if self.task_bank is not None:
            self.task_bank.add_task(
                task_id,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )

    def _expand_mod_mask(self, mod, x):
        if not torch.is_tensor(mod):
            mod = torch.as_tensor(mod, device=x.device)
        else:
            mod = mod.to(x.device)

        mod = mod.reshape(-1)
        if mod.numel() != x.shape[0]:
            raise ValueError(
                "mod shape mismatch: expected batch size {}, got {}".format(x.shape[0], mod.numel())
            )

        expand_shape = [x.shape[0]] + [1] * (x.dim() - 1)
        return mod.view(*expand_shape)

    def forward(self, x, mod=None, task_id=None):
        out = self.base(x)

        if self.shared_adapter is not None:
            out = out + self.gamma_k1 * self.shared_adapter(x)

        if self.use_k2:
            if mod is None:
                raise ValueError("mod must be provided when K2 branches are enabled")

            mod_mask = self._expand_mod_mask(mod, x)
            rgb_mask = (mod_mask == 1).to(dtype=out.dtype)
            ir_mask = (mod_mask == 0).to(dtype=out.dtype)

            delta_rgb = self.rgb_adapter(x)
            delta_ir = self.ir_adapter(x)
            out = out + rgb_mask * (self.gamma_rgb * delta_rgb) + ir_mask * (self.gamma_ir * delta_ir)

        if self.task_bank is not None and task_id is not None:
            out = out + self.gamma_k3 * self.task_bank(x, task_id)

        return out
