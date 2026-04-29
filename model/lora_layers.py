import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


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
    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        alpha=8,
        dropout=0.0,
        route_tau=1.0,
        route_momentum=0.9,
        route_eps=1e-8,
        train_old_scale=1.0,
        eval_old_scale=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.route_tau = route_tau
        self.route_momentum = route_momentum
        self.route_eps = route_eps
        self.train_old_scale = train_old_scale
        self.eval_old_scale = eval_old_scale
        self.experts = nn.ModuleDict()
        self.last_old_keys = []
        self.last_old_weights = None
        self.last_old_similarities = None
        self.last_old_weight_entropy = None
        self.last_mu_cur_norm = None
        self.last_old_mu_norms = None

    def _task_key(self, task_id):
        if torch.is_tensor(task_id):
            return str(int(task_id.item()))
        return str(int(task_id))

    def _route_mu_name(self, key):
        return f"route_mu_{key}"

    def _ensure_route_buffer(self, key, device=None):
        buffer_name = self._route_mu_name(key)
        if buffer_name not in self._buffers:
            self.register_buffer(
                buffer_name,
                torch.zeros(self.in_features, device=device, dtype=torch.float32),
            )

    def _get_route_mu(self, key):
        return getattr(self, self._route_mu_name(key))

    def _reduce_batch_mean(self, x):
        x_detached = x.detach()
        if x_detached.dim() == 3:
            x_detached = x_detached.mean(dim=1)
        elif x_detached.dim() != 2:
            raise ValueError("expected x to have shape [B, C] or [B, N, C], got {}".format(tuple(x.shape)))
        batch_sum = x_detached.sum(dim=0).to(dtype=torch.float32)
        batch_count = torch.tensor([x_detached.size(0)], device=x_detached.device, dtype=torch.float32)
        if self.training and dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        return batch_sum / batch_count.clamp_min(1.0)

    def _update_current_route_mu(self, key, mu_cur):
        if not self.training:
            return
        with torch.no_grad():
            route_mu = self._get_route_mu(key)
            route_mu.mul_(self.route_momentum).add_(mu_cur * (1.0 - self.route_momentum))

    def _get_fusion_task_keys(self, current_key):
        current_id = int(current_key)
        if self.training:
            fusion_keys = [key for key in self.experts.keys() if int(key) < current_id]
        else:
            fusion_keys = [key for key in self.experts.keys() if int(key) != current_id]
        return sorted(fusion_keys, key=int)

    def _compute_old_weights(self, mu_cur, old_keys):
        if not old_keys:
            return None, None, None

        similarities = []
        old_mu_norms = []
        mu_cur_2d = mu_cur.unsqueeze(0)
        for old_key in old_keys:
            mu_old = self._get_route_mu(old_key).detach().to(device=mu_cur.device, dtype=torch.float32)
            sim = F.cosine_similarity(mu_cur_2d, mu_old.unsqueeze(0), dim=-1, eps=self.route_eps).squeeze(0)
            sim = (sim + 1.0) / 2.0
            similarities.append(sim)
            old_mu_norms.append(mu_old.norm())

        similarities = torch.stack(similarities, dim=0)
        weights = F.softmax(similarities / self.route_tau, dim=0)
        weights = weights + self.route_eps
        weights = weights / weights.sum()
        old_mu_norms = torch.stack(old_mu_norms, dim=0)
        return weights, similarities, old_mu_norms

    def _record_route_debug(self, old_keys, weights, similarities, mu_cur, old_mu_norms):
        self.last_old_keys = list(old_keys)
        self.last_old_weights = None if weights is None else weights.detach().cpu()
        self.last_old_similarities = None if similarities is None else similarities.detach().cpu()
        self.last_old_weight_entropy = None
        if weights is not None and weights.numel() > 0:
            entropy = -(weights * torch.log(weights + self.route_eps)).sum()
            self.last_old_weight_entropy = entropy.detach().cpu()
        self.last_mu_cur_norm = mu_cur.detach().norm().cpu()
        self.last_old_mu_norms = None if old_mu_norms is None else old_mu_norms.detach().cpu()

    def get_route_debug_snapshot(self):
        active_old_scale = self.train_old_scale if self.training else self.eval_old_scale
        return {
            "old_keys": list(self.last_old_keys),
            "weights": None if self.last_old_weights is None else self.last_old_weights.clone(),
            "similarities": None if self.last_old_similarities is None else self.last_old_similarities.clone(),
            "entropy": self.last_old_weight_entropy,
            "mu_cur_norm": self.last_mu_cur_norm,
            "old_mu_norms": None if self.last_old_mu_norms is None else self.last_old_mu_norms.clone(),
            "tau": self.route_tau,
            "old_scale": active_old_scale,
            "train_old_scale": self.train_old_scale,
            "eval_old_scale": self.eval_old_scale,
        }

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
        self._ensure_route_buffer(key, device=device)

    def forward(self, x, task_id=None):
        if task_id is None:
            return x.new_zeros(*x.shape[:-1], self.out_features)

        key = self._task_key(task_id)
        if key not in self.experts:
            raise KeyError("task expert {} is not initialized".format(key))

        mu_cur = self._reduce_batch_mean(x)
        self._update_current_route_mu(key, mu_cur)

        current_out = self.experts[key](x)
        old_keys = self._get_fusion_task_keys(key)
        if not old_keys:
            self._record_route_debug(old_keys, None, None, mu_cur, None)
            return current_out

        weights, similarities, old_mu_norms = self._compute_old_weights(mu_cur, old_keys)
        old_out = x.new_zeros(*x.shape[:-1], self.out_features)
        with torch.no_grad():
            for idx, old_key in enumerate(old_keys):
                old_out = old_out + weights[idx].to(device=x.device, dtype=x.dtype) * self.experts[old_key](x)

        self._record_route_debug(old_keys, weights, similarities, mu_cur, old_mu_norms)
        old_scale = self.train_old_scale if self.training else self.eval_old_scale
        return current_out + old_scale * old_out


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
