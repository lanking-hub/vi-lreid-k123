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
        eval_fusion_mode="all_except_current",
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
        self.eval_fusion_mode = eval_fusion_mode
        self.experts = nn.ModuleDict()
        self.last_old_keys = []
        self.last_old_weights = None
        self.last_old_similarities = None
        self.last_old_weight_entropy = None
        self.last_mu_cur_norm = None
        self.last_old_mu_norms = None
        self.enable_branch_stats = False
        self.last_branch_outputs = None

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
            if self.eval_fusion_mode == "all_except_current":
                fusion_keys = [key for key in self.experts.keys() if int(key) != current_id]
            elif self.eval_fusion_mode == "previous":
                fusion_keys = [key for key in self.experts.keys() if int(key) < current_id]
            elif self.eval_fusion_mode == "current_only":
                fusion_keys = []
            else:
                raise ValueError("Unsupported eval_fusion_mode: {}".format(self.eval_fusion_mode))
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
            "eval_fusion_mode": self.eval_fusion_mode,
        }

    def _record_branch_outputs(self, current_out, old_scaled_out):
        if not self.enable_branch_stats:
            self.last_branch_outputs = None
            return
        self.last_branch_outputs = {
            "current": current_out.detach(),
            "old_scaled": old_scaled_out.detach(),
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
        self.last_branch_outputs = None
        if task_id is None:
            empty_out = x.new_zeros(*x.shape[:-1], self.out_features)
            self._record_branch_outputs(empty_out, empty_out)
            return empty_out

        key = self._task_key(task_id)
        if key not in self.experts:
            raise KeyError("task expert {} is not initialized".format(key))

        mu_cur = self._reduce_batch_mean(x)
        self._update_current_route_mu(key, mu_cur)

        current_out = self.experts[key](x)
        old_keys = self._get_fusion_task_keys(key)
        if not old_keys:
            self._record_route_debug(old_keys, None, None, mu_cur, None)
            self._record_branch_outputs(current_out, torch.zeros_like(current_out))
            return current_out

        weights, similarities, old_mu_norms = self._compute_old_weights(mu_cur, old_keys)
        old_out = x.new_zeros(*x.shape[:-1], self.out_features)
        with torch.no_grad():
            for idx, old_key in enumerate(old_keys):
                old_out = old_out + weights[idx].to(device=x.device, dtype=x.dtype) * self.experts[old_key](x)

        self._record_route_debug(old_keys, weights, similarities, mu_cur, old_mu_norms)
        old_scale = self.train_old_scale if self.training else self.eval_old_scale
        old_scaled_out = old_scale * old_out
        self._record_branch_outputs(current_out, old_scaled_out)
        return current_out + old_scaled_out


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
        self.enable_branch_stats = False
        self.last_branch_stats = None

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

    def _branch_stat_record(self, branch, delta, scaled_delta, mod=None, extra=None):
        with torch.no_grad():
            delta_detached = delta.detach().float()
            scaled_detached = scaled_delta.detach().float()
            batch_size = delta_detached.shape[0]
            delta_flat = delta_detached.reshape(batch_size, -1)
            scaled_flat = scaled_detached.reshape(batch_size, -1)
            delta_norm = torch.linalg.vector_norm(delta_flat, dim=1)
            scaled_norm = torch.linalg.vector_norm(scaled_flat, dim=1)
            delta_abs = delta_flat.abs().mean(dim=1)
            record = {
                'branch': branch,
                'count': torch.tensor(float(batch_size), device=delta.device),
                'delta_norm_sum': delta_norm.sum(),
                'delta_norm_sq_sum': (delta_norm * delta_norm).sum(),
                'delta_abs_sum': delta_abs.sum(),
                'scaled_delta_norm_sum': scaled_norm.sum(),
                'scaled_delta_norm_sq_sum': (scaled_norm * scaled_norm).sum(),
            }

            if mod is not None:
                mod_flat = mod.detach().reshape(-1).to(device=delta.device)
                rgb_index = mod_flat == 1
                ir_index = mod_flat == 0
                record['rgb_count'] = torch.tensor(0.0, device=delta.device)
                record['rgb_scaled_delta_norm_sum'] = torch.tensor(0.0, device=delta.device)
                record['rgb_scaled_delta_norm_sq_sum'] = torch.tensor(0.0, device=delta.device)
                record['ir_count'] = torch.tensor(0.0, device=delta.device)
                record['ir_scaled_delta_norm_sum'] = torch.tensor(0.0, device=delta.device)
                record['ir_scaled_delta_norm_sq_sum'] = torch.tensor(0.0, device=delta.device)
                if rgb_index.any():
                    rgb_norm = scaled_norm[rgb_index]
                    record['rgb_count'] = torch.tensor(float(rgb_norm.numel()), device=delta.device)
                    record['rgb_scaled_delta_norm_sum'] = rgb_norm.sum()
                    record['rgb_scaled_delta_norm_sq_sum'] = (rgb_norm * rgb_norm).sum()
                if ir_index.any():
                    ir_norm = scaled_norm[ir_index]
                    record['ir_count'] = torch.tensor(float(ir_norm.numel()), device=delta.device)
                    record['ir_scaled_delta_norm_sum'] = ir_norm.sum()
                    record['ir_scaled_delta_norm_sq_sum'] = (ir_norm * ir_norm).sum()

            if extra is not None:
                record.update(extra)
            return record

    def _k3_split_stat_record(self, current_scaled_delta, old_scaled_delta):
        with torch.no_grad():
            current_flat = current_scaled_delta.detach().float().reshape(current_scaled_delta.shape[0], -1)
            old_flat = old_scaled_delta.detach().float().reshape(old_scaled_delta.shape[0], -1)
            current_norm = torch.linalg.vector_norm(current_flat, dim=1)
            old_norm = torch.linalg.vector_norm(old_flat, dim=1)
            denom = current_norm + old_norm + 1e-12
            old_current_ratio = old_norm / (current_norm + 1e-12)
            old_fraction = old_norm / denom
            return {
                'k3_current_scaled_norm_sum': current_norm.sum(),
                'k3_current_scaled_norm_sq_sum': (current_norm * current_norm).sum(),
                'k3_old_scaled_norm_sum': old_norm.sum(),
                'k3_old_scaled_norm_sq_sum': (old_norm * old_norm).sum(),
                'k3_old_current_ratio_sum': old_current_ratio.sum(),
                'k3_old_current_ratio_sq_sum': (old_current_ratio * old_current_ratio).sum(),
                'k3_old_fraction_sum': old_fraction.sum(),
                'k3_old_fraction_sq_sum': (old_fraction * old_fraction).sum(),
            }

    def forward(self, x, mod=None, task_id=None):
        out = self.base(x)
        branch_stats = [] if self.enable_branch_stats else None
        mod_for_stats = None
        if self.enable_branch_stats and mod is not None:
            if torch.is_tensor(mod):
                mod_for_stats = mod.to(x.device).reshape(-1)
            else:
                mod_for_stats = torch.as_tensor(mod, device=x.device).reshape(-1)

        if self.shared_adapter is not None:
            delta_k1 = self.shared_adapter(x)
            scaled_k1 = self.gamma_k1 * delta_k1
            out = out + scaled_k1
            if branch_stats is not None:
                branch_stats.append(self._branch_stat_record(
                    'k1',
                    delta_k1,
                    scaled_k1,
                    mod=mod_for_stats,
                    extra={'gamma': self.gamma_k1.detach().float()},
                ))

        if self.use_k2:
            if mod is None:
                raise ValueError("mod must be provided when K2 branches are enabled")

            mod_mask = self._expand_mod_mask(mod, x)
            rgb_mask = (mod_mask == 1).to(dtype=out.dtype)
            ir_mask = (mod_mask == 0).to(dtype=out.dtype)

            delta_rgb = self.rgb_adapter(x)
            delta_ir = self.ir_adapter(x)
            scaled_rgb = self.gamma_rgb * delta_rgb
            scaled_ir = self.gamma_ir * delta_ir
            selected_delta = rgb_mask * delta_rgb + ir_mask * delta_ir
            selected_scaled = rgb_mask * scaled_rgb + ir_mask * scaled_ir
            out = out + selected_scaled
            if branch_stats is not None:
                branch_stats.append(self._branch_stat_record(
                    'k2',
                    selected_delta,
                    selected_scaled,
                    mod=mod_for_stats,
                    extra={
                        'gamma_rgb': self.gamma_rgb.detach().float(),
                        'gamma_ir': self.gamma_ir.detach().float(),
                    },
                ))

        if self.task_bank is not None and task_id is not None:
            self.task_bank.enable_branch_stats = branch_stats is not None
            delta_k3 = self.task_bank(x, task_id)
            scaled_k3 = self.gamma_k3 * delta_k3
            out = out + scaled_k3
            if branch_stats is not None:
                k3_extra = {'gamma': self.gamma_k3.detach().float()}
                branch_outputs = getattr(self.task_bank, 'last_branch_outputs', None)
                if branch_outputs is not None:
                    current_scaled = self.gamma_k3 * branch_outputs['current']
                    old_scaled = self.gamma_k3 * branch_outputs['old_scaled']
                    k3_extra.update(self._k3_split_stat_record(current_scaled, old_scaled))
                branch_stats.append(self._branch_stat_record(
                    'k3',
                    delta_k3,
                    scaled_k3,
                    mod=mod_for_stats,
                    extra=k3_extra,
                ))

        if branch_stats is not None:
            self.last_branch_stats = branch_stats

        return out
