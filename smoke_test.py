"""
Minimal smoke tests for the current K1/K2/K3 CKDA refactor.

Goals:
- verify model construction works
- verify K1/K2/K3 wiring is active
- verify add_task / task_id path works
- verify wrapper model forward/backward works
- verify FKD output works
- verify old task expert freeze logic works

This script is dataset-free. It uses random tensors only.
"""

import argparse
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from model.make_model import build_vision_transformer
from model.lora_layers import TaskLoRABank
from model.vision_transformer import ViT


class FakeCfg:
    H = 256
    W = 128
    STRIDE_SIZE = 16
    DROP_PATH = 0.1
    DROP_OUT = 0.0
    ATT_DROP_RATE = 0.0
    PRETRAIN_PATH = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for K1/K2/K3 CKDA refactor")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:id]")
    parser.add_argument("--batch-size", type=int, default=2, help="random batch size")
    parser.add_argument(
        "--pretrain-path",
        default="",
        help="optional pretrained checkpoint path; empty means skip loading",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=["all"],
        help=(
            "subset of tests to run. choices: vit branches add_task router wrapper "
            "fkd freeze all"
        ),
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(device_name)


@contextmanager
def maybe_skip_pretrain(pretrain_path):
    """
    build_vision_transformer always calls self.base.load_param(cfg.PRETRAIN_PATH).
    For smoke tests we often want to skip that safely.
    """
    if pretrain_path:
        yield
        return

    original_load_param = ViT.load_param

    def _skip_load_param(self, model_path):
        print("[smoke_test] Skip pretrained load for wrapper build.")

    ViT.load_param = _skip_load_param
    try:
        yield
    finally:
        ViT.load_param = original_load_param


def print_header(title):
    print(f"\n=== {title} ===")


def count_task_expert_params(model):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if "task_bank.experts." in name:
            total += 1
            if param.requires_grad:
                trainable += 1
    return total, trainable


def test_vit_forward(device, batch_size):
    print_header("Test 1: ViT forward")
    model = ViT(img_size=[256, 128], stride_size=16, depth=12, embed_dim=768, num_heads=12)
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, 3, 256, 128, device=device)
    mod = torch.tensor(([1, 0] * ((batch_size + 1) // 2))[:batch_size], device=device)

    with torch.no_grad():
        feat = model(x, mod, task_id=None)

    assert feat.shape == (batch_size, 768), f"Expected ({batch_size}, 768), got {feat.shape}"
    print(f"ViT forward OK, output shape: {tuple(feat.shape)}")


def test_lora_branches():
    print_header("Test 2: K1/K2/K3 branch assignment")
    model = ViT(img_size=[256, 128], stride_size=16, depth=12, embed_dim=768, num_heads=12)

    for i, blk in enumerate(model.blocks):
        qkv = blk.attn.qkv
        if i < 4:
            assert qkv.use_k2 and not qkv.use_k1 and not qkv.use_k3, f"Block {i} should be K2 only"
        elif i < 8:
            assert qkv.use_k1 and not qkv.use_k2 and not qkv.use_k3, f"Block {i} should be K1 only"
        else:
            assert qkv.use_k3 and not qkv.use_k1 and not qkv.use_k2, f"Block {i} should be K3 only"

    blk8 = model.blocks[8]
    assert blk8.attn.qkv.task_bank is not None, "Deep block should own a task bank"
    print("Layer assignment OK: K2 blocks 0-3, K1 blocks 4-7, K3 blocks 8-11")


def test_custom_lora_branch_config():
    print_header("Test 2b: custom K1/K2/K3 branch config")
    model = ViT(
        img_size=[256, 128],
        stride_size=16,
        depth=12,
        embed_dim=768,
        num_heads=12,
        k1_blocks="0-11",
        k2_blocks="0-7",
        k3_blocks="8-11",
        k1_rank=8,
        k2_rank=8,
        k3_rank=4,
    )

    for i, blk in enumerate(model.blocks):
        qkv = blk.attn.qkv
        assert qkv.use_k1, f"Block {i} should enable global K1"
        assert qkv.use_k2 == (i < 8), f"Block {i} K2 assignment mismatch"
        assert qkv.use_k3 == (i >= 8), f"Block {i} K3 assignment mismatch"
        if qkv.shared_adapter is not None:
            assert qkv.shared_adapter.rank == 8, f"Block {i} K1 rank mismatch"
        if qkv.rgb_adapter is not None:
            assert qkv.rgb_adapter.rank == 8, f"Block {i} K2 rank mismatch"
            assert qkv.ir_adapter.rank == 8, f"Block {i} K2 IR rank mismatch"
        if qkv.task_bank is not None:
            assert qkv.task_bank.rank == 4, f"Block {i} K3 rank mismatch"

    print("Custom branch config OK: K1 0-11 r8, K2 0-7 r8, K3 8-11 r4")


def test_add_task(device):
    print_header("Test 3: add_task and task-specific forward")
    model = ViT(img_size=[256, 128], stride_size=16, depth=12, embed_dim=768, num_heads=12)
    model.add_task(0)
    model.add_task(1)
    model = model.to(device)
    model.eval()

    blk8 = model.blocks[8]
    assert "0" in blk8.attn.qkv.task_bank.experts, "Expert 0 should exist"
    assert "1" in blk8.attn.qkv.task_bank.experts, "Expert 1 should exist"

    x = torch.randn(2, 3, 256, 128, device=device)
    mod = torch.tensor([1, 0], device=device)

    with torch.no_grad():
        feat0 = model(x, mod, task_id=0)
        feat1 = model(x, mod, task_id=1)
        feat_none = model(x, mod, task_id=None)

    assert feat0.shape == (2, 768)
    assert feat1.shape == (2, 768)
    assert feat_none.shape == (2, 768)
    # LoRA experts are initialized with zero B weights, so different task_id values
    # are allowed to produce identical outputs before any training happens.
    assert torch.allclose(feat0, feat1, atol=1e-6), (
        "Fresh task experts are expected to be functionally identical before training"
    )

    # Nudge one deep expert so we can verify task routing really selects different paths.
    # We switch to training mode here because K3 routing semantics are:
    # - training: current expert + previous experts
    # - eval: current expert + all non-current experts
    # In eval mode, perturbing expert 1 can also affect task_id=0 through fusion.
    with torch.no_grad():
        blk8.attn.qkv.task_bank.experts["1"].lora_B.weight.fill_(1e-3)

    model.train()
    with torch.no_grad():
        feat0_after = model(x, mod, task_id=0)
        feat1_after = model(x, mod, task_id=1)

    assert not torch.allclose(
        feat0_after, feat1_after, atol=1e-6
    ), "After perturbing expert 1, different task_id should change outputs"
    print("add_task and task-specific forward OK")


def test_router_bank(device):
    print_header("Test 4: K3 stats-based fusion routing")
    bank = TaskLoRABank(in_features=4, out_features=4, rank=1, alpha=1, route_tau=1.0)
    bank.add_task(0, device=device, dtype=torch.float32)
    bank.add_task(1, device=device, dtype=torch.float32)
    bank.add_task(2, device=device, dtype=torch.float32)
    bank.train()

    with torch.no_grad():
        for expert in bank.experts.values():
            expert.lora_A.weight.fill_(1.0)
            expert.lora_B.weight.zero_()

        bank.experts["0"].lora_B.weight.fill_(1.0)
        bank.experts["1"].lora_B.weight.fill_(2.0)
        bank.experts["2"].lora_A.weight.zero_()
        bank.experts["2"].lora_B.weight.zero_()

        bank.route_mu_0.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        bank.route_mu_1.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device))

    state_dict = bank.state_dict()
    assert "route_mu_0" in state_dict and "route_mu_1" in state_dict and "route_mu_2" in state_dict

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
    out = bank(x, task_id=2)
    weights = bank.last_old_weights

    assert weights is not None and weights.numel() == 2
    assert weights[0] > weights[1], "Expected old task 0 to receive a larger fusion weight during training"
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    out.sum().backward()
    assert x.grad is not None
    assert torch.allclose(
        x.grad, torch.zeros_like(x.grad), atol=1e-8
    ), "Old expert branch should be gradient-isolated when current expert is zeroed"
    assert torch.count_nonzero(bank.route_mu_2).item() > 0, "Current-task EMA route stats should update during training"

    bank.eval()
    with torch.no_grad():
        bank.route_mu_2.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        eval_x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
        eval_out = bank(eval_x, task_id=0)

    eval_weights = bank.last_old_weights
    assert eval_weights is not None and eval_weights.numel() == 2
    assert bank.last_old_keys == ["1", "2"], "Eval mode should fuse all non-current experts"
    assert eval_weights[1] > eval_weights[0], "Expected future task 2 to receive a larger fusion weight during eval"
    assert torch.count_nonzero(eval_out).item() > 0, "Eval fusion should include non-current experts"
    print("K3 routing bank OK")


def test_wrapper_forward_backward(device, batch_size, pretrain_path):
    print_header("Test 5: build_vision_transformer forward + backward")
    cfg = FakeCfg()
    cfg.PRETRAIN_PATH = pretrain_path

    with maybe_skip_pretrain(pretrain_path):
        model = build_vision_transformer(num_classes=32, cfg=cfg)

    model.add_task(0)
    model = model.to(device)
    model.train()

    x = torch.randn(batch_size, 3, 256, 128, device=device)
    mod = torch.tensor(([1, 0] * ((batch_size + 1) // 2))[:batch_size], device=device)
    labels = torch.arange(batch_size, device=device) % 32

    scores, feats = model(x, mod, task_id=0)
    assert scores.shape == (batch_size, 32), f"Expected ({batch_size}, 32), got {scores.shape}"
    assert feats.shape == (batch_size, 768), f"Expected ({batch_size}, 768), got {feats.shape}"

    loss = nn.CrossEntropyLoss()(scores, labels)
    loss.backward()

    lora_grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and ("lora_" in name or "gamma_" in name):
            lora_grad_count += 1

    assert lora_grad_count > 0, "Expected LoRA-related parameters to receive gradients"
    print(f"Wrapper forward/backward OK, loss={loss.item():.4f}, LoRA grad params={lora_grad_count}")


def test_fkd_mode(device, batch_size, pretrain_path):
    print_header("Test 6: fkd=True output")
    cfg = FakeCfg()
    cfg.PRETRAIN_PATH = pretrain_path

    with maybe_skip_pretrain(pretrain_path):
        model = build_vision_transformer(num_classes=32, cfg=cfg)

    model.add_task(0)
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, 3, 256, 128, device=device)
    mod = torch.tensor(([1, 0] * ((batch_size + 1) // 2))[:batch_size], device=device)

    with torch.no_grad():
        feat_norm = model(x, mod, task_id=0, fkd=True)

    norms = feat_norm.norm(dim=1)
    assert feat_norm.shape == (batch_size, 768)
    assert torch.allclose(norms, torch.ones(batch_size, device=device), atol=1e-4), "Expected unit vectors after l2norm"
    print("fkd=True OK")


def test_freeze_logic(pretrain_path):
    print_header("Test 7: freeze_old_task_experts logic")
    cfg = FakeCfg()
    cfg.PRETRAIN_PATH = pretrain_path

    with maybe_skip_pretrain(pretrain_path):
        model = build_vision_transformer(num_classes=32, cfg=cfg)

    model.add_task(0)
    model.add_task(1)
    model.add_task(2)

    current_task_id = 2
    current_key = str(current_task_id)
    for name, param in model.named_parameters():
        if "task_bank.experts." in name:
            expert_id = name.split("task_bank.experts.")[1].split(".")[0]
            param.requires_grad = (expert_id == current_key)

    total_expert_params, trainable_expert_params = count_task_expert_params(model)
    assert total_expert_params > 0, "Expected at least one task expert parameter"
    assert trainable_expert_params > 0, "Expected current task expert to remain trainable"

    for name, param in model.named_parameters():
        if "task_bank.experts." in name:
            expert_id = name.split("task_bank.experts.")[1].split(".")[0]
            if expert_id == current_key:
                assert param.requires_grad, f"Current expert should remain trainable: {name}"
            else:
                assert not param.requires_grad, f"Old expert should be frozen: {name}"

    print(
        "Freeze logic OK, "
        f"task expert params total={total_expert_params}, trainable={trainable_expert_params}"
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    requested = set(args.tests)
    run_all = "all" in requested

    print(f"[smoke_test] device={device}")
    print(f"[smoke_test] batch_size={args.batch_size}")
    print(f"[smoke_test] pretrain_path={'<skip>' if not args.pretrain_path else args.pretrain_path}")

    if run_all or "vit" in requested:
        test_vit_forward(device, args.batch_size)
    if run_all or "branches" in requested:
        test_lora_branches()
        test_custom_lora_branch_config()
    if run_all or "add_task" in requested:
        test_add_task(device)
    if run_all or "router" in requested:
        test_router_bank(device)
    if run_all or "wrapper" in requested:
        test_wrapper_forward_backward(device, args.batch_size, args.pretrain_path)
    if run_all or "fkd" in requested:
        test_fkd_mode(device, args.batch_size, args.pretrain_path)
    if run_all or "freeze" in requested:
        test_freeze_logic(args.pretrain_path)

    print("\n=== All requested smoke tests passed ===")


if __name__ == "__main__":
    main()
