"""Debug-oriented copy of train.py for small-scale validation runs."""

import argparse
import builtins as __builtin__
from contextlib import nullcontext
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from config.config import cfg
from dataloader import (
    LLCMData,
    RegDBData,
    SYSUData,
    TestData_Inf,
    TestData_Vis,
    TrainData,
    VCMData,
    GenIdx,
    IdentitySampler,
)
from datamanager import (
    VCM,
    VideoDataset_test_Inf,
    VideoDataset_test_Vis,
    process_gallery_llcm,
    process_gallery_sysu,
    process_gallery_vcm,
    process_query_llcm,
    process_query_sysu,
    process_query_vcm,
    process_test_regdb,
    process_train_llcm,
    process_train_regdb,
    process_train_sysu,
    process_train_vcm,
)
from eval_metrics import eval_regdb, eval_sysu
from loss.Triplet import TripletLoss
from loss.TripletCross import TripletCrossLoss
from model.make_model import build_vision_transformer
from optimizer import make_optimizer
from scheduler import create_scheduler
from transforms import transform_rgb, transform_thermal, transform_test
from utils import AverageMeter, Logger, cosine_similarity, get_normal_affinity, get_old_proto, set_seed


parser = argparse.ArgumentParser(description="PMT Training (Debug Copy)")
parser.add_argument('--config_file', default='config/SYSU.yml',
                    help='path to config file', type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--model_path', default='save_model/',
                    help='model save path', type=str)
parser.add_argument('--num_workers', default=8,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=128,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
parser.add_argument("--logs-dir", help="Modify config options using the command-line",
                    default='logs_debug_3stage', type=str)
parser.add_argument('--setting', type=int, default=6, choices=[1, 2, 3, 4, 5, 6, 7], help="training order setting")
parser.add_argument('--ema_weight', type=float, default=0.7)
parser.add_argument('--proto_weight', type=float, default=1.0)
parser.add_argument('--inter_weight', type=float, default=0.5)
parser.add_argument('--new_weight', type=float, default=1.0)
parser.add_argument('--cross_weight', type=float, default=0.5)
parser.add_argument('--debug_batch_size', type=int, default=8, help='smaller batch size for quick validation')
parser.add_argument('--debug_test_batch', type=int, default=32, help='smaller test batch for quick validation')
parser.add_argument('--debug_num_workers', type=int, default=4, help='smaller worker count for quick validation')
parser.add_argument('--debug_max_epoch', type=int, default=3, help='run only a few epochs in debug validation')
parser.add_argument('--ddp_consistency_debug', action='store_true', help='reduce nondeterminism for single-vs-DDP debug comparisons')
parser.add_argument('--freeze_backbone', action='store_true', help='freeze pretrained ViT backbone weights and keep LoRA/task heads trainable')
parser.add_argument('--route_tau', type=float, default=1.0, help='softmax temperature for K3 old-expert routing')
parser.add_argument('--train_k3_old_scale', type=float, default=1.0, help='scaling factor for fused old K3 experts during training')
parser.add_argument('--train_k3_old_scale_start', type=float, default=None,
                    help='optional starting scale for scheduled training-time old K3 expert fusion')
parser.add_argument('--train_k3_old_scale_warmup_epochs', type=int, default=0,
                    help='number of epochs used to ramp train_k3_old_scale_start to train_k3_old_scale')
parser.add_argument('--train_k3_old_scale_schedule', type=str, default='constant',
                    choices=['constant', 'linear', 'cosine'],
                    help='schedule for training-time old K3 expert fusion scale')
parser.add_argument('--eval_k3_old_scale', type=float, default=1.0, help='scaling factor for fused old K3 experts during evaluation')
parser.add_argument('--eval_k3_fusion_mode', type=str, default='all_except_current',
                    choices=['all_except_current', 'previous', 'current_only'],
                    help='K3 expert fusion mode used by the model during evaluation/testing')
parser.add_argument('--distill_k3_fusion_mode', type=str, default='all_except_current',
                    choices=['all_except_current', 'previous', 'current_only'],
                    help='K3 expert fusion mode used by the frozen old model during distillation')
parser.add_argument('--compare_eval_k3_fusion_modes', type=str, default='',
                    help='optional comma-separated extra eval K3 fusion modes to test at each stage end')
parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend for torchrun launches')
parser.add_argument('--log_branch_stats', action='store_true', help='record K1/K2/K3 branch contribution statistics during training')
parser.add_argument('--branch_log_interval', type=int, default=1, help='record branch statistics every N epochs')
parser.add_argument('--branch_log_blocks', type=str, default='0,3,4,7,8,11', help='comma-separated ViT block ids for branch statistics')
parser.add_argument('--k1_blocks', type=str, default='4-7',
                    help='comma-separated/range ViT block ids for K1 shared LoRA, e.g. 0-11')
parser.add_argument('--k2_blocks', type=str, default='0-3',
                    help='comma-separated/range ViT block ids for K2 modality LoRA, e.g. 0-7')
parser.add_argument('--k3_blocks', type=str, default='8-11',
                    help='comma-separated/range ViT block ids for K3 task LoRA, e.g. 8-11')
parser.add_argument('--k1_rank', type=int, default=4, help='LoRA rank for K1 shared adapters')
parser.add_argument('--k2_rank', type=int, default=4, help='LoRA rank for K2 modality adapters')
parser.add_argument('--k3_rank', type=int, default=4, help='LoRA rank for K3 task adapters')
parser.add_argument('--k1_alpha', type=int, default=8, help='LoRA alpha for K1 shared adapters')
parser.add_argument('--k2_alpha', type=int, default=8, help='LoRA alpha for K2 modality adapters')
parser.add_argument('--k3_alpha', type=int, default=8, help='LoRA alpha for K3 task adapters')
parser.add_argument('--k1_xmod_align_weight', type=float, default=0.0,
                    help='weight for K1 cross-modality identity SupCon alignment')
parser.add_argument('--k1_xmod_align_temp', type=float, default=0.1,
                    help='temperature for K1 cross-modality identity SupCon alignment')
parser.add_argument('--k1_xmod_align_source', type=str, default='block_output',
                    choices=['block_output', 'scaled_delta', 'scaled_delta_detached'],
                    help='feature source for K1 cross-modality alignment')
parser.add_argument('--k1_align_blocks', type=str, default='4,7',
                    help='comma-separated ViT block ids used for K1 cross-modality alignment')
parser.add_argument('--k1_align_modules', type=str, default='qkv,proj',
                    help='comma-separated attention modules used for K1 delta alignment')
parser.add_argument('--k1_norm_guard_weight', type=float, default=0.0,
                    help='optional weight for K1 scaled-delta norm floor loss')
parser.add_argument('--k1_norm_guard_target', type=float, default=0.02,
                    help='target mean norm for optional K1 scaled-delta norm floor loss')
parser.add_argument('--k3_topk_old', type=int, default=0,
                    help='top-k old experts for K3 fusion (0=all, 1=top-1, etc.)')
parser.add_argument('--k3_gate_mode', type=str, default='none',
                    choices=['none', 'max_weight', 'margin'],
                    help='K3 old fusion confidence gate mode')
parser.add_argument('--k3_gate_threshold', type=float, default=0.0,
                    help='confidence threshold for K3 gate (gate=0 below this)')
parser.add_argument('--k3_gate_min', type=float, default=0.0,
                    help='minimum gate value even when confidence is below threshold')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

ARTIFACT_WAIT_SECONDS = 300.0
STAGE_DONE_WAIT_SECONDS = 1800.0


def setup_for_distributed(is_master):
    builtin_print = __builtin__.print

    def print(*print_args, **print_kwargs):
        force = print_kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*print_args, **print_kwargs)

    __builtin__.print = print


def init_distributed_mode():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1

    if distributed:
        backend = args.dist_backend if torch.cuda.is_available() else 'gloo'
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://')

    return distributed, rank, world_size, local_rank


distributed, rank, world_size, local_rank = init_distributed_mode()
device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
setup_for_distributed(rank == 0)

cfg.merge_from_list(args.opts)
cfg.defrost()
cfg.BATCH_SIZE = args.debug_batch_size
cfg.MAX_EPOCH = args.debug_max_epoch
cfg.K1_BLOCKS = args.k1_blocks
cfg.K2_BLOCKS = args.k2_blocks
cfg.K3_BLOCKS = args.k3_blocks
cfg.K1_LORA_RANK = args.k1_rank
cfg.K2_LORA_RANK = args.k2_rank
cfg.K3_LORA_RANK = args.k3_rank
cfg.K1_LORA_ALPHA = args.k1_alpha
cfg.K2_LORA_ALPHA = args.k2_alpha
cfg.K3_LORA_ALPHA = args.k3_alpha
cfg.freeze()
print("==========\nArgs:{}\n==========".format(args))
if cfg.BATCH_SIZE % cfg.NUM_POS != 0:
    raise ValueError('cfg.BATCH_SIZE={} must be divisible by cfg.NUM_POS={} so IdentitySampler keeps whole identities per rank'.format(
        cfg.BATCH_SIZE, cfg.NUM_POS))
if distributed:
    print('[!INFO] DDP batch config: global_batch_size={}, local_batch_size={}, world_size={}'.format(
        cfg.BATCH_SIZE * world_size, cfg.BATCH_SIZE, world_size))
if args.ddp_consistency_debug:
    print('[!INFO] DDP consistency debug enabled: shared seed, num_workers=0, amp_disabled, light_determinism')

seed_value = cfg.SEED if args.ddp_consistency_debug else (cfg.SEED + rank)
set_seed(seed_value)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = not args.ddp_consistency_debug
    torch.backends.cudnn.deterministic = args.ddp_consistency_debug
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.setting == 1:
    training_set = ['regdb', 'sysu', 'llcm', 'vcm']
elif args.setting == 6:
    training_set = ['regdb', 'sysu', 'llcm']
elif args.setting == 7:
    training_set = ['regdb']
else:
    raise ValueError("Unsupported setting {} in current script".format(args.setting))


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def is_main_process():
    return rank == 0


def barrier():
    if distributed:
        if device.type == 'cuda':
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()


if rank == 0 and os.path.exists(args.logs_dir) is False:
    os.makedirs(args.logs_dir)
if rank == 0:
    for filename in os.listdir(args.logs_dir):
        if filename.endswith('_stage.done'):
            os.remove(osp.join(args.logs_dir, filename))
barrier()
if rank == 0:
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))


def reduce_mean_scalar(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float64)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def all_gather_tensor(tensor, with_grad=False):
    if not distributed:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    if with_grad:
        gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def wait_for_file(path, timeout_seconds=ARTIFACT_WAIT_SECONDS, poll_seconds=0.2):
    if osp.exists(path):
        return
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if osp.exists(path):
            return
        time.sleep(poll_seconds)
    raise FileNotFoundError('Timed out after {:.1f}s waiting for file: {}'.format(timeout_seconds, path))


def autocast_context():
    if device.type == 'cuda' and not args.ddp_consistency_debug:
        return torch.amp.autocast('cuda', enabled=True)
    return nullcontext()


def create_grad_scaler():
    if device.type == 'cuda' and not args.ddp_consistency_debug:
        return torch.amp.GradScaler('cuda', enabled=True)
    return torch.amp.GradScaler('cuda', enabled=False)


def stage_done_path(stage_name):
    return osp.join(args.logs_dir, '{}_stage.done'.format(stage_name))


def append_stage_results(stage_name, stage_idx, metrics, eval_k3_fusion_mode=None):
    results_path = osp.join(args.logs_dir, 'results.jsonl')
    summary_path = osp.join(args.logs_dir, 'results_summary.tsv')
    eval_mode = args.eval_k3_fusion_mode if eval_k3_fusion_mode is None else eval_k3_fusion_mode
    row = {
        'stage': stage_name,
        'stage_idx': int(stage_idx),
        'metrics': metrics,
        'avg_mAP': float(np.mean([item['mAP'] for item in metrics])),
        'avg_R1': float(np.mean([item['R1'] for item in metrics])),
        'route_tau': float(args.route_tau),
        'train_k3_old_scale': float(args.train_k3_old_scale),
        'train_k3_old_scale_start': None if args.train_k3_old_scale_start is None else float(args.train_k3_old_scale_start),
        'train_k3_old_scale_warmup_epochs': int(args.train_k3_old_scale_warmup_epochs),
        'train_k3_old_scale_schedule': args.train_k3_old_scale_schedule,
        'eval_k3_old_scale': float(args.eval_k3_old_scale),
        'debug_max_epoch': int(args.debug_max_epoch),
        'debug_batch_size': int(args.debug_batch_size),
        'world_size': int(world_size),
        'eval_k3_fusion_mode': eval_mode,
        'distill_k3_fusion_mode': args.distill_k3_fusion_mode,
        'k1_xmod_align_source': args.k1_xmod_align_source,
        'k1_xmod_align_weight': float(args.k1_xmod_align_weight),
        'k1_xmod_align_temp': float(args.k1_xmod_align_temp),
        'k1_align_blocks': args.k1_align_blocks,
        'k1_align_modules': args.k1_align_modules,
        'k1_norm_guard_weight': float(args.k1_norm_guard_weight),
        'k1_norm_guard_target': float(args.k1_norm_guard_target),
        'k1_blocks': args.k1_blocks,
        'k2_blocks': args.k2_blocks,
        'k3_blocks': args.k3_blocks,
        'k1_rank': int(args.k1_rank),
        'k2_rank': int(args.k2_rank),
        'k3_rank': int(args.k3_rank),
        'k1_alpha': int(args.k1_alpha),
        'k2_alpha': int(args.k2_alpha),
        'k3_alpha': int(args.k3_alpha),
        'k3_topk_old': int(args.k3_topk_old),
        'k3_gate_mode': args.k3_gate_mode,
        'k3_gate_threshold': float(args.k3_gate_threshold),
        'k3_gate_min': float(args.k3_gate_min),
    }

    with open(results_path, 'a', encoding='utf-8') as results_file:
        results_file.write(json.dumps(row, sort_keys=True) + '\n')

    write_header = not osp.exists(summary_path)
    with open(summary_path, 'a', encoding='utf-8') as summary_file:
        if write_header:
            summary_file.write(
                'stage\tstage_idx\tdataset\tmAP\tR1\tavg_mAP\tavg_R1\troute_tau\t'
                'train_k3_old_scale\ttrain_k3_old_scale_start\ttrain_k3_old_scale_warmup_epochs\t'
                'train_k3_old_scale_schedule\teval_k3_old_scale\tdebug_max_epoch\tdebug_batch_size\t'
                'world_size\teval_k3_fusion_mode\tdistill_k3_fusion_mode\t'
                'k1_xmod_align_source\tk1_xmod_align_weight\tk1_xmod_align_temp\t'
                'k1_align_blocks\tk1_align_modules\tk1_norm_guard_weight\tk1_norm_guard_target\t'
                'k1_blocks\tk2_blocks\tk3_blocks\tk1_rank\tk2_rank\tk3_rank\tk1_alpha\tk2_alpha\tk3_alpha\t'
                'k3_topk_old\tk3_gate_mode\tk3_gate_threshold\tk3_gate_min\n'
            )
        for item in metrics:
            summary_file.write(
                '{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    stage_name,
                    int(stage_idx),
                    item['dataset'],
                    item['mAP'] * 100.0,
                    item['R1'] * 100.0,
                    row['avg_mAP'] * 100.0,
                    row['avg_R1'] * 100.0,
                    args.route_tau,
                    args.train_k3_old_scale,
                    '' if args.train_k3_old_scale_start is None else args.train_k3_old_scale_start,
                    args.train_k3_old_scale_warmup_epochs,
                    args.train_k3_old_scale_schedule,
                    args.eval_k3_old_scale,
                    args.debug_max_epoch,
                    args.debug_batch_size,
                    world_size,
                    eval_mode,
                    args.distill_k3_fusion_mode,
                    args.k1_xmod_align_source,
                    args.k1_xmod_align_weight,
                    args.k1_xmod_align_temp,
                    args.k1_align_blocks,
                    args.k1_align_modules,
                    args.k1_norm_guard_weight,
                    args.k1_norm_guard_target,
                    args.k1_blocks,
                    args.k2_blocks,
                    args.k3_blocks,
                    args.k1_rank,
                    args.k2_rank,
                    args.k3_rank,
                    args.k1_alpha,
                    args.k2_alpha,
                    args.k3_alpha,
                    args.k3_topk_old,
                    args.k3_gate_mode,
                    args.k3_gate_threshold,
                    args.k3_gate_min,
                )
            )


def get_local_batch_size():
    return cfg.BATCH_SIZE


def get_global_batch_size():
    return cfg.BATCH_SIZE * world_size if distributed else cfg.BATCH_SIZE


def get_debug_num_workers():
    if args.ddp_consistency_debug:
        return 0
    return args.debug_num_workers


def parse_branch_log_blocks(value):
    if value is None or str(value).strip() == '':
        return []
    block_ids = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        block_ids.append(int(part))
    return block_ids


def parse_k1_align_modules(value):
    valid_modules = {'qkv', 'proj'}
    if value is None or str(value).strip() == '':
        return []
    module_names = []
    for part in str(value).split(','):
        module_name = part.strip()
        if not module_name:
            continue
        if module_name not in valid_modules:
            raise ValueError('Unsupported K1 alignment module: {}'.format(module_name))
        if module_name not in module_names:
            module_names.append(module_name)
    return module_names


def parse_attn_branch_module_name(name):
    parts = name.split('.')
    if len(parts) != 5:
        return None
    if parts[0] != 'base' or parts[1] != 'blocks' or parts[3] != 'attn':
        return None
    try:
        block_id = int(parts[2])
    except ValueError:
        return None
    branch_name = parts[4]
    if branch_name not in ('qkv', 'proj'):
        return None
    return block_id, branch_name


def parse_eval_k3_fusion_modes():
    valid_modes = {'all_except_current', 'previous', 'current_only'}
    modes = [args.eval_k3_fusion_mode]
    if args.compare_eval_k3_fusion_modes:
        for part in args.compare_eval_k3_fusion_modes.split(','):
            mode = part.strip()
            if not mode:
                continue
            if mode not in valid_modes:
                raise ValueError('Unsupported eval K3 fusion mode: {}'.format(mode))
            if mode not in modes:
                modes.append(mode)
    return modes


def build_epoch_sampler_indices(trainset_rgb, color_pos_rgb, thermal_pos_rgb, stage_id, epoch):
    global_batch_size = get_global_batch_size()
    sampler_seed = cfg.SEED + stage_id * 1000 + epoch
    np_state = np.random.get_state()
    np.random.seed(sampler_seed)
    try:
        sampler_rgb = IdentitySampler(
            trainset_rgb.train_color_label,
            trainset_rgb.train_thermal_label,
            color_pos_rgb,
            thermal_pos_rgb,
            global_batch_size,
            per_img=cfg.NUM_POS,
        )
    finally:
        np.random.set_state(np_state)
    return np.asarray(sampler_rgb.index1), np.asarray(sampler_rgb.index2)


def merge_feat(features_all, labels_all):
    label_list = [int(x) for x in labels_all.detach().cpu().tolist()]
    features_collect = {}
    for feature, label in zip(features_all, label_list):
        features_collect.setdefault(label, []).append(feature)

    labels_named = sorted(features_collect.keys())
    features_mean = []
    for label in labels_named:
        feats = torch.stack(features_collect[label], dim=0)
        features_mean.append(feats.mean(dim=0))

    labels_tensor = torch.tensor(labels_named, device=labels_all.device, dtype=labels_all.dtype)
    return torch.stack(features_mean, dim=0), labels_tensor


def freeze_old_task_experts(model, current_task_id):
    bare_model = unwrap_model(model)
    current_key = str(int(current_task_id))
    for name, param in bare_model.named_parameters():
        if 'task_bank.experts.' in name:
            expert_id = name.split('task_bank.experts.')[1].split('.')[0]
            param.requires_grad = (expert_id == current_key)


def freeze_backbone_except_lora(model):
    bare_model = unwrap_model(model)
    lora_keywords = (
        'lora_A',
        'lora_B',
        'shared',
        'rgb',
        'infrared',
        'task_bank',
        'gamma_k1',
        'gamma_k2',
        'gamma_k3',
    )
    frozen_params = 0
    trainable_params = 0
    for name, param in bare_model.named_parameters():
        if name.startswith('base.') and not any(keyword in name for keyword in lora_keywords):
            param.requires_grad = False
            frozen_params += param.numel()
        elif param.requires_grad:
            trainable_params += param.numel()
    print('[!INFO] Freeze backbone enabled: frozen base params={}, remaining trainable params={}'.format(
        frozen_params, trainable_params))


def configure_k3_router(model, route_tau, train_old_scale, eval_old_scale, eval_fusion_mode,
                        topk_old=0, gate_mode='none', gate_threshold=0.0, gate_min=0.0):
    bare_model = unwrap_model(model)
    configured = 0
    for module in bare_model.modules():
        task_bank = getattr(module, 'task_bank', None)
        if task_bank is None:
            continue
        task_bank.route_tau = float(route_tau)
        if hasattr(task_bank, 'train_old_scale'):
            task_bank.train_old_scale = float(train_old_scale)
        if hasattr(task_bank, 'eval_old_scale'):
            task_bank.eval_old_scale = float(eval_old_scale)
        if hasattr(task_bank, 'eval_fusion_mode'):
            task_bank.eval_fusion_mode = str(eval_fusion_mode)
        task_bank.k3_topk_old = int(topk_old)
        task_bank.k3_gate_mode = str(gate_mode)
        task_bank.k3_gate_threshold = float(gate_threshold)
        task_bank.k3_gate_min = float(gate_min)
        configured += 1
    if configured > 0:
        gate_info = ''
        if gate_mode != 'none':
            gate_info = ', gate={}:t={}:min={}'.format(gate_mode, gate_threshold, gate_min)
        print('[!INFO] Configure K3 router: banks={}, route_tau={}, train_k3_old_scale={}, eval_k3_old_scale={}, eval_k3_fusion_mode={}, topk_old={}{}'.format(
            configured, route_tau, train_old_scale, eval_old_scale, eval_fusion_mode, topk_old, gate_info))


def compute_train_k3_old_scale(epoch):
    end_scale = float(args.train_k3_old_scale)
    if args.train_k3_old_scale_schedule == 'constant':
        return end_scale

    start_scale = 0.0 if args.train_k3_old_scale_start is None else float(args.train_k3_old_scale_start)
    warmup_epochs = int(args.train_k3_old_scale_warmup_epochs)
    if warmup_epochs <= 0:
        return end_scale
    if warmup_epochs == 1:
        progress = 1.0
    else:
        progress = float(epoch - cfg.START_EPOCH) / float(warmup_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)

    if args.train_k3_old_scale_schedule == 'linear':
        factor = progress
    elif args.train_k3_old_scale_schedule == 'cosine':
        factor = 0.5 - 0.5 * np.cos(np.pi * progress)
    else:
        raise ValueError('Unsupported train_k3_old_scale_schedule: {}'.format(args.train_k3_old_scale_schedule))
    return start_scale + (end_scale - start_scale) * factor


def set_k3_train_old_scale(model, train_old_scale):
    bare_model = unwrap_model(model)
    configured = 0
    for module in bare_model.modules():
        task_bank = getattr(module, 'task_bank', None)
        if task_bank is None or not hasattr(task_bank, 'train_old_scale'):
            continue
        task_bank.train_old_scale = float(train_old_scale)
        configured += 1
    return configured


def set_k3_eval_fusion_mode(model, eval_fusion_mode):
    bare_model = unwrap_model(model)
    configured = 0
    for module in bare_model.modules():
        task_bank = getattr(module, 'task_bank', None)
        if task_bank is None or not hasattr(task_bank, 'eval_fusion_mode'):
            continue
        task_bank.eval_fusion_mode = str(eval_fusion_mode)
        configured += 1
    return configured


def _fmt_router_values(value):
    if value is None:
        return 'None'
    if torch.is_tensor(value):
        if value.dim() == 0:
            return '{:.3f}'.format(value.item())
        return '[' + ', '.join('{:.3f}'.format(v) for v in value.tolist()) + ']'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(str(v) for v in value) + ']'
    return str(value)


def log_k3_router_stats(model, phase, stage_name, epoch=None, eval_name=None):
    bare_model = unwrap_model(model)
    if not hasattr(bare_model, 'base'):
        return

    logs = []
    for block_idx, blk in enumerate(bare_model.base.blocks):
        for branch_name, branch in (('qkv', blk.attn.qkv), ('proj', blk.attn.proj)):
            task_bank = getattr(branch, 'task_bank', None)
            if task_bank is None:
                continue
            snapshot = task_bank.get_route_debug_snapshot()
            old_keys = snapshot['old_keys']
            if not old_keys:
                continue
            logs.append(
                'b{}.{} keys={} w={} sim={} ent={} tau={} old_scale={} gate={} eff_scale={}'.format(
                    block_idx,
                    branch_name,
                    _fmt_router_values(old_keys),
                    _fmt_router_values(snapshot['weights']),
                    _fmt_router_values(snapshot['similarities']),
                    _fmt_router_values(snapshot['entropy']),
                    _fmt_router_values(snapshot['tau']),
                    _fmt_router_values(snapshot.get('old_scale')),
                    _fmt_router_values(snapshot.get('gate')),
                    _fmt_router_values(snapshot.get('effective_old_scale')),
                )
            )

    prefix = '[!INFO][K3Router][{}][stage={}]'.format(phase, stage_name)
    if epoch is not None:
        prefix += '[epoch={}]'.format(epoch)
    if eval_name is not None:
        prefix += '[eval={}]'.format(eval_name)

    if logs:
        print(prefix)
        for line in logs:
            print('  ' + line)
    else:
        print(prefix + ' no_old_expert_fusion')


class BranchStatsCollector:
    def __init__(self, model, logs_dir, block_ids, device, distributed=False):
        self.model = unwrap_model(model)
        self.logs_dir = logs_dir
        self.block_ids = set(int(block_id) for block_id in block_ids)
        self.device = device
        self.distributed = distributed
        self.handles = []
        self.records = {}
        self.stat_scope = 'global_ddp_reduced' if distributed else 'single_process'
        self._register_hooks()

    def _parse_branch_module(self, name):
        parts = name.split('.')
        if len(parts) != 5:
            return None
        if parts[0] != 'base' or parts[1] != 'blocks' or parts[3] != 'attn':
            return None
        try:
            block_id = int(parts[2])
        except ValueError:
            return None
        branch_name = parts[4]
        if block_id not in self.block_ids or branch_name not in ('qkv', 'proj'):
            return None
        return block_id, branch_name

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            parsed = self._parse_branch_module(name)
            if parsed is None or not hasattr(module, 'enable_branch_stats'):
                continue
            module.enable_branch_stats = True
            module.last_branch_stats = None
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append((handle, module))

    def _make_hook(self, module_name):
        def hook(module, inputs, output):
            branch_stats = getattr(module, 'last_branch_stats', None)
            if not branch_stats:
                return
            for stat in branch_stats:
                branch = stat.get('branch')
                if branch is None:
                    continue
                key = (module_name, branch)
                accumulator = self.records.setdefault(key, {})
                self._accumulate(accumulator, stat)
        return hook

    def _accumulate(self, accumulator, stat):
        stat_count = stat.get('count')
        for key, value in stat.items():
            if key == 'branch':
                continue
            if not torch.is_tensor(value):
                continue
            value = value.detach().to(device=self.device, dtype=torch.float64)
            if key in ('gamma', 'gamma_rgb', 'gamma_ir') and torch.is_tensor(stat_count):
                value = value * stat_count.detach().to(device=self.device, dtype=torch.float64)
            if key not in accumulator:
                accumulator[key] = torch.zeros((), device=self.device, dtype=torch.float64)
            accumulator[key] = accumulator[key] + value

    def _reduced_records(self):
        reduced = {}
        for record_key, accumulator in self.records.items():
            reduced_accumulator = {}
            for metric_name, value in accumulator.items():
                tensor = value.detach().clone()
                if self.distributed:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced_accumulator[metric_name] = tensor.item()
            reduced[record_key] = reduced_accumulator
        return reduced

    def _safe_mean(self, values, count_name, sum_name):
        count = values.get(count_name, 0.0)
        if count <= 0 or sum_name not in values:
            return None
        return values.get(sum_name, 0.0) / count

    def _safe_std(self, values, count_name, sum_name, sq_sum_name):
        count = values.get(count_name, 0.0)
        if count <= 0 or sum_name not in values or sq_sum_name not in values:
            return None
        mean = values.get(sum_name, 0.0) / count
        mean_sq = values.get(sq_sum_name, 0.0) / count
        return max(mean_sq - mean * mean, 0.0) ** 0.5

    def _format_row(self, module_name, branch, values, stage_name, stage_idx, epoch):
        count = values.get('count', 0.0)
        if count <= 0:
            return None

        row = {
            'stage': stage_name,
            'stage_idx': int(stage_idx),
            'epoch': int(epoch),
            'phase': 'train',
            'module': module_name,
            'branch': branch,
            'stat_scope': self.stat_scope,
            'count': float(count),
            'delta_norm_mean': self._safe_mean(values, 'count', 'delta_norm_sum'),
            'delta_norm_std': self._safe_std(values, 'count', 'delta_norm_sum', 'delta_norm_sq_sum'),
            'delta_abs_mean': self._safe_mean(values, 'count', 'delta_abs_sum'),
            'scaled_delta_norm_mean': self._safe_mean(values, 'count', 'scaled_delta_norm_sum'),
            'scaled_delta_norm_std': self._safe_std(values, 'count', 'scaled_delta_norm_sum', 'scaled_delta_norm_sq_sum'),
            'rgb_scaled_delta_norm_mean': self._safe_mean(values, 'rgb_count', 'rgb_scaled_delta_norm_sum'),
            'ir_scaled_delta_norm_mean': self._safe_mean(values, 'ir_count', 'ir_scaled_delta_norm_sum'),
            'k3_current_scaled_norm_mean': self._safe_mean(values, 'count', 'k3_current_scaled_norm_sum'),
            'k3_current_scaled_norm_std': self._safe_std(values, 'count', 'k3_current_scaled_norm_sum', 'k3_current_scaled_norm_sq_sum'),
            'k3_old_scaled_norm_mean': self._safe_mean(values, 'count', 'k3_old_scaled_norm_sum'),
            'k3_old_scaled_norm_std': self._safe_std(values, 'count', 'k3_old_scaled_norm_sum', 'k3_old_scaled_norm_sq_sum'),
            'k3_old_current_ratio_mean': self._safe_mean(values, 'count', 'k3_old_current_ratio_sum'),
            'k3_old_fraction_mean': self._safe_mean(values, 'count', 'k3_old_fraction_sum'),
        }
        rgb_norm = row['rgb_scaled_delta_norm_mean']
        ir_norm = row['ir_scaled_delta_norm_mean']
        if rgb_norm is not None and ir_norm is not None:
            row['modality_gap'] = abs(rgb_norm - ir_norm) / (rgb_norm + ir_norm + 1e-12)
        else:
            row['modality_gap'] = None

        if branch == 'k2':
            gamma_rgb = self._safe_mean(values, 'count', 'gamma_rgb')
            gamma_ir = self._safe_mean(values, 'count', 'gamma_ir')
            row['gamma_rgb'] = gamma_rgb
            row['gamma_ir'] = gamma_ir
            row['selected_scaled_delta_norm_mean'] = row['scaled_delta_norm_mean']
            row['gamma'] = None
        else:
            row['gamma'] = self._safe_mean(values, 'count', 'gamma')
            row['gamma_rgb'] = None
            row['gamma_ir'] = None
            row['selected_scaled_delta_norm_mean'] = None
        return row

    def _router_rows(self, stage_name, stage_idx, epoch):
        rows = []
        for block_idx, blk in enumerate(getattr(self.model.base, 'blocks', [])):
            if block_idx not in self.block_ids:
                continue
            for branch_name, branch in (('qkv', blk.attn.qkv), ('proj', blk.attn.proj)):
                task_bank = getattr(branch, 'task_bank', None)
                if task_bank is None:
                    continue
                snapshot = task_bank.get_route_debug_snapshot()
                weights = snapshot.get('weights')
                if weights is None or weights.numel() == 0:
                    continue
                rows.append({
                    'stage': stage_name,
                    'stage_idx': int(stage_idx),
                    'epoch': int(epoch),
                    'phase': 'train',
                    'module': 'base.blocks.{}.attn.{}'.format(block_idx, branch_name),
                    'branch': 'k3_router',
                    'stat_scope': 'rank0_snapshot',
                    'old_keys': snapshot.get('old_keys'),
                    'max_weight': float(weights.max().item()),
                    'entropy': None if snapshot.get('entropy') is None else float(snapshot['entropy'].item()),
                    'tau': float(snapshot.get('tau')),
                    'old_scale': float(snapshot.get('old_scale')),
                    'weights': [float(value) for value in weights.tolist()],
                    'similarities': None if snapshot.get('similarities') is None else [float(value) for value in snapshot['similarities'].tolist()],
                    'raw_weights': None if snapshot.get('raw_weights') is None else [float(value) for value in snapshot['raw_weights'].tolist()],
                    'sparse_weights': None if snapshot.get('sparse_weights') is None else [float(value) for value in snapshot['sparse_weights'].tolist()],
                    'selected_old_keys': snapshot.get('selected_old_keys'),
                    'confidence': None if snapshot.get('confidence') is None else float(snapshot['confidence'].item()),
                    'gate': None if snapshot.get('gate') is None else float(snapshot['gate'].item()),
                    'effective_old_scale': None if snapshot.get('effective_old_scale') is None else float(snapshot['effective_old_scale'].item()),
                    'topk_old': snapshot.get('topk_old'),
                    'gate_mode': snapshot.get('gate_mode'),
                    'gate_threshold': snapshot.get('gate_threshold'),
                    'gate_min': snapshot.get('gate_min'),
                })
        return rows

    def dump_epoch(self, stage_name, stage_idx, epoch):
        if not self.records:
            return
        reduced_records = self._reduced_records()
        rows = []
        for (module_name, branch), values in sorted(reduced_records.items()):
            row = self._format_row(module_name, branch, values, stage_name, stage_idx, epoch)
            if row is not None:
                rows.append(row)

        if is_main_process():
            jsonl_path = osp.join(self.logs_dir, 'branch_stats.jsonl')
            summary_path = osp.join(self.logs_dir, 'branch_summary.tsv')
            with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
                for row in rows:
                    jsonl_file.write(json.dumps(row, sort_keys=True) + '\n')
                for row in self._router_rows(stage_name, stage_idx, epoch):
                    jsonl_file.write(json.dumps(row, sort_keys=True) + '\n')

            write_header = not osp.exists(summary_path)
            with open(summary_path, 'a', encoding='utf-8') as summary_file:
                if write_header:
                    summary_file.write(
                        'stage\tstage_idx\tepoch\tmodule\tbranch\tstat_scope\tgamma\tgamma_rgb\tgamma_ir\t'
                        'delta_norm\tdelta_norm_std\tscaled_delta_norm\tselected_scaled_delta_norm\t'
                        'rgb_norm\tir_norm\tmodality_gap\tk3_current_norm\tk3_old_norm\t'
                        'k3_old_current_ratio\tk3_old_fraction\tcount\n'
                    )
                for row in rows:
                    summary_file.write(
                        '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            row['stage'],
                            row['stage_idx'],
                            row['epoch'],
                            row['module'],
                            row['branch'],
                            row['stat_scope'],
                            '' if row['gamma'] is None else '{:.8f}'.format(row['gamma']),
                            '' if row['gamma_rgb'] is None else '{:.8f}'.format(row['gamma_rgb']),
                            '' if row['gamma_ir'] is None else '{:.8f}'.format(row['gamma_ir']),
                            '' if row['delta_norm_mean'] is None else '{:.8f}'.format(row['delta_norm_mean']),
                            '' if row['delta_norm_std'] is None else '{:.8f}'.format(row['delta_norm_std']),
                            '' if row['scaled_delta_norm_mean'] is None else '{:.8f}'.format(row['scaled_delta_norm_mean']),
                            '' if row['selected_scaled_delta_norm_mean'] is None else '{:.8f}'.format(row['selected_scaled_delta_norm_mean']),
                            '' if row['rgb_scaled_delta_norm_mean'] is None else '{:.8f}'.format(row['rgb_scaled_delta_norm_mean']),
                            '' if row['ir_scaled_delta_norm_mean'] is None else '{:.8f}'.format(row['ir_scaled_delta_norm_mean']),
                            '' if row['modality_gap'] is None else '{:.8f}'.format(row['modality_gap']),
                            '' if row['k3_current_scaled_norm_mean'] is None else '{:.8f}'.format(row['k3_current_scaled_norm_mean']),
                            '' if row['k3_old_scaled_norm_mean'] is None else '{:.8f}'.format(row['k3_old_scaled_norm_mean']),
                            '' if row['k3_old_current_ratio_mean'] is None else '{:.8f}'.format(row['k3_old_current_ratio_mean']),
                            '' if row['k3_old_fraction_mean'] is None else '{:.8f}'.format(row['k3_old_fraction_mean']),
                            '{:.0f}'.format(row['count']),
                        )
                    )
        self.reset()

    def reset(self):
        self.records = {}

    def close(self):
        for handle, module in self.handles:
            handle.remove()
            module.enable_branch_stats = False
            module.last_branch_stats = None
        self.handles = []
        self.records = {}


class K1AlignHook:
    def __init__(self, model, block_ids, branch_names, source):
        self.model = unwrap_model(model)
        self.block_ids = [int(block_id) for block_id in block_ids]
        self.branch_names = list(branch_names)
        self.source = str(source)
        self.handles = []
        self.outputs = {}
        self.active = False
        self._register_hooks()

    def _register_hooks(self):
        if not hasattr(self.model, 'base') or not hasattr(self.model.base, 'blocks'):
            raise ValueError('K1 alignment requires model.base.blocks')
        if self.source == 'block_output':
            for block_id in self.block_ids:
                if block_id < 0 or block_id >= len(self.model.base.blocks):
                    raise ValueError('K1 alignment block id out of range: {}'.format(block_id))
                block = self.model.base.blocks[block_id]
                module_name = 'base.blocks.{}'.format(block_id)
                handle = block.register_forward_hook(self._make_block_hook(module_name))
                self.handles.append(handle)
            return

        if self.source not in ('scaled_delta', 'scaled_delta_detached'):
            raise ValueError('Unsupported K1 alignment source: {}'.format(self.source))

        block_set = set(self.block_ids)
        branch_set = set(self.branch_names)
        for name, module in self.model.named_modules():
            parsed = parse_attn_branch_module_name(name)
            if parsed is None:
                continue
            block_id, branch_name = parsed
            if block_id not in block_set or branch_name not in branch_set:
                continue
            if not hasattr(module, 'compute_k1_alignment_delta'):
                continue
            if getattr(module, 'shared_adapter', None) is None:
                continue
            handle = module.register_forward_hook(self._make_delta_hook(name))
            self.handles.append(handle)
        if not self.handles:
            raise ValueError('No K1 alignment modules matched blocks={} modules={}'.format(
                self.block_ids, self.branch_names))

    def _pool_cls(self, tensor):
        if tensor.dim() < 3:
            raise ValueError('Expected K1 alignment tensor [B, N, C], got shape {}'.format(tuple(tensor.shape)))
        return tensor[:, 0]

    def _make_block_hook(self, module_name):
        def hook(module, inputs, output):
            if not self.active:
                return
            self.outputs[module_name] = self._pool_cls(output)
        return hook

    def _make_delta_hook(self, module_name):
        def hook(module, inputs, output):
            if not self.active:
                return
            detach_input = self.source == 'scaled_delta_detached'
            scaled_delta = module.compute_k1_alignment_delta(inputs[0], detach_input=detach_input, scaled=True)
            if scaled_delta is None:
                return
            self.outputs[module_name] = self._pool_cls(scaled_delta)
        return hook

    def reset(self):
        self.outputs = {}

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def features(self):
        if not self.outputs:
            return None
        return torch.cat([self.outputs[key] for key in sorted(self.outputs)], dim=1)

    def feature_norm(self):
        features = self.features()
        if features is None:
            return None
        flat = features.float().reshape(features.shape[0], -1)
        return torch.linalg.vector_norm(flat, dim=1).mean()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.outputs = {}
        self.active = False


def cross_modal_supcon_loss(features, labels, mods, temperature=0.1):
    if features is None:
        return None
    if features.shape[0] <= 1:
        return None

    labels = labels.reshape(-1)
    mods = mods.reshape(-1)
    if features.shape[0] != labels.shape[0] or labels.shape[0] != mods.shape[0]:
        raise ValueError('K1 SupCon feature/label/mod batch size mismatch')

    features = F.normalize(features.float(), p=2, dim=1)
    logits = torch.matmul(features, features.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    batch_size = labels.shape[0]
    self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & mods.unsqueeze(0).ne(mods.unsqueeze(1)) & (~self_mask)
    logits_mask = ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if not valid_anchor.any():
        return None

    exp_logits = torch.exp(logits) * logits_mask.to(dtype=logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_log_prob = (positive_mask.to(dtype=log_prob.dtype) * log_prob).sum(dim=1) / positive_count.clamp_min(1).to(dtype=log_prob.dtype)
    return -positive_log_prob[valid_anchor].mean()


def build_train_loader(trainset_rgb, color_pos_rgb, thermal_pos_rgb, stage_id, epoch):
    index1, index2 = build_epoch_sampler_indices(trainset_rgb, color_pos_rgb, thermal_pos_rgb, stage_id, epoch)
    local_batch_size = get_local_batch_size()
    if distributed:
        global_batch_size = get_global_batch_size()
        total_global_batches = len(index1) // global_batch_size
        usable_samples = total_global_batches * global_batch_size
        if usable_samples == 0:
            raise ValueError('Not enough sampled batches for distributed training: total_global_batches={}, world_size={}'.format(
                total_global_batches, world_size))
        if is_main_process() and usable_samples != len(index1):
            print('[!INFO] Truncate sampler output for DDP: {} -> {} samples'.format(len(index1), usable_samples))
        index1 = index1[:usable_samples].reshape(total_global_batches, world_size, local_batch_size)
        index2 = index2[:usable_samples].reshape(total_global_batches, world_size, local_batch_size)
        trainset_rgb.cIndex = index1[:, rank, :].reshape(-1)
        trainset_rgb.tIndex = index2[:, rank, :].reshape(-1)
        sampler = data.SequentialSampler(range(len(trainset_rgb.cIndex)))
    else:
        trainset_rgb.cIndex = index1
        trainset_rgb.tIndex = index2
        sampler = data.SequentialSampler(range(len(trainset_rgb.cIndex)))

    return data.DataLoader(
        trainset_rgb,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=get_debug_num_workers(),
        drop_last=True,
        pin_memory=True,
    )


def build_test_loaders(current_stage_idx):
    if not is_main_process():
        return [], [], [], [], [], []

    query_loaders, gall_loaders = [], []
    query_labels, gall_labels = [], []
    query_cams, gall_cams = [], []

    for ii in range(current_stage_idx + 1):
        query_cam = None
        gall_cam = None

        if training_set[ii] == 'sysu':
            data_path = cfg.DATA_PATH_SYSU
            query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
            queryset = TestData_Inf(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0, gall_mode='single')
            gallset = TestData_Vis(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'regdb':
            data_path = cfg.DATA_PATH_RegDB
            query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
            queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
            gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'llcm':
            data_path = cfg.DATA_PATH_LLCM
            query_img, query_label, query_cam = process_query_llcm(data_path, modal=2)
            queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, modal=1)
            gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'vcm':
            processed_data = VCM(root=cfg.DATA_PATH_VCM)
            queryset = VideoDataset_test_Inf(processed_data.query, 1, "video_test", transform_test, processed_data.query_cam)
            gallset = VideoDataset_test_Vis(processed_data.gallery, 1, "video_test", transform_test, processed_data.gallary_cam)
            queryset.test_label = processed_data.query_labels
            gallset.test_label = processed_data.gallery_labels
        else:
            raise ValueError('Unsupported test dataset {}'.format(training_set[ii]))

        query_loaders.append(data.DataLoader(queryset, batch_size=args.debug_test_batch, shuffle=False, num_workers=get_debug_num_workers()))
        gall_loaders.append(data.DataLoader(gallset, batch_size=args.debug_test_batch, shuffle=False, num_workers=get_debug_num_workers()))
        query_labels.append(queryset.test_label)
        gall_labels.append(gallset.test_label)
        query_cams.append(query_cam if training_set[ii] == 'sysu' else None)
        gall_cams.append(gall_cam if training_set[ii] == 'sysu' else None)

    return query_loaders, gall_loaders, query_labels, gall_labels, query_cams, gall_cams


def train(epoch, stage_id, model, old_model, scheduler, optimizer, scaler, trainloader,
          criterion_id, criterion_tri, criterion_tricros, criterion_tricent, kldiv_loss,
          vis_features_mean, inf_features_mean, k1_align_hook=None):
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    loss_k1_xmod_meter = AverageMeter()
    loss_k1_norm_guard_meter = AverageMeter()
    k1_delta_norm_meter = AverageMeter()
    acc_rgb_meter = AverageMeter()
    acc_ir_meter = AverageMeter()
    epoch_start_time = time.time()

    scheduler.step(epoch)
    model.train()

    for idx_, (input1, input2, label1, label2) in enumerate(trainloader):
        optimizer.zero_grad()
        input1 = input1.to(device, non_blocking=True)
        input2 = input2.to(device, non_blocking=True)
        label1 = label1.to(device, non_blocking=True)
        label2 = label2.to(device, non_blocking=True)
        labels = torch.cat((label1, label2), 0)
        mods = torch.cat([torch.ones_like(label1), torch.zeros_like(label2)])

        if k1_align_hook is not None:
            k1_align_hook.reset()
            k1_align_hook.activate()

        with autocast_context():
            scores, feats = model(torch.cat([input1, input2]), mods, task_id=stage_id)
            score1, score2 = scores.chunk(2, 0)
            feat1, feat2 = feats.chunk(2, 0)

            global_feats = all_gather_tensor(feats, with_grad=True)
            global_labels = all_gather_tensor(labels, with_grad=False)
            global_feat1 = all_gather_tensor(feat1, with_grad=True)
            global_feat2 = all_gather_tensor(feat2, with_grad=True)
            global_label1 = all_gather_tensor(label1, with_grad=False)
            global_label2 = all_gather_tensor(label2, with_grad=False)

            loss_id = criterion_id(score1, label1.long()) + criterion_id(score2, label2.long())
            loss_tri = criterion_tri(global_feats, global_feats, global_labels)
            loss = loss_id + loss_tri * world_size

            feat1_mean, feat1_mean_label = merge_feat(global_feat1, global_label1)
            feat2_mean, feat2_mean_label = merge_feat(global_feat2, global_label2)

            loss_tri_cent = 0.5 * criterion_tricent(feat1_mean, feat2_mean, feat1_mean_label, feat2_mean_label) \
                + 0.5 * criterion_tricent(feat2_mean, feat1_mean, feat2_mean_label, feat1_mean_label)
            loss_tri_cross = 0.5 * criterion_tricros(global_feat1, feat2_mean, global_label1, feat2_mean_label) \
                + 0.5 * criterion_tricros(feat1_mean, global_feat2, feat1_mean_label, global_label2)

            global_pair_loss = (1 - args.cross_weight) * loss_tri_cent + args.cross_weight * loss_tri_cross
            loss += args.new_weight * global_pair_loss * world_size

            loss_k1_xmod = None
            loss_k1_norm_guard = None
            k1_delta_norm = None
            if k1_align_hook is not None:
                k1_features = k1_align_hook.features()
                k1_delta_norm = k1_align_hook.feature_norm()
                loss_k1_xmod = cross_modal_supcon_loss(k1_features, labels, mods, args.k1_xmod_align_temp)
                if loss_k1_xmod is not None:
                    loss += args.k1_xmod_align_weight * loss_k1_xmod
                if (
                    k1_delta_norm is not None
                    and args.k1_norm_guard_weight > 0
                    and args.k1_xmod_align_source != 'block_output'
                ):
                    target_norm = torch.tensor(
                        float(args.k1_norm_guard_target),
                        device=k1_delta_norm.device,
                        dtype=k1_delta_norm.dtype,
                    )
                    loss_k1_norm_guard = F.relu(target_norm - k1_delta_norm)
                    loss += args.k1_norm_guard_weight * loss_k1_norm_guard

            if old_model is not None:
                old_model.eval()
                with torch.no_grad():
                    feats_old_norm = old_model(torch.cat([input1, input2]), mods, task_id=stage_id - 1, fkd=True).detach()
                    feat1_old, feat2_old = feats_old_norm.chunk(2, 0)

                vis_features_new = cosine_similarity(feat1, vis_features_mean)
                vis_features_old = cosine_similarity(feat1_old, vis_features_mean)
                inf_features_new = cosine_similarity(feat2, inf_features_mean)
                inf_features_old = cosine_similarity(feat2_old, inf_features_mean)

                vis_features_new = F.softmax(vis_features_new / 0.1, dim=1)
                vis_features_old = F.softmax(vis_features_old / 0.1, dim=1)
                inf_features_new = F.softmax(inf_features_new / 0.1, dim=1)
                inf_features_old = F.softmax(inf_features_old / 0.1, dim=1)

                vis_rel_new = get_normal_affinity(vis_features_new, 0.1)
                vis_rel_old = get_normal_affinity(vis_features_old, 0.1)
                inf_rel_new = get_normal_affinity(inf_features_new, 0.1)
                inf_rel_old = get_normal_affinity(inf_features_old, 0.1)

                div_1 = 0.5 * kldiv_loss(torch.log(vis_rel_new), vis_rel_old) + 0.5 * kldiv_loss(torch.log(inf_rel_new), inf_rel_old)

                vis_features_new_ = cosine_similarity(feat1, inf_features_mean)
                vis_features_old_ = cosine_similarity(feat1_old, inf_features_mean)
                inf_features_new_ = cosine_similarity(feat2, vis_features_mean)
                inf_features_old_ = cosine_similarity(feat2_old, vis_features_mean)

                vis_features_new_ = F.softmax(vis_features_new_ / 0.1, dim=1)
                vis_features_old_ = F.softmax(vis_features_old_ / 0.1, dim=1)
                inf_features_new_ = F.softmax(inf_features_new_ / 0.1, dim=1)
                inf_features_old_ = F.softmax(inf_features_old_ / 0.1, dim=1)

                vis_rel_new_ = get_normal_affinity(vis_features_new_, 0.1)
                vis_rel_old_ = get_normal_affinity(vis_features_old_, 0.1)
                inf_rel_new_ = get_normal_affinity(inf_features_new_, 0.1)
                inf_rel_old_ = get_normal_affinity(inf_features_old_, 0.1)

                div_2 = 0.5 * kldiv_loss(torch.log(vis_rel_new_), vis_rel_old_) + 0.5 * kldiv_loss(torch.log(inf_rel_new_), inf_rel_old_)

                loss += args.proto_weight * ((1 - args.inter_weight) * div_1 + args.inter_weight * div_2)

        if k1_align_hook is not None:
            k1_align_hook.deactivate()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc_rgb = (score1.max(1)[1] == label1).float().mean()
        acc_ir = (score2.max(1)[1] == label2).float().mean()

        loss_tri_meter.update(loss_tri.item())
        loss_ce_meter.update(loss_id.item())
        loss_meter.update(loss.item())
        loss_k1_xmod_meter.update(0.0 if loss_k1_xmod is None else loss_k1_xmod.item())
        loss_k1_norm_guard_meter.update(0.0 if loss_k1_norm_guard is None else loss_k1_norm_guard.item())
        k1_delta_norm_meter.update(0.0 if k1_delta_norm is None else k1_delta_norm.item())
        acc_rgb_meter.update(acc_rgb.item(), 1)
        acc_ir_meter.update(acc_ir.item(), 1)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        should_log_progress = args.ddp_consistency_debug and ((idx_ + 1) % 200 == 0)
        should_log_epoch_end = (idx_ + 1) % len(trainloader) == 0
        if should_log_progress or should_log_epoch_end:
            elapsed = max(time.time() - epoch_start_time, 1e-6)
            samples_seen = (idx_ + 1) * get_global_batch_size()
            samples_per_second = samples_seen / elapsed
            print('Epoch[{}] Iteration[{}/{}]'
                  ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                  'K1XMod:{:.3f}, K1Norm:{:.4f}, K1Guard:{:.4f}, '
                  'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                  'Base Lr: {:.2e}, Time: {:.1f}s, Global Img/s: {:.2f} '.format(
                      epoch,
                      (idx_ + 1),
                      len(trainloader),
                      reduce_mean_scalar(loss_meter.avg),
                      reduce_mean_scalar(loss_tri_meter.avg),
                      reduce_mean_scalar(loss_ce_meter.avg),
                      reduce_mean_scalar(loss_k1_xmod_meter.avg),
                      reduce_mean_scalar(k1_delta_norm_meter.avg),
                      reduce_mean_scalar(loss_k1_norm_guard_meter.avg),
                      reduce_mean_scalar(acc_rgb_meter.avg),
                      reduce_mean_scalar(acc_ir_meter.avg),
                      optimizer.state_dict()['param_groups'][0]['lr'],
                      elapsed,
                      samples_per_second))

    return model


def test(model, query_loader, gall_loader, dataset='sysu', query_label=None, gall_label=None,
         query_cam=None, gall_cam=None, task_id=None, stage_name=None):
    bare_model = unwrap_model(model)
    bare_model.eval()

    print('[!INFO] Testing...')
    gall_feat = []
    with torch.no_grad():
        for input_data, label, cam in gall_loader:
            input_data = Variable(input_data.to(device, non_blocking=True))
            feat = bare_model(input_data, cam, task_id=task_id)
            gall_feat.append(feat.detach().cpu().numpy())
    gall_feat = np.concatenate(gall_feat, axis=0)

    query_feat = []
    with torch.no_grad():
        for input_data, label, cam in query_loader:
            input_data = Variable(input_data.to(device, non_blocking=True))
            feat = bare_model(input_data, cam, task_id=task_id)
            query_feat.append(feat.detach().cpu().numpy())
    query_feat = np.concatenate(query_feat, axis=0)

    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    if dataset == 'sysu':
        cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    else:
        cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)

    log_k3_router_stats(bare_model, phase='eval', stage_name=stage_name, eval_name=dataset)
    return cmc, mAP, mInp


for idx, dataset_name in enumerate(training_set):
    print("[!INFO]: Start Training on:", dataset_name)
    if dataset_name == 'sysu':
        data_path = cfg.DATA_PATH_SYSU
        trainset_rgb = SYSUData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'regdb':
        data_path = cfg.DATA_PATH_RegDB
        trainset_rgb = RegDBData(data_path, args.trial, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'llcm':
        data_path = cfg.DATA_PATH_LLCM
        trainset_rgb = LLCMData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'vcm':
        data_path = cfg.DATA_PATH_VCM
        trainset_rgb = VCMData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    else:
        raise ValueError('Unsupported training dataset {}'.format(dataset_name))

    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)
    num_classes = len(np.unique(trainset_rgb.train_color_label))

    base_model = build_vision_transformer(num_classes=num_classes, cfg=cfg)
    base_model.to(device)
    configure_k3_router(base_model, args.route_tau, args.train_k3_old_scale, args.eval_k3_old_scale, args.eval_k3_fusion_mode,
                        topk_old=args.k3_topk_old, gate_mode=args.k3_gate_mode,
                        gate_threshold=args.k3_gate_threshold, gate_min=args.k3_gate_min)

    if idx > 0:
        args.resume = training_set[idx - 1] + '.pth'
        model_path = os.path.join(args.logs_dir, args.resume)
        proto_path = model_path.replace('.pth', '_proto.pth')
        prev_stage_done = stage_done_path(training_set[idx - 1])

        # Wait for the previous stage's rank0-only finalize path to finish first.
        # The stage-done marker is written only after checkpoint/prototype save and evaluation.
        wait_for_file(prev_stage_done, timeout_seconds=STAGE_DONE_WAIT_SECONDS)
        wait_for_file(model_path, timeout_seconds=STAGE_DONE_WAIT_SECONDS)
        wait_for_file(proto_path, timeout_seconds=STAGE_DONE_WAIT_SECONDS)

        for task_id in range(idx):
            base_model.add_task(task_id)
        base_model.load_param(model_path)
        base_model.add_task(idx)
        print('[!INFO] Loaded old model: {}'.format(args.resume))

        old_param_dict = torch.load(model_path, map_location='cpu')
        old_num_classes = None
        for key in old_param_dict:
            if key == 'classifier.weight':
                old_num_classes = old_param_dict[key].shape
                break
        if old_num_classes is None:
            raise ValueError('classifier.weight not found in {}'.format(model_path))

        old_model = build_vision_transformer(num_classes=old_num_classes[0], cfg=cfg)
        old_model.to(device)
        configure_k3_router(old_model, args.route_tau, args.train_k3_old_scale, args.eval_k3_old_scale, args.distill_k3_fusion_mode,
                            topk_old=args.k3_topk_old, gate_mode=args.k3_gate_mode,
                            gate_threshold=args.k3_gate_threshold, gate_min=args.k3_gate_min)
        for task_id in range(idx):
            old_model.add_task(task_id)
        old_model.load_param(model_path)
        old_model.eval()
        print('[!INFO] Froze old model: {}'.format(args.resume))

        old_proto = torch.load(proto_path, map_location='cpu')
        vis_features_mean = old_proto['vis_features_mean'].to(device)
        vis_labels_named = old_proto['vis_labels_named']
        inf_features_mean = old_proto['inf_features_mean'].to(device)
        inf_labels_named = old_proto['inf_labels_named']
    else:
        base_model.add_task(idx)
        old_model = None
        vis_features_mean = None
        vis_labels_named = None
        inf_features_mean = None
        inf_labels_named = None

    freeze_old_task_experts(base_model, idx)
    if args.freeze_backbone:
        freeze_backbone_except_lora(base_model)

    criterion_ID = nn.CrossEntropyLoss()
    criterion_Tri = TripletLoss(margin=cfg.MARGIN, feat_norm='no')
    criterion_TriCros = TripletCrossLoss(margin=cfg.MARGINCROSS, feat_norm='no')
    criterion_TriCent = TripletCrossLoss(margin=cfg.MARGINCENTER, feat_norm='no')
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    optimizer = make_optimizer(cfg, base_model)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = create_grad_scaler()

    model = base_model
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    branch_stats_collector = None
    if args.log_branch_stats:
        branch_stats_collector = BranchStatsCollector(
            model,
            args.logs_dir,
            parse_branch_log_blocks(args.branch_log_blocks),
            device,
            distributed=distributed,
        )
        if is_main_process():
            print('[!INFO] Branch stats enabled: blocks={}, stat_scope={}'.format(
                args.branch_log_blocks,
                branch_stats_collector.stat_scope,
            ))

    k1_align_hook = None
    if args.k1_xmod_align_weight > 0 or args.k1_norm_guard_weight > 0:
        k1_align_blocks = parse_branch_log_blocks(args.k1_align_blocks)
        if not k1_align_blocks:
            raise ValueError('--k1_align_blocks must not be empty when K1 alignment is enabled')
        k1_align_modules = parse_k1_align_modules(args.k1_align_modules)
        if args.k1_xmod_align_source != 'block_output' and not k1_align_modules:
            raise ValueError('--k1_align_modules must not be empty for K1 delta alignment')
        k1_align_hook = K1AlignHook(
            model,
            k1_align_blocks,
            k1_align_modules,
            args.k1_xmod_align_source,
        )
        if is_main_process():
            print('[!INFO] K1 cross-modal alignment enabled: source={}, blocks={}, modules={}, weight={}, temp={}, norm_guard_weight={}, norm_guard_target={}'.format(
                args.k1_xmod_align_source,
                args.k1_align_blocks,
                args.k1_align_modules,
                args.k1_xmod_align_weight,
                args.k1_xmod_align_temp,
                args.k1_norm_guard_weight,
                args.k1_norm_guard_target,
            ))

    query_loaders, gall_loaders, query_labels, gall_labels, query_cams, gall_cams = build_test_loaders(idx)

    print('==> Start Training...')
    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH + 1):
        current_train_k3_old_scale = compute_train_k3_old_scale(epoch)
        configured_k3_banks = set_k3_train_old_scale(model, current_train_k3_old_scale)
        if is_main_process() and configured_k3_banks > 0:
            print('[!INFO] Train K3 old scale schedule: stage={}, epoch={}, scale={:.6f}, banks={}, schedule={}, start={}, end={}, warmup_epochs={}'.format(
                dataset_name,
                epoch,
                current_train_k3_old_scale,
                configured_k3_banks,
                args.train_k3_old_scale_schedule,
                args.train_k3_old_scale_start,
                args.train_k3_old_scale,
                args.train_k3_old_scale_warmup_epochs,
            ))
        trainloader = build_train_loader(trainset_rgb, color_pos_rgb, thermal_pos_rgb, idx, epoch)
        model = train(
            epoch,
            idx,
            model,
            old_model,
            scheduler,
            optimizer,
            scaler,
            trainloader,
            criterion_ID,
            criterion_Tri,
            criterion_TriCros,
            criterion_TriCent,
            KLDivLoss,
            vis_features_mean,
            inf_features_mean,
            k1_align_hook=k1_align_hook,
        )

        if is_main_process():
            log_k3_router_stats(model, phase='train', stage_name=dataset_name, epoch=epoch)

        if branch_stats_collector is not None and args.branch_log_interval > 0 and (epoch % args.branch_log_interval == 0):
            branch_stats_collector.dump_epoch(dataset_name, idx, epoch)

        if epoch == cfg.MAX_EPOCH and branch_stats_collector is not None:
            branch_stats_collector.close()
            branch_stats_collector = None

        if epoch == cfg.MAX_EPOCH and k1_align_hook is not None:
            k1_align_hook.close()
            k1_align_hook = None

        if epoch == cfg.MAX_EPOCH:
            if is_main_process():
                bare_model = unwrap_model(model)
                if idx > 0:
                    model_path = os.path.join(args.logs_dir, args.resume)
                    bare_model.merge_param(model_path, args.ema_weight)
                    print('[!INFO] Merge old model ......')
                torch.save(bare_model.state_dict(), osp.join(args.logs_dir, training_set[idx] + '.pth'))

                if dataset_name == 'regdb':
                    train_img, train_label, train_mod = process_train_regdb(data_path)
                elif dataset_name == 'sysu':
                    train_img, train_label, train_mod = process_train_sysu(data_path)
                elif dataset_name == 'llcm':
                    train_img, train_label, train_mod = process_train_llcm(data_path)
                elif dataset_name == 'vcm':
                    train_img, train_label, train_mod = process_train_vcm(data_path)
                else:
                    raise ValueError('Unsupported dataset {}'.format(dataset_name))

                trainset = TrainData(train_img, train_label, train_mod, transform=transform_test, img_size=(cfg.W, cfg.H))
                train_loader = data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=get_debug_num_workers())
                vis_features_mean, vis_labels_named, inf_features_mean, inf_labels_named = get_old_proto(train_loader, bare_model, task_id=idx)
                proto_type = {
                    'vis_features_mean': vis_features_mean,
                    'vis_labels_named': vis_labels_named,
                    'inf_features_mean': inf_features_mean,
                    'inf_labels_named': inf_labels_named
                }
                torch.save(proto_type, osp.join(args.logs_dir, dataset_name + '_proto.pth'))

                for eval_mode in parse_eval_k3_fusion_modes():
                    set_k3_eval_fusion_mode(bare_model, eval_mode)
                    print('[!INFO] Eval K3 fusion mode: {}'.format(eval_mode))

                    head_str, results_str, copy_str = '|', '|', ''
                    stage_metrics = []
                    mean_R1, mean_mAP = 0, 0
                    for ii, testset_name in enumerate(training_set):
                        if ii > idx:
                            continue
                        cmc, mAP, mINP = test(
                            bare_model,
                            query_loaders[ii],
                            gall_loaders[ii],
                            testset_name,
                            query_labels[ii],
                            gall_labels[ii],
                            query_cams[ii],
                            gall_cams[ii],
                            task_id=ii,
                            stage_name=dataset_name,
                        )
                        head_str += testset_name + '|\t'
                        results_str += '{:.2f}/{:.2f}|\t'.format(mAP * 100, cmc[0] * 100)
                        copy_str += '{:.2f}\t{:.2f}\t'.format(mAP * 100, cmc[0] * 100)
                        mean_R1 += cmc[0]
                        mean_mAP += mAP
                        stage_metrics.append({
                            'dataset': testset_name,
                            'mAP': float(mAP),
                            'R1': float(cmc[0]),
                            'mINP': float(mINP),
                        })

                    mean_R1 /= (idx + 1)
                    mean_mAP /= (idx + 1)
                    head_str += 'AVG|\t'
                    results_str += '{:.2f}/{:.2f}|\t'.format(mean_mAP * 100, mean_R1 * 100)
                    copy_str += '{:.2f}\t{:.2f}'.format(mean_mAP * 100, mean_R1 * 100)

                    print("Results:")
                    print(head_str)
                    print(results_str)
                    print(copy_str)
                    append_stage_results(dataset_name, idx, stage_metrics, eval_k3_fusion_mode=eval_mode)
                with open(stage_done_path(dataset_name), 'w', encoding='utf-8') as stage_done_file:
                    stage_done_file.write('done\n')

    if branch_stats_collector is not None:
        branch_stats_collector.close()
    if k1_align_hook is not None:
        k1_align_hook.close()

if distributed:
    dist.destroy_process_group()
