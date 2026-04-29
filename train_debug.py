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
parser.add_argument('--eval_k3_old_scale', type=float, default=1.0, help='scaling factor for fused old K3 experts during evaluation')
parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend for torchrun launches')

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


def append_stage_results(stage_name, stage_idx, metrics):
    results_path = osp.join(args.logs_dir, 'results.jsonl')
    summary_path = osp.join(args.logs_dir, 'results_summary.tsv')
    row = {
        'stage': stage_name,
        'stage_idx': int(stage_idx),
        'metrics': metrics,
        'avg_mAP': float(np.mean([item['mAP'] for item in metrics])),
        'avg_R1': float(np.mean([item['R1'] for item in metrics])),
        'route_tau': float(args.route_tau),
        'train_k3_old_scale': float(args.train_k3_old_scale),
        'eval_k3_old_scale': float(args.eval_k3_old_scale),
        'debug_max_epoch': int(args.debug_max_epoch),
        'debug_batch_size': int(args.debug_batch_size),
        'world_size': int(world_size),
    }

    with open(results_path, 'a', encoding='utf-8') as results_file:
        results_file.write(json.dumps(row, sort_keys=True) + '\n')

    write_header = not osp.exists(summary_path)
    with open(summary_path, 'a', encoding='utf-8') as summary_file:
        if write_header:
            summary_file.write('stage\tstage_idx\tdataset\tmAP\tR1\tavg_mAP\tavg_R1\troute_tau\ttrain_k3_old_scale\teval_k3_old_scale\tdebug_max_epoch\tdebug_batch_size\tworld_size\n')
        for item in metrics:
            summary_file.write(
                '{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    stage_name,
                    int(stage_idx),
                    item['dataset'],
                    item['mAP'] * 100.0,
                    item['R1'] * 100.0,
                    row['avg_mAP'] * 100.0,
                    row['avg_R1'] * 100.0,
                    args.route_tau,
                    args.train_k3_old_scale,
                    args.eval_k3_old_scale,
                    args.debug_max_epoch,
                    args.debug_batch_size,
                    world_size,
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


def configure_k3_router(model, route_tau, train_old_scale, eval_old_scale):
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
        configured += 1
    if configured > 0:
        print('[!INFO] Configure K3 router: banks={}, route_tau={}, train_k3_old_scale={}, eval_k3_old_scale={}'.format(
            configured, route_tau, train_old_scale, eval_old_scale))


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
                'b{}.{} keys={} w={} sim={} ent={} tau={} old_scale={}'.format(
                    block_idx,
                    branch_name,
                    _fmt_router_values(old_keys),
                    _fmt_router_values(snapshot['weights']),
                    _fmt_router_values(snapshot['similarities']),
                    _fmt_router_values(snapshot['entropy']),
                    _fmt_router_values(snapshot['tau']),
                    _fmt_router_values(snapshot.get('old_scale')),
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
          vis_features_mean, inf_features_mean):
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc_rgb = (score1.max(1)[1] == label1).float().mean()
        acc_ir = (score2.max(1)[1] == label2).float().mean()

        loss_tri_meter.update(loss_tri.item())
        loss_ce_meter.update(loss_id.item())
        loss_meter.update(loss.item())
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
                  'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                  'Base Lr: {:.2e}, Time: {:.1f}s, Global Img/s: {:.2f} '.format(
                      epoch,
                      (idx_ + 1),
                      len(trainloader),
                      reduce_mean_scalar(loss_meter.avg),
                      reduce_mean_scalar(loss_tri_meter.avg),
                      reduce_mean_scalar(loss_ce_meter.avg),
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
    configure_k3_router(base_model, args.route_tau, args.train_k3_old_scale, args.eval_k3_old_scale)

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
        configure_k3_router(old_model, args.route_tau, args.train_k3_old_scale, args.eval_k3_old_scale)
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

    query_loaders, gall_loaders, query_labels, gall_labels, query_cams, gall_cams = build_test_loaders(idx)

    print('==> Start Training...')
    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH + 1):
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
        )

        if is_main_process():
            log_k3_router_stats(model, phase='train', stage_name=dataset_name, epoch=epoch)

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
                append_stage_results(dataset_name, idx, stage_metrics)
                with open(stage_done_path(dataset_name), 'w', encoding='utf-8') as stage_done_file:
                    stage_done_file.write('done\n')

if distributed:
    dist.destroy_process_group()
