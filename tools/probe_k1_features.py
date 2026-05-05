"""Probe K1 identity and modality information from trained checkpoints.

This script is intentionally read-only for training behavior. It extracts three
feature views for the selected K1 blocks:

- scaled_k1_delta: direct K1 branch contribution.
- block_output: normal output of the K1 blocks.
- block_output_disable_k1: output of the same blocks with K1 gamma zeroed.

The goal is to separate K1 branch contribution from backbone/block identity
capacity before adding any K1 alignment loss.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
import os.path as osp
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.config import cfg
from dataloader import LLCMData, RegDBData, SYSUData, VCMData
from model.make_model import build_vision_transformer
from transforms import transform_test
from utils import set_seed


DATASET_ORDER = ['regdb', 'sysu', 'llcm', 'vcm']


def parse_args():
    parser = argparse.ArgumentParser(description='Probe K1 feature sources')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--dataset', required=True, choices=DATASET_ORDER)
    parser.add_argument('--task_id', required=True, type=int)
    parser.add_argument('--trial', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--max-samples-per-modality', default=1000, type=int)
    parser.add_argument('--k1-blocks', default='4,7', type=str)
    parser.add_argument('--k1-modules', default='qkv,proj', type=str)
    parser.add_argument('--eval-k3-fusion-mode', default='current_only',
                        choices=['all_except_current', 'previous', 'current_only'])
    parser.add_argument('--route_tau', default=0.25, type=float)
    parser.add_argument('--train_k3_old_scale', default=0.4, type=float)
    parser.add_argument('--eval_k3_old_scale', default=0.4, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def parse_int_list(value):
    if value is None or str(value).strip() == '':
        return []
    return [int(part.strip()) for part in str(value).split(',') if part.strip()]


def parse_str_list(value):
    if value is None or str(value).strip() == '':
        return []
    return [part.strip() for part in str(value).split(',') if part.strip()]


class FlatModalityDataset(data.Dataset):
    def __init__(self, color_images, color_labels, thermal_images, thermal_labels,
                 transform=None, max_samples_per_modality=0, seed=0):
        self.transform = transform
        samples = []
        for index, label in enumerate(color_labels):
            samples.append((1, index, int(label)))
        for index, label in enumerate(thermal_labels):
            samples.append((0, index, int(label)))

        if max_samples_per_modality and max_samples_per_modality > 0:
            rng = random.Random(seed)
            limited = []
            for mod in (1, 0):
                mod_samples = [item for item in samples if item[0] == mod]
                if len(mod_samples) > max_samples_per_modality:
                    mod_samples = rng.sample(mod_samples, max_samples_per_modality)
                limited.extend(mod_samples)
            samples = limited

        self.color_images = color_images
        self.color_labels = color_labels
        self.thermal_images = thermal_images
        self.thermal_labels = thermal_labels
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mod, image_index, label = self.samples[index]
        if mod == 1:
            image = self.color_images[image_index]
        else:
            image = self.thermal_images[image_index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, mod


def build_flat_dataset(dataset_name, trial, max_samples_per_modality, seed):
    if dataset_name == 'regdb':
        dataset = RegDBData(cfg.DATA_PATH_RegDB, trial, transform1=transform_test, transform2=transform_test)
    elif dataset_name == 'sysu':
        dataset = SYSUData(cfg.DATA_PATH_SYSU, transform1=transform_test, transform2=transform_test)
    elif dataset_name == 'llcm':
        dataset = LLCMData(cfg.DATA_PATH_LLCM, transform1=transform_test, transform2=transform_test)
    elif dataset_name == 'vcm':
        dataset = VCMData(cfg.DATA_PATH_VCM, transform1=transform_test, transform2=transform_test)
    else:
        raise ValueError('Unsupported dataset: {}'.format(dataset_name))

    return FlatModalityDataset(
        dataset.train_color_image,
        dataset.train_color_label,
        dataset.train_thermal_image,
        dataset.train_thermal_label,
        transform=transform_test,
        max_samples_per_modality=max_samples_per_modality,
        seed=seed,
    )


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
    return checkpoint


def normalized_key(key):
    return key.replace('module.', '')


def infer_checkpoint_metadata(checkpoint_path, fallback_task_id):
    checkpoint = unwrap_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    num_classes = None
    max_task_id = fallback_task_id
    task_pattern = re.compile(r'task_bank\.experts\.(\d+)')

    for raw_key, value in checkpoint.items():
        key = normalized_key(raw_key)
        if key == 'classifier.weight':
            num_classes = int(value.shape[0])
        match = task_pattern.search(key)
        if match is not None:
            max_task_id = max(max_task_id, int(match.group(1)))

    if num_classes is None:
        raise ValueError('Could not infer classifier.weight from {}'.format(checkpoint_path))
    return checkpoint, num_classes, max_task_id


def load_checkpoint_into_model(model, state_dict):
    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for raw_key, value in state_dict.items():
        key = normalized_key(raw_key)
        if key not in model_state:
            if '.attn.qkv.' in key:
                key = key.replace('.attn.qkv.', '.attn.qkv.base.')
            elif '.attn.proj.' in key:
                key = key.replace('.attn.proj.', '.attn.proj.base.')
        if key not in model_state or model_state[key].shape != value.shape:
            skipped += 1
            continue
        model_state[key].copy_(value)
        loaded += 1
    print('[!INFO] Loaded checkpoint tensors: loaded={}, skipped={}'.format(loaded, skipped))


def configure_k3_router(model, route_tau, train_old_scale, eval_old_scale, eval_fusion_mode):
    configured = 0
    for module in model.modules():
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
        configured += 1
    print('[!INFO] Configured K3 router banks={}'.format(configured))


def parse_branch_module_name(name):
    parts = name.split('.')
    if len(parts) != 5:
        return None
    if parts[0] != 'base' or parts[1] != 'blocks' or parts[3] != 'attn':
        return None
    if parts[4] not in ('qkv', 'proj'):
        return None
    try:
        block_id = int(parts[2])
    except ValueError:
        return None
    return block_id, parts[4]


def pooled_cls(tensor):
    if tensor.dim() >= 3:
        return tensor[:, 0].detach().float().flatten(1)
    return tensor.detach().float().flatten(1)


class K1FeatureCollector:
    def __init__(self, model, block_ids, branch_names):
        self.model = model
        self.block_ids = set(block_ids)
        self.branch_names = set(branch_names)
        self.handles = []
        self.scaled_k1_delta = {}
        self.block_output = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            parsed = parse_branch_module_name(name)
            if parsed is not None:
                block_id, branch_name = parsed
                if block_id in self.block_ids and branch_name in self.branch_names and getattr(module, 'shared_adapter', None) is not None:
                    self.handles.append(module.register_forward_hook(self._make_k1_hook(name)))

        for block_id in self.block_ids:
            if block_id < 0 or block_id >= len(self.model.base.blocks):
                raise ValueError('K1 block id out of range: {}'.format(block_id))
            block = self.model.base.blocks[block_id]
            self.handles.append(block.register_forward_hook(self._make_block_hook('base.blocks.{}'.format(block_id))))

    def _make_k1_hook(self, name):
        def hook(module, inputs, output):
            x = inputs[0]
            with torch.no_grad():
                scaled = module.gamma_k1 * module.shared_adapter(x)
                self.scaled_k1_delta[name] = pooled_cls(scaled).cpu()
        return hook

    def _make_block_hook(self, name):
        def hook(module, inputs, output):
            self.block_output[name] = pooled_cls(output).cpu()
        return hook

    def reset(self):
        self.scaled_k1_delta = {}
        self.block_output = {}

    def get_scaled_k1_delta(self):
        return self._concat(self.scaled_k1_delta)

    def get_block_output(self):
        return self._concat(self.block_output)

    def _concat(self, values):
        if not values:
            return None
        return torch.cat([values[key] for key in sorted(values)], dim=1)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@contextmanager
def zero_k1_gamma(model, block_ids, branch_names):
    saved = []
    block_ids = set(block_ids)
    branch_names = set(branch_names)
    for name, module in model.named_modules():
        parsed = parse_branch_module_name(name)
        if parsed is None:
            continue
        block_id, branch_name = parsed
        if block_id not in block_ids or branch_name not in branch_names:
            continue
        gamma = getattr(module, 'gamma_k1', None)
        if gamma is None:
            continue
        saved.append((gamma, gamma.detach().clone()))
        gamma.data.zero_()
    try:
        yield
    finally:
        for gamma, value in saved:
            gamma.data.copy_(value)


def extract_features(model, loader, device, task_id, block_ids, branch_names):
    collector = K1FeatureCollector(model, block_ids, branch_names)
    outputs = defaultdict(list)
    label_chunks = []
    mod_chunks = []
    model.eval()

    with torch.no_grad():
        for images, labels, mods in loader:
            images = images.to(device, non_blocking=True)
            mods = mods.to(device, non_blocking=True)

            collector.reset()
            model(images, mods, task_id=task_id)
            scaled_k1 = collector.get_scaled_k1_delta()
            block_output = collector.get_block_output()
            if scaled_k1 is not None:
                outputs['scaled_k1_delta'].append(scaled_k1)
            if block_output is not None:
                outputs['block_output'].append(block_output)

            collector.reset()
            with zero_k1_gamma(model, block_ids, branch_names):
                model(images, mods, task_id=task_id)
            disabled_output = collector.get_block_output()
            if disabled_output is not None:
                outputs['block_output_disable_k1'].append(disabled_output)

            label_chunks.append(labels.cpu().long())
            mod_chunks.append(mods.cpu().long())

    collector.close()
    features = {key: torch.cat(value, dim=0) for key, value in outputs.items() if value}
    labels = torch.cat(label_chunks, dim=0)
    mods = torch.cat(mod_chunks, dim=0)
    return features, labels, mods


def l2_normalize(features):
    return F.normalize(features.float(), p=2, dim=1)


def nearest_centroid_accuracy(features, targets):
    features = l2_normalize(features)
    targets = targets.long()
    classes = torch.unique(targets).sort()[0]
    centroids = []
    for cls in classes:
        centroids.append(features[targets == cls].mean(dim=0))
    centroids = l2_normalize(torch.stack(centroids, dim=0))
    scores = features @ centroids.t()
    pred = classes[scores.argmax(dim=1)]
    return float((pred == targets).float().mean().item())


def same_id_cross_modality_margin(features, labels, mods):
    features = l2_normalize(features)
    labels = labels.long()
    mods = mods.long()
    rgb_means = {}
    ir_means = {}
    for label in torch.unique(labels).tolist():
        label_mask = labels == int(label)
        rgb_mask = label_mask & (mods == 1)
        ir_mask = label_mask & (mods == 0)
        if rgb_mask.any():
            rgb_means[int(label)] = features[rgb_mask].mean(dim=0)
        if ir_mask.any():
            ir_means[int(label)] = features[ir_mask].mean(dim=0)

    shared_labels = sorted(set(rgb_means).intersection(ir_means))
    if not shared_labels:
        return None, None, None, 0

    rgb = l2_normalize(torch.stack([rgb_means[label] for label in shared_labels], dim=0))
    ir = l2_normalize(torch.stack([ir_means[label] for label in shared_labels], dim=0))
    sim = rgb @ ir.t()
    same = sim.diag()
    if sim.numel() == same.numel():
        diff_mean = None
        margin = None
    else:
        diff_mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
        diff = sim[diff_mask]
        diff_mean = float(diff.mean().item())
        margin = float((same.mean() - diff.mean()).item())
    return float(same.mean().item()), diff_mean, margin, len(shared_labels)


def feature_norm_mean(features):
    return float(torch.linalg.vector_norm(features.float(), dim=1).mean().item())


def summarize_source(source_name, features, labels, mods):
    identity_acc = nearest_centroid_accuracy(features, labels)
    modality_acc = nearest_centroid_accuracy(features, mods)
    same_mean, diff_mean, margin, shared_ids = same_id_cross_modality_margin(features, labels, mods)
    return {
        'source': source_name,
        'num_samples': int(features.shape[0]),
        'feature_dim': int(features.shape[1]),
        'feature_norm_mean': feature_norm_mean(features),
        'identity_centroid_acc': identity_acc,
        'modality_centroid_acc': modality_acc,
        'same_id_rgb_ir_cosine': same_mean,
        'diff_id_rgb_ir_cosine': diff_mean,
        'same_diff_margin': margin,
        'shared_rgb_ir_id_count': int(shared_ids),
    }


def write_outputs(output_dir, rows, metadata):
    os.makedirs(output_dir, exist_ok=True)
    json_path = osp.join(output_dir, 'k1_probe_summary.json')
    tsv_path = osp.join(output_dir, 'k1_probe_summary.tsv')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump({'metadata': metadata, 'rows': rows}, handle, indent=2, sort_keys=True)

    columns = [
        'source',
        'num_samples',
        'feature_dim',
        'feature_norm_mean',
        'identity_centroid_acc',
        'modality_centroid_acc',
        'same_id_rgb_ir_cosine',
        'diff_id_rgb_ir_cosine',
        'same_diff_margin',
        'shared_rgb_ir_id_count',
    ]
    with open(tsv_path, 'w', encoding='utf-8') as handle:
        handle.write('\t'.join(columns) + '\n')
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col)
                if isinstance(value, float):
                    values.append('{:.8f}'.format(value))
                elif value is None:
                    values.append('')
                else:
                    values.append(str(value))
            handle.write('\t'.join(values) + '\n')
    print('[!INFO] Wrote {}'.format(json_path))
    print('[!INFO] Wrote {}'.format(tsv_path))


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.opts:
        cfg.merge_from_list(args.opts)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block_ids = parse_int_list(args.k1_blocks)
    branch_names = parse_str_list(args.k1_modules)
    if not block_ids:
        raise ValueError('--k1-blocks must not be empty')
    if not branch_names:
        raise ValueError('--k1-modules must not be empty')

    state_dict, num_classes, max_task_id = infer_checkpoint_metadata(args.checkpoint, args.task_id)
    print('[!INFO] checkpoint num_classes={}, max_task_id={}'.format(num_classes, max_task_id))

    dataset = build_flat_dataset(
        args.dataset,
        args.trial,
        args.max_samples_per_modality,
        args.seed,
    )
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print('[!INFO] Probe samples={}'.format(len(dataset)))

    model = build_vision_transformer(num_classes=num_classes, cfg=cfg)
    for task_id in range(max_task_id + 1):
        model.add_task(task_id)
    load_checkpoint_into_model(model, state_dict)
    configure_k3_router(
        model,
        args.route_tau,
        args.train_k3_old_scale,
        args.eval_k3_old_scale,
        args.eval_k3_fusion_mode,
    )
    model.to(device)
    model.eval()

    features, labels, mods = extract_features(
        model,
        loader,
        device,
        args.task_id,
        block_ids,
        branch_names,
    )
    rows = [summarize_source(source, features[source], labels, mods) for source in sorted(features)]
    metadata = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'task_id': int(args.task_id),
        'k1_blocks': block_ids,
        'k1_modules': branch_names,
        'max_samples_per_modality': int(args.max_samples_per_modality),
        'eval_k3_fusion_mode': args.eval_k3_fusion_mode,
    }
    write_outputs(args.output_dir, rows, metadata)


if __name__ == '__main__':
    main()
