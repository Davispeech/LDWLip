import ast
import logging
import os.path

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter  # 查看中间的输出过程

from ldw_utils.eval_matrix import compute_WordorChar_level_distance
from ldw_utils.parseCFG import parse_cfg
from ldw_utils.ini_utils import ini_cfg, get_filetype
from ldw_datas.av_dataset import AVDataset
from torch.utils.data import DataLoader
from ldw_nets.model import Model
from ldw_performance import results_analysis


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def run_register(args):
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model)
    cfg = ini_cfg(args, cfg)

    if isinstance(cfg.train.device, list):
        total_gpus = len(cfg.train.device)
    else:
        total_gpus = 1
    if isinstance(cfg.train.device, list):
        device = cfg.train.device
    else:
        device = ast.literal_eval(cfg.train.device)

    model = Model(cfg)
    # 加载权重
    if total_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=device)
    model = model.cuda(device[0])
    if cfg.default.weights or cfg.default.aux_weights:
        model.load(cfg.train.freeze)

    dataset_eval = AVDataset(cfg, "test")
    eval_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=None)

    # register -----------------------------------------------------
    cer_per=0
    cer_total=0
    model.eval()
    with torch.no_grad():
        val_num = len(eval_loader)
        eval_loader = tqdm(eval_loader, total=val_num, desc=f"eval data: ", ncols=150)
        for index, data in enumerate(eval_loader, 0):  # 使用enumerate 返回下标和值 追踪
            if 'target_0_' in data:
                current_ids = data['target_1_'].item()
            else:
                current_ids = data['target'].item()
            if current_ids not in cfg.model.Output.output_param.ids_user:
                continue
            batch = {}
            batch['videos'] = data['video'].unsqueeze(0)
            B,T,_,_,_ = batch['videos'].shape
            batch['video_lengths'] = torch.full((B,), T)
            batch['path'] = data['path']
            for key, value in data.items():
                if 'target_0_' in key:
                    batch['target_0_s'] = data['target_0_'].unsqueeze(0)
                    batch['target_1__lengths'] = torch.tensor([len(batch['target_0_s'])])
                    batch['target_1_s'] = data['target_1_']
                    batch['target_1__lengths'] = torch.tensor([1])
                    break
                if 'target' in key:
                    batch['targets'] = data['target'].unsqueeze(0)
                    batch['target_lengths'] = torch.tensor([len(batch['targets'])])
                    break

            if isinstance(batch, dict):
                for key in batch:
                    if not isinstance(batch[key], str):
                        batch[key] = batch[key].cuda(device[0])
            else:
                batch = batch.cuda(device[0])
            ## 前馈计算
            losses = model.loss(batch)

            # 更新 tqdm 的 postfix（注意：这不是实时更新的，会在下一次迭代开始时显示）
            eval_loader.set_postfix(cer_per=f"{cer_per:.4f}", cer_total=f"{cer_total:.4f}")

