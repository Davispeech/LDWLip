import ast
import logging
import os.path

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 查看中间的输出过程

import torch

from ldw_utils.parseCFG import parse_cfg
from ldw_utils.ini_utils import ini_cfg
from ldw_datas.av_dataset import AVDataset
from torch.utils.data import DataLoader
from ldw_nets.model import Model
from ldw_utils.cosine import WarmupCosineScheduler

from ldw_datas.samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)


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
        pad_val = -1 if "target" in data_type else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def run_train(args):
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model)
    cfg = ini_cfg(args, cfg)

    log_dir = cfg.default.root_dir+'/'+cfg.train.run_exp_dir+'/'+cfg.train.model_name + '/log'
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)

    if isinstance(cfg.train.device, list):
        total_gpus = len(cfg.train.device)
    else:
        total_gpus = 1
    if isinstance(cfg.train.device, list):
        device = cfg.train.device
    else:
        device = ast.literal_eval(cfg.train.device)

    model = Model(cfg)
    if total_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=device)
    model = model.cuda(device[0])
    # 加载权重
    if cfg.default.weights or cfg.train.aux_weights:
        model.load(cfg.train.freeze)

    if "_webtar.yaml" in args.data:  # == "lrwAuthWord_webtar.yaml":
        from ldw_datas.web_dataset import WebAVDataset
        dataset_train = WebAVDataset(cfg=cfg, subset="train")
        train_loader = DataLoader(
            dataset_train,
            batch_size=cfg.data.data_utils.batch_size,
            num_workers=cfg.train.workers,
            pin_memory=True,
            collate_fn=collate_pad,
            persistent_workers=True,
            prefetch_factor=4
        )
        dataset_val = WebAVDataset(cfg=cfg, subset="val")
        val_loader = DataLoader(
            dataset_val,
            batch_size=cfg.data.data_utils.batch_size,
            num_workers=1,  # cfg.train.workers,
            pin_memory=True,
            collate_fn=collate_pad,
            persistent_workers=True,
            prefetch_factor=4
        )
        train_num = getattr(cfg.train, "approx_len", int(dataset_train.count/cfg.data.data_utils.batch_size)+1)
        val_num = getattr(cfg.train, "approx_len", int(dataset_val.count/cfg.data.data_utils.batch_size)+1)
    else:
        dataset_train = AVDataset(cfg, "train")
        sampler = ByFrameCountSampler(dataset_train, cfg.data.data_utils.max_frames)
        if total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
        train_loader = DataLoader(dataset_train,
                                  num_workers=cfg.train.workers,
                                  pin_memory=True,
                                  batch_sampler=sampler,
                                  collate_fn=collate_pad,
                                  persistent_workers=True if cfg.train.workers else False,
                                  prefetch_factor=4 if cfg.train.workers else None
                                  )
        dataset_val = AVDataset(cfg, "val")
        sampler = ByFrameCountSampler(dataset_val, cfg.data.data_utils.max_frames_val, shuffle=False)
        if total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        else:
            sampler = RandomSamplerWrapper(sampler)
        val_loader = DataLoader(dataset_val,
                                num_workers=cfg.train.workers,
                                pin_memory=True,
                                batch_sampler=sampler,
                                collate_fn=collate_pad,
                                persistent_workers=True if cfg.train.workers else False,
                                prefetch_factor=4 if cfg.train.workers else None
                                )
        train_num = len(train_loader)
        val_num = len(val_loader)

    optimizer = torch.optim.AdamW(
        [{"name": "model", "params": model.parameters(), "lr": cfg.train.lr}],
        weight_decay=cfg.train.weight_decay, betas=(0.9, 0.98))
    scheduler = WarmupCosineScheduler(optimizer, cfg.train.warmup_epochs, cfg.train.max_epochs, train_num, cfg.train.lr_min)

    # ---------------- Resume 部分 ----------------
    # ---------------- Resume or Pretrain ----------------
    start_epoch = 0
    if cfg.default.weights or cfg.train.aux_weights:
        model.load(cfg.train.freeze)

        if cfg.default.mode == "resume":
            # 恢复训练
            resume_path = cfg.default.weights
            if os.path.isfile(resume_path):
                print(f"=> Resuming from checkpoint '{resume_path}'")
                checkpoint = torch.load(resume_path, map_location="cuda:{}".format(device[0]))

                # 恢复 optimizer
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # 恢复 scheduler
                if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"=> Resumed training from epoch {start_epoch}")
            else:
                raise FileNotFoundError(f"=> No checkpoint found at '{resume_path}'")

    for epoch in range(start_epoch, cfg.train.max_epochs):
        # train -----------------------------------------------------
        model.train()
        train_loader = tqdm(train_loader, total=train_num, desc=f"train epoch[{epoch}/{cfg.train.max_epochs}]", ncols=150)
        for index, data in enumerate(train_loader, 0):  # 使用enumerate 返回下标和值 追踪
            if isinstance(data, dict):
                for key in data:
                    data[key] = data[key].cuda(device[0], non_blocking=True)
            else:
                data = data.cuda(device[0])
            ## 前馈计算
            losses = model.loss(data)
            loss = losses["loss_total"]

            ## 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # # save to tensorboard per batch------------------------------
            # save train loss
            loss = loss.item()
            for key, value in losses.items():
                tb_writer.add_scalar("train_" + key, value.item(), index)
            # tb_writer.add_scalar("train_loss", loss, index)
            train_loss_ctc = losses["loss_ctc"].item()
            # tb_writer.add_scalar("train_loss_ctc", train_loss_ctc, index)
            train_loss_att = losses["loss_att"].item()
            # tb_writer.add_scalar("train_loss_att", train_loss_att, index)
            # save lr
            current_lr = optimizer.param_groups[0]["lr"]
            tb_writer.add_scalar("learning_rate", current_lr, index)

            # 更新 tqdm 的 postfix（注意：这不是实时更新的，会在下一次迭代开始时显示）
            train_loader.set_postfix(loss=f"{loss:.4f}", loss_ctc=f"{train_loss_ctc:.4f}",
                                     loss_att=f"{train_loss_att:.4f}")  # , acc=f"{acc:.4f}")

        # # save to tensorboard per epoch------------------------------
        # save train loss
        for key, value in losses.items():
            tb_writer.add_scalar("train_"+key, value.item(), epoch)
        # tb_writer.add_scalar("train_loss", loss, epoch)
        # train_loss_ctc = losses["loss_ctc"].item()
        # tb_writer.add_scalar("train_loss_ctc", train_loss_ctc, epoch)
        # train_loss_att = losses["loss_att"].item()
        # tb_writer.add_scalar("train_loss_att", train_loss_att, epoch)
        # save lr
        current_lr = optimizer.param_groups[0]["lr"]
        tb_writer.add_scalar("learning_rate", current_lr, epoch)

        # val -----------------------------------------------------
        model.eval()
        loss_val = {"val_loss": 0, "val_loss_ctc": 0, "val_loss_att": 0}
        with torch.no_grad():
            val_loader = tqdm(val_loader, total=val_num, desc=f"val epoch[{epoch}/{cfg.train.max_epochs}]", ncols=150)
            for index, data in enumerate(val_loader, 0):  # 使用enumerate 返回下标和值 追踪
                if isinstance(data, dict):
                    for key in data:
                        data[key] = data[key].cuda(device[0])
                else:
                    data = data.cuda(device[0])
                ## 前馈计算
                losses_val = model.loss(data)
                if "targets" not in data:
                    sampler_num = data['target_0_s'].size(0)
                else:
                    sampler_num = data["targets"].size(0)
                loss_val["val_loss"] += losses_val["loss_total"].item() * sampler_num  # 累积损失
                loss_val["val_loss_ctc"] += losses_val["loss_ctc"].item() * sampler_num  # 累积损失
                loss_val["val_loss_att"] += losses_val["loss_att"].item() * sampler_num  # 累积损失
                # acc_avg += acc * sampler_num
        # # save to tensorboard per epoch------------------------------
        # save val loss
        val_loss = loss_val["val_loss"]/val_num
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        val_loss_ctc = loss_val["val_loss_ctc"]/val_num
        tb_writer.add_scalar("val_loss_ctc", val_loss_ctc, epoch)
        val_loss_att = loss_val["val_loss_att"]/val_num
        tb_writer.add_scalar("val_loss_att", val_loss_att, epoch)
        # save lr
        current_lr = optimizer.param_groups[0]["lr"]
        tb_writer.add_scalar("learning_rate", current_lr, epoch)

        # 更新 tqdm 的 postfix（注意：这不是实时更新的，会在下一次迭代开始时显示）
        val_loader.set_postfix(val_loss=f"{val_loss:.4f}", val_loss_ctc=f"{val_loss_ctc:.4f}",
                                 val_loss_att=f"{val_loss_att:.4f}")  # , val_acc=f"{acc:.4f}")

        path = cfg.default.root_dir+'/'+cfg.train.run_exp_dir+'/'+cfg.train.model_name
        # model.save(path, epoch)
        model.save(path, epoch, optimizer=optimizer, scheduler=scheduler)

    tb_writer.close()

