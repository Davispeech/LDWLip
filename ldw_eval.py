import ast
import logging
import os.path

from accelerate import Accelerator
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter # 查看中间的输出过程
import time
from ldw_train import getConcatDataset
from ldw_datas.samplers import ByFrameCountSampler
from ldw_utils.eval_matrix import compute_WordorChar_level_distance
from ldw_utils.parseCFG import parse_cfg
from ldw_utils.ini_utils import ini_cfg
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
    """Collate tensors and keep integer indices for robust multi-GPU logging.

    - Pads tensor fields (video, audio, target) and their lengths.
    - Aggregates integer `index` into a tensor `indices`.
    - Skips other non-tensor metadata.
    """
    # Filter out None items from batch (dataset may return None on failed load)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    batch_out = {}
    first = batch[0]
    for key in first.keys():
        # Keep indices as tensor (safe across devices)
        if key == "index":
            batch_out["indices"] = torch.tensor([int(s["index"]) for s in batch], dtype=torch.long)
            continue
        # Tensor-like fields only
        val0 = batch[0][key]
        if not torch.is_tensor(val0):
            # skip non-tensors (e.g., meta info)
            continue
        pad_val = -1 if key == "target" else 0.0
        c_batch, sample_lengths = pad([s[key] for s in batch if s[key] is not None], pad_val)
        batch_out[key + "s"] = c_batch
        batch_out[key + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out

def custom_collate_fn(batch):
    """
    一个自定义的 collate_fn，它会过滤掉值为 None 的样本。
    batch: 是一个列表，其中包含了从 __getitem__ 返回的样本。
    """
    # 过滤掉所有值为 None 的项 (这些是加载失败的样本)
    batch = [item for item in batch if item is not None]
    
    # 如果整个批次都加载失败了，返回 None 或者一个空的 batch
    if not batch:
        return None # 或者根据你的循环逻辑返回一个可处理的空对象
        
    # 使用 PyTorch 默认的 collate 函数来处理剩下的、有效的样本
    return torch.utils.data.default_collate(batch)

def run_eval(args):
    # 初始化 Accelerator
    accelerator = Accelerator()
    accelerator.even_batches = False
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model, root_dir=args.root_dir)
    cfg = ini_cfg(args, cfg)

    if args.outfile:
        cfg.eval.outfile=args.outfile
    # timestamp = time.time()
    # local_time = time.localtime(timestamp)
    # formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

    log_dir = os.path.join(
        cfg.default.root_dir,
        cfg.train.run_exp_dir,
        # "test--"+formatted_time.replace(':', '：'),
        "test--" + cfg.train.model_name,
        'log'
    )
    # Ensure log dir exists on all ranks; barrier is unnecessary here and can warn under NCCL
    # log_dir = log_dir.replace(':', '：')
    os.makedirs(log_dir, exist_ok=True)

    # 单独评估只编码和解码，不计算训练损失。束搜索仍与验证相同。
    cfg.default.mode = "eval"
    model = Model(cfg)
    # 加载权重
    if cfg.default.weights:
        model.load(cfg.train.freeze)
    # 挂载lora
    if cfg.train.lora and cfg.default.lora_weights:
        # model.load_state_dict(cfg.default.lora_weights, strict=False)
        lora_oarams = cfg.model.LoRA
        # [3, LoRALinear, [8, 16], ["feed_forward_macaron"]]
        for lora_oaram in lora_oarams:
            from ldw_nets.lora.inject_lora_to_model import inject_lora_to_model, get_submodule_by_name
            inject_lora_to_model(model.model[lora_oaram[0]], lora_oaram)  # get_submodule_by_name(model, lora_oaram[0])
        lora_path = cfg.default.lora_weights
        if not os.path.isabs(lora_path):
            lora_path = cfg.default.root_dir + '/' + lora_path
        lora_state = torch.load(lora_path, map_location="cpu")
        if isinstance(lora_state, dict) and 'model' in lora_state:
            lora_state = lora_state['model']
        elif isinstance(lora_state, dict) and 'state_dict' in lora_state:
            lora_state = lora_state['state_dict']
        first_key = next(iter(lora_state))
        # Training saves the inner network, so those keys have no "model." prefix.
        target = model if str(first_key).startswith("model.") else model.model
        incompatible = target.load_state_dict(lora_state, strict=False)
        loaded = len(lora_state) - len(incompatible.unexpected_keys)
        print(f"loading lora: {lora_path} loaded {loaded}/{len(lora_state)}")
        if loaded == 0:
            raise RuntimeError(f"no LoRA weights matched: {lora_path}")

    if hasattr(cfg.data, "data_list"):
        dataset_eval = getConcatDataset(cfg, "test")
    else:
        dataset_eval = AVDataset(cfg, "test")
    model.src_paths = dataset_eval.metadata_df.relative_path

    # 使用与训练/验证一致的按帧采样器，构建多样本批次
    val_sampler = ByFrameCountSampler(
        dataset_eval,
        cfg.data.data_utils.max_frames_val,
        shuffle=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset_eval,
        num_workers=cfg.train.workers,
        pin_memory=True,
        batch_sampler=val_sampler,
        collate_fn=collate_pad,
    )

    # 使用 Accelerator 准备模型、优化器、数据加载器
    model, eval_loader = accelerator.prepare(
        model, eval_loader
    )
    # eval -----------------------------------------------------
    outfile = os.path.join(log_dir, f"result_on_device{torch.cuda.current_device()}.txt")
    with open(outfile, 'a') as f:
        f.write(f"eval on device {torch.cuda.current_device()}!!!\n")
        for attr_name, attr_value in vars(args).items():
            f.write(f'{attr_name}: {attr_value}\n')
        f.write(f"loading weight from {cfg.default.weights}\n")
        f.write("---------------------------分割线-----------------------------\n")
        f.close()
    distance_per = 0
    length_per = 0
    dis_total = 0
    len_total = 0

    model.eval()
    with torch.no_grad():
        val_num = len(eval_loader)
        if accelerator.is_main_process:
            eval_loader = tqdm(eval_loader, total=val_num, desc=f"eval data: ", ncols=150)
        for batch_idx, data in enumerate(eval_loader, 0):  # 使用enumerate 返回下标和值 追踪
            if data is None:
                continue
            # data_cpu = {k: v.detach().cpu() for k, v in data.items()}
            # torch.save(data_cpu, r"./eval_input_data.pt")

            out = model(data)
            # 兼容：val路径返回 (losses, preds, actuals)，老的eval路径返回 (predict, actual)
            if isinstance(out, tuple) and len(out) == 3:
                _, preds, actuals = out
                inds_t = data.get("indices", None)
                inds = inds_t.detach().cpu().tolist() if torch.is_tensor(inds_t) else list(range(len(preds)))
                src_paths = []
                for i in inds:
                    try:
                        ds_name = dataset_eval.metadata_df.at[int(i), 'dataset_name']
                        rel = dataset_eval.metadata_df.at[int(i), 'relative_path']
                        src_paths.append(os.path.join(dataset_eval.data_root_dir, str(ds_name), str(rel)))
                    except Exception:
                        if 0 <= int(i) < len(getattr(dataset_eval, '_relative_paths', [])):
                            src_paths.append(os.path.join(
                                dataset_eval.data_root_dir,
                                str(dataset_eval._dataset_names[int(i)]),
                                str(dataset_eval._relative_paths[int(i)])
                            ))
                        else:
                            src_paths.append("")
                lines = []
                for predict, actual, src in zip(preds, actuals, src_paths):
                    # if 'en' in cfg.data.char_path:
                    #     distance_per = compute_WordorChar_level_distance(actual, predict, 'en')
                    #     length_per = len(actual.split())
                    # elif 'zh' in cfg.data.char_path:
                    #     distance_per = compute_WordorChar_level_distance(actual, predict, 'zh')
                    #     length_per = len(list(actual.replace('\n', '')))
                    language = cfg.data.char_path.split('dictionary/')[1].split('/')[0]
                    if language == 'en':
                        distance_per = compute_WordorChar_level_distance(actual, predict, 'en')
                        length_per = len(actual.split())
                    elif language == 'zh-cn':
                        distance_per = compute_WordorChar_level_distance(actual, predict, 'zh')
                        length_per = len(list(actual.replace('\n', '')))
                    dis_total += distance_per
                    len_total += length_per
                    cer_per = distance_per / max(1, length_per)
                    cer_total = dis_total / max(1, len_total)
                    lines.append(
                        "path: " + str(src) +
                        "   actual: " + actual +
                        "   predict: " + predict +
                        "   cer_per: " + str(cer_per) +
                        "   cer_total: " + str(cer_total) + "\n"
                    )
                if lines:
                    with open(outfile, "a") as f:
                        f.writelines(lines)
                    if accelerator.is_main_process:
                        eval_loader.set_postfix(cer_per=f"{cer_per:.4f}", cer_total=f"{cer_total:.4f}")
            else:
                predict, actual = out
                inds_t = data.get("indices", None)
                inds = inds_t.detach().cpu().tolist() if torch.is_tensor(inds_t) else []
                src_paths = [""]
                if isinstance(inds, list) and len(inds) > 0:
                    i = int(inds[0])
                    try:
                        ds_name = dataset_eval.metadata_df.at[i, 'dataset_name']
                        rel = dataset_eval.metadata_df.at[i, 'relative_path']
                        src_paths = [os.path.join(dataset_eval.data_root_dir, str(ds_name), str(rel))]
                    except Exception:
                        if 0 <= i < len(getattr(dataset_eval, '_relative_paths', [])):
                            src_paths = [os.path.join(
                                dataset_eval.data_root_dir,
                                str(dataset_eval._dataset_names[i]),
                                str(dataset_eval._relative_paths[i])
                            )]
                if 'en' in cfg.data.char_path:
                    distance_per = compute_WordorChar_level_distance(actual, predict, 'en')
                    length_per = len(actual.split())
                elif 'zh' in cfg.data.char_path:
                    distance_per = compute_WordorChar_level_distance(actual, predict, 'zh')
                    length_per = len(list(actual.replace('\n', '')))
                dis_total += distance_per
                len_total += length_per
                cer_per = distance_per/length_per
                cer_total = dis_total/len_total
                if accelerator.is_main_process:
                    eval_loader.set_postfix(cer_per=f"{cer_per:.4f}", cer_total=f"{cer_total:.4f}")
                with open(outfile, 'a') as f:
                    f.write(
                        "path: " + str(src_paths[0]) +
                        "   actual: " + actual +
                        "   predict: " + predict +
                        "   cer_per: " + str(cer_per) +
                        "   cer_total: " + str(cer_total) + "\n"
                    )
                    f.flush()
    with open(outfile,"a") as f:
        f.write("*"*100+"\n")
        f.write(f"average cer: {cer_total}")
    # results_analysis(cfg.default.root_dir, args.data, outfile)


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        import glob
        merged_path = os.path.join(log_dir, "result_all.txt")
        part_files = sorted(glob.glob(os.path.join(log_dir, "result_on_device*.txt")))
        total_dis = 0
        total_len = 0
        total_samples = 0
        with open(merged_path, 'w') as fout:
            for pf in part_files:
                if not os.path.exists(pf):
                    continue
                with open(pf, 'r') as fin:
                    for line in fin:
                        if ("actual:" in line) and ("predict:" in line):
                            # Copy sample line to merged file
                            fout.write(line)
                            # Parse actual / predict
                            parts = [p.strip() for p in line.strip().split("   ") if p.strip()]
                            actual = ""
                            predict = ""
                            for p in parts:
                                if p.startswith("actual: "):
                                    actual = p[len("actual: ") :]
                                elif p.startswith("predict: "):
                                    predict = p[len("predict: ") :]
                            if actual == "":
                                continue
                            # Recompute distances for global aggregation
                            if 'en' in cfg.data.char_path:
                                distance_per = compute_WordorChar_level_distance(actual, predict, 'en')
                                length_per = len(actual.split())
                            else:
                                distance_per = compute_WordorChar_level_distance(actual, predict, 'zh')
                                length_per = len(list(actual.replace('\n', '')))
                            total_dis += distance_per
                            total_len += max(1, length_per)
                            total_samples += 1
        global_cer = (total_dis / total_len) if total_len > 0 else 0.0
        with open(merged_path, 'a') as fout:
            fout.write("*" * 100 + "\n")
            fout.write(f"total samples: {total_samples}\n")
            fout.write(f"global average cer: {global_cer}\n")

        # if cfg.eval.outfile:
        #     with open(merged_path, 'rb') as sf, open(cfg.eval.outfile, 'wb') as df:
        #         df.write(sf.read())

