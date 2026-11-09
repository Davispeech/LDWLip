import copy
from email.policy import strict
import os
import time
import csv
import torch.distributed as dist
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 查看中间的输出过程
from tqdm import tqdm

from ldw_datas.av_dataset import AVDataset


def enroll_identity(model, dataset):
    """Attach the identity target selected by SID_LOSS.

    ce freezes the pretrained speaker network and classifies each clip against
    that network. proto rebuilds one cosine prototype from the first readable
    adaptation clips through the adapting visual frontend.
    """
    sid_path = os.environ.get("SID_WEIGHTS", "").strip()
    if not sid_path:
        return
    if not os.path.isfile(sid_path):
        raise FileNotFoundError(f"identity checkpoint not found: {sid_path}")

    frontend = None
    for module in model.modules():
        if module.__class__.__name__ == "Conv3D_2DNet":
            frontend = module
            break
    if frontend is None:
        raise RuntimeError("identity enrollment could not find Conv3D_2DNet")
    frontend._record_id = True

    from ldw_sid import identity_loss_mode

    mode = identity_loss_mode()
    if mode == "proto":
        _enroll_clip_prototype(model, dataset, frontend, sid_path)
        return
    _enroll_identity_gallery(model, dataset, sid_path)


def _enroll_clip_prototype(model, dataset, frontend, sid_path):
    """Original identity loss: cosine to the first adaptation clips."""
    checkpoint = torch.load(sid_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    weight = state["proj.weight"]
    bias = state["proj.bias"]
    projection = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        projection.weight.copy_(weight)
        projection.bias.copy_(bias)
    projection.eval()
    for param in projection.parameters():
        param.requires_grad_(False)
    device = next(frontend.parameters()).device
    projection.to(device)
    model.id_proj = projection

    was_training = frontend.training
    frontend.eval()
    shots = int(os.environ.get("SID_K_SHOT", "2") or "2")
    embeds = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if not isinstance(sample, dict) or "video" not in sample:
                continue
            video = sample["video"]
            if not torch.is_tensor(video):
                video = torch.as_tensor(video)
            video = video.float()
            if video.dim() == 3:
                video = video.unsqueeze(1)
            video = video.unsqueeze(0).to(device)
            features = frontend(video)
            pooled = features.mean(dim=1)
            embeds.append(torch.nn.functional.normalize(projection(pooled), dim=-1).squeeze(0))
            if len(embeds) >= shots:
                break
    frontend.train(was_training)
    frontend._id_features = None
    if not embeds:
        raise RuntimeError("identity enrollment found no adaptation video")
    prototype = torch.nn.functional.normalize(torch.stack(embeds, 0).mean(dim=0), dim=0)
    model.register_buffer("id_prototype", prototype.detach())
    print(f"identity prototype from {len(embeds)} adaptation clips, dim {prototype.numel()}, sid {sid_path}")


def _enroll_identity_gallery(model, dataset, sid_path):
    from ldw_sid import LipIdentityNet

    checkpoint = torch.load(sid_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    id_net = LipIdentityNet()
    missing, unexpected = id_net.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"identity checkpoint mismatch missing={missing[:8]} unexpected={unexpected[:8]}")
    id_net.eval()
    for param in id_net.parameters():
        param.requires_grad_(False)

    build_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_net.to(build_device)
    gallery, speakers = _build_speaker_gallery(id_net, dataset, build_device)
    home = next(model.parameters()).device
    id_net.to(home)
    model.id_net = id_net
    model.register_buffer("id_gallery", gallery.to(device=home), persistent=False)
    model.id_gallery_names = speakers
    print(
        f"identity gallery {len(speakers)} speakers x {os.environ.get('SID_K_SHOT', '2')} clips, "
        f"per-clip teacher, sid {sid_path}"
    )
    print("identity negatives: " + ", ".join(speakers))

def _adaptation_speakers(dataset):
    from ldw_sid import speaker_key

    names = getattr(dataset, "_dataset_names", None)
    paths = getattr(dataset, "_relative_paths", None)
    if names is None or paths is None:
        raise RuntimeError("adaptation dataset has no speaker paths for identity enrollment")
    return {speaker_key(str(name), str(path)) for name, path in zip(names, paths)}


def _build_speaker_gallery(id_net, dataset, device):
    import pandas as pd
    from ldw_sid import speaker_key
    from ldw_datas.av_dataset import load_video

    root = dataset.root_dir
    csv_path = os.environ.get("SID_GALLERY", "").strip()
    if not csv_path:
        csv_path = os.path.join(root, "code", "labels", "sid", "sid_unseen_train.csv")
    elif not os.path.isabs(csv_path):
        csv_path = os.path.join(root, csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"identity gallery list not found: {csv_path}")

    k_shot = int(os.environ.get("SID_K_SHOT", "2") or "2")
    n_way = int(os.environ.get("SID_N_WAY", "8") or "8")
    n_neg = max(n_way - 1, 1)
    max_frames = int(os.environ.get("SID_MAX_FRAMES", "250") or "250")
    frame = pd.read_csv(csv_path, usecols=lambda name: name in {"dataset_name", "relative_path", "frame_length"})
    if "frame_length" in frame.columns:
        frame = frame[frame["frame_length"] <= max_frames]
    names = frame["dataset_name"].astype(str).to_numpy()
    paths = frame["relative_path"].astype(str).to_numpy()
    targets = _adaptation_speakers(dataset)
    transform = dataset.video_transform
    data_root = dataset.data_root_dir
    partial = {}
    selected = []
    speakers = []
    failures = 0
    id_net.eval()
    with torch.no_grad():
        for index in range(len(names)):
            speaker = speaker_key(names[index], paths[index])
            if speaker in targets:
                continue
            if speaker not in partial and len(selected) + len(partial) >= n_neg:
                continue
            video_path = os.path.join(data_root, names[index], paths[index])
            try:
                video = transform(load_video(video_path))
                if not torch.is_tensor(video):
                    video = torch.as_tensor(video)
                video = video.float()
                if video.dim() == 3:
                    video = video.unsqueeze(1)
                video = video.unsqueeze(0).to(device, non_blocking=True)
                length = torch.tensor([video.size(1)], device=device)
                embedding = id_net(video, length).squeeze(0)
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    print(f"identity gallery skip {video_path}: {exc}")
                continue
            bucket = partial.setdefault(speaker, [])
            bucket.append(embedding)
            if len(bucket) >= k_shot:
                prototype = torch.nn.functional.normalize(torch.stack(bucket[:k_shot], 0).mean(dim=0), dim=0)
                selected.append(prototype)
                speakers.append(speaker)
                partial.pop(speaker, None)
                if len(selected) >= n_neg:
                    break
    if len(selected) < n_neg:
        raise RuntimeError(f"identity gallery found {len(selected)} of {n_neg} speakers in {csv_path}")
    return torch.stack(selected, 0).detach(), speakers

from ldw_datas.samplers import (
    ByFrameCountSampler,
    RandomSamplerWrapper, )
from ldw_nets.model import Model
from ldw_utils.cosine import WarmupCosineScheduler
from ldw_utils.ini_utils import ini_cfg
from ldw_utils.parseCFG import parse_cfg
from ldw_utils.eval_matrix import compute_WordorChar_level_distance

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


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
    # Filter out None items from batch
    batch = [item for item in batch if item is not None]

    # If all items are None, return empty batch
    if not batch:
        return {}

    batch_out = {}
    first = batch[0]
    for key in first.keys():
        # Collect src_path as a simple python list if present
        if key == "src_path":
            batch_out["src_paths"] = [s["src_path"] for s in batch]
            continue
        if key == "index":
            batch_out["indices"] = torch.tensor([int(s["index"]) for s in batch], dtype=torch.long)
            continue
        # Only pad tensor fields
        val0 = first[key]
        if not torch.is_tensor(val0):
            continue
        pad_val = -1 if key == "target" else 0.0
        c_batch, sample_lengths = pad([s[key] for s in batch if s[key] is not None], pad_val)
        batch_out[key + "s"] = c_batch
        batch_out[key + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def ConcatDatasetWithJIT(datasets):
    """
    Concatenate multiple JIT datasets efficiently by combining their compact DataFrames.
    This maintains the lightweight memory footprint of the JIT approach.
    """
    if not datasets:
        raise ValueError("Cannot concatenate empty dataset list")

    # Use the first dataset as the base
    total_dataset = datasets[0]

    if len(datasets) == 1:
        return total_dataset

    # Collect DataFrames and numpy arrays efficiently
    all_dataframes = [dataset.metadata_df for dataset in datasets]
    all_dataset_names = [dataset._dataset_names for dataset in datasets]
    all_relative_paths = [dataset._relative_paths for dataset in datasets]
    all_frame_lengths = [dataset._frame_lengths for dataset in datasets]
    all_labels = [dataset._labels for dataset in datasets]

    # Concatenate DataFrames
    import pandas as pd
    import numpy as np

    concatenated_df = pd.concat(all_dataframes, ignore_index=True)

    # Concatenate numpy arrays efficiently
    concatenated_dataset_names = np.concatenate(all_dataset_names)
    concatenated_relative_paths = np.concatenate(all_relative_paths)
    concatenated_frame_lengths = np.concatenate(all_frame_lengths)
    concatenated_labels = np.concatenate(all_labels)

    # Update the total dataset with concatenated data
    total_dataset.metadata_df = concatenated_df
    total_dataset._dataset_names = concatenated_dataset_names
    total_dataset._relative_paths = concatenated_relative_paths
    total_dataset._frame_lengths = concatenated_frame_lengths
    total_dataset._labels = concatenated_labels

    return total_dataset


def getConcatDatasetVal(cfg, subset):
    datasets = []
    cfg_data_copy = copy.deepcopy(cfg.data)
    for data in cfg_data_copy.commonval_list:
        cfg.data = data
        # 保证mix.yaml中的最高控制优先级
        cfg.data.data_utils = cfg_data_copy.data_utils
        cfg.data.char_path = cfg_data_copy.char_path
        dataset = AVDataset(cfg, subset)
        datasets.append(dataset)
    cfg.data = cfg_data_copy
    return ConcatDatasetWithJIT(datasets)


def getConcatDataset(cfg, subset):
    datasets = []
    cfg_data_copy = copy.deepcopy(cfg.data)
    for data in cfg_data_copy.data_list:
        cfg.data = data
        # 保证mix.yaml中的最高控制优先级
        cfg.data.data_utils = cfg_data_copy.data_utils
        cfg.data.char_path = cfg_data_copy.char_path
        dataset = AVDataset(cfg, subset)
        datasets.append(dataset)
    cfg.data = cfg_data_copy
    return ConcatDatasetWithJIT(datasets)


def freeze_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d,
                               nn.BatchNorm2d,
                               nn.BatchNorm3d,
                               nn.LayerNorm
                               )):
            module.eval()


"""
根据指定策略加载预训练模型的部分或全部权重
"""


def load_weights_from_pt(model: nn.Module,
                         pt_file_path: str,
                         load_strategy="full",
                         exclude_prefixes: list = None):
    """
    :param model: 当前的模型实例
    :param pt_file_path: 预训练pt或者checkpoint的文件路径
    :param load_strategy: 加载策略a
        - ”full“直接全部加载
        - ”partial_exclude“: 排除指定的层前缀
    :param exclude_prefixes: 当load_strategy为p-e时，一个字符串列表
    """
    if not os.path.exists(pt_file_path):
        print(f"Warning:Checkpoint path {pt_file_path} does not exist!Skipping weight loading.")
        return

    print(f"Attempting to load weights from: {pt_file_path}")

    try:
        # 直接加载pt文件，state_dict
        pretrained_state_dict = torch.load(pt_file_path, map_location="cpu")
        # 如果加载的是一个完整的checkpoint字典（例如包含了'model'键），则提取模型状态
        if isinstance(pretrained_state_dict, dict) and 'model' in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict['model']
            print("Detected checkpoint format within PT file, extracting 'model' state_dict.")
        else:
            print("Detected raw state_dict format in PT file.")



    except Exception as e:
        print(f"Error loading PT file {pt_file_path}: {e}. Skipping weight loading.")
        return

    # 获取当前模型的 state_dict
    model_state_dict = model.state_dict()
    if load_strategy == "full":
        # 加载所有匹配的权重。
        # strict=False 允许在模型和预训练权重之间存在不匹配的键。
        print("Loading full model weights (strict=False).")
        model.load_state_dict(pretrained_state_dict, strict=False)
        print("Successfully loaded all matching weights from PT file.")

    elif load_strategy == "partial_exclude":
        if exclude_prefixes is None or not isinstance(exclude_prefixes, list):
            raise ValueError("When load_strategy is 'partial_exclude', 'exclude_prefixes' must be a list of strings.")

        new_state_dict = {}
        for k, v in pretrained_state_dict.items():
            should_exclude = False
            for prefix in exclude_prefixes:
                if k.startswith(prefix):
                    should_exclude = True
                    break
            if not should_exclude:
                # 只有当键在当前模型中存在且形状匹配时才加载，避免警告或错误
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    new_state_dict[k] = v
                else:
                    print(f"Skipping key '{k}' from PT file: not found in current model or shape mismatch.")
            else:
                print(f"Excluded key: '{k}' (matches prefix in {exclude_prefixes})")

        # 使用过滤后的 state_dict 加载，strict=False 确保未加载的层保持不变
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded weights from PT file, excluding prefixes: {exclude_prefixes}")

    else:
        raise ValueError(f"Unknown load_strategy: {load_strategy}. Supported: 'full', 'partial_exclude'.")


def set_train(model, strategy):
    model.train()
    if strategy is not None and getattr(strategy, "name", None) == "lora":
        freeze_norm(model)


def run_lora_recalibration(args, cfg=None, model=None, accelerator=None):
    # === step1: 初始化 cfg 和模型 ===
    if cfg is None:
        cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model, root_dir=args.root_dir)
        cfg = ini_cfg(args, cfg)

    if model is None:
        model = Model(cfg)

    # === step2: 加载 LoRA 权重 ===
    if cfg.train.lora:
        lora_oarams = cfg.model.LoRA
        for lora_oaram in lora_oarams:
            from ldw_nets.lora.inject_lora_to_model import inject_lora_to_model
            inject_lora_to_model(model.model[lora_oaram[0]], lora_oaram)
    # 冻结全部参数
    for param in model.parameters():
        param.requires_grad = False

    # === step3: 构建训练集 DataLoader，用于 BN recalibration ===
    if hasattr(cfg.data, "data_list"):
        dataset_train = getConcatDataset(cfg, "train")
    else:
        dataset_train = AVDataset(cfg, "train")

    train_sampler = ByFrameCountSampler(dataset_train, cfg.data.data_utils.max_frames)
    train_sampler = RandomSamplerWrapper(train_sampler)
    train_loader = DataLoader(
        dataset_train,
        batch_sampler=train_sampler,
        num_workers=cfg.train.workers,
        pin_memory=True,
        collate_fn=collate_pad,
        persistent_workers=True
    )

    # === step4: Accelerator 准备 ===
    if accelerator is None:
        accelerator = Accelerator()
    model, train_loader = accelerator.prepare(model, train_loader)

    # === step5: BN recalibration ===
    model.train()  # 保持 train 模式以更新 BN
    print("开始 LoRA BN recalibration...")
    with torch.no_grad():  # 不计算梯度
        for index, batch in enumerate(train_loader):
            model(batch)  # forward 仅更新 BN running_mean / running_var

    # === step6: 保存 recalibrated 模型 ===
    save_path = os.path.join(cfg.default.root_dir, cfg.train.run_exp_dir, f"{cfg.train.model_name}_lora_recalibrated.pt")
    model_to_save = accelerator.unwrap_model(model)
    torch.save({'model': model_to_save.state_dict()}, save_path)
    print(f"LoRA recalibration 完成，模型已保存到: {save_path}")


def run_train(args, cfg=None, parallel=True, model=None, stop_event=None, accelerator=None):
    # === 打印所有传入参数 ===
    print("\n" + "=" * 80)
    print("进入 run_train 函数，参数信息:")
    print("-" * 80)
    print(f"{'参数名':<20} | {'值':<40} | {'类型':<15}")
    print("-" * 80)

    # 打印函数参数
    print(
        f"{'args':<20} | {str(vars(args))[:60] + '...' if len(str(vars(args))) > 60 else str(vars(args)):<40} | {type(args).__name__}")
    print(
        f"{'cfg':<20} | {str(cfg)[:60] + '...' if cfg and len(str(cfg)) > 60 else str(cfg):<40} | {type(cfg).__name__ if cfg else 'NoneType'}")
    print(f"{'parallel':<20} | {parallel:<40} | {type(parallel).__name__}")
    print(
        f"{'model':<20} | {str(model)[:60] + '...' if model and len(str(model)) > 60 else str(model):<40} | {type(model).__name__ if model else 'NoneType'}")
    print(f"{'stop_event':<20} | {str(stop_event):<40} | {type(stop_event).__name__ if stop_event else 'NoneType'}")
    print(
        f"{'accelerator':<20} | {str(accelerator)[:60] + '...' if accelerator and len(str(accelerator)) > 60 else str(accelerator):<40} | {type(accelerator).__name__ if accelerator else 'NoneType'}")

    # === step1: 配置处理 ===
    if cfg is None:
        print("\n" + "=" * 80)
        print("开始解析配置文件")
        print(f"数据配置文件: {args.data}")
        print(f"模型配置文件: {args.model}")
        print(f"根目录: {args.root_dir}")

        cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model, root_dir=args.root_dir)
        cfg = ini_cfg(args, cfg)

        print("\n" + "=" * 80)
        print("解析后的配置信息 (cfg):")

    if parallel is False:
        torch.cuda.set_device(cfg.train.device[0])

    if accelerator is None:
        accelerator = Accelerator()

    accelerator.even_batches = False

    # 进行输出检查：
    print(f"设置的学习率是{cfg.train.lr}")
    print(f"设置的warmup是{cfg.train.warmup_epochs}")
    print(f"设置的max_epochs是{cfg.train.max_epochs}")

    # === step2: 初始化日志路径和Tensorboard ===
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)

    log_dir = os.path.join(
        cfg.default.root_dir,
        cfg.train.run_exp_dir,
        # "train--" + formatted_time,
        "train--" + cfg.train.model_name,
        'log'
    )
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=log_dir)
        print(tb_writer.get_logdir())
        # cfg.save(os.path.join("/".join(log_dir.split("/")[:-1]),"config.json"))
        cfg.save(os.path.join(log_dir.replace('log', ''),"config.json"))

        # ====== 新增：将配置信息添加到TensorBoard ======
        # 将配置转换为Markdown格式的字符串
        def cfg_to_markdown(cfg_obj, indent=0):
            """递归将配置对象转换为Markdown字符串"""
            indent_str = "  " * indent
            result = ""
            for key, value in cfg_obj.__dict__.items():
                if key.startswith("__"):  # 跳过内部属性
                    continue
                if hasattr(value, "__dict__"):  # 嵌套对象
                    result += f"{indent_str}- **{key}**:\n{cfg_to_markdown(value, indent + 1)}"
                elif isinstance(value, list):  # 列表类型
                    list_items = "\n".join([f"{indent_str}  - {item}" for item in value])
                    result += f"{indent_str}- **{key}**:\n{list_items}\n"
                else:  # 基本类型
                    result += f"{indent_str}- **{key}**: `{value}`\n"
            return result

        # 创建Markdown格式的配置信息
        config_md = f"## 训练配置信息\n{cfg_to_markdown(cfg)}"

        # 添加到TensorBoard
        tb_writer.add_text("Configuration", config_md, 0)
        # ====== 新增内容结束 ======

    # === step3: 初始化模型结构 ===
    if model is None:
        model = Model(cfg)

    # === step4: 加载模型权重：优先checkpoint，其次pt文件 ===
    resume_checkpoint = None
    start_epoch = 0
    if hasattr(cfg.train, "resume_from") and cfg.train.resume_from is not None:
        if os.path.exists(cfg.train.resume_from):
            resume_checkpoint = torch.load(cfg.train.resume_from, map_location="cpu")
            model.load_state_dict(resume_checkpoint["model"])
            start_epoch = resume_checkpoint["epoch"] + 1
            print(f"Resume from checkpoint at epoch {start_epoch}")
        else:
            print(f"Warning: Specified resume path does not exist: {cfg.train.resume_from}")
    elif cfg.default.weights:
        model.load(cfg.train.freeze)
        print(f"Loading initial weights from PT file: {cfg.default.weights}")
    else:
        if accelerator.is_main_process:
            print(f"no weight!!")

    # === step5: 附加结构Lora ===
    if cfg.train.lora:
        lora_oarams = cfg.model.LoRA
        # [ 3, LoRALinear, [ 16, 64 ], [ "self_attn" ]]
        for lora_oaram in lora_oarams:
            from ldw_nets.lora.inject_lora_to_model import inject_lora_to_model, get_submodule_by_name
            inject_lora_to_model(model.model[lora_oaram[0]], lora_oaram)  # get_submodule_by_name(model, lora_oaram[0])
        # 冻结其它权重
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        # 冻结BN
        # for m in model.modules():
        #     if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        #         m.eval()

    # # 冻结全模型
    # for param in model.parameters():
    #     param.requires_grad = False
    # # 遍历 ConformerLayer
    # conformer_seq = model.model[3]  # 你的 MultiSequential
    # for layer in conformer_seq:
    #     # LayerNorm bias 解冻
    #     for ln_name in ["norm_ff", "norm_mha", "norm_conv", "norm_final", "norm_ff_macaron"]:
    #         ln = getattr(layer, ln_name, None)
    #         if ln is not None and ln.bias is not None:
    #             ln.bias.requires_grad = True
    #     # ConvModule pointwise_conv bias 解冻
    #     conv = getattr(layer, "conv_module", None)
    #     if conv is not None:
    #         for conv_name in ["pointwise_cov1", "pointwise_cov2"]:
    #             pw = getattr(conv, conv_name, None)
    #             if pw is not None and pw.bias is not None:
    #                 pw.bias.requires_grad = True
    # print("解冻完成：只训练 LayerNorm.bias 和 ConvModule pointwise_conv.bias")


    # === step6: 构建Dataset和DataLoader ===
    if hasattr(cfg.data, "data_list"):
        dataset_train = getConcatDataset(cfg, "train")
    else:
        dataset_train = AVDataset(cfg, "train")
    enroll_identity(model, dataset_train)

    train_sampler = ByFrameCountSampler(
        dataset_train,
        cfg.data.data_utils.max_frames
    )
    train_sampler = RandomSamplerWrapper(train_sampler)
    train_loader = DataLoader(
        dataset_train,
        num_workers=cfg.train.workers,
        pin_memory=True,
        batch_sampler=train_sampler,
        # batch_size=8,
        collate_fn=collate_pad,
        persistent_workers=True
    )

    # 同样逻辑处理val_dataset
    """添加公共验证集逻辑***********************************************"""
    if hasattr(cfg.data, "commonval_list"):
        dataset_val = getConcatDatasetVal(cfg, "val")
    elif hasattr(cfg.data, "data_list"):
        dataset_val = getConcatDataset(cfg, "val")
    else:
        dataset_val = AVDataset(cfg, "val")
    """*************************************************************"""

    val_sampler = ByFrameCountSampler(
        dataset_val,
        cfg.data.data_utils.max_frames_val,
        shuffle=False
    )

    val_loader = DataLoader(
        dataset_val,
        num_workers=cfg.train.workers,
        pin_memory=True,
        batch_sampler=val_sampler,
        # batch_size=8,
        collate_fn=collate_pad,
    )

    # === step7: 初始化 optimizer 和 scheduler ===
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"name": "model", "params": params, "lr": cfg.train.lr}],
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.98)
    )
    scheduler = WarmupCosineScheduler(optimizer, cfg.train.warmup_epochs, cfg.train.max_epochs,
                                      len(train_loader))

    # === Step 8. 若resume则加载 optimizer 和 scheduler 状态 ===
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])

    # === Step 9. accelerator.prepare ===
    # 使用 Accelerator 准备模型、优化器、数据加载器
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    for epoch in range(start_epoch, cfg.train.max_epochs):  # 修改起始epoch
        # for epoch in range(cfg.train.max_epochs):
        # train -----------------------------------------------------
        # 如果是LoRA，需要保证其冻结Norm
        cfg.default.mode = "train"
        set_train(model, cfg.strategy)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        model.set_epoch(epoch)

        train_num = len(train_loader)
        if accelerator.is_main_process:
            train_loader = tqdm(train_loader, total=train_num, desc=f"train epoch[{epoch}/{cfg.train.max_epochs}]",
                                ncols=150)
        for index, data in enumerate(train_loader, 0):
            # 使用enumerate 返回下标和值 追踪
            #  处理
            if "video_lengths" in data and "audio_lengths" in data:
                # 找出最长视频和音频
                max_video_len = max(data["video_lengths"])
                max_audio_len = max(data["audio_lengths"])

                # 计算最大保留数量（按视频/音频都考虑）
                max_video_num = int(cfg.data.data_utils.max_frames / max_video_len)
                max_audio_num = int(
                    cfg.data.data_utils.max_frames / 25 * cfg.data.data_utils.audio_ratio / max_audio_len)

                # 取两者最小值，保证同步截取
                max_num = min(max_video_num, max_audio_num)

                # 找出最长视频和最长音频的索引
                max_video_index = next(id for id, l in enumerate(data["video_lengths"]) if l == max_video_len)
                max_audio_index = next(id for id, l in enumerate(data["audio_lengths"]) if l == max_audio_len)

                # 如果最长视频或音频会被截掉，则提前到0号位（同步移动）
                if max_video_index >= max_num or max_audio_index >= max_num:
                    print("拯救最大视频和音频")
                    # 选择最长的视频/音频索引（可以优先视频或音频）
                    preserve_index = max_video_index if max_video_index >= max_audio_index else max_audio_index

                    data["videos"][0] = data["videos"][preserve_index, :, :, :, :]
                    data["video_lengths"][0] = data["video_lengths"][preserve_index]
                    data["audios"][0] = data["audios"][preserve_index, :, :]
                    data["audio_lengths"][0] = data["audio_lengths"][preserve_index]
                    data["targets"][0] = data["targets"][preserve_index, :, :]
                    data["target_lengths"][0] = data["target_lengths"][preserve_index]

                # 截断同步数据
                data["videos"] = data["videos"][:max_num, :, :, :, :]
                data["video_lengths"] = data["video_lengths"][:max_num]
                data["audios"] = data["audios"][:max_num, :, :]
                data["audio_lengths"] = data["audio_lengths"][:max_num]
                data["targets"] = data["targets"][:max_num, :, :]
                data["target_lengths"] = data["target_lengths"][:max_num]
            elif "video_lengths" in data:
                max_len = max(data["video_lengths"])
                max_num = int(cfg.data.data_utils.max_frames / max_len)

                # print("ss:", data["video_lengths"])
                max_index = 0
                for id, leni in enumerate(data["video_lengths"]):
                    if leni == max_len:
                        max_index = id
                        break
                if max_index >= max_num:  # 如果最大视频即将被删，将其提前到0号位
                    print("拯救最大视频")
                    data["videos"][0] = data["videos"][max_index, :, :, :, :]
                    data["video_lengths"][0] = data["video_lengths"][max_index]
                    data["targets"][0] = data["targets"][max_index, :, :]
                    data["target_lengths"][0] = data["target_lengths"][max_index]

                data["videos"] = data["videos"][:max_num, :, :, :, :]
                data["video_lengths"] = data["video_lengths"][:max_num]
                data["targets"] = data["targets"][:max_num, :, :]
                data["target_lengths"] = data["target_lengths"][:max_num]
            elif "audio_lengths" in data:
                max_len = max(data["audio_lengths"])
                max_num = int(cfg.data.data_utils.max_frames /25 * cfg.data.data_utils.audio_ratio / max_len)

                # print("ss:", data["video_lengths"])
                max_index = 0
                for id, leni in enumerate(data["audio_lengths"]):
                    if leni == max_len:
                        max_index = id
                        break
                if max_index >= max_num:  # 如果最大音频即将被删，将其提前到0号位
                    print("拯救最大音频")
                    data["audios"][0] = data["audios"][max_index, :, :]
                    data["audio_lengths"][0] = data["audio_lengths"][max_index]
                    data["targets"][0] = data["targets"][max_index, :, :]
                    data["target_lengths"][0] = data["target_lengths"][max_index]

                data["audios"] = data["audios"][:max_num, :, :]
                data["audio_lengths"] = data["audio_lengths"][:max_num]
                data["targets"] = data["targets"][:max_num, :, :]
                data["target_lengths"] = data["target_lengths"][:max_num]

            # print("tt:", data["video_lengths"])

            # # 对 video 重新 pad，保证时序维度一致（否则 mask 会错）
            # device = data["videos"].device if isinstance(data["videos"], torch.Tensor) else torch.device("cpu")
            # videos = [v[:l] for v, l in zip(data["videos"], data["video_lengths"])]  # 裁剪有效帧
            # padded_videos = pad_sequence(videos, batch_first=True)  # [B, T, C, H, W]
            # data["videos"] = padded_videos.to(device)
            # data["video_lengths"] = torch.tensor([v.shape[0] for v in videos], device=device)
            # # --------------------------------------------------------------------------------------------

            # if isinstance(data, dict):
            #     for key in data:
            #         data[key] = accelerator.prepare(data[key])
            # else:
            #     data = accelerator.prepare(data)
            if int(os.environ.get("EAAI_STAGE1_EPOCHS", "0") or "0") > 0:
                raw_model = accelerator.unwrap_model(model)
                for module in raw_model.modules():
                    if module.__class__.__name__ == "LoRAMoEPaper":
                        module._slot = index
            ## 前馈计算
            losses = model(data)
            loss = losses["loss_total"]
            ## 反向传播，更新参数
            optimizer.zero_grad()
            accelerator.backward(loss)
            # 梯度裁剪
            accelerator.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_val)
            optimizer.step()
            scheduler.step()
            # # save to tensorboard per batch------------------------------
            # save train loss
            loss = loss.item()

            if accelerator.is_main_process:
                show_data = dict()
                tb_writer.add_scalar("train_loss", loss, index + epoch * train_num)
                show_data["loss"] = f"{loss:.4f}"
                if "loss_ctc" in losses:
                    train_loss_ctc = losses["loss_ctc"].item()
                    show_data["loss_ctc"] = f"{train_loss_ctc:.4f}"
                    tb_writer.add_scalar("train_loss_ctc", train_loss_ctc, index + epoch * train_num)

                if "loss_att" in losses:
                    train_loss_att = losses["loss_att"].item()
                    show_data["loss_att"] = f"{train_loss_att:.4f}"
                    tb_writer.add_scalar("train_loss_att", train_loss_att, index + epoch * train_num)

                if "loss_id" in losses:
                    train_loss_id = losses["loss_id"].item()
                    show_data["loss_id"] = f"{train_loss_id:.4f}"
                    tb_writer.add_scalar("train_loss_id", train_loss_id, index + epoch * train_num)

                if "loss_corr" in losses:
                    train_loss_corr = losses["loss_corr"].item()
                    show_data["loss_corr"] = f"{train_loss_corr:.4f}"
                    tb_writer.add_scalar("train_loss_corr", train_loss_corr, index + epoch * train_num)

                # save lr
                current_lr = optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar("learning_rate", current_lr, index + epoch * train_num)

                # 更新 tqdm 的 postfix（注意：这不是实时更新的，会在下一次迭代开始时显示）
                train_loader.set_postfix(**show_data)  # , acc=f"{acc:.4f}")

        # val -----------------------------------------------------
        """val_cer计算初始化***********************************************"""
        if epoch % cfg.train.save_every_epoch != 0:  # or epoch == 0:
            continue
        cfg.default.mode = "val"
        model.eval()
        loss_val = {"val_loss": 0, "val_loss_ctc": 0, "val_loss_att": 0}
        total_samples=0
        # CSV临时文件路径，每个进程写不同文件
        rank = accelerator.process_index
        val_cer_csv = os.path.join(log_dir, f"val_cer_epoch{epoch}_rank{rank}.csv")
        val_loss_csv = os.path.join(log_dir, f"val_loss_epoch{epoch}_rank{rank}.csv")

        # 打开 CSV 文件准备写入
        with open(val_cer_csv, "w", newline='') as f_cer, \
            open(val_loss_csv, "w", newline='') as f_loss:

            cer_writer = csv.writer(f_cer)
            cer_writer.writerow(["distance", "length"])
            loss_writer = csv.writer(f_loss)
            loss_writer.writerow(["val_loss", "val_loss_ctc", "val_loss_att","num_samples"])

            with torch.no_grad():
                val_num = len(val_loader)
                if accelerator.is_main_process:
                    val_loader = tqdm(val_loader, total=val_num, desc=f"val epoch[{epoch}/{cfg.train.max_epochs}]",
                                      ncols=150)
                for index, data in enumerate(val_loader, 0):  # 使用enumerate 返回下标和值 追踪
                    # if isinstance(data, dict):
                    #     for key in data:
                    #         data[key] = accelerator.prepare(data[key])
                    # else:
                    #     data = accelerator.prepare(data)

                    # 前馈计算
                    losses_val, preds, actuals = model(data)
                    sampler_num = data["targets"].size(0)
                    total_samples += sampler_num
                    loss_val["val_loss"] += losses_val["loss_total"].item() * sampler_num
                    if "val_loss_ctc" in losses_val:
                        loss_val["val_loss_ctc"] += losses_val["loss_ctc"].item() * sampler_num
                    if "val_loss_att" in losses_val:
                        loss_val["val_loss_att"] += losses_val["loss_att"].item() * sampler_num
                    loss_writer.writerow([
                        losses_val["loss_total"].item(),
                        losses_val["loss_ctc"].item(),
                        losses_val["loss_att"].item(),
                        sampler_num
                    ])
                    # 写入每个样本 distance / length 到 CSV
                    for i in range(len(preds)):
                        language = cfg.data.char_path.split('dictionary/')[1].split('/')[0]
                        if language == 'en':
                            distance_per = compute_WordorChar_level_distance(actuals[i], preds[i], 'en')
                            length_per = len(actuals[i].split())
                        elif language == 'zh-cn':
                            distance_per = compute_WordorChar_level_distance(actuals[i], preds[i], 'zh')
                            length_per = len(list(actuals[i].replace('\n', '')))
                        cer_writer.writerow([distance_per, length_per])

        # 主进程读取所有 CSV 聚合
        if accelerator.is_main_process:
            import glob

            # 聚合 CER
            dis_total, len_total = 0, 0
            for csv_file in glob.glob(os.path.join(tb_writer.get_logdir(), f"val_cer_epoch{epoch}_rank*.csv")):
                with open(csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dis_total += float(row["distance"])
                        len_total += float(row["length"])
            cer_total = dis_total / len_total

            # 聚合 Loss
            val_loss, val_loss_ctc, val_loss_att, total_samples = 0, 0, 0, 0
            for csv_file in glob.glob(os.path.join(tb_writer.get_logdir(), f"val_loss_epoch{epoch}_rank*.csv")):
                with open(csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        n = int(row["num_samples"])
                        val_loss += float(row["val_loss"]) * n
                        val_loss_ctc += float(row["val_loss_ctc"]) * n
                        val_loss_att += float(row["val_loss_att"]) * n
                        total_samples += n

            val_loss /= total_samples
            val_loss_ctc /= total_samples
            val_loss_att /= total_samples

            # 写入 TensorBoard
            tb_writer.add_scalar("val_cer", cer_total, epoch)
            tb_writer.add_scalar("val_loss", val_loss, epoch)
            tb_writer.add_scalar("val_loss_ctc", val_loss_ctc, epoch)
            tb_writer.add_scalar("val_loss_att", val_loss_att, epoch)
            # 更新 tqdm postfix
            val_loader.set_postfix(
                val_loss=f"{val_loss:.4f}",
                val_loss_ctc=f"{val_loss_ctc:.4f}",
                val_loss_att=f"{val_loss_att:.4f}",
                val_cer=f"{cer_total:.4f}"
            )

            print(f"[Epoch {epoch}] Global CER = {cer_total:.4f}")
            path = tb_writer.get_logdir()

            # ！！！！！！！！！！！！！！！！！！！！！！！！保存新的checkpoint
            # save_path = os.path.join(path.replace('log', ''), f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint_path = os.path.join(path.replace('log', ''), f"model_last.pt")
            torch.save({
                'model': accelerator.get_state_dict(model),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }, save_checkpoint_path)
            # ！！！！！！！！！！！！！！！

            model_to_save = accelerator.unwrap_model(model)
            model_to_save.save(path.replace('log', ''), epoch, strategy=cfg.strategy, lora=cfg.train.lora)

    if accelerator.is_main_process:
        tb_writer.close()
