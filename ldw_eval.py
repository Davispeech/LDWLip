import ast
import logging
import os.path
import time

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


def run_eval(args):
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model)
    cfg = ini_cfg(args, cfg)

    outfile = cfg.eval.outfile

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

    # eval -----------------------------------------------------
    with open(outfile, 'w', encoding="utf-8") as f:
        f.write("Let's Start the eval!!!\n")
        for attr_name, attr_value in vars(args).items():
            f.write(f'{attr_name}: {attr_value}\n')
        f.write("---------------------------分割线-----------------------------\n")
        f.close()

    if cfg.default.task in ['vsr', 'asr', 'avsr']:
        distance_per = 0
        length_per = 0
        dis_total = 0
        len_total = 0
    elif cfg.default.task in ['classifier']:
        right = 0
        wrong = 0

    model.eval()
    with torch.no_grad():
        val_num = len(eval_loader)
        eval_loader = tqdm(eval_loader, total=val_num, desc=f"eval data: ", ncols=150)
        for index, data in enumerate(eval_loader, 0):  # 使用enumerate 返回下标和值 追踪
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda(device[0])
            else:
                data = data.cuda(device[0])
            ## 前馈计算
            # start = time.time()
            predict, actual = model.predict(data)
            # print("Time model.predict", time.time() - start)

            if cfg.default.task in ['vsr', 'asr', 'avsr']:
                if cfg.data.language == 'en':
                    distance_per = compute_WordorChar_level_distance(actual, predict, 'en')
                    length_per = len(actual.split())
                elif cfg.data.language == 'zh':
                    distance_per = compute_WordorChar_level_distance(actual, predict, 'zh')
                    length_per = len(list(actual.replace('\n', '')))

                dis_total += distance_per
                len_total += length_per
                cer_per = distance_per/length_per
                cer_total = dis_total/len_total
                # 更新 tqdm 的 postfix（注意：这不是实时更新的，会在下一次迭代开始时显示）
                eval_loader.set_postfix(cer_per=f"{cer_per:.4f}", cer_total=f"{cer_total:.4f}")
                print("actual: " + str(actual) + "   predict: " + str(predict.replace(' ', '')))
                with open(outfile, 'a', encoding=get_filetype(outfile)) as f:
                    f.write("path: " + str(data['path']) + "   actual: " + str(actual) + "   predict: " + str(predict.replace(' ', '')) + "   cer_per: " + str(cer_per) +"   cer_total: " + str(cer_total) + "\n")
                f.close()
            else:
                predict, score = predict
                if predict == 'None' or predict == None:
                    predict = 0
                if str(predict)==str(actual):
                    right = right+1
                else:
                    wrong = wrong+1
                acc = right/(right+wrong)
                eval_loader.set_postfix(Accuracy=f"{acc:.4f}", num=f"{right+wrong}")
                with open(outfile, 'a', encoding=get_filetype(outfile)) as f:
                    f.write("actual: " + str(actual) + "   predict: " + str(predict) + "   score: " + str(score) + "   |  ACC: "+str(acc)+ "\n")
                f.close()
    if cfg.default.task in ['vsr', 'asr', 'avsr']:
        results_analysis(args.root_dir, args.data, outfile)
