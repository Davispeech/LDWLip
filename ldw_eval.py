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


def eval_lipAuth(args):
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model)
    cfg = ini_cfg(args, cfg)
    outfile = cfg.eval.outfile
    with open(outfile, 'w', encoding="utf-8") as f:
        f.write("Let's start to LipAuth:\n")

    from ldw_nets.nn import LipAuthOutput
    LipAuth = LipAuthOutput(
        cfg.default.root_dir,
        cfg.model.Output.output_param.support,
        cfg.model.Output.output_param.ids_user,
        cfg.model.Output.output_param.save_layerFeatures
    )
    is_vsr = 0
    for fea in cfg.model.Output.output_param.save_layerFeatures:
        if fea[1] in ['ce', 'ctc', 'att']:
            is_vsr = 1
        elif fea[1] in ['cer']:
            is_vsr = 2
    if is_vsr > 0:
        # vsr_path = (cfg.default.model).split("/")[-1].replace(".yaml", "")
        # vsr_path = cfg.default.root_dir + "/output/" + vsr_path + "/" + vsr_path + ".txt"
        vsr_path = cfg.default.root_dir + "/" + cfg.model.Output.output_param.vsr_res
        lines = open(vsr_path, "r", encoding="utf-8").readlines()
        cers = {}
        for line in tqdm(lines):
            if "actual: " in line:
                # path: F:/Datasets/lipreading/icslr\icslr_video_seg24s/main/PID0001/0001.mp4   actual: 啊呀   predict: 啊 呀    cer_per: 0.0   cer_total: 0.0
                # path: F:/Datasets/lipreading/grid\grid_video_seg24s/s22/mpg_6000/swwt8a.mp4   actual: S E T <space> W H I T E <space> W I T H <space> T <space> 8 <space> A G A I N   predict: S E T <space> W H I T E <space> W I T H <space> T <space> 8 <space> A G A I N    cer_per: 0.0   cer_total: 0.08968589325180071
                video_path = line.split('   ')[0].split('_video_seg24s')[-1]
                if "icslr" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-2].replace("PID", "")))
                elif "grid" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-3].replace("s", "")))
                video_number = video_path.split('/')[-1].replace(".mp4", "")
                cer_per = float(line.split('   ')[3].split(": ")[-1])
                cers[PID+"_"+video_number] = cer_per
    if is_vsr == 2:
        # vsr_path = (cfg.default.model).split("/")[-1].replace(".yaml", "")
        # vsr_path = cfg.default.root_dir + "/output/" + vsr_path + "/" + vsr_path + ".txt"
        vsr_path2 = cfg.default.root_dir + "/" + cfg.model.Output.output_param.vsr_res2
        lines2 = open(vsr_path2, "r", encoding="utf-8").readlines()
        cers2 = {}
        for line in tqdm(lines2):
            if "actual: " in line:
                # path: F:/Datasets/lipreading/icslr\icslr_video_seg24s/main/PID0001/0001.mp4   actual: 啊呀   predict: 啊 呀    cer_per: 0.0   cer_total: 0.0
                # path: F:/Datasets/lipreading/grid\grid_video_seg24s/s22/mpg_6000/swwt8a.mp4   actual: S E T <space> W H I T E <space> W I T H <space> T <space> 8 <space> A G A I N   predict: S E T <space> W H I T E <space> W I T H <space> T <space> 8 <space> A G A I N    cer_per: 0.0   cer_total: 0.08968589325180071
                video_path = line.split('   ')[0].split('_video_seg24s')[-1]
                if "icslr" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-2].replace("PID", "")))
                elif "grid" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-3].replace("s", "")))
                video_number = video_path.split('/')[-1].replace(".mp4", "")
                cer_per = float(line.split('   ')[3].split(": ")[-1])
                cers2[PID+"_"+video_number] = cer_per

    # 获取pairs列表
    pairfile_path = cfg.default.root_dir + '/' + args.lipauthEval_data
    with open(pairfile_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    # model_name = cfg.model.Output.output_param.support.split("/")[-2]
    # rep = cfg.model.Output.output_param.support.split("/")[-1].split("_")[0]
    all_sim_scores = []
    for line in tqdm(lines):
        # 'Pos 19_0302.mp4 19_0546.mp4
        label, support, query = line.split(' ')
        if is_vsr==1:
            if '_light' in query:
                query_cer_index = query.split(".")[0].replace('_light', '')
            elif '_DF' in query:
                query_cer_index = query.split('_')[0]+'_'+query.split('_')[4]
            else:
                query_cer_index = query.split(".")[0]
            cer = cers[query_cer_index]
            sim_scores = LipAuth.authentication(label, support, query, cer)
        elif is_vsr==2:
            if '_light' in query:
                query_cer_index = query.split(".")[0].replace('_light', '')
            elif '_DF' in query:
                query_cer_index = query.split('_')[0]+'_'+query.split('_')[4]
            else:
                query_cer_index = query.split(".")[0]
            cer = [cers[query_cer_index], cers2[query_cer_index]]
            sim_scores = LipAuth.authentication(label, support, query, cer)
        else:
            sim_scores = LipAuth.authentication(label, support, query)
        all_sim_scores.append(sim_scores)
        with open(outfile, 'a', encoding="utf-8") as f:
            f.write(line+"  scores: "+str(sim_scores))
            f.write("\n")
        # if len(all_sim_scores)>2000:
        #     break

    from ldw_utils.eval_matrix import plot_lipauth_threshold_hist, plot_lipauth_err, cal_lipauth_matrics
    savefig_path = outfile.replace(outfile.split('/')[-1], "")
    plot_lipauth_threshold_hist(all_sim_scores, save_dir=savefig_path)
    keys_err = plot_lipauth_err(all_sim_scores, save_dir=savefig_path)
    key_thresholds = {}
    for key, _, threshold in cfg.model.Output.output_param.save_layerFeatures:
        key_thresholds[key] = float(threshold)
    opt_thresholds, confusion_metrics, key_metrics, Total_confusion_metrics, Total_metrics = cal_lipauth_matrics(all_sim_scores, key_thresholds)
    with open(outfile, 'a', encoding="utf-8") as f:
        f.write("\n\n\nopt_thresholds: " + str(opt_thresholds))
        f.write("\nconfusion_metrics: " + str(confusion_metrics))
        f.write("\nkey_metrics: " + str(key_metrics))
        f.write("\n\nkey_thresholds: " + str(key_thresholds))
        f.write("\nTotal_confusion_metrics: " + str(Total_confusion_metrics))
        f.write("\nTotal_metrics: " + str(Total_metrics))
        f.write("\n")


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
