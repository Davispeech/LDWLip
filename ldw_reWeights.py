import ast
import os
import re

import torch
from collections import OrderedDict
from ldw_predict import mkPath, mkModelDir


def log_model_state(pth_file_path, layer_depth=0):
    if not pth_file_path:
        pth_file_path = "run_exp/preweights/CMLR_V_WER8.0/model.pth"
    if layer_depth == 0:
        layer_depth = 100
    # 使用 torch.load 加载 .pth 文件
    loaded_data = torch.load(pth_file_path, map_location=torch.device('cpu'))

    # 检查加载的数据是否是一个字典
    if not isinstance(loaded_data, dict):
        return print("加载的数据不是一个字典!!!!!!!!!!!")
    if not isinstance(loaded_data, OrderedDict):
        isFlag = False
        for key in loaded_data:
            if 'state_dict' in key:
                isFlag = key
        loaded_data = loaded_data[isFlag] if isFlag else loaded_data

    name = ["layers", "name", "parameter", "FLOPs"]
    print(f"{name[0]:>10}：{name[1]:^70}{name[2]:^10}{name[3]:^15}")
    # 按顺序打印键值对
    num_params = 0
    num_param = 0
    is_next = False
    for i, (key, value) in enumerate(loaded_data.items(), start=0):
        now = key.split('.')
        if i > 0:
            num = layer_depth if layer_depth<len(now) else len(now)
            for j in range(num):
                if j >= len(last):
                    break
                if now[j] != last[j]:
                    is_next = True
                    break
        if is_next:
            num = layer_depth if layer_depth < len(last) else len(last)
            lastkey = '.'.join(last[:num])
            print(f"{i:>10}：{lastkey:<70}{num_param:>10}")
            num_params += num_param
            num_param = 0
            is_next = False
        num_param = num_param + value.numel() * 4  # float32通常占用4字节
        last = key.split('.')
    i = i + 1
    num = layer_depth if layer_depth < len(last) else len(last)
    lastkey = '.'.join(last[:num])
    print(f"{i:>10}：{lastkey:<70}{num_param:>10}")
    num_params += num_param
    print(f"total parameter: {num_params / (1024 * 1024): .2f} MB")


def export_model_state(pth_file_path, export_name, args=''):
    if not args:
        return print("need to set export layers!!!!")
    save_path = os.getcwd()+'/run_exp/export'
    mkPath(os.getcwd(), save_path)
    export_name = mkModelDir(save_path, export_name)
    save_path = save_path +'/'+ export_name+'.pth'
    if isinstance(args, str):
        args = ast.literal_eval(args)
    if isinstance(args, (int, tuple)):
        args = [args]
    if not pth_file_path:
        pth_file_path = "run_exp/preweights/CMLR_V_WER8.0/model.pth"
    # 使用 torch.load 加载 .pth 文件
    loaded_data = torch.load(pth_file_path, map_location=torch.device('cpu'))
    # 检查加载的数据是否是一个字典
    if isinstance(loaded_data, dict):
        for key in loaded_data:
            if 'state_dict' in key:
                loaded_data = loaded_data[key]
                break
    else:
        return print("加载的数据不是一个字典。")

    extracted_weights = OrderedDict()
    for i, arg in enumerate(args, start=0):
        for j, (key, value) in enumerate(loaded_data.items(), start=0):
            if isinstance(arg, int):
                if arg == j:
                    extracted_weights[key] = value
            elif isinstance(arg, tuple):
                if arg[0] <= j < arg[1]:
                    extracted_weights[key] = value
            else:
                return print("提取权重参数设置错误！！！")
    torch.save(extracted_weights, save_path)
    log_model_state(save_path, layer_depth=0)
    print(f"extracted weights finished!!!")


def re_create_weights(pth_file_path, export_name, reWeights_name, args=''):
    if not args:
        return print("need to set export layers!!!!")
    args = ast.literal_eval(args)

    # 使用 torch.load 加载 .pth 文件
    if not pth_file_path:
        pth_file_path = "run_exp/preweights/CMLR_V_WER8.0/model.pth"
    loaded_data = torch.load(pth_file_path, map_location=torch.device('cpu'))
    # 检查加载的数据是否是一个字典
    if not isinstance(loaded_data, dict):
        return print("加载的数据不是一个字典。")
    search_ids = []
    start = -1
    start_key = ''
    for i, (key, value) in enumerate(loaded_data.items(), start=0):
        now_key = key.split('.')
        if start==-1:
            if reWeights_name in now_key:
                start = i
                pattern = r'^-?\d+$'
                s = now_key[now_key.index(reWeights_name) + 1]
                if bool(re.match(pattern, s)):
                    start_key = now_key[:now_key.index(reWeights_name) + 3]
                else:
                    start_key = now_key[:now_key.index(reWeights_name) + 2]
        else:
            if start_key != now_key[:len(start_key)]:
                end = i
                search_ids.append((start, end))
                start = -1
                start_key = ''
                if reWeights_name in now_key:
                    start = i
                    pattern = r'^-?\d+$'
                    s = now_key[now_key.index(reWeights_name) + 1]
                    if bool(re.match(pattern, s)):
                        start_key = now_key[:now_key.index(reWeights_name) + 3]
                    else:
                        start_key = now_key[:now_key.index(reWeights_name) + 2]
    print(search_ids)
    length = len(loaded_data.items())
    over_ids = []
    for ids in search_ids:
        if not over_ids:
            start = 0 if 0 < ids[0] < length else ids[0]
            end = ids[1] if 0 < ids[1] < length else length
            over_ids.append((start, end))
        else:
            if over_ids[-1][1]<ids[0]:
                start = over_ids[-1][1]
                end = ids[1]
                over_ids.append((start, end))
            over_ids.append(ids)

        if over_ids[-1][1]>=length:
            break
    if over_ids[-1][1]<length:
        start = over_ids[-1][1]
        end = length
        over_ids.append((start, end))
    print(over_ids)
    sort_ids = over_ids[:]
    j = 0
    for i, ids in enumerate(over_ids, start=0):
        if search_ids[j] == ids:
            sort_ids[i] = search_ids[args[j]]
            j = j + 1
            if j >= len(search_ids):
                break
    print(sort_ids)
    export_model_state(pth_file_path, export_name, args=sort_ids)


def combine_weights(weights1, export_name, weights2):
    w1 = torch.load(weights1)
    if not isinstance(w1, OrderedDict):
        print("error!!!")
    w2 = torch.load(weights2)
    if not isinstance(w2, OrderedDict):
        print("error!!!")
        w2 = w2['state_dict']

    merged_dict = OrderedDict()
    for key, value in w1.items():
        if key in merged_dict:
            # 如果键已经存在，将新值添加到列表中
            i = 1
            key_ = key
            while key_ in merged_dict:
                i = i + 1
                key_ = key + '_' + str(i)
            merged_dict[key_].append(value)
        else:
            # 如果键不存在，创建一个新条目，其值为一个包含单个元素的列表
            merged_dict[key] = value

    for key, value in w2.items():
        if key in merged_dict:
            # 如果键已经存在，将新值添加到列表中
            i = 1
            key_ = key
            while key_ in merged_dict:
                i = i + 1
                key_ = key + '_' + str(i)
            merged_dict[key_].append(value)
        else:
            # 如果键不存在，创建一个新条目，其值为一个包含单个元素的列表
            merged_dict[key] = value

    torch.save(merged_dict, "run_exp/export/"+export_name+".pth")  # mpcVSR256_pre80_icslr_75_cer417.pth

