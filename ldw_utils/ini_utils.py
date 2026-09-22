import os
import copy
import chardet
import logging


def get_filetype(filepath):
    with open(filepath, 'rb') as file:
        rawdata = file.read(10000)
        result = chardet.detect(rawdata)
        file.close()
    return result['encoding']


def mkPath(root_dir, path):
    # 判断路径是否存在，如果不存在则创建文件夹
    path = path.replace(root_dir, '')
    dirs = path.split('/')
    path = root_dir
    for dir in dirs:
        if not dir:
            continue
        path = path + '/' + dir
        if not os.path.exists(path):
            os.mkdir(path)


def get_model_dir_name(path, name):
    # 如果目录文件已存在，则依序新建+1
    # E:\Sproject\DaviLip\LDWLip\data\lipauth_support\icslr_FeatureHog\icslr_deepfake1
    if os.path.exists(path + '/' + name):
        i = 2
        isFlag = True
        while isFlag:
            if not os.path.exists(path + '/' + name + '_' + str(i)):
                isFlag = False
                name = name + '_' + str(i)
            else:
                i = i + 1
    return name


def data_preprocess_method(cfg, data_type):
    if data_type == 'audio':
        for input in cfg.model.Input:
            if 'audio' in input[3]:
                return input[3]
    # to be continue


def get_dis_model_list(id_del, id_cro, model_list):
    dis = 0
    if id_cro[0]<id_del[0] or (id_cro[0]==id_del[0] and id_cro[1]<=id_del[1]):
        return dis
    for i in range(id_del[0], len(model_list)):
        module = model_list[i]
        for j in range(len(module)):
            if i == id_del[0] and j <= id_del[1]:
                continue
            dis = dis + 1
            if i == id_cro[0] and j == id_cro[1]:
                return dis
    print("something wrong with id_del and id_cro!!!!!!!!!!")
    return dis


def ini_cfg_del_auxmodel_old(cfg):
    if cfg.default.mode in ["eval", "test", "predict"]:
        aux_ids = []
        cross_ids = []
        model_list = [cfg.model.Input, cfg.model.Encoder, cfg.model.Decoder]  #, model.Loss]
        model_list_new = copy.deepcopy(model_list)

        for i in range(len(model_list)):
            module = model_list[i]
            for j in range(len(module)):
                if 'aux_' in module[j][2]:  # type
                    aux_ids.append([i, j])
                    continue
                f = module[j][0]
                if isinstance(f, int):
                    if f != -1 and f != 0:
                        cross_ids.append([i, j])
                else:
                    for f_ in f:
                        if f_ != -1 and f_ != 0:
                            cross_ids.append([i, j])
                            break
        for id_ in range(len(aux_ids)):
            i, j = aux_ids[id_]
            # 重置同一个list下的aux_ids和cross_ids
            for id__ in range(id_, len(aux_ids)):
                if aux_ids[id__][0] == i and aux_ids[id__][1] > j:
                    aux_ids[id__][1] = aux_ids[id__][1] - 1
            for id__ in range(id_, len(cross_ids)):
                if cross_ids[id__][0] == i and cross_ids[id__][1] > j:
                    cross_ids[id__][1] = cross_ids[id__][1] - 1
            # 重置cross_ids的f
            for id_c in cross_ids:
                dis_del = get_dis_model_list((0, 0), (i, j), model_list_new)
                dis_cros = get_dis_model_list((0, 0), id_c, model_list_new)
                # dis = get_dis_model_list((i, j), id_c, model_list)
                if dis_del < dis_cros:
                    f = model_list_new[id_c[0]][id_c[1]][0]
                    if isinstance(f, int):
                        if f < -1 and f + dis_cros < dis_del:
                            model_list_new[id_c[0]][id_c[1]][0] = f+1
                        elif f > 0 and f > dis_del:
                            model_list_new[id_c[0]][id_c[1]][0] = f-1
                    else:
                        for f_ in f:
                            if f_ < -1 and f_ + dis_cros < dis_del:
                                model_list_new[id_c[0]][id_c[1]][0] = f_ + 1
                            elif f_ > 0 and f_ > dis_del:
                                model_list_new[id_c[0]][id_c[1]][0] = f_ - 1
            del model_list_new[i][j]
        cfg.model.Input = model_list_new[0]
        cfg.model.Encoder = model_list_new[1]
        cfg.model.Decoder = model_list_new[2]
    return cfg


def ini_cfg_model(cfg):
    odim = cfg.default.odim
    for i in range(len(cfg.model.Decoder)):
        a = cfg.model.Decoder[i]
        if cfg.model.Decoder[i][3] in ['CTCDecoder', 'ATTDecoder', 'TextDecoder']:
            cfg.model.Decoder[i][4][0] = odim
    for i in range(len(cfg.model.Loss)):
        if cfg.model.Loss[i][3] in ['loss_ctc', 'loss_att']:
            cfg.model.Loss[i][4][0] = odim
    return cfg


def ini_cfg_del_auxmodel(cfg):
    if cfg.default.mode in ["eval", "test", "predict"]:
        aux_ids = []
        cross_ids = []
        model_list = [cfg.model.Input, cfg.model.Encoder, cfg.model.Decoder]  #, model.Loss]

        # 将所有层输入源序号置为正
        layer_num = 0
        for i in range(len(model_list)):
            for j in range(len(model_list[i])):
                if model_list[i][j][2] in ["input", "aux_input"]:
                    layer_num = layer_num + 1
                    continue
                layer_from = model_list[i][j][0]
                if isinstance(layer_from, int):
                    if layer_from < 0:
                        model_list[i][j][0] = model_list[i][j][0] + layer_num
                else:
                    for layer_from_i in range(len(layer_from)):
                        if model_list[i][j][0][layer_from_i]<0:
                            model_list[i][j][0][layer_from_i] = model_list[i][j][0][layer_from_i] + layer_num
                layer_num = layer_num + 1

        model_list_new = copy.deepcopy(model_list)

        # 获取aux列表
        for i in range(len(model_list)):
            module = model_list[i]
            for j in range(len(module)):
                if 'aux_' in module[j][2]:  # type
                    aux_ids.append([i, j])
                    continue
                f = module[j][0]
                if isinstance(f, int):
                    if f != -1 and f != 0:
                        cross_ids.append([i, j])
                else:
                    for f_ in f:
                        if f_ != -1 and f_ != 0:
                            cross_ids.append([i, j])
                            break

        # 遍历aux_ids，并删除

        while 1:
            # 顺序查找aux_层，如果查找不到，跳出循环
            aux_id = [0, 0]
            isflag = False
            for i in range(len(model_list)):
                for j in range(len(model_list[i])):
                    if 'aux_' in model_list[i][j][2]:
                        aux_id = [i, j]
                        isflag = True
                        break
                if isflag:
                    break
            if not isflag:
                break
            del_i, del_j = aux_id
            if del_i==0 and del_j==0:
                print("ini_cfg_del_auxmodel is wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            layer_num = -1
            del_id = -1
            for i in range(len(model_list)):
                for j in range(len(model_list[i])):
                    layer_num = layer_num + 1
                    if i == del_i and j == del_j:
                        del_id = layer_num
                    if del_id < 0:
                        continue
                    layer_from = model_list[i][j][0]
                    if isinstance(layer_from, int):
                        if layer_from > del_id:
                            model_list[i][j][0] = model_list[i][j][0] - 1
                        elif layer_from == del_id and "aux_" not in model_list[i][j][2]:
                            model_list[i][j][2] = "aux_" + model_list[i][j][2]
                    else:
                        for layer_from_i in range(len(layer_from)):
                            if model_list[i][j][0][layer_from_i] > del_id:
                                model_list[i][j][0][layer_from_i] = model_list[i][j][0][layer_from_i] - 1
                            elif layer_from == del_id and "aux_" not in model_list[i][j][2]:
                                model_list[i][j][2] = "aux_" + model_list[i][j][2]
            del model_list[del_i][del_j]

        cfg.model.Input = model_list[0]
        cfg.model.Encoder = model_list[1]
        cfg.model.Decoder = model_list[2]
    return cfg


def get_register_subset(test_path):
    # root_dir + 'data/lipauth_support/icslr_Lipnet/'
    # labels/icslrAuth/icslrAuth_testAll.csv
    # labels/icslrAuth/DeepfakeAttack/Lip_deepfake1_test.csv
    csv_name = test_path.split('/')[-1].replace('.csv', '')
    register_subset = csv_name

    return register_subset


def ensure_folder_structure(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"已创建文件夹路径: {dir_path}")
        except OSError as e:
            print(f"创建文件夹时出错: {e}")


def ini_cfg(args, cfg):
    # custom
    is_have_sr = False
    if args.task:
        cfg.default.task = args.task
    if args.mode:
        cfg.default.mode = args.mode
    if args.model:
        cfg.default.model = args.model
        if not args.model_name:
            args.model_name = args.model.split('/')[-1].replace('.yaml', '')
    if args.data:
        cfg.default.data = args.data
    if args.root_dir:
        cfg.default.root_dir = args.root_dir
    if args.weights:
        cfg.default.weights = args.weights
    if args.aux_weights:
        cfg.train.aux_weights = args.aux_weights

    # Train
    if args.task in ['lipauth', 'vsr', 'asr', 'avsr', 'classifier'] and args.mode in ['train', 'resume']:
        if args.run_exp_dir:
            cfg.train.run_exp_dir = args.run_exp_dir
            mkPath(cfg.default.root_dir, cfg.train.run_exp_dir)

        if args.model_name:
            path = cfg.default.root_dir + '/' + cfg.train.run_exp_dir
            cfg.train.model_name = get_model_dir_name(path, args.model_name)
        else:
            cfg.train.model_name = get_model_dir_name(cfg.default.root_dir+'/'+cfg.train.run_exp_dir, cfg.train.model_name)
        if args.mode == "resume":
            cfg.train.model_name = args.weights.split('/')[1]

        path = cfg.default.root_dir + '/' + cfg.train.run_exp_dir + '/' + cfg.train.model_name
        mkPath(cfg.default.root_dir, path)
    if args.max_epochs:
        cfg.train.max_epochs = args.max_epochs
    if args.batch:
        cfg.train.batch = args.batch
    if args.save_every_epoch:
        cfg.train.save_every_epoch = args.save_every_epoch
    if args.save_dir:
        cfg.train.save_dir = args.save_dir
    if args.device:
        cfg.train.device = args.device
    if args.pretrained:
        cfg.train.pretrained = args.pretrained
    if args.freeze:
        cfg.train.freeze = args.freeze
    if args.optimizer:
        cfg.train.optimizer = args.optimizer
    if args.lr:
        cfg.train.lr = args.lr
    if args.warmup_epochs:
        cfg.train.warmup_epochs = args.warmup_epochs
        if cfg.train.max_epochs < 15:
            cfg.train.warmup_epochs = 1
    if args.weight_decay:
        cfg.train.weight_decay = args.weight_decay
    if args.workers!=8:
        cfg.train.workers = args.workers
    if args.resume:
        cfg.train.resume = args.resume

    # if args.multi_scale:
    #     cfg.train.multi_scale = args.multi_scale

    # Eval
    attack = ""
    if args.attack_type and args.attack_level:
        attack = args.attack_type+args.attack_level
    cfg.default.attack_type = args.attack_type
    cfg.default.attack_level = args.attack_level
    if args.test_data:
        cfg.data.test = args.test_data
    if args.task in ['lipauth', 'vsr', 'asr', 'avsr', 'classifier'] and args.mode in ['eval', 'test']:
        if cfg.default.task in ["vsr", "asr", "avsr"]:
            a = cfg.data.char_path.split("/")[-1].replace(".tiktoken", "").replace(".txt", "")
            if a in ["cmlr", "mnt", "icslr", "cmlr_icslr", "multilingual"]:
                cfg.data.language = "zh"
            else:
                cfg.data.language = "en"
        if args.save_per_sample:
            cfg.eval.save_per_sample = args.save_per_sample
            outfilr_name = get_model_dir_name(cfg.default.root_dir + '/output/', args.model_name)
            cfg.eval.outfile = cfg.default.root_dir + '/output/' + outfilr_name+'/'+args.model_name+'.txt'
            mkPath(cfg.default.root_dir, cfg.default.root_dir + '/output/' + outfilr_name)
        if args.half:
            cfg.eval.half = args.half
        if cfg.model.Output.name == 'HybrideLipOutput':
            if args.ctc_weight:
                cfg.model.Output.output_param.ctc_weight = args.ctc_weight
            if args.beam_size:
                cfg.model.Output.output_param.beam_size = args.beam_size
            if args.lm_weight:
                if args.lm_weight == 0.0:
                    cfg.model.Output.output_param.lm_model = False
                else:
                    cfg.model.Output.output_param.lm_model = True
                    cfg.model.Output.output_param.lm_weight = args.lm_weight

    # lipauth register
    # cfg.default.root_dir + '/' + cfg.model.Output.output_param.support
    # data/lipauth_support/icslr_Lipnet/
    if args.task == 'lipauth':  # and args.mode == 'register':
        for layer in cfg.model.Decoder:
            if layer[3] in ['CTCDecoder', 'ATTDecoder', 'TextDecoder']:
                is_have_sr = True
                break
        # 获取子集名称 icslr_valid, icslr_test, icslr_attack1, icslr_attack2, icslr_attack4
        register_subset = get_register_subset(cfg.data.test)
        # 获取注册特征层数
        save_layerFeatures = []
        for key, value in vars(cfg.model.Output.output_param).items():
            if 'Auth_feature_' in key:
                save_layerFeatures.append(value.split(','))
                suppot_path = cfg.model.Output.output_param.support + save_layerFeatures[-1][0] + '_' + register_subset
                if attack:
                    suppot_path = suppot_path+"_"+attack
                if args.mode in ['register']:
                    if os.path.exists(suppot_path):
                        dirname = suppot_path.split('/')[-1]
                        dirname_new = get_model_dir_name(cfg.default.root_dir + '/' + suppot_path.replace(dirname, ''),
                                                         dirname)
                        suppot_path = suppot_path.replace(dirname, dirname_new)
                    ensure_folder_structure(cfg.default.root_dir + '/' + suppot_path + '/saber')
        cfg.model.Output.output_param.support = suppot_path
        cfg.model.Output.output_param.save_layerFeatures = save_layerFeatures

        if args.ids_user:
            import json
            cfg.model.Output.output_param.ids_user = json.loads(args.ids_user)

    # Predict
    if args.source:
        cfg.predict.source = args.source
    # else
    # 其中训练模式根据input层判断数据类型，非训练模式根据任务类型选择
    if args.mode in ["train", "resume"]:
        inputs = []
        for i in range(len(cfg.model.Input)):
            Input_layer = cfg.model.Input[i]
            if Input_layer[3] == "Input_label":
                continue
            if Input_layer[3] in ["Input_video", "Input_video2imgSingle"]:
                if "video" not in inputs:
                    inputs.append("video")
            elif Input_layer[3] in ["Input_audio", "Input_audio_mel"]:
                if "audio" not in inputs:
                    inputs.append("audio")
            else:
                logging.info("task-modality ERROR!!!!!!")
        if len(inputs) == 1:
            cfg.default.modality = inputs[0]
        elif "video" in inputs and "audio" in inputs:
            cfg.default.modality = "audio_video"
        else:
            logging.info("task-modality ERROR!!!!!!")
    else:
        if args.task in ["vsr", "lipauth", "classifier"]:
            cfg.default.modality = "video"  # ASR蒸馏注意，好像无需在意，根据任务直接选择输入了
        elif args.task == "asr":
            cfg.default.modality = "audio"
        elif args.task == "avsr":
            cfg.default.modality = "audio_video"
        else:
            logging.info("task-modality ERROR!!!!!!")

    if cfg.default.task in ["vsr", "asr", "avsr"] or is_have_sr:
        dict_path = cfg.default.root_dir + '/' + cfg.data.char_path
        if cfg.data.char_path.split(".")[-1] == "txt":
            from ldw_datas.transforms import TextTransform
            cfg.default.odim = len(TextTransform(dict_path=dict_path).token_list)
            cfg.default.sos = cfg.default.odim - 1
            cfg.default.eos = cfg.default.odim - 1
            cfg.default.sot = cfg.default.sos
            cfg.default.eot = cfg.default.eos
        elif cfg.data.char_path.split(".")[-1] == "tiktoken":
            from ldw_nets.nn.decoder.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(multilingual=True, num_languages=99)
            cfg.default.odim = tokenizer.encoding.n_vocab
            cfg.default.sot = tokenizer.special_tokens["<|startoftranscript|>"]  # <sot>
            cfg.default.eot = tokenizer.special_tokens["<|endoftext|>"]  # <eot>
            cfg.default.sos = cfg.default.sot
            cfg.default.eos = cfg.default.eot
            del tokenizer
        cfg = ini_cfg_model(cfg)
    cfg = ini_cfg_del_auxmodel(cfg)
    return cfg
