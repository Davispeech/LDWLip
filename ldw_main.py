import sys
import json
import argparse
# print("DEBUG: 命令行参数 ->", sys.argv)


def preprocess_argv():
    # 创建一个新的 argv 列表，用于传递给 argparse
    new_argv = []

    for arg in sys.argv:
        if '=' in arg:
            # 分离 key 和 value
            key, value = arg.split('=', 1)
            # 转换为 argparse 可以接受的格式
            new_argv.append(f'--{key}')
            new_argv.append(value)
        else:
            new_argv.append(arg)

    return new_argv


sys.argv = preprocess_argv()
parser = argparse.ArgumentParser(description="Your script description")
# 必选参数
# parser.add_argument('--task', type=str, default="vsr", required=True, help='predict mode')
# parser.add_argument('--mode', type=str, default="predict", required=True, help='predict mode')
parser.add_argument('task', type=str, default="vsr", help='predict mode')
parser.add_argument('mode', type=str, default="predict", help='predict mode')
# 可选参数，覆盖默认参数
## 常规参数
parser.add_argument('--model', type=str, default="Lipv2_base.yaml", help='load model')
parser.add_argument('--data', type=str, default="cmlr.yaml", help='load dataset')
parser.add_argument('--root_dir', type=str, default="")  # /home/ldw/Project/Lipreading/LDWLip
parser.add_argument('--modality', type=str, default="video")  # audio, video, audio_video
parser.add_argument('--weights', type=str, default="")  # run_exp/preweights/CMLR_V_WER8.0/model.pth
parser.add_argument('--aux_weights', type=str, default="")  # run_exp/export/ASR_encoder.pth
## train参数
parser.add_argument('--model_name', type=str, default="")
parser.add_argument('--run_exp_dir', type=str, default="run_exp")
parser.add_argument('--max_epochs', type=int, default=75)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--save_every_epoch', type=int, default=5)
parser.add_argument('--save_dir', type=str, default="run_exp")
parser.add_argument('--device', type=str, default=[0])  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
parser.add_argument('--pretrained', type=str, default="")  # run_exp/preweights/CMLR_V_WER8.0/model.pth
parser.add_argument('--freeze', type=str, default="")  # 冻结model.yaml中第几层权重
parser.add_argument('--optimizer', type=str, default="Adam")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.03)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--workers', type=int, default=8)
## eval参数
parser.add_argument('--save_per_sample', type=bool, default=True)
parser.add_argument('--half', type=bool, default=False)
parser.add_argument('--ctc_weight', type=float, default=None)
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--lm_weight', type=float, default=0.0)
parser.add_argument('--test_data', type=str, default=None)
parser.add_argument('--lipauthEval_data', type=str, default=None)
parser.add_argument('--ids_user', type=str, help='JSON list of integers', default=None)
parser.add_argument('--save_featuresPath', type=str, default="/data/support_icslr/")
parser.add_argument('--attack_type', type=str, default="")
parser.add_argument('--attack_level', type=str, default="")
## predict参数
parser.add_argument('--source', type=str, default="/dataset/icslr/icslr_video_seg24s/main/PID0001/0001.mp4")
# - reWeights
parser.add_argument('--layer_depth', type=float, default=0)
parser.add_argument('--export_name', type=str, default='exportModel')
parser.add_argument('--export_layers', type=str, default='[10]')
parser.add_argument('--reWeights_name', type=str, default='frontend')
# - performance
parser.add_argument('--results', type=str, default='')


def main():
    # 解析命令行参数
    args = parser.parse_args()
    task = args.task
    mode = args.mode

    if task in ["lipauth", "vsr", "asr", "avsr", "classifier"]:
        if mode in ["train", "resume"]:
            from ldw_train import run_train
            run_train(args)
        elif mode == "eval" or mode == "test":
            if task == "lipauth":
                from ldw_eval import eval_lipAuth
                eval_lipAuth(args)
            else:
                from ldw_eval import run_eval
                run_eval(args)
        elif mode == "predict":
            from ldw_predict import run_predict
            run_predict(args)
        elif mode == "register":
            if task == "lipauth":
                from ldw_register import run_register
                run_register(args)
            else:
                print("task should be lipauth")
        else:
            print("vsr task: print --weights --level")

    elif task == "reWeights":
        if mode == "print":
            from ldw_reWeights import log_model_state
            log_model_state(args.weights, args.layer_depth)
        elif mode== "export":
            from ldw_reWeights import export_model_state
            export_model_state(args.weights, args.export_name, args=args.export_layers)
        elif mode== "reCreate":
            from ldw_reWeights import re_create_weights
            re_create_weights(args.weights, args.export_name, args.reWeights_name, args=args.export_layers)
        elif mode== "combine":
            from ldw_reWeights import combine_weights
            combine_weights(args.weights, args.export_name, args.aux_weights)
        else:
            print("reWeights task: print --weights --level")
    elif task == "performance":
        if mode == "dataset":
            from ldw_performance import dataset_analysis
            dataset_analysis(args.root_dir, args.data)
        elif mode == "model":
            from ldw_performance import model_analysis
            model_analysis(args.root_dir, args.data)
        elif mode == "results":
            from ldw_performance import results_analysis
            results_analysis(args.root_dir, args.data, args.results)
        else:
            print("performance task !!!!!!!!!!!")
    else:
        print("nothing!!")


# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # Windows 下安全使用 num_workers>0
    main()


