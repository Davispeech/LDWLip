import ast
import time
import logging
import os.path

import torch
import torchaudio
import torchvision
from ldw_utils.parseCFG import parse_cfg
from ldw_utils.ini_utils import ini_cfg, data_preprocess_method
from ldw_datas.transforms import TextTransform, AudioTransform, VideoTransform
from ldw_nets.model import Model


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


def mkPath(root_dir, path):
    path = path.replace(root_dir, '')
    dirs = path.split('/')
    path = root_dir
    for dir in dirs:
        if not dir:
            continue
        path = path + '/' + dir
        if not os.path.exists(path):
            os.mkdir(path)


def mkModelDir(path, name):
    if not os.path.exists(path + '/' + name):
        os.mkdir(path + '/' + name)
        return name
    i = 2
    isFlag = True
    while isFlag:
        if not os.path.exists(path + '/' + name + '_' + str(i)):
            os.mkdir(path + '/' + name + '_' + str(i))
            isFlag = False
        else:
            i = i + 1
    return name + '_' + str(i)


class preprocess(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(preprocess, self).__init__()
        self.cfg = cfg
        self.modality = cfg.default.modality
        if self.modality in ["audio", "audio_video"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audio_video"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(visual_attack="", attack_level="", subset="test")

    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audio_video"]:
            input_ = data_preprocess_method(self.cfg, "audio")
            if input_ == "Input_audio":
                audio, sample_rate = self.load_audio(data_filename)
                audio = self.audio_process(audio, sample_rate)
                audio = audio.transpose(1, 0)
                audio = self.audio_transform(audio)
            elif input_ == "Input_audio_mel":
                from ldw_datas.av_dataset import load_audio_whisper, pad_or_trim, log_mel_spectrogram
                audio = load_audio_whisper(data_filename)  # 加载音频文件
                audio = pad_or_trim(audio)  # 补齐或裁剪音频到合适的长度
                audio = log_mel_spectrogram(audio)  # .to(device)  # 转换为 Mel 频谱图
            else:
                assert False, f"data load error!!!!!!!!!!!"


        if self.modality in ["video", "audio_video"]:
            video = self.load_video(data_filename)
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            # import cv2
            # import numpy as np
            # frames, height, width, channels = video.shape
            # # 创建视频编写器，输出文件为MP4格式
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择编码格式
            # out = cv2.VideoWriter("out96.mp4", fourcc, 25, (width, height))  # RGB图像（isColor=True）
            # # 将每一帧写入视频
            # for i in range(frames):
            #     frame = video[i]  # 获取第i帧
            #     frame = np.uint8(frame)  # 转换为8位整数类型
            #     frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            #     out.write(frame_bgr)  # 写入视频
            # out.release()

            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)

        if self.modality == "video":
            return video
        elif self.modality == "audio":
            return audio

        elif self.modality == "audiovisual":
            print(len(audio), len(video))
            assert 530 < len(audio) // len(video) < 670, "The video frame rate should be between 24 and 30 fps."

            rate_ratio = len(audio) // len(video)
            if rate_ratio == 640:
                pass
            else:
                from ldw_datas.av_dataset import cut_or_pad
                print(
                    f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video) * 16000 / len(audio):.1f}, which may affect the performance.")
                audio = cut_or_pad(audio, len(video) * 640)
            return audio, video

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


def run_predict(args):
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model)

    # -----------------------------------------------------
    # 【Step 1】初始化：数据加载与预处理器
    cfg = ini_cfg(args, cfg)
    data_preprocess = preprocess(cfg)
    source = cfg.predict.source
    import time
    start = time.time()
    data = data_preprocess(source)
    print(time.time()-start)

    # -----------------------------------------------------
    # 【Step 2】模型构建
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

    # -----------------------------------------------------
    # 【Step 3】predict
    model.eval()
    with torch.no_grad():
        data = data.cuda(device[0])
        ## 前馈计算
        # import time
        # start = time.time()
        predict = model.predict(data)
        # print(time.time()-start)
    print(f"预测结果为：{predict}")
