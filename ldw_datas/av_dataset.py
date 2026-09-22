import logging
import os
import math
import numpy as np
from subprocess import CalledProcessError, run
import torch
import torchaudio
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from ldw_utils.ini_utils import data_preprocess_method
from ldw_datas.transforms import TextTransform, AudioTransform, VideoTransform


def load_audio_whisper(file, sr=16000):
    # cmd = [
    #     "ffmpeg",
    #     "-nostdin",
    #     "-threads", "0",
    #     "-i", file,
    #     "-f", "s16le",
    #     "-ac", "1",
    #     "-acodec", "pcm_s16le",
    #     "-ar", str(sr),
    #     "-"
    # ]
    # try:
    #     out = run(cmd, capture_output=True, check=True).stdout
    # except CalledProcessError as e:
    #     raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    # return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # try:
    if 1:
        # Extract audio from the file using torchvision.io.read_video
        audio, original_sr = torchvision.io.read_video(file, pts_unit="sec")[1:3]
        original_sr = original_sr['audio_fps']

        # Convert audio to a PyTorch tensor
        audio_tensor = audio.clone().detach().to(torch.float32)

        # If audio is stereo, average channels to convert to mono
        if audio_tensor.dim() > 1 and audio_tensor.size(0) > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        # Resample audio if necessary
        if original_sr != sr:
            resample_ratio = sr / original_sr
            target_length = int(audio_tensor.size(1) * resample_ratio)

            # Use linear interpolation to resample
            audio_tensor = torch.nn.functional.interpolate(
                audio_tensor.unsqueeze(0),  # Add batch dimension
                size=target_length,
                mode="linear",
                align_corners=False
            ).squeeze(0)

        # Normalize the waveform to range [-1.0, 1.0]
        audio_tensor = audio_tensor / max(abs(audio_tensor).max(), 1.0)

        # Convert to NumPy array
        return audio_tensor.squeeze().numpy()

    # except Exception as e:
    #     raise RuntimeError(f"Failed to load audio: {e}")


def pad_or_trim(array, length: int = 16000*30, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(audio, n_mels=80, padding=0, device=None,):
    N_FFT = 400
    HOP_LENGTH = 160
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class ShardedAVDataset(Dataset):
    def __init__(self, base_dataset, shard_idx=0, num_shards=5):
        self.base_dataset = base_dataset
        self.num_shards = num_shards
        self.shard_idx = shard_idx

        total_len = len(base_dataset)
        shard_size = math.ceil(total_len / num_shards)

        start = shard_idx * shard_size
        end = min(start + shard_size, total_len)
        self.indices = list(range(start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]


class AVDataset(Dataset):
    def __init__(
        self,
        cfg,
        subset,
        rate_ratio=640,
    ):
        self.cfg = cfg
        self.root_dir = cfg.default.root_dir
        self.data_root_dir = cfg.data.data_path  #  "F:/Datasets/lipreading/"  # cfg.default.root_dir+'/data/'
        self.modality = cfg.default.modality
        self.rate_ratio = rate_ratio

        if subset=="train":
            label_path = self.root_dir + '/' + cfg.data.train
        elif subset=="test" or subset=="eval":
            label_path = self.root_dir + '/' + cfg.data.test
        elif subset=="val":
            label_path = self.root_dir + '/' + cfg.data.val
        else:
            label_path = ""
            logging.info("label path ERROR!!!!!!")
        self.list = self.load_list(label_path)

        self.audio_transform = AudioTransform(subset=subset)
        self.video_transform = VideoTransform(
            visual_attack=cfg.default.attack_type,
            attack_level=cfg.default.attack_level,
            subset=subset
        )

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            # dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            data_elem = path_count_label.split(",")
            dataset_name = data_elem[0]
            rel_path = data_elem[1]
            input_length = data_elem[2]
            if len(data_elem) == 4:
                token_id = data_elem[3]
                token_id = torch.tensor([int(_) for _ in token_id.split()])
            else:
                token_id = []
                for i in range(3, len(data_elem)):
                    token = torch.tensor([int(_) for _ in data_elem[i].split()])
                    token_id.append(token)
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    token_id,
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        # import time
        # start = time.time()
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.data_root_dir, dataset_name, rel_path)

        Data_Dict = {}

        if self.modality in ["audio", "audio_video"]:
            audio_path = path[:-4] + ".wav"
            input_ = data_preprocess_method(self.cfg, "audio")
            if input_ == "Input_audio":
                audio = load_audio(audio_path)
                audio = self.audio_transform(audio)
                Data_Dict["audio"] = audio
            elif input_ == "Input_audio_mel":
                audio = load_audio_whisper(audio_path)  # 加载音频文件
                audio = pad_or_trim(audio)  # 补齐或裁剪音频到合适的长度
                audio = log_mel_spectrogram(audio)  # .to(device)  # 转换为 Mel 频谱图
                Data_Dict["audio"] = audio
            else:
                assert False, f"audio data load error!!!!!!!!!!!"
        if self.modality in ["video", "audio_video"]:
            # t0 = time.time()
            video = load_video(path)
            # t1 = time.time()
            # print(f"getitem {idx} load_video {t1 - t0:.3f}")
            video = self.video_transform(video)
            Data_Dict["video"] = video

        if isinstance(token_id, list):
            for i in range(len(token_id)):
                Data_Dict["target_"+str(i)+"_"] = token_id[i]
        else:
            Data_Dict["target"] = token_id

        if self.cfg.default.task in ["classifier", "vsr", "asr", "avsr", "lipauth"] and self.cfg.default.mode not in ["train", "val", "resume"]:
            Data_Dict["path"] = path

        return Data_Dict

    def __len__(self):
        return len(self.list)
