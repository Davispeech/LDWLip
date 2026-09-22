
import os
import io
import numpy as np
import webdataset as wds
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from ldw_utils.ini_utils import data_preprocess_method
from ldw_datas.transforms import TextTransform, AudioTransform, VideoTransform


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


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


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


class WebAVDataset(IterableDataset):
    def __init__(self, cfg, subset="train"):
        self.cfg = cfg

        if subset=="train":
            tar_path = cfg.data.data_path + cfg.data.train
        elif subset=="test":
            tar_path = cfg.data.data_path + cfg.data.test
        else:
            tar_path = cfg.data.data_path + cfg.data.val

        self.count = int(tar_path.split('_')[-2].replace('train', '').replace('test', '').replace('val', ''))
        self.subset = subset
        self.modality = cfg.default.modality

        self.audio_transform = AudioTransform(subset=subset)
        self.video_transform = VideoTransform(subset=subset)
        self.input_method = data_preprocess_method(cfg, "audio")

        self.dataset = (
            wds.WebDataset(tar_path, shardshuffle=False, empty_check=False)
            .decode()  # keep raw bytes
            .to_tuple("mp4", "tokens.npy", "spk", "video_path")  # 注意这里带点
            .map(self._process_sample)
        )

    def _process_sample(self, sample):
        video_bytes, tokens_bytes, spk_bytes, video_path_bytes = sample
        data_dict = {}

        # tokens_bytes 直接转 list
        tokens = list(tokens_bytes)
        data_dict["target_0_"] = torch.tensor(tokens, dtype=torch.long)

        # spk 解码为 int tensor
        spk = int(spk_bytes.decode("utf-8"))
        data_dict["target_1_"] = torch.tensor([spk], dtype=torch.long)

        # video_path 解码为字符串
        # video_path = video_path_bytes.decode("utf-8")
        # data_dict["path"] = video_path

        # video 解码与变换
        if self.modality in ["video", "audio_video"]:
            video_tensor = self.decode_video_from_bytes(video_bytes)
            video_tensor = self.video_transform(video_tensor)
            data_dict["video"] = video_tensor

        # audio 解码与变换
        if self.modality in ["audio", "audio_video"]:
            if self.input_method == "Input_audio":
                audio_tensor = self.decode_audio_from_mp4(video_bytes)
                audio_tensor = self.audio_transform(audio_tensor)
                data_dict["audio"] = audio_tensor
            elif self.input_method == "Input_audio_mel":
                audio = self.decode_audio_from_mp4(video_bytes)
                audio = pad_or_trim(audio)
                audio = log_mel_spectrogram(audio)
                data_dict["audio"] = audio
            else:
                raise ValueError("Unsupported audio method")

        return data_dict

    def decode_video_from_bytes(self, video_bytes):
        import av
        container = av.open(io.BytesIO(video_bytes))
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()
            frames.append(img)
        video_tensor = torch.tensor(np.stack(frames), dtype=torch.uint8)  # [T, H, W, C]
        return video_tensor.permute(0, 3, 1, 2)  # 转为 [T, C, H, W]

    def decode_audio_from_mp4(self, video_bytes):
        import torchaudio
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(video_bytes)
            f.flush()
            waveform, sr = torchaudio.load(f.name)
        return waveform

    def __iter__(self):
        return iter(self.dataset)
