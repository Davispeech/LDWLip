"""LDWLip unit."""
import torch
import torch.nn as nn
from ldw_nets.utils import make_non_pad_mask


__all__ = (
    "Input_video2imgSingle",
    "Input_video",
    "Input_audio",
    "Input_audio_mel",
    "Input_label",
)


class Input_video2imgSingle(nn.Module):
    def forward(self, x):
        if "videos" in x:  # [15, 77, 1, 88, 88]
            inputs = x["videos"][:, 0, :, :, :]
            return {"inputs": inputs}
        elif "input" in x:
            x = x["input"].unsqueeze(0)  # [1, 77, 1, 88, 88]
            return x[:, 0, :, :, :]
        else:  # [1, 77, 1, 88, 88]
            x = x["video"].unsqueeze(0)
            return x[:, 0, :, :, :]


class Input_video(nn.Module):
    def forward(self, x):
        if "videos" in x:
            inputs = x["videos"]
            lengths = x["video_lengths"]
            mask = make_non_pad_mask(lengths).to(
                inputs[0].device if isinstance(inputs, tuple) else inputs.device).unsqueeze(-2)
            return {"inputs": inputs, "lengths": lengths, "mask": mask}
        elif "input" in x:
            return x["input"].unsqueeze(0)
        else:
            return x["video"].unsqueeze(0)


class Input_label(nn.Module):
    def __init__(self, label_id):
        super().__init__()
        self.label_id = label_id

    def forward(self, x):
        if 'target_0_s' in x:
            target = x['target_'+str(self.label_id)+'_s']
            return {"inputs": target, "label": True}
        else:
            return {"inputs": x['targets'], "label": True}


class Input_audio(nn.Module):
    def forward(self, x):
        if "audios" in x:
            inputs = x["audios"]
            lengths = x["audio_lengths"]
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
            mask = make_non_pad_mask(lengths).to(inputs[0].device if isinstance(inputs, tuple) else inputs.device).unsqueeze(-2)
            return {"inputs": inputs, "lengths": lengths, "mask": mask}
        elif "input" in x:
            return x["input"].unsqueeze(0)
        else:
            return x["audio"].unsqueeze(0)


class Input_audio_mel(nn.Module):
    def forward(self, x):
        if "audios" in x:
            inputs = x["audios"]
            lengths = x["audio_lengths"]
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
            mask = make_non_pad_mask(lengths).to(inputs[0].device if isinstance(inputs, tuple) else inputs.device).unsqueeze(-2)
            return {"inputs": inputs, "lengths": lengths, "mask": mask}
        elif "input" in x:
            mel = x["input"].squeeze()
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)
            return mel
        else:
            mel = x["audio"].squeeze()
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)
            return mel

