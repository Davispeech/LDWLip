import torch


__all__ = (
    "featurefusionmodule",
)

class featurefusionmodule(torch.nn.Module):

    def __init__(self, channels=512):
        super(featurefusionmodule, self).__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        out = out1 + out2

        return out.permute(0, 2, 1)


