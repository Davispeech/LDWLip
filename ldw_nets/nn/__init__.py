#
from .input import Input_video, Input_audio, Input_audio_mel
from .peft_module import Adapter
from .unit import (
    Swish,
    View,
    ViewtoBatch,
    LayerNorm,
    AddWithDecayEpoch,
    nnLinear,
    tensor3Dto2D,
)
from .block import (
    Conv,  # CBS
    Conv3Dblock,
    Conv3Dfrontend,
    DownsampleCB,  # downsample_basic_block
    DownsampleCB_1D,
    ResidualBlock,
    ResidualBlock131,
    ResidualBlock_1D,
    ResidualBlock131_1D,
    ConvDictBlock,
    featurefusionmodule,
    PositionwiseFeedForward,  # conformerBlock
    PositionalEncoding,
    RelPositionalEncoding,
    embed,
    ConformerLayer,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    SwinTransformerBlock,  # Visual Transformer Block
    BranchformerEncoderLayer, # branchformerBlock
    EBranchformerEncoderLayer, # ebranchformerBlock
    ConvolutionalGatingMLP,
    FastSelfAttention,
    # SwinLip 2d组件
    Conv3D_BN_PReLU,
    SpatialAvgPooling,
    ConvAttentionModule,
    SwinBlock,
    SwinStage,
    SwinTransformer,
    SwinTransformerBlockR,
    PatchEmbed,
    PatchMerging,
    # 3D SwinLip Block
    SwinTransformerBlock3D,
    PatchMerging3D,
    SwinStage3D,
    PatchEmbed3D,
    SpatialAvgPooling3D,
)
from .network import (
    ResNet,
    ResNet1D,
    UNet1D,
    Conv3D_2DNet,
    Conv_1DNet,
    AudioEncoder,
    TextDecoder,
    Whisper,
    LipNet,
)
from .decoder import (
    HybrideLipOutput,
    CTCDecoder,
    ATTDecoder,
    whisperDecoder,
)
from .loss import (
    loss_total,
    loss_ctc,
    loss_att,
    loss_Focal_ctc,
    loss_Focal_att,
    loss_CrossEntropy,
    loss_L1,
    loss_relation,
    loss_corr,
    loss_id,
    loss_Cosine,
    loss_MPAD,
    loss_rc,
)

__all__ = (
    # input ####################################################
    Input_video,
    Input_audio,
    Input_audio_mel,
    # peft ####################################################
    Adapter,
    # units ####################################################
    Swish,
    View,
    ViewtoBatch,
    LayerNorm,
    AddWithDecayEpoch,
    nnLinear,
    tensor3Dto2D,
    # block ####################################################
    Conv,
    Conv3Dblock,
    Conv3Dfrontend,
    DownsampleCB,  # downsample_basic_block
    DownsampleCB_1D,
    ResidualBlock,
    ResidualBlock131,
    ResidualBlock_1D,
    ResidualBlock131_1D,
    ConvDictBlock,
    featurefusionmodule,
    PositionwiseFeedForward,
    PositionalEncoding,
    PositionwiseFeedForward,
    RelPositionalEncoding,
    embed,
    ConformerLayer, # ConformerBlock
    BranchformerEncoderLayer,  # branchformerBlock
    EBranchformerEncoderLayer,  # ebranchformerBlock
    SwinTransformerBlock,  # Visual Transformer Block
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    # SwinLip 2d组件
    Conv3D_BN_PReLU,
    SpatialAvgPooling,
    ConvAttentionModule,
    SwinBlock,
    SwinStage,
    SwinTransformer,
    SwinTransformerBlockR,
    PatchEmbed,
    PatchMerging,
    # 3D SwinLip Block
    SwinTransformerBlock3D,
    PatchMerging3D,
    SwinStage3D,
    PatchEmbed3D,
    SpatialAvgPooling3D,
    # network ####################################################
    ResNet,
    ResNet1D,
    UNet1D,
    Conv3D_2DNet,
    Conv_1DNet,
    AudioEncoder,
    TextDecoder,
    Whisper,
    LipNet,
    # decoder ####################################################
    HybrideLipOutput,
    CTCDecoder,
    ATTDecoder,
    whisperDecoder,
    # loss ####################################################
    loss_total,
    loss_ctc,
    loss_att,
    loss_Focal_ctc,
    loss_Focal_att,
    loss_CrossEntropy,
    loss_L1,
    loss_relation,
    loss_corr,
    loss_id,
    loss_Cosine,
    loss_MPAD,
    loss_rc,
)
