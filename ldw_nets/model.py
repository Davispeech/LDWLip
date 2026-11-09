import os
import sys
import ast
import thop
import logging
import contextlib

from ldw_utils.model_tools import save_NBbest_to_txt

from ldw_datas.transforms import TextTransform
from ldw_nets.utils import MultiSequential, make_non_pad_mask, th_accuracy, audio_video_pad, align_to_ref

import torch
import torch.nn as nn
from ldw_nets.nn import (
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
    RelPositionalEncoding,
    embed,
    SwinTransformerBlock,  # Visual Transformer Block

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
    # module ####################################################
    BranchformerEncoderLayer, # branchformerBlock
    EBranchformerEncoderLayer, # ebranchformerBlock
    ConformerLayer,
    FastSelfAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
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
from loralib import lora_state_dict

def fix_args(args):
    # 1. 将 args 中的元素分离成“字典”和“非字典”两组
    dicts = [arg for arg in args if isinstance(arg, dict)]
    pos_args_list = [arg for arg in args if not isinstance(arg, dict)]

    # 2. 根据找到的字典数量进行判断和赋值
    if len(dicts) == 0:
        # 情况一：args 中没有字典
        kw_args = {}
        pos_args = tuple(pos_args_list)

    elif len(dicts) == 1:
        # 情况二：args 中有且仅有一个字典，这是我们期望的
        kw_args = dicts[0]
        pos_args = tuple(pos_args_list)

    else:
        # 情况三：args 中有多于一个字典，这是一个模糊不清的异常情况
        # 为了避免程序静默地产生错误结果，最好是直接报错，让我们知道输入有问题
        raise ValueError(f"在args中发现了多个字典，无法确定哪一个是用作kwargs: {dicts}")
    return pos_args, kw_args

def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def pharseModel(cfg):
    layers, save = [], []  # layers, savelist, ch out
    for i, (f, n, t, m, args) in enumerate(cfg.Input + cfg.Encoder + cfg.Decoder):  # from, number, type, module, args
        name = m
        # m = importlib.import_module(m)  # get module
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        # unit #####################################################
        if m in {
            # input ####################################################
            Input_video,
            Input_audio,
            Input_audio_mel,
            # units ####################################################
            Adapter,
            Swish,
            View,
            ViewtoBatch,
            LayerNorm,
            AddWithDecayEpoch,
            nnLinear,
            tensor3Dto2D,
            # block ####################################################
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
            RelPositionalEncoding,
            embed,
            SwinTransformerBlock,  # Visual Transformer Block
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

            # module ####################################################
            FastSelfAttention,
            MultiHeadedAttention,
            RelPositionMultiHeadedAttention,
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
            CTCDecoder,
        }:
            args = args
        # module #####################################################
        elif m in {ConformerLayer}:  # Para: adim, aheads, eunits, dropout_rate, attn_layer_type
            attention_dim = args[0] if args[0] else 256
            attention_heads = args[1] if args[1] else 4
            linear_units = args[2] if args[2] else 2048
            dropout_rate = args[3] if args[3] else 0.1
            encoder_attn_layer_type = args[4] if args[4] else "rel_mha"
            args = [attention_dim, attention_heads, linear_units, dropout_rate, encoder_attn_layer_type,
                31, True, True, False, True, False,
            ]
        elif m in {BranchformerEncoderLayer}:
            attention_dim = args[0] if args[0] else 256
            attention_heads = args[1] if args[1] else 4
            linear_units = args[2] if args[2] else 2048
            dropout_rate = args[3] if args[3] else 0.1
            encoder_attn_layer_type = args[4] if args[4] else "fast_selfattn"
            args = [attention_dim, attention_heads, linear_units, dropout_rate, encoder_attn_layer_type,
                    True, True, 'learned_ave', 0.5, 0.0, 0.0, False
            ]
        elif m in {EBranchformerEncoderLayer}:
            attention_dim = args[0] if args[0] else 256
            attention_heads = args[1] if args[1] else 4
            linear_units = args[2] if args[2] else 2048
            dropout_rate = args[3] if args[3] else 0.1
            encoder_attn_layer_type = args[4] if args[4] else "fast_selfattn"
            args = [attention_dim, attention_heads, linear_units, dropout_rate, encoder_attn_layer_type,
                    True, True, True, True, 3, False, 'swish']
        # network #####################################################
        # decoder #####################################################
        elif m in {ATTDecoder}:
            odim = args[0] if args[0] else 3363
            attention_dim = args[1] if args[1] else 256
            attention_heads = args[2] if args[2] else 4
            linear_units = args[3] if args[3] else 2048
            num_blocks = args[4] if args[4] else 6
            dropout_rate = args[5] if args[5] else 0.1
            args = [odim, attention_dim, attention_heads, linear_units, num_blocks, dropout_rate,
                    0.0, 0.0, "embed", True, PositionalEncoding, True, False]
        else:
            raise ValueError("unknown module: " + m)

        if m in {SwinStage}:
            pos_args, kw_args = fix_args(args)  # 分离args中的字典和非字典参数
        else:
            pos_args, kw_args = args, {}
        if m in {ConformerLayer, BranchformerEncoderLayer, EBranchformerEncoderLayer, ATTDecoder, ViewtoBatch,SwinBlock,SwinStage, }:  # torch.nn.Sequential是一个顺序容器，仅支持单输入和单输出；MultiSequential旨在支持多输入多输出。
            m_ = MultiSequential(*(m(*pos_args, **kw_args) for _ in range(n))) if n > 1 else m(*pos_args, **kw_args)
        else:
            m_ = nn.Sequential(*(m(*pos_args, **kw_args) for _ in range(n))) if n > 1 else m(*pos_args, **kw_args)

        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        if isinstance(f, int):
            f = f + i if f < -1 else f
        else:
            for f_i in range(len(f)):
                f[f_i] = f[f_i] + i if f[f_i] < 0 else f[f_i]
        m_.i, m_.f, m_.type, m_.name = i, f, t, name  # attach index, 'from' index, type, name
        # if verbose:
        #     LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print function is unavailable
        # save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                if x < 0:
                    save.append(i + x)
                else:
                    save.append(x)
        layers.append(m_)

    #############################################################
    # loss function
    #############################################################
    losses = []  # layers, savelist, ch out
    for i_, (f, n, t, m, args) in enumerate(cfg.Loss):  # from, number, type, module, args
        name = m
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        # unit #####################################################
        if m in {
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
        }:
            args = args
        elif m in {
            loss_total,
        }:
            args = [args]
        else:
            raise ValueError("unknown module: " + m)
        # print(*args)
        m_ = MultiSequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        if isinstance(f, int):
            f = f + i + 1 + i_ if f < -1 else f
        else:
            for f_i in range(len(f)):
                f[f_i] = f[f_i] + i + 1 + i_ if f[f_i] < -1 else f[f_i]  # add i_+1
        m_.i, m_.f, m_.type, m_.name = i_, f, t, name  # attach index, 'from' index, type, name
        # save.extend(x % i_ for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                if x < 0:
                    save.append(i + 1 + i_ + x)
                else:
                    save.append(x)
        losses.append(m_)

    return nn.Sequential(*layers), losses, sorted(save)


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.current_epoch = None
        self.max_epoch = cfg.train.max_epochs

        self.odim = cfg.default.odim
        self.save_every_epoch = cfg.train.save_every_epoch
        if cfg.default.weights and os.path.isabs(cfg.default.weights):
            self.weights = cfg.default.weights
        else:
            self.weights = cfg.default.root_dir + '/' + cfg.default.weights if cfg.default.weights else ""
        self.aux_weights = cfg.default.root_dir + '/' + cfg.train.aux_weights if cfg.train.aux_weights else ""

        if cfg.default.model[-5:] == ".yaml":
            self.model, self.criterions, self.save_featensor = pharseModel(self.cfg.model)
        else:
            print("暂不支持直接加载模型文件!!!")

        if cfg.default.save_mid == "NBest_LLM":
            self._ctc_matrix = None
            self.file_dir_path = cfg.default.root_dir + "/output/test--" + cfg.train.model_name + "/save_mid"
            # with open(file_path, "w", encoding="utf-8") as f:
            #     f.write("Let's start save NBest_LLM!!!\n")
            os.mkdir(self.file_dir_path)
        self.src_paths = None

        # 考虑淘汰CTC
        self.aux_list = []
        if cfg.default.mode in ["train", "val"]:
            #self.beam_search = None
            # 存储辅助模型的层数，用于训练时权重加载、冻结和保存
            for i in range(len(self.model)):
                m = self.model[i]
                if "aux_" in m.type:
                    self.aux_list.append(str(i))
            """添加解码部分用于val输出pred和actual*******************************************************************"""
            self.outputName = cfg.model.Output.name
            output_param = cfg.model.Output.output_param  # output_param=lm_model

            # self.text_transform = TextTransform(dict_path=cfg.default.root_dir + '/' + cfg.data.char_path)
            # self.token_list = self.text_transform.token_list

            if self.outputName == "HybrideLipOutput":
                self.text_transform = TextTransform(dict_path=cfg.default.root_dir + '/' + cfg.data.char_path)
                self.token_list = self.text_transform.token_list
                self.beam_search = HybrideLipOutput(output_param, self.token_list, self.model, self.odim)
            elif self.outputName == "whisperDecoder":
                self.beam_search = whisperDecoder(self.model, output_param)
            else:
                print("未识别Decoder!!!")
            """*************************************************************************************************"""
            # 缓存：验证阶段复用encoder特征，避免重复前向
            self._enc_feats_batch = None
        else:
            self.outputName = cfg.model.Output.name
            output_param = cfg.model.Output.output_param  # output_param=lm_model

            # self.text_transform = TextTransform(dict_path=cfg.default.root_dir + '/' + cfg.data.char_path)
            # self.token_list = self.text_transform.token_list

            if self.outputName == "HybrideLipOutput":
                self.text_transform = TextTransform(dict_path=cfg.default.root_dir + '/' + cfg.data.char_path)
                self.token_list = self.text_transform.token_list
                self.beam_search = HybrideLipOutput(output_param, self.token_list, self.model, self.odim)
            elif self.outputName == "whisperDecoder":
                self.beam_search = whisperDecoder(self.model, output_param)
            else:
                print("未识别Decoder!!!")
            # 缓存：验证阶段复用encoder特征，避免重复前向
            self._enc_feats_batch = None
    def reset_ctc_weight(self, ctc_weight):
        self.cfg.model.Output.output_param.ctc_weight=ctc_weight
        self.beam_search = HybrideLipOutput(self.cfg.model.Output.output_param, self.token_list, self.model, self.odim)

    def set_epoch(self, epoch: int):
        """在每个训练 epoch 开始时调用，更新当前 epoch"""
        self.current_epoch = epoch
        decay = os.environ.get("EAAI_AUX_DECAY", "").strip().lower()
        if decay not in ("", "off", "none"):
            if decay == "att":
                weights = [0.0, 0.0, 0.9, 0.1]
            elif decay == "2":
                if epoch <= 0:
                    weights = [0.15, 0.05, 0.72, 0.08]
                elif epoch == 1:
                    weights = [0.075, 0.025, 0.81, 0.09]
                else:
                    weights = [0.0, 0.0, 0.9, 0.1]
            else:
                raise ValueError("EAAI_AUX_DECAY must be off, att, or 2")
            holders = list(self.modules())
            criterions = getattr(self, "criterions", None)
            if criterions is not None:
                holders.extend(list(criterions))
            applied = 0
            for module in holders:
                if module.__class__.__name__ == "loss_total" and len(list(getattr(module, "weights", []))) == 4:
                    module._epoch_weights = [float(w) for w in weights]
                    applied += 1
            print(
                "EAAI_AUX_DECAY epoch %d weights id/corr/att/ctc %s applied %d"
                % (epoch, weights, applied),
                flush=True,
            )
        per = int(os.environ.get("EAAI_EXPERT_EPOCHS", "0") or "0")
        if per > 0:
            active = None
            for module in self.modules():
                if module.__class__.__name__ == "LoRAMoEPaper":
                    module._stage = None
                    if epoch < per:
                        module._active = (0,)
                    else:
                        module._active = tuple(range(1, module.num_experts))
                    active = module._active
            print(
                "EAAI_EXPERT_EPOCHS epoch %d active_experts %s" % (epoch, active),
                flush=True,
            )
            return
        stage1 = int(os.environ.get("EAAI_STAGE1_EPOCHS", "0") or "0")
        if stage1 <= 0:
            return
        stage = 0 if epoch < stage1 else 1
        for module in self.modules():
            if module.__class__.__name__ == "LoRAMoEPaper":
                module._stage = stage

    def origin_forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.
        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.
        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def forward(self, x):
        # 判断是否是验证集batch
        """添加输出pred和actual的逻辑*******************************************************************"""
        if isinstance(x, dict) and "targets" in x and self.cfg.default.mode in ("val", "eval"):
            # Standalone eval only needs hypotheses. Skip teacher-forced losses.
            if self.cfg.default.mode == "eval":
                enc_feats = self._encode_batch_for_decode(x)
                losses = {"loss_total": x["targets"].new_zeros(())}
            else:
                # 1) compute batch loss once (batched encoder forward)
                losses = self.loss(x)

                # 2) 复用 loss() 过程中缓存的 encoder 输出，避免第二次encoder前向
                if isinstance(getattr(self, "_enc_feats_batch", None), torch.Tensor):
                    enc_feats_tensor = self._enc_feats_batch
                    enc_feats = [enc_feats_tensor[i].detach() for i in range(enc_feats_tensor.shape[0])]
                else:
                    enc_feats = self._encode_batch_for_decode(x)

            preds = []
            actuals = []
            # prepare ground-truth strings in batch
            if self.outputName == "HybrideLipOutput":
                for i in range(x["targets"].shape[0]):
                    token_id = x["targets"][i]
                    actual = self.text_transform.post_process(token_id)
                    actuals.append(actual)
            else:
                # fallback: keep original behavior if needed
                for i in range(x["targets"].shape[0]):
                    token_id = x["targets"][i]
                    actuals.append(token_id)

            # decode with multi-sample batched beam search
            # import time
            # time_now = time.time()
            if self.outputName == "HybrideLipOutput":
                xs = torch.stack(enc_feats, dim=0)  # [B, T, C]
                # 保存N-best候选，用于纠错微调
                if self.cfg.default.save_mid == "NBest_LLM":
                    best_list, n_best_list = self.beam_search.forward_batch(xs, True)
                    save_mid_src_paths = [self.src_paths[int(id.item())] for id in x['indices']]
                    save_NBbest_to_txt(self.text_transform, self.file_dir_path, save_mid_src_paths, self._ctc_matrix, n_best_list, actuals)
                else:
                    best_list = self.beam_search.forward_batch(xs)

                for best in best_list:
                    if best is None or best.yseq is None or best.yseq.numel() == 0:
                        preds.append("")
                        continue
                    predicted_token_id = torch.tensor(list(map(int, best.yseq.tolist()[1:])), device=xs.device)
                    pred = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
                    preds.append(pred)
            elif self.outputName == "whisperDecoder":
                # keep existing whisper path: expects batched features (model-specific)
                preds = self.beam_search.run(enc_feats)
            else:
                print("未识别Decoder!!!")
            # print("decoder: ", 1000*(time.time()-time_now))
            return losses, preds, actuals
        elif isinstance(x, dict) and "targets" in x:
            return self.loss(x)
        else:
            return self.predict(x)
        """*****************************************************************************************"""

    def _encode_batch_for_decode(self, batch):
        """Run the network up to encoder outputs in batch and return per-sample features.

        Returns a list of encoder features: each tensor [T_i, C].
        """
        label = batch["targets"]
        Batch_size = batch["targets"].shape[0]

        y = []  # outputs cache when needed
        x = None
        for i in range(len(self.model)):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                # work on copies to avoid in-place side effects
                x = x.copy() if isinstance(x, dict) else x
            if m.type.replace('aux_', '') == "input":
                x = m(batch)
            elif m.name == "ViewtoBatch":
                out = m(x["inputs"], Batch_size)
                x["inputs"] = out
            elif m.name == "ConvDictBlock":
                out, _ = m(x["inputs"])
                x["inputs"] = out
            elif m.name in {"ConformerLayer", "BranchformerEncoderLayer", "EBranchformerEncoderLayer"}:
                out, _ = m(x["inputs"], x["mask"])
                x["inputs"] = out
            elif m.name == "LayerNorm":
                if isinstance(x["inputs"], tuple):
                    out = x["inputs"][0]
                else:
                    out = x["inputs"]
                out = m(out)
                x["inputs"] = out
            elif m.name == "featurefusionmodule":
                x_a, x_v = audio_video_pad(x[0]["inputs"], x[1]["inputs"])
                out = m(x_a, x_v)
                x = x[0]
                x["inputs"] = out
            elif m.name in {"ATTDecoder", "TextDecoder"}:
                # stop before decoders; we only need encoder features
                break
            elif m.name in {"CTCDecoder"}:
                out = m(x["inputs"])  # T×B×C
                if self.cfg.default.save_mid == "NBest_LLM":
                    self._ctc_matrix = out
                x["inputs"] = out
            elif m.name == "SwinStage":
                out = m(*x["inputs"])
                x["inputs"] = out[:2]
            elif m.name == "PatchMerging":
                out = m(*x["inputs"])
                x["inputs"] = out
            elif m.name == "SpatialAvgPooling":
                out = m(x["inputs"])  # [B*T, C]
                T = x["mask"].shape[-1]
                B = out.shape[0] // T
                out = out.view(B, T, -1)  # [B, T, C]
                x["inputs"] = out
            elif m.name == "PatchEmbed3D":
                out = m(x["inputs"])
                N = out.shape[2]
                T = x["inputs"].shape[2]
                import math
                patch_size = math.ceil(T / N)
                pad_len = N * patch_size - T
                if pad_len > 0:
                    pad = torch.zeros((x["mask"].shape[0], x["mask"].shape[1], pad_len), dtype=x["mask"].dtype, device=x["mask"].device)
                    mask_padded = torch.cat([x["mask"], pad], dim=2)
                else:
                    mask_padded = x["mask"]
                mask_patches = mask_padded.reshape(x["mask"].shape[0], 1, N, patch_size)
                patch_mask = mask_patches.all(dim=-1)  # [B, 1, N]
                x["mask"] = patch_mask
                x["inputs"] = out
                x["lengths"] = torch.ceil(x["lengths"] / patch_size).to(x["lengths"].dtype)
            else:
                if isinstance(x["inputs"], tuple):
                    out = m(x["inputs"][0])
                else:
                    out = m(x["inputs"])  # run
                x["inputs"] = out
            y.append(x.copy() if isinstance(x, dict) and (m.i in self.save_featensor) else None)

        # x["inputs"] is [B, T, C]; split into per-sample [T, C]
        feats = [x["inputs"][i].detach() for i in range(x["inputs"].shape[0])]
        return feats

    def predict(self, x):
        """
        Perform a forward pass through the network.
        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.
        Returns:
            (torch.Tensor): The last output of the model.
        """
        if isinstance(x, dict):
            return self._predict_eval(x)  # 后续考虑采用batch求解
        else:
            return self._predict_once(x)

    def _predict_once(self, input):
        """Perform augmentations on input image x and return augmented inference."""
        sample = {"input": input}
        Batch_size = 1
        mask = None
        y = []  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if m.type == "input":
                x = m(sample)
            elif m.name == "ViewtoBatch":
                x = m(x, Batch_size)
            elif m.name == "ConvDictBlock":
                x, _ = m(x)
            elif m.name in {"ConformerLayer", "BranchformerEncoderLayer", "EBranchformerEncoderLayer"}:
                x, _ = m(x, mask)
            elif m.name == "LayerNorm":
                if isinstance(x, tuple):
                    x = x[0]
                x = m(x)
            elif m.type in ["decoder_", "decoder", "loss"]:
                x = x
            elif m.name == "SwinStage":
                out = m(*x)
                x = out[:2]
            elif m.name == "PatchMerging":
                out = m(*x)
                x = out
            elif m.name == "SpatialAvgPooling":
                out = m(x)  # [B*T, 512]
                # B 在你的函数里是 1. T 是时间/序列长度.
                T = out.shape[0] // Batch_size  # 自动计算T
                # 将形状从 (B*T, C) 恢复为 (B, T, C)
                # 这是为后续的时序建模层做准备
                out = out.view(Batch_size, T, -1)
                x = out
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save_featensor else None)  # save output

        if self.outputName == "HybrideLipOutput":
            enc_feat = x.squeeze(0)
            nbest_hyps = self.beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            predicts = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        elif self.outputName == "whisperDecoder":
            enc_feat = x
            predicts = self.beam_search.run(enc_feat)
        else:
            print("未识别Decoder!!!")
        return predicts

    def _predict_eval(self, sample):
        """Perform augmentations on input image x and return augmented inference."""
        # x = sample["input"].unsqueeze(0)
        token_id = sample["target"]
        actuals = self.text_transform.post_process(token_id)
        Batch_size = 1
        mask = None
        y = []  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if m.type == "input":
                x = m(sample)
            elif m.name == "ViewtoBatch":
                x = m(x, Batch_size)
            elif m.name == "ConvDictBlock":
                x, _ = m(x)
            elif m.name in {"ConformerLayer", "BranchformerEncoderLayer", "EBranchformerEncoderLayer"}:
                x, _ = m(x, mask)
            elif m.name == "LayerNorm":
                if isinstance(x, tuple):
                    x = x[0]
                x = m(x)
            elif m.type in ["decoder_", "decoder", "loss"]:
                x = x
            elif m.name == "SwinStage":
                out = m(*x)
                x = out[:2]
            elif m.name == "PatchMerging":
                out = m(*x)
                x = out
            elif m.name == "SpatialAvgPooling":
                out = m(x)  # [B*T, 512]
                T = out.shape[0]  # 时间长度
                B = 1    #  batch_size固定为1
                x = out.view(B, T, -1)  # [B, T, 512]
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save_featensor else None)  # save output

        if self.outputName == "HybrideLipOutput":
            enc_feat = x.squeeze(0)
            nbest_hyps = self.beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            predicts = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        elif self.outputName == "whisperDecoder":
            enc_feat = x
            predicts = self.beam_search.run(enc_feat)
        else:
            print("未识别Decoder!!!")

        return predicts, actuals

    def info(self, input_data=None):
        if input_data == None:
            input_data = torch.randn(1, 77, 1, 88, 88)
        cfg = self.cfg
        layers, save = [], []  # layers, savelist, ch out
        for i, (f, n, t, m, args) in enumerate(cfg.Encoder + cfg.Decoder):  # from, number, type, module, args
            name = m
            # m = importlib.import_module(m)  # get module
            m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
            print(m)

        raise NotImplementedError("model_info() needs to be implemented by model pharse")


    def reload(self,path,freeze=""):
        if path!=self.weights:
            self.weights = path
            self.load(freeze=freeze)

    def load(self, freeze=""):
        '''
        Args:
            weights: path or weights
            aux_weights: path or weights
        '''

        # 加载模型权重(基于self.aux_list，避开aux层)
        # 加载aux权重
        # 冻结aux权重和指定model权重

        # 获取权重（state_dict）
        print(f"loading weight from {self.weights}")
        if self.weights:
            weights = torch.load(self.weights,map_location="cpu")
            isFlag = False
            for key in weights:
                if 'state_dict' in key or 'model' == key:
                    isFlag = key
                    break
            weights = weights[isFlag] if isFlag else weights
            keys_model = [key for key in weights]
        # 获取aux_weights
        if self.aux_weights:
            aux_weights = torch.load(self.aux_weights)
            isFlag = False
            for key in aux_weights:
                if 'state_dict' in key or 'model' == key:
                    isFlag = key
                    break
            aux_weights = aux_weights[isFlag] if isFlag else aux_weights
            keys_auxmodel = [key for key in aux_weights]

        # 构建临时模型权重，用于加载
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        # 一层一层给临时模型权重赋值
        i_model = 0
        i_auxmodel = 0
        for key in state_dict:
            if key.split(".")[0] in self.aux_list:  # 如果是aux层
                if i_auxmodel >= len(list(aux_weights.keys())):
                    print(f'auxmodel超出索引{i_auxmodel}！！！！！！！！！！！！！！！！')
                    break
                key_auxmodel = list(aux_weights.keys())[i_auxmodel]
                if state_dict[key].size() == aux_weights[key_auxmodel].size():
                    state_dict[key] = aux_weights[key_auxmodel]
                else:
                    print(f'第【{key}】层权重尺寸不对！！！！！！！！！！！！！！！！')
                i_auxmodel = i_auxmodel + 1
            else:
                if i_model >= len(list(weights.keys())):
                    print(f'model超出索引{i_model}！！！！！！！！！！！！！！！！')
                    break
                key_model = list(weights.keys())[i_model]
                if state_dict[key].size() == weights[key_model].size():
                    state_dict[key] = weights[key_model]
                else:
                    print(f'expect:{state_dict[key].size()},get:{weights[key_model].size()}')
                    print(f'第【{key}】层权重尺寸不对！！！！！！！！！！！！！！！！')
                i_model = i_model + 1

        # 冻结权重
        # freeze_list = []
        # if self.aux_list:
        #     for layer in self.aux_list:
        #         freeze_list.append(str(layer))
        # if freeze:
        #     if isinstance(freeze, list):
        #         for x in freeze:
        #             freeze_list.append(str(x))
        #     else:
        #         for x in range(freeze[0], freeze[1]):
        #             freeze_list.append(str(x))
        # freeze_list = list(set(freeze_list))
        freeze_list = {}
        if self.aux_list:
            for layer in self.aux_list:
                freeze_list[str(layer)] = 'all'
        if freeze:
            for layer in freeze:
                freeze_list[str(layer[0])] = layer[1]
        for name, module in self.model.named_modules():
            if name.split('.')[0] in freeze_list:
                for param in module.parameters():
                    param.requires_grad = False

        self.model.load_state_dict(state_dict)

    def loss(self, batch):
        """
        Compute loss.
        Args:
            batch (dict): Batch to compute loss on
        """
        label = batch["targets"]
        Batch_size = batch["targets"].shape[0]
        losses = {"loss_total": 0}  # outputs
        # 重置缓存（按batch）
        self._enc_feats_batch = None
        self._id_videos = batch.get("videos") if isinstance(batch, dict) else None
        self._id_lengths = batch.get("video_lengths") if isinstance(batch, dict) else None
        for module in self.modules():
            if getattr(module, "_record_id", False):
                module._id_features = None

        # import time
        y = []  # outputs
        for i in range(len(self.model)):
            # start = time.time()
            m = self.model[i]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = x.copy()  # 避免修改y
            if m.type.replace('aux_', '') == "input":
                x = m(batch)
            elif m.name == "ViewtoBatch":
                out = m(x["inputs"], Batch_size)
                x["inputs"] = out
            elif m.name == "ConvDictBlock":
                out, rc_loss = m(x["inputs"])
                x["inputs"] = out
                losses["loss_CSC_r"] = losses["loss_CSC_r"] + rc_loss[0] if "loss_CSC_r" in losses else rc_loss[0]
                losses["loss_CSC_c"] = losses["loss_CSC_c"] + rc_loss[1] if "loss_CSC_c" in losses else rc_loss[1]
            elif m.name in {"ConformerLayer", "BranchformerEncoderLayer", "EBranchformerEncoderLayer"}:
                out, _ = m(x["inputs"], x["mask"])
                x["inputs"] = out
            elif m.name == "LayerNorm":
                if isinstance(x["inputs"], tuple):
                    out = x["inputs"][0]
                out = m(out)
                x["inputs"] = out
            elif m.name == "featurefusionmodule":
                x_a, x_v = audio_video_pad(x[0]["inputs"], x[1]["inputs"])
                out = m(x_a, x_v)
                x = x[0]
                x["inputs"] = out
            elif m.name == "AddWithDecayEpoch":
                # x_v, x_a = align_to_ref(x[0]["inputs"], x[1]["inputs"])
                # out = m(x_v, x_a, self.current_epoch)
                out = m(x[0]["inputs"], x[1]["inputs"], self.current_epoch)
                x = x[0]
                x["inputs"] = out
            elif m.name in {"CTCDecoder"}:
                out = m(x["inputs"])  # T×B×C
                if self.cfg.default.save_mid == "NBest_LLM":
                    self._ctc_matrix = out
                x["inputs"] = out
            elif m.name == "ATTDecoder":
                from ldw_nets.utils import add_sos_eos, target_mask
                ys_in_pad, ys_out_pad = add_sos_eos(label, self.cfg.default.sos, self.cfg.default.eos, -1)
                ys_mask = target_mask(ys_in_pad, -1)
                # 在进入解码器前缓存encoder输出供验证解码复用
                if self.cfg.default.mode == "val" and isinstance(x, dict) and "inputs" in x:
                    self._enc_feats_batch = x["inputs"].detach()
                out, _ = m(ys_in_pad, ys_mask, x["inputs"], x["mask"])
                # pred_pad = out
                x["inputs"] = out
            elif m.name == "TextDecoder":
                from ldw_nets.utils import add_sot_eot
                label_token = add_sot_eot(label, self.cfg.default.sot, self.cfg.default.eot).to(x["inputs"].device)
                if self.cfg.default.mode == "val" and isinstance(x, dict) and "inputs" in x:
                    self._enc_feats_batch = x["inputs"].detach()
                out = m(label_token[:, :-1], x["inputs"])
                x["inputs"] = out
            elif m.name == "SwinStage":
                out = m(*x["inputs"])
                x["inputs"] = out[:2]
            elif m.name == "PatchMerging":
                out = m(*x["inputs"])
                x["inputs"] = out
            elif m.name == "SpatialAvgPooling":
                out = m(x["inputs"])  # [B*T, 512]
                #T = x["mask"].shape[-1]  # 时间长度
                #B = out.shape[0] // T    # 自动推算 batch_size
                #out = out.view(B, T, -1)  # [B, T, 512]
                x["inputs"] = out
            elif m.name == "PatchEmbed3D":
                out = m(x["inputs"])
                N = out.shape[2]  # patch数量=帧数/patch时间维上的长度
                T = x["inputs"].shape[1]  # 原始帧数
                import math
                patch_size = math.ceil(T / N)  # patch在时间维上的长度
                pad_len = N * patch_size - T
                if pad_len > 0:
                    # 无效帧用False填充
                    pad = torch.zeros((x["mask"].shape[0], x["mask"].shape[1], pad_len), dtype=x["mask"].dtype, device=x["mask"].device)
                    mask_padded = torch.cat([x["mask"], pad], dim=2)
                else:
                    mask_padded = x["mask"]
                # 将 mask 按 patch_size 分组，只有组内所有帧都有效才算有效patch
                mask_patches = mask_padded.reshape(x["mask"].shape[0], 1, N, patch_size)
                #print(mask_patches.shape)
                patch_mask = mask_patches.all(dim=-1)  # [B, 1, N]
                x["mask"] = patch_mask
                x["inputs"] = out
                x["lengths"] = torch.ceil(x["lengths"] / patch_size).to(x["lengths"].dtype)
            else:
                if isinstance(x["inputs"], tuple):
                    out = m(x["inputs"][0])
                else:
                    out = m(x["inputs"])  # run
                x["inputs"] = out
            # print(f"{m.name} run speed: {1000*(time.time()-start)}")
            y.append(x.copy() if m.i in self.save_featensor else None)  # save output

        # losses = {"loss_total": 0}  # outputs
        for j in range(len(self.criterions)):
            m = self.criterions[j]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(x, tuple):
                    x = x[0]
            if m.name in ["loss_ctc", "loss_Focal_ctc"]:
                # m = m.cuda(x.device)  # 创建CTC时要根据参数选择device，模型创建后执行model.cuda(0)，因此这里需要将其给到GPU
                x = m(x["inputs"], x["lengths"], label)
                losses["loss_ctc"] = x
            elif m.name in ["loss_att", "loss_Focal_att"]:
                x = m(x["inputs"], ys_out_pad)
                losses["loss_att"] = x
            elif m.name == "loss_CrossEntropy":
                x = m(x["inputs"], label_token[:, 1:])
                losses["loss_CrossEntropy"] = x
            elif m.name == "loss_rc":
                x = m(losses["loss_CSC_r"], losses["loss_CSC_c"])
                losses["loss_rc"] = x
            elif m.name == "loss_corr":
                x = m(self.model)
                losses["loss_corr"] = x
            elif m.name == "loss_id":
                x = m(self)
                losses["loss_id"] = x
            elif m.name == "loss_total":
                losses["loss_total"] = m(x)
            elif m.name in ["loss_L1", "loss_relation", "loss_Cosine"]:
                x_ = x[0]["inputs"][0] if isinstance(x[0]["inputs"], tuple) else x[0]["inputs"]
                x_aux = x[1]["inputs"][0] if isinstance(x[1]["inputs"], tuple) else x[1]["inputs"]
                x = x_
                x_aux, x = audio_video_pad(x_aux, x)
                x = m(x, x_aux)
                losses[m.name] = x
            elif m.name in ["loss_MPAD"]:
                v = x[0]["inputs"][0] if isinstance(x[0]["inputs"], tuple) else x[0]["inputs"]
                vs = [v]
                for t_i in range(1, len(x)):
                    t = x[t_i]["inputs"][0] if isinstance(x[t_i]["inputs"], tuple) else x[t_i]["inputs"]
                    _, t = align_to_ref(v, t)
                    vs.append(t)
                x = m(vs)
                losses[m.name] = x
            else:
                logging.info("Loss ERROR: unkown loss function!!!!")
            y.append(x if m.i + 1 + i in self.save_featensor else None)  # save output
        # acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=-1)
        if losses["loss_total"] == 0:
            for key in losses:
                losses["loss_total"] = losses[key] if key != "loss_total" else losses["loss_total"]
        if "loss_ctc" not in losses:
            losses["loss_att"] = torch.tensor(0)
            losses["loss_ctc"] = torch.tensor(0)
        return losses  # , acc

    def save(self, path, current_epoch, strategy=None, lora=False):
        """
        Compute loss.
        Args:
            epoch (int): epoch to save model
        """
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict().copy()
        else:
            state_dict = self.model.state_dict().copy()

        for key, _ in list(state_dict.items()):  # 使用list()创建items()的副本
            if key.split(".")[0] in self.aux_list:
                del state_dict[key]

        data = {'epoch': current_epoch, 'state_dict': state_dict}
        # lora存储：只需要保存LoRA权重即可
        # if strategy.name == "lora":
            # data['state_dict'] = lora_state_dict(self.model)
        if lora:
            from collections import OrderedDict
            data['state_dict'] = OrderedDict((k, v.cpu()) for k, v in self.model.state_dict().items()if "lora_" in k)

        if current_epoch % self.save_every_epoch == 0:
            torch.save(data, path + "/model_" + str(current_epoch) + ".pth")
        # torch.save(data, path + "/model_last.pth")
