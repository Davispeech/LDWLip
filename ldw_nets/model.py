import os
import sys
import ast
import thop
import logging
import contextlib

from ldw_datas.transforms import TextTransform
from ldw_nets.utils import MultiSequential, make_non_pad_mask, th_accuracy, audio_video_pad

import torch
import torch.nn as nn
from ldw_nets.nn import (
    # input ####################################################
    Input_video2imgSingle,
    Input_video,
    Input_audio,
    Input_audio_mel,
    Input_label,
    # units ####################################################
    Swish,
    View,
    ViewtoBatch,
    Concat,
    Transpose,
    LayerNorm,
    BatchNorm3d,
    Dropout3d,
    GlobalPool,
    MaxPool,
    AvgPool,
    AvgPool1d,
    AvgPool3d,
    AdaptiveAvgPool3d,
    tensor3Dto2D,
    nnLinear,
    BiGRU,
    GramMatrix_self,
    GramMatrix_co,
    # block ####################################################
    HOGLayer,
    HOGFeatureExtractor,
    Conv,
    Conv3D,
    Conv3DBlock,
    Conv3Dfrontend,
    SpatioTemporalConv,
    SlowFast,
    SlowFastLite,
    DownsampleCB,  # downsample_basic_block
    DownsampleCB_1D,
    ResidualBlock,
    ResidualBlock131,
    ResidualBlock_1D,
    ResidualBlock131_1D,
    ConvDictBlock,
    ResNetBlockLayer,
    featurefusionmodule,
    PositionwiseFeedForward,
    PositionalEncoding,
    RelPositionalEncoding,
    embed,
    SwinTransformerBlockR,  # Visual Transformer Block
    # module ####################################################
    ConformerLayer,
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
    TSNetwork,
    TCN,
    MTCN,
    DenseTCN,
    MobileNetV2,
    MobileNetV3Large,
    ShuffleNetV2,
    ShuffleNetV2lrw,
    SegFormerEncoder,
    EfficientNet,
    EfficientNetLite0,
    ResNet18,
    SqueezeNet,
    MobileNetV2_035,
    R2Plus1D10,
    SqueezeTime,
    MobileViCLIP,
    Mobile3DNet,
    X3DNet_M,
    TimeSformerLite,
    TimeSformerTiny,
    Lip3DA2D,
    FlowNet,
    C3DNet,
    LipNet,
    Whisper,
    SNet,
    CNet,
    DNet,
    LipAuth_Conv3DFront,
    LipAuth_Conv2DBackbone,
    LipAuth_SeqEncoder,
    # decoder ####################################################
    HybrideLipOutput,
    CTCDecoder,
    ATTDecoder,
    whisperDecoder,
    LipAuthOutput,
    ClassifierOutput,
    # loss ####################################################
    loss_total,
    LossRelation,
    nnCrossEntropy,
    nnFocalLoss,
    ArcFaceLoss,
    loss_ctc,
    loss_att,
    loss_CrossEntropy,
    loss_SoftLabelCE,
    loss_KLDiv,
    loss_CosineSimilarity,
    loss_MSE,
    loss_L1,
    loss_rc,
)


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
            Input_video2imgSingle,
            Input_video,
            Input_audio,
            Input_audio_mel,
            Input_label,
            # units ####################################################
            Swish,
            View,
            ViewtoBatch,
            Concat,
            Transpose,
            LayerNorm,
            BatchNorm3d,
            Dropout3d,
            GlobalPool,
            MaxPool,
            AvgPool,
            AvgPool1d,
            AvgPool3d,
            AdaptiveAvgPool3d,
            tensor3Dto2D,
            nnLinear,
            BiGRU,
            GramMatrix_self,
            GramMatrix_co,
            # block ####################################################
            HOGLayer,
            HOGFeatureExtractor,
            Conv,
            Conv3D,
            Conv3DBlock,
            Conv3Dfrontend,
            SpatioTemporalConv,
            DownsampleCB,  # downsample_basic_block
            DownsampleCB_1D,
            ResidualBlock,
            ResidualBlock131,
            ResidualBlock_1D,
            ResidualBlock131_1D,
            ConvDictBlock,
            ResNetBlockLayer,
            featurefusionmodule,
            PositionwiseFeedForward,
            PositionalEncoding,
            RelPositionalEncoding,
            embed,
            SwinTransformerBlockR,  # Visual Transformer Block
            # module ####################################################
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
            Lip3DA2D,
            TSNetwork,
            TCN,
            MTCN,
            DenseTCN,
            MobileNetV2,
            MobileNetV3Large,
            ShuffleNetV2,
            ShuffleNetV2lrw,
            SegFormerEncoder,
            EfficientNet,
            EfficientNetLite0,
            ResNet18,
            SqueezeNet,
            MobileNetV2_035,
            R2Plus1D10,
            SqueezeTime,
            MobileViCLIP,
            Mobile3DNet,
            SlowFast,
            SlowFastLite,
            X3DNet_M,
            TimeSformerLite,
            TimeSformerTiny,
            FlowNet,
            C3DNet,
            LipNet,
            Whisper,
            SNet,
            CNet,
            DNet,
            LipAuth_Conv3DFront,
            LipAuth_Conv2DBackbone,
            LipAuth_SeqEncoder,
            # decoder ####################################################
            CTCDecoder,
        }:
            args = args
        # module #####################################################
        elif m in {ConformerLayer}:  # Para: adim, aheads, eunits, dropout_rate
            attention_dim = args[0] if args[0] else 256
            attention_heads = args[1] if args[1] else 4
            linear_units = args[2] if args[2] else 2048
            dropout_rate = args[3] if args[3] else 0.1
            encoder_attn_layer_type = args[4] if args[4] else "rel_mha"
            args = [attention_dim, attention_heads, linear_units, dropout_rate, encoder_attn_layer_type,
                31, True, True, False, True, False,
            ]
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
        if m in {ConformerLayer, ATTDecoder, ViewtoBatch}:  # torch.nn.Sequential是一个顺序容器，仅支持单输入和单输出；MultiSequential旨在支持多输入多输出。
            m_ = MultiSequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        if isinstance(f, int):
            f = f+i if f<-1 else f
        else:
            for f_i in range(len(f)):
                f[f_i] = f[f_i]+i if f[f_i]<0 else f[f_i]
        m_.i, m_.f, m_.type, m_.name = i, f, t, name  # attach index, 'from' index, type, name
        # if verbose:
        #     LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print function is unavailable
        # save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                if x<0:
                    save.append(i+x)
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
            LossRelation,
            nnCrossEntropy,
            nnFocalLoss,
            ArcFaceLoss,
            loss_CrossEntropy,
            loss_SoftLabelCE,
            loss_KLDiv,
            loss_CosineSimilarity,
            loss_MSE,
            loss_L1,
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
            f = f+i+1+i_ if f<-1 else f
        else:
            for f_i in range(len(f)):
                f[f_i] = f[f_i]+i+1+i_ if f[f_i]<-1 else f[f_i]  # add i_+1
        m_.i, m_.f, m_.type, m_.name = i_, f, t, name  # attach index, 'from' index, type, name
        # save.extend(x % i_ for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                if x < 0:
                    save.append(i+1+i_+x)
                else:
                    save.append(x)
        losses.append(m_)

    return nn.Sequential(*layers), losses, sorted(save)


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.save_every_epoch = cfg.train.save_every_epoch
        self.weights = cfg.default.root_dir+'/'+cfg.default.weights if cfg.default.weights else ""
        self.aux_weights = cfg.default.root_dir+'/'+cfg.train.aux_weights if cfg.train.aux_weights else ""

        if cfg.default.model[-5:] == ".yaml":
            self.model, self.criterions, self.save_featensor = pharseModel(self.cfg.model)
        else:
            print("暂不支持直接加载模型文件!!!")

        self.saveFeatureFLAG = False
        if cfg.default.task in ["vsr", "asr", "avsr"]:
            self.odim = cfg.default.odim

        # 考虑淘汰CTC
        self.aux_list = []
        if cfg.default.mode in ["train", "val"]:
            self.outdecoder = None
            # 存储辅助模型的层数，用于训练时权重加载、冻结和保存
            for i in range(len(self.model)):
                m = self.model[i]
                if "aux_" in m.type:
                    self.aux_list.append(str(i))
        else:
            self.outputName = cfg.model.Output.name
            output_param = cfg.model.Output.output_param  # output_param=lm_model

            if self.outputName == "HybrideLipOutput":
                dict_path = cfg.default.root_dir + '/' + cfg.data.char_path
                if cfg.data.char_path[-4:] == ".txt":
                    self.text_transform = TextTransform(dict_path=dict_path)
                    self.token_list = self.text_transform.token_list
                else:
                    from ldw_datas.load_tiktoken import get_encoding
                    self.tiktokenDecoder = get_encoding(dict_path, 99)
                    self.token_list = range(self.tiktokenDecoder.n_vocab)  # n_vocab  51865
                    self.token_list = list(map(str, self.token_list))
                self.outdecoder = HybrideLipOutput(output_param, self.token_list, self.model, self.odim)
            elif self.outputName == "whisperDecoder":
                self.outdecoder = whisperDecoder(self.model, output_param)
            elif self.outputName == "LipAuthOutput":
                if cfg.default.mode == "register":
                    # 存储辅助模型的层数，用于训练时权重加载、冻结和保存
                    for i in range(len(self.model)):
                        m = self.model[i]
                        if "aux_" in m.type:
                            self.aux_list.append(str(i))
                # model, support_path: str, id_list: List[str], threshold: float = 0.6
                self.saveFeatureFLAG = True
                # [[3,ctc,0.75],[5,cos,0.75]]  # 层数，相似度类型，阈值
                self.save_layers = [int(x[0]) for x in output_param.save_layerFeatures]
                support_path = cfg.default.root_dir+'/'+output_param.support
                id_list = output_param.ids_user
                self.outdecoder = LipAuthOutput(cfg.default.root_dir, support_path, id_list, output_param.save_layerFeatures)
            elif self.outputName == "ClassifierOutput":
                self.outdecoder = ClassifierOutput()
            else:
                print("未识别Decoder!!!")

    def forward(self, x, *args, **kwargs):
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

    def predict(self, x):
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
            elif m.name == "ConformerLayer":
                x = (x, x_embed)
                x, _ = m(x, mask)
            elif m.type in ["decoder_", "loss"]:
                x = x
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save_featensor else None)  # save output
            if isinstance(x, tuple):
                x_embed = x[1]
                x = x[0]

        if self.outputName == "HybrideLipOutput":
            enc_feat = x.squeeze(0)
            nbest_hyps = self.outdecoder(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            import torch
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            # predicts = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
            if self.cfg.data.char_path[-4:] == ".txt":
                predicts = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")  # tensor([1183, 1862,  769,  441, 1144, 1961,  802, 3362])
            else:
                predicts = self.tiktokenDecoder.decode(predicted_token_id)

        elif self.outputName == "whisperDecoder":
            enc_feat = x

            # tensor = x
            # import matplotlib
            # matplotlib.use('Agg')
            # import matplotlib.pyplot as plt
            # import torch
            # # 预处理张量
            # tensor = tensor.squeeze(0)  # 移除 batch 维度，使其变成 (77, 384)
            # tensor = tensor.detach().cpu().numpy()
            # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)  # 避免除零错误
            # # 设置 colormap
            # cmap = "viridis"
            # # 绘制热力图
            # fig, ax = plt.subplots(figsize=(16, 5))  # 调整 figsize 以适应 (77, 384) 的长宽比
            # heatmap = ax.imshow(tensor, aspect="auto", cmap=cmap)
            # # 添加颜色条
            # cbar = plt.colorbar(heatmap, ax=ax)
            # cbar.set_label("Feature Intensity", fontsize=30)
            # # 设置轴标签和标题
            # ax.set_xlabel("Feature Dimension (384)", fontsize=30)
            # ax.set_ylabel("Time Step (77)", fontsize=30)
            # ax.set_title("Feature Tensor Heatmap", fontsize=32)
            # # 设置刻度字体大小
            # ax.tick_params(axis='both', labelsize=28)
            # cbar.ax.tick_params(labelsize=28)  # 颜色条刻度字体
            # # 保存图片，不显示
            # plt.savefig("./heat.png", bbox_inches="tight")
            # plt.close(fig)  # 关闭图像，释放内存

            # import time
            # start = time.time()
            predicts = self.outdecoder.run(enc_feat)
            # print(time.time()-start)
        else:
            predicts = None
            print("未识别Decoder!!!")
        return predicts

    def _predict_eval(self, sample):
        """Perform augmentations on input image x and return augmented inference."""
        # x = sample["input"].unsqueeze(0)
        if "target" in sample:
            token_id = sample["target"]
        else:
            token_id = sample["target_0_"]
        if 'video' in sample:
            lengths = torch.tensor([sample["video"].shape[0]])

        Batch_size = 1
        mask = None
        y = []  # outputs
        # if self.saveFeatureFLAG:
        #     save_layerFeaturesTensor = {}
        for i in range(len(self.model)):
            m = self.model[i]
            if 'aux_' in m.type:
                y.append(None)  # 添加None保障占位
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if m.type == "input":
                x = m(sample)
            elif m.name == "ViewtoBatch":
                x = m(x, Batch_size)
            elif m.name == "ConvDictBlock":
                x, _ = m(x)
            elif m.name == "ConformerLayer":
                x, _ = m(x, mask)
            elif m.name in ["MTCN", "DenseTCN"]:
                out = m(x, lengths, Batch_size)
                x = out
            elif m.name in ["LayerNorm", "nnLinear"]:
                if isinstance(x, tuple):
                    x = x[0]
                x = m(x)
            elif m.type in ["decoder_", "loss"]:
                x = x
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save_featensor else None)  # save output
            # if self.saveFeatureFLAG:
            #     if i in self.save_layers:
            #         save_layerFeaturesTensor[i] = x

        if self.outputName == "HybrideLipOutput":
            enc_feat = x.squeeze(0)
            nbest_hyps = self.outdecoder(enc_feat)

            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            if self.cfg.data.char_path[-4:] == ".txt":
                actuals = self.text_transform.post_process(token_id)
                predicts = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")  # tensor([1183, 1862,  769,  441, 1144, 1961,  802, 3362])
            else:
                actuals = self.tiktokenDecoder.decode(token_id.tolist())
                predicts = self.tiktokenDecoder.decode(predicted_token_id.tolist()).replace("<|30.00|>", "")
        elif self.outputName == "whisperDecoder":
            token_id = token_id.tolist()
            actuals = self.outdecoder.tokenizer.decode(token_id).strip()
            enc_feat = x
            predicts = self.outdecoder.run(enc_feat)[0]
        elif self.outputName == "LipAuthOutput":
            if "target" in sample:
                token_id = sample["target"]
            else:
                token_id = sample["target_1_"]
            if token_id.shape[0] == 2:
                actuals = [token_id[0].item(), token_id[1].item()]
            else:
                actuals = token_id.item()
            enc_feat = x
            # if self.cfg.default.mode == 'register':
            #     self.outdecoder.save_registered_features(actuals, save_layerFeaturesTensor, sample['path'])
            #     predicts = None
            # else:
            predicts, score = self.outdecoder.face_authentication(enc_feat)
            return [str(predicts), str(score)], str(actuals)
        elif self.outputName == "ClassifierOutput":
            if "target" in sample:
                token_id = sample["target"]
            else:
                token_id = sample["target_0_"]
            if token_id.shape[0] == 2:
                actuals = [token_id[0].item(), token_id[1].item()]
            else:
                actuals = int(token_id.item())-1
            enc_feat = x
            predicts = self.outdecoder(enc_feat)

            predict = predicts['top5_idx'][0, 0].item()
            score = predicts['top5_prob'][0, 0].item()
            return [str(predict), str(score)], str(actuals)
        else:
            print("未识别Decoder!!!")
        return predicts, actuals

    def info(self, input_data=None):
        if input_data==None:
            input_data = torch.randn(1, 77, 1, 88, 88)
        cfg = self.cfg
        layers, save = [], []  # layers, savelist, ch out
        for i, (f, n, t, m, args) in enumerate(cfg.Encoder + cfg.Decoder):  # from, number, type, module, args
            name = m
            # m = importlib.import_module(m)  # get module
            m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
            print(m)

        raise NotImplementedError("model_info() needs to be implemented by model pharse")

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
        if self.weights:
            weights = torch.load(self.weights)
            isFlag = False
            for key in weights:
                if 'state_dict' == key or 'model_state_dict' == key:
                    isFlag = key
            weights = weights[isFlag] if isFlag else weights
            keys_model = [key for key in weights]
        # 获取aux_weights
        if self.aux_weights:
            aux_weights = torch.load(self.aux_weights)
            isFlag = False
            for key in aux_weights:
                if 'state_dict' in key:
                    isFlag = key
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
                if self.aux_weights:
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
                if self.weights:
                    if i_model >= len(list(weights.keys())):
                        print(f'model超出索引{i_model}！！！！！！！！！！！！！！！！')
                        break
                    key_model = list(weights.keys())[i_model]
                    if state_dict[key].size() == weights[key_model].size():
                        state_dict[key] = weights[key_model]
                    else:
                        print(f'{state_dict[key].size()}不等于{weights[key_model].size()}，第【{key}-{key_model}】层权重尺寸不对！！！！！！！！！！！！！！！！')
                    i_model = i_model + 1

        # 冻结权重
        freeze_list = []
        if self.aux_list:
            for layer in self.aux_list:
                freeze_list.append(str(layer))
        if freeze:
            if isinstance(freeze, str):
                freeze = ast.literal_eval(freeze)
            if isinstance(freeze, (int, tuple)):
                freeze = [freeze]
            for i, arg in enumerate(freeze, start=0):
                if isinstance(arg, int):
                    freeze_list.append(str(arg))
                elif isinstance(arg, tuple):
                    for x in range(arg[0], arg[1]):
                        freeze_list.append(str(x))
                else:
                    print("冻结权重参数设置错误！！！")
        freeze_list = list(set(freeze_list))
        for name, module in self.model.named_modules():
            if name.split('.')[0] in freeze_list:
                for param in module.parameters():
                    param.requires_grad = False

        self.model.load_state_dict(state_dict)

    def modelforward(self, batch):
        Batch_size = batch['videos'].shape[0]  # torch.Size([15, 77, 1, 88, 88])
        lengths = batch['video_lengths']
        if "targets" in batch:
            label = batch["targets"]
        else:
            label = batch["target_0_s"]  # 默认是唇读token
        losses = {"loss_total": 0}  # outputs

        extraDict = {}
        y = []  # outputs
        save_layerFeaturesTensor = {}
        for i in range(len(self.model)):
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

            elif m.name in ["MTCN", "DenseTCN"]:
                out = m(x["inputs"], lengths, Batch_size)
                x["inputs"] = out
            elif m.name == "ConformerLayer":
                out, _ = m(x["inputs"], x["mask"])
                x["inputs"] = out
            elif m.name == "GramMatrix_co":
                out = m(x[0]["inputs"], x[1]["inputs"])
                x = x[0].copy()
                x["inputs"] = out
            elif m.name == "Concat":
                if isinstance(x, list):
                    x_list = []
                    for x_ in x:
                        x_list.append((x_["inputs"]))
                    out = m(x_list)
                    x = x[0].copy()
                    x["inputs"] = out
                else:
                    x = x
            elif m.name in ["LayerNorm", "nnLinear"]:
                if isinstance(x["inputs"], tuple):
                    out = x["inputs"][0]
                    out = m(out)
                else:
                    out = m(x["inputs"])
                x["inputs"] = out
            elif m.name == "featurefusionmodule":
                x_a, x_v = audio_video_pad(x[0]["inputs"], x[1]["inputs"])
                out = m(x_a, x_v)
                x = x[0].copy()
                x["inputs"] = out
            elif m.name == "ATTDecoder":
                from ldw_nets.utils import add_sos_eos, target_mask
                ys_in_pad, ys_out_pad = add_sos_eos(label, self.cfg.default.sos, self.cfg.default.eos, -1)
                ys_mask = target_mask(ys_in_pad, -1)
                out, _ = m(ys_in_pad, ys_mask, x["inputs"], x["mask"])
                x["inputs"] = out
                extraDict["ys_out_pad"] = ys_out_pad
            elif m.name == "TextDecoder":
                from ldw_nets.utils import add_sot_eot
                label_token = add_sot_eot(label, self.cfg.default.sot, self.cfg.default.eot).to(x["inputs"].device)
                out = m(label_token[:, :-1], x["inputs"], )  # logits
                x["inputs"] = out
                extraDict["label_token"] = label_token
            else:
                out = m(x["inputs"])  # run
                x["inputs"] = out
            # print(f"{m.name} run speed: {1000*(time.time()-start)}")
            y.append(x.copy() if m.i in self.save_featensor else None)  # save output
            if self.saveFeatureFLAG:
                if i in self.save_layers:
                    save_layerFeaturesTensor[i] = x.copy()
        return losses, x, y, label, save_layerFeaturesTensor, extraDict, i

    def loss(self, batch):
        """
        Compute loss.
        Args:
            batch (dict): Batch to compute loss on
        """
        losses, x, y, label, save_layerFeaturesTensor, extraDict, i = self.modelforward(batch)

        # losses = {"loss_total": 0}  # outputs
        for j in range(len(self.criterions)):
            m = self.criterions[j]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(x, tuple):
                    x = x[0]

            isflag = False
            # 如果x是list，读取x中的lebel，更新target
            if isinstance(x, list):
                for item in x:
                    if isinstance(item, dict):
                        if 'label' in item:
                            tartget = item['inputs']  # {'inputs':tensor,'label':True}
                            isflag = True
                            break
            # 如果x是list且有label，读取x中的tensor
            if isflag:
                for item in x:
                    if 'label' not in item:
                        x = item['inputs']
                        break
            else:
                if m.type == "loss":
                    if m.name != "loss_ctc":
                        x = x["inputs"]  # 获取x的tensor
                    tartget = label  # target就是label
                    if m.name == "loss_att":
                        tartget = extraDict["ys_out_pad"]
                    elif m.name == "loss_CrossEntropy":
                        label_token = extraDict["label_token"]
                        tartget = label_token[:, 1:]
                elif m.type == "aux_loss":
                    x, tartget = x
                    if m.name in ["loss_L1", "loss_MSE", "loss_CosineSimilarity"]:
                        x, tartget = audio_video_pad(x["inputs"], tartget["inputs"])
                    elif m.name in ["loss_KLDiv", "loss_SoftLabelCE"]:
                        x = x["inputs"]
                        tartget = tartget["inputs"]

            if m.name == "loss_ctc":
                # m = m.cuda(x.device)  # 创建CTC时要根据参数选择device，模型创建后执行model.cuda(0)，因此这里需要将其给到GPU
                x = m(x["inputs"], x["lengths"], tartget)
            elif m.name in ["loss_att", "loss_CrossEntropy",
                            "loss_L1", "loss_MSE", "loss_CosineSimilarity", "loss_KLDiv", "loss_SoftLabelCE"]:
                x = m(x, tartget)
            elif m.name == "loss_rc":
                x = m(losses["loss_CSC_r"], losses["loss_CSC_c"])
            elif m.name in ["nnCrossEntropy", "nnFocalLoss", "ArcFaceLoss"]:
                x = m(x, tartget)
            elif m.name == "loss_total":
                x = m(x)
            else:
                x = m(x)
            losses[m.name] = x
            y.append(x if m.i+1+i in self.save_featensor else None)  # save output
            if self.saveFeatureFLAG:
                if m.i+1+i in self.save_layers:
                    save_layerFeaturesTensor[m.i+1+i] = x / tartget.size(1)
        # acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=-1)
        if losses["loss_total"] == 0:
            for key in losses:
                losses["loss_total"] = losses[key] if key != "loss_total" else losses["loss_total"]
        if "loss_ctc" not in losses:
            losses["loss_att"] = torch.tensor(0)
            losses["loss_ctc"] = torch.tensor(0)

        if self.cfg.default.mode == 'register':
            if "target_0_s" in batch:
                actuals = batch['target_1_s'].item()
            else:
                actuals = batch['targets'].item()
            self.outdecoder.save_registered_features(actuals, save_layerFeaturesTensor, batch['path'])

        return losses  # , acc

    def save_src(self, path, current_epoch):
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
        if current_epoch % self.save_every_epoch == 0:
            torch.save(data, path+"/model_"+str(current_epoch)+".pth")
        torch.save(data, path + "/model_last.pth")

    def save(self, path, current_epoch, optimizer=None, scheduler=None):
        """
        Save model checkpoint.
        - 每 save_every_epoch 个 epoch 保存一次 model_xx.pth（只含权重）
        - 始终更新 model_last.pth（含权重+优化器+调度器+epoch）
        """
        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = self.model.module.state_dict().copy()
        else:
            state_dict = self.model.state_dict().copy()

        # 去掉 aux_list 中的参数
        for key, _ in list(state_dict.items()):
            if key.split(".")[0] in self.aux_list:
                del state_dict[key]

        # 1. 定期保存：仅权重
        if current_epoch % self.save_every_epoch == 0:
            torch.save({'state_dict': state_dict},
                       os.path.join(path, f"model_{current_epoch}.pth"))

        # 2. 始终保存：完整信息
        last_ckpt = {
            'epoch': current_epoch,
            'model_state_dict': state_dict,
        }
        if optimizer is not None:
            last_ckpt['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            last_ckpt['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(last_ckpt, os.path.join(path, "model_last.pth"))


