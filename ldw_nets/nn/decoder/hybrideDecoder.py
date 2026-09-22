import logging
import numpy as np
from typing import Any, List, Tuple, Dict
from distutils.version import LooseVersion
import six
import torch
import torch.nn as nn
import torch.nn.functional as F


from ldw_nets.nn.unit import LayerNorm
from ldw_nets.nn.block import PositionalEncoding, PositionwiseFeedForward, MultiHeadedAttention
from ldw_nets.utils import repeat
from ldw_nets.utils import subsequent_mask

# from ldw_nets.lm.ctc_scores import CTCPrefixScorer
# from ldw_nets.lm.utils import BatchBeamSearch, get_model_conf, dynamic_import_lm, torch_load
from ldw_nets.scorer_interface import BatchScorerInterface
# from ldw_nets.scorers.length_bonus import LengthBonus


__all__ = (
    "HybrideLipOutput",
    "CTCDecoder",
    "ATTDecoder",
)


class HybrideLipOutput(nn.Module):
    def __init__(self, lm_hype, token_list, model, odim):
        super().__init__()

        sos = odim - 1
        eos = odim - 1
        if lm_hype.lm_model:
            # from espnet.nets.lm_utils import get_model_conf, dynamic_import_lm, torch_load
            # lm_args = get_model_conf(lm_hype.rnnlm, lm_hype.rnnlm_conf)
            # lm_model_module = getattr(lm_args, "model_module", "default")
            # lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
            # lm = lm_class(len(token_list), lm_args)
            # torch_load(lm_hype.rnnlm, lm)
            # lm.eval()
            # #
            from ldw_nets.nn.decoder.LMDecoder import LMDecoder
            from ldw_nets.lm.utils import torch_load_byLayer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lm = LMDecoder(lm_hype.rnnlm_conf, device)
            torch_load_byLayer(lm_hype.rnnlm, lm)
            lm.eval()
        else:
            lm = None
        for i in range(len(model)):
            m = model[i]
            if m.name == "ATTDecoder":
                ATTDecoder = model[i]
            if m.name == "CTCDecoder":
                CTCDecoder = model[i]
        from ldw_nets.scorers.ctc import CTCPrefixScorer
        from ldw_nets.scorers.length_bonus import LengthBonus
        scorers = {
            "decoder": ATTDecoder,
            "ctc": CTCPrefixScorer(CTCDecoder, eos),
            "length_bonus": LengthBonus(len(token_list)),
            "lm": lm
        }
        weights = {
            "decoder": 1.0 - lm_hype.ctc_weight,
            "ctc": lm_hype.ctc_weight,
            "lm": lm_hype.lm_weight,
            "length_bonus": lm_hype.penalty,
        }
        from ldw_nets.lm.utils import BatchBeamSearch
        self.beam_search = BatchBeamSearch(
            beam_size=lm_hype.beam_size,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=sos,
            eos=eos,
            token_list=token_list,
            pre_beam_score_key=None if lm_hype.ctc_weight == 1.0 else "decoder",
        )

    def forward(self, xs):
        # tensor = xs
        # cmap = "viridis"
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # tensor = tensor.detach().cpu().numpy()
        # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)  # 避免除零错误
        #
        # # 绘制热力图
        # fig, ax = plt.subplots(figsize=(12, 4))
        # heatmap = ax.imshow(tensor, aspect="auto", cmap=cmap)
        # cbar = plt.colorbar(heatmap, ax=ax) # 添加颜色条
        # cbar.set_label("Feature Intensity", fontsize=14)  # 设置颜色条标签的字体大小
        # ax.set_xlabel("Feature Dimension (256)", fontsize=14)
        # ax.set_ylabel("Time Step (77)", fontsize=14)
        # ax.set_title("Feature Tensor Heatmap", fontsize=16)
        # ax.tick_params(axis='both', labelsize=12)
        # cbar.ax.tick_params(labelsize=12)  # 颜色条刻度字体
        #
        # # 保存图片，不显示
        # plt.savefig("./heat.png", dpi=300, bbox_inches="tight")
        # plt.close(fig)  # 关闭图像，释放内存
        return self.beam_search(xs)

class ATTDecoderLayer(nn.Module):
    """Single decoder layer module.
    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward: feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(ATTDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.
        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


def rename_state_dict(
    old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]
):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class CTCDecoder(torch.nn.Module):
    """CTC module
    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param str ctc_type: builtin or warpctc
    :param bool reduce: reduce the CTC loss into a scalar
    """
    def __init__(self, odim, eprojs, dropout_rate, ctc_type="builtin", reduce=True):
        super().__init__()
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # In case of Pytorch >= 1.7.0, CTC will be always builtin
        self.ctc_type = (
            ctc_type
            if LooseVersion(torch.__version__) < LooseVersion("1.7.0")
            else "builtin"
        )
        if ctc_type != self.ctc_type:
            logging.debug(f"CTC was set to {self.ctc_type} due to PyTorch version.")

    def forward(self, hs_pad):
        """CTC forward
        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # zero padding for hs
        ys_hat = self.ctc_lo(self.dropout(hs_pad))
        if self.ctc_type != "gtnctc":
            ys_hat = ys_hat.transpose(0, 1)

        if not self.ctc_type == "builtin":
            dtype = ys_hat.dtype
            if self.ctc_type == "warpctc" or dtype == torch.float16:
                # warpctc only supports float32
                # torch.ctc does not support float16 (#1751)
                ys_hat = ys_hat.to(dtype=torch.float32)
        return ys_hat

    def softmax(self, hs_pad):
        """softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=-1)
        return self.probs

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=-1)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=-1)

    def forced_align(self, h, y, blank_id=0):
        """forced alignment.

        :param torch.Tensor h: hidden state sequence, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])
            return label

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)

        y_int = interpolate_blank(y, blank_id)

        logdelta = np.zeros((lpz.size(0), len(y_int))) - 100000000000.0  # log of zero
        state_path = (
            np.zeros((lpz.size(0), len(y_int)), dtype=np.int16) - 1
        )  # state path

        logdelta[0, 0] = lpz[0][y_int[0]]
        logdelta[0, 1] = lpz[0][y_int[1]]

        for t in six.moves.range(1, lpz.size(0)):
            for s in six.moves.range(len(y_int)):
                if y_int[s] == blank_id or s < 2 or y_int[s] == y_int[s - 2]:
                    candidates = np.array([logdelta[t - 1, s], logdelta[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = np.array(
                        [
                            logdelta[t - 1, s],
                            logdelta[t - 1, s - 1],
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = np.max(candidates) + lpz[t][y_int[s]]
                state_path[t, s] = prev_state[np.argmax(candidates)]

        state_seq = -1 * np.ones((lpz.size(0), 1), dtype=np.int16)

        candidates = np.array(
            [logdelta[-1, len(y_int) - 1], logdelta[-1, len(y_int) - 2]]
        )
        prev_state = [len(y_int) - 1, len(y_int) - 2]
        state_seq[-1] = prev_state[np.argmax(candidates)]
        for t in six.moves.range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_state_seq = []
        for t in six.moves.range(0, lpz.size(0)):
            output_state_seq.append(y_int[state_seq[t, 0]])

        return output_state_seq

    def forced_align_batch(self, hs_pad, ys_pad, ilens, blank_id=0):
        """forced alignment with batch processing.

        :param torch.Tensor hs_pad: hidden state sequence, 3d tensor (T, B, D)
        :param torch.Tensor ys_pad: id sequence tensor 2d tensor (B, L)
        :param torch.Tensor ilens: Input length of each utterance (B,)
        :param int blank_id: blank symbol index
        :return: best alignment results
        :rtype: list of numpy.array
        """

        def interpolate_blank(label, olens_int):
            """Insert blank token between every two label token."""
            lab_len = label.shape[1] * 2 + 1
            label_out = np.full((label.shape[0], lab_len), blank_id, dtype=np.int64)
            label_out[:, 1::2] = label
            for b in range(label.shape[0]):
                label_out[b, olens_int[b] * 2 + 1 :] = self.ignore_id
            return label_out

        neginf = float("-inf")  # log of zero
        # lpz = self.log_softmax(hs_pad).cpu().detach().numpy()
        # hs_pad = hs_pad.transpose(1,0)
        lpz = F.log_softmax(hs_pad, dim=-1).cpu().detach().numpy()
        ilens = ilens.cpu().detach().numpy()

        ys_pad = ys_pad.cpu().detach().numpy()
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        olens = np.array([len(s) for s in ys])
        olens_int = olens * 2 + 1
        ys_int = interpolate_blank(ys_pad, olens_int)

        Tmax, B, _ = lpz.shape
        Lmax = ys_int.shape[-1]
        logdelta = np.full((Tmax, B, Lmax), neginf, dtype=lpz.dtype)
        state_path = -np.ones(logdelta.shape, dtype=np.int16)  # state path

        b_indx = np.arange(B, dtype=np.int64)
        t_0 = np.zeros(B, dtype=np.int64)
        logdelta[0, :, 0] = lpz[t_0, b_indx, ys_int[:, 0]]
        logdelta[0, :, 1] = lpz[t_0, b_indx, ys_int[:, 1]]

        s_indx_mat = np.arange(Lmax)[None, :].repeat(B, 0)
        notignore_mat = ys_int != self.ignore_id
        same_lab_mat = np.zeros((B, Lmax), dtype=np.bool)
        same_lab_mat[:, 3::2] = ys_int[:, 3::2] == ys_int[:, 1:-2:2]
        Lmin = olens_int.min()
        for t in range(1, Tmax):
            s_start = max(0, Lmin - (Tmax - t) * 2)
            s_end = min(Lmax, t * 2 + 2)
            candidates = np.full((B, Lmax, 3), neginf, dtype=logdelta.dtype)
            candidates[:, :, 0] = logdelta[t - 1, :, :]
            candidates[:, 1:, 1] = logdelta[t - 1, :, :-1]
            candidates[:, 3::2, 2] = logdelta[t - 1, :, 1:-2:2]
            candidates[same_lab_mat, 2] = neginf
            candidates_ = candidates[:, s_start:s_end, :]
            idx = candidates_.argmax(-1)
            b_i, s_i = np.ogrid[:B, : idx.shape[-1]]
            nignore = notignore_mat[:, s_start:s_end]
            logdelta[t, :, s_start:s_end][nignore] = (
                candidates_[b_i, s_i, idx][nignore]
                + lpz[t, b_i, ys_int[:, s_start:s_end]][nignore]
            )
            s = s_indx_mat[:, s_start:s_end]
            state_path[t, :, s_start:s_end][nignore] = (s - idx)[nignore]

        alignments = []
        prev_states = logdelta[
            ilens[:, None] - 1,
            b_indx[:, None],
            np.stack([olens_int - 2, olens_int - 1], -1),
        ].argmax(-1)
        for b in range(B):
            T, L = ilens[b], olens_int[b]
            prev_state = prev_states[b] + L - 2
            ali = np.empty(T, dtype=ys_int.dtype)
            ali[T - 1] = ys_int[b, prev_state]
            for t in range(T - 2, -1, -1):
                prev_state = state_path[t + 1, b, prev_state]
                ali[t] = ys_int[b, prev_state]
            alignments.append(ali)

        return alignments


class ATTDecoder(BatchScorerInterface, torch.nn.Module):
    """Transfomer decoder module.

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        odim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        positional_dropout_rate = dropout_rate
        self._register_load_state_dict_pre_hook(_pre_hook)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda: ATTDecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
                                 if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels)
                                 in the other cases
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
                                         dtype=torch.uint8 in PyTorch 1.2-
                                         dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
                   if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim)
                   in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
                                      dtype=torch.uint8 in PyTorch 1.2-
                                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
            `y.shape` is (batch, maxlen_out, token)
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][l] for b in range(n_batch)])
                for l in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[l][b] for l in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
