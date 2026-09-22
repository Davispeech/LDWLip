import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
from typing import Any, List, Tuple
from ldw_nets.nn.unit import LayerNorm



# ------------------------
# 工具函数
# ------------------------
class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
        RelPositionMultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.transformer.convolution.
        ConvolutionModule feed_foreard:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool macaron_style: whether to use macaron style for PositionwiseFeedForward

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        macaron_style=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.ff_scale = 1.0
        self.conv_module = conv_module
        self.macaron_style = macaron_style
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if self.macaron_style:
            self.feed_forward_macaron = copy.deepcopy(feed_forward)
            self.ff_scale = 0.5
            # for another FNN module in macaron style
            self.norm_ff_macaron = LayerNorm(size)
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.macaron_style:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask


# ------------------------
# 自定义 Transformer LMDecoder
# ------------------------
class LMDecoder(nn.Module):
    def __init__(self, json_path, device="cpu"):
        super().__init__()
        self.device = device

        # === 读取 json 配置 ===
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        n_vocab = cfg["n_vocab"]
        embed_unit = cfg["embed_unit"]
        attention_dim = cfg["att_unit"]
        attention_heads = cfg.get("head", 8)
        dropout_rate = cfg.get("dropout_rate", 0.0)
        linear_units = cfg.get("unit", 2048)
        tie_weights = cfg.get("tie_weights", False)
        encoder_attn_layer_type = "mha"
        zero_triu=False
        positionwise_layer_type = "linear"
        cnn_module_kernel = 31
        use_cnn_module = False
        self.normalize_before = True
        concat_after = False
        macaron_style = False

        # === 组件 ===
        if encoder_attn_layer_type == "mha":
            from ldw_nets.nn.block.conformerBlock import MultiHeadedAttention
            encoder_attn_layer = MultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                dropout_rate,
            )
        elif encoder_attn_layer_type == "legacy_rel_mha":
            from ldw_nets.nn.block.conformerBlock import LegacyRelPositionMultiHeadedAttention
            encoder_attn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                dropout_rate,
            )
        elif encoder_attn_layer_type == "rel_mha":
            from ldw_nets.nn.block.conformerBlock import RelPositionMultiHeadedAttention
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer_type)

        if positionwise_layer_type == "linear":
            from ldw_nets.nn.block.conformerBlock import PositionwiseFeedForward
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # === 模型 ===
        self.embed = nn.Embedding(n_vocab, embed_unit).cuda()
        if dropout_rate == 0.0:
            self.embed_drop = None
        else:
            self.embed_drop = nn.Dropout(dropout_rate)

        self.encoderEmbed = torch.nn.Sequential(
                torch.nn.Linear(embed_unit, attention_dim),  # idim=128, attention_dim=512
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),  # dropout_rate=0.0
                torch.nn.ReLU(),
                self.pos_enc_class(attention_dim, 0.1),  # positional_dropout_rate=0.1
            )
        from ldw_nets.nn.block.conformerBlock import ConvolutionModule
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)
        from ldw_nets.utils import repeat
        self.encoders = repeat(
            cfg["layer"],
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                self.normalize_before,
                concat_after,
                macaron_style,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.decoder = nn.Linear(attention_dim, n_vocab)

        if tie_weights:
            assert (
                attention_dim == embed_unit
            ), "Tie Weights: True need embedding and final dimensions to match"
            self.decoder.weight = self.embed.weight

    def pos_enc_class(*args, **kwargs):
        return nn.Sequential()  # indentity

    # ------------------------
    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        from ldw_nets.utils import subsequent_mask
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    # ------------------------
    def forward(self, x, masks):
        x = self.encoderEmbed(x)
        x, masks = self.encoders(x, masks)
        if isinstance(x, tuple):
            x = x[0]
        if self.normalize_before:
            x = self.after_norm(x)
        return x, masks

    # ------------------------
    def forward_one_step(self, x, masks, cache=None):
        x = self.encoderEmbed(x)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            x, masks = e(x, masks, cache=c)
            new_cache.append(x)
        if self.normalize_before:
            x = self.after_norm(x)
        return x, masks, new_cache

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y = y.unsqueeze(0)

        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(y))
        else:
            emb = self.embed(y)

        h, _, cache = self.forward_one_step(
            emb, self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

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
        n_layers = len(self.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(ys))
        else:
            emb = self.embed(ys)

        # batch decoding
        h, _, states = self.forward_one_step(
            emb, self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def batch_init_state(self, x=None):
        return None

    def select_state(self, state, i, new_id=None):
        return None if state is None else state[i]
        # return state

    def final_score(self, state):
        return 0.0
