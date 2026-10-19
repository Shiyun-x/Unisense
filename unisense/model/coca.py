from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np
import math
from dataclasses import dataclass

try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StopStringCriteria,
        EosTokenCriteria,
        StoppingCriteriaList
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            inner_norm: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            norm_layer: Type[nn.Module] = LayerNormFp32,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        assert not (scaled_cosine and qk_norm), "Cannot activate both scaled cosine and QK normalization"
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        # QK normalization (with LN) from https://arxiv.org/abs/2106.04560 and related to other QK Norm ideas
        if qk_norm:
            self.ln_q = norm_layer(self.head_dim)
            self.ln_k = norm_layer(self.head_dim)
        else:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()

        # Scaled cosine attention (from Swin Transformer V2, https://arxiv.org/abs/2111.09883)
        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None

        self.attn_drop = nn.Dropout(attn_drop)

        # Per-head attention logit scaling (from NormFormer, https://arxiv.org/abs/2110.09456)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

        # Normalization of attention logits, before final projection.
        # Origin likely Sub-LN in (Foundation Transformers, https://arxiv.org/abs/2210.06423)
        if inner_norm:
            self.ln_inner = norm_layer(dim)
        else:
            self.ln_inner = nn.Identity()

        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        N, L, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(N, L, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(N, L, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(N, L, self.num_heads, -1).transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                # this module works with (L, L), or (N, num_heads, L, L) masks
                attn_mask = attn_mask.reshape(N, self.num_heads, L, L)
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            else:
                attn_mask = attn_mask.to(dtype=q.dtype)

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1),
                F.normalize(k, dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn * logit_scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            q = self.ln_q(q)
            k = self.ln_k(k)
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        # N, num_heads, L, head_dim
        if self.head_scale is not None:
            x = x * self.head_scale
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.ln_inner(x)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x
    

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x,
            need_weights=False,
            attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            qk_norm: bool = False,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()
        assert batch_first, 'batch_first must be True for CustomResidualAttentionBlock'

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            qk_norm=qk_norm,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            inner_norm=scale_attn_inner,
            norm_layer=norm_layer,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),  # from NormFormer / Foundation Transformers
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_weight_dtype(self) -> torch.dtype:
        if hasattr(self.mlp.c_fc, 'int8_original_dtype'):
            return self.mlp.c_fc.int8_original_dtype
        return self.mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            batch_first: bool = True,
            block_type: Optional[str] = None,
            qk_norm: bool = False,
            scaled_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn_inner: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        # Auto-select custom block if any custom features are enabled
        if block_type is None:
            if any([qk_norm, scaled_cosine_attn, scale_heads, scale_attn_inner, scale_attn, scale_fc]):
                block_type = 'custom'
            else:
                block_type = 'default'

        if block_type == 'custom':
            self.resblocks = nn.ModuleList([
                CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    scale_cosine_attn=scaled_cosine_attn,
                    scale_heads=scale_heads,
                    scale_attn_inner=scale_attn_inner,
                    scale_attn=scale_attn,
                    scale_fc=scale_fc,
                    batch_first=batch_first,
                )
                for _ in range(layers)
            ])
        else:
            self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
                for _ in range(layers)
            ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].get_weight_dtype()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 512,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            output_dim: int = 512,
            batch_first: bool = True,
            text_embedding: Optional[torch.dtype] = None,
            input_width: Optional[int] = None,
            external_vocab_projection: bool = False,

    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
        )
        self.context_length = context_length
        input_width = input_width or width
        self.image_input_projection = (
            nn.Identity() if input_width == width else nn.Linear(input_width, width, bias=False)
        )
        self.text_input_projection = (
            nn.Identity() if input_width == width else nn.Linear(input_width, width, bias=False)
        )
        self.output_projection = (
            nn.Identity() if input_width == width else nn.Linear(width, input_width, bias=False)
        )
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.ln_final = norm_layer(width)
        # text_projection maps CoCa output → vocab logits using the LLM's embedding matrix.
        # It is intentionally frozen (requires_grad=False): we want the CoCa decoder to learn
        # to output features in the LLM's embedding space, not to drift the projection matrix.
        # Keeping it trainable creates a ~1 GB gradient tensor (248K×2048) that must be
        # AllReduced across GPUs every step, causing NVLink pressure and NCCL crashes.
        # Legacy CoCa owns a frozen copy of the vocabulary projection. CoCa-Lite
        # instead receives the existing frozen Qwen lm_head at forward time, so it
        # does not duplicate the very large vocab_size x LLM_hidden matrix.
        self.external_vocab_projection = external_vocab_projection
        if external_vocab_projection:
            self.register_parameter('text_projection', None)
        else:
            self.text_projection = nn.Parameter(
                text_embedding.clone().detach(), requires_grad=False)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
            self, image_embs, text_embs, objects_len: int = 0,
            image_embs_projected: bool = False,
            text_embs_projected: bool = False):
        """
        Args:
            image_embs:  (B, img_tokens, input_width), or decoder-width features
                         when ``image_embs_projected=True``.
            text_embs:   (B, seq_len, input_width), or decoder-width unimodal
                         states when ``text_embs_projected=True``.
            objects_len: number of leading object tokens (including separator).
                         When > 0, only caption tokens (text_embs[:, objects_len:])
                         participate in cross-attention with image_embs; object tokens
                         go through self-attention only.
        """
        if not image_embs_projected:
            image_embs = self.image_input_projection(image_embs)
        seq_len = text_embs.shape[1]
        if not text_embs_projected:
            text_embs = self.text_input_projection(text_embs)
            text_embs = text_embs + self.positional_embedding[:seq_len]

        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)    # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # Self-attention over the full sequence (causal mask applied)
                text_embs = checkpoint(
                    resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len], use_reentrant=False)
                # Cross-attention: only caption tokens attend to image
                if objects_len > 0:
                    obj_part = text_embs[:, :objects_len]
                    cap_part = text_embs[:, objects_len:]
                    cap_part = checkpoint(cross_attn, cap_part, image_embs, image_embs, None, use_reentrant=False)
                    text_embs = torch.cat([obj_part, cap_part], dim=1)
                else:
                    text_embs = checkpoint(
                        cross_attn, text_embs, image_embs, image_embs, None, use_reentrant=False)
            else:
                # Self-attention over the full sequence (causal mask applied)
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                # Cross-attention: only caption tokens attend to image
                if objects_len > 0:
                    obj_part = text_embs[:, :objects_len]
                    cap_part = text_embs[:, objects_len:]
                    cap_part = cross_attn(cap_part, k_x=image_embs, v_x=image_embs)
                    text_embs = torch.cat([obj_part, cap_part], dim=1)
                else:
                    text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        out = self.output_projection(out)
        if self.text_projection is not None:
            out = out @ self.text_projection.T
        return out

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


@dataclass
class CLIPTextCfg:
    context_length: int = 512
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_mode: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    eos_id: int = 2  # only used for when pool_type == 'eos', must match tokenizer eos
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # Custom attention block settings
    block_type: Optional[str] = None  # attention block type ('default', 'custom'), auto-selects 'custom' if any custom features enabled
    qk_norm: bool = False  # apply layer norm to q and k in attention
    scaled_cosine_attn: bool = False  # use scaled cosine attention
    scale_heads: bool = False  # learnable head-specific scale applied to attention logits
    scale_attn_inner: bool = False  # apply layer norm on attention context, before output projection
    scale_attn: bool = False  # apply layer norm after full attention block
    scale_fc: bool = False  # apply layer norm in MLP block

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8


def build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        text_embedding: Optional[torch.dtype] = None,
        decoder_width: Optional[int] = None,
        external_vocab_projection: bool = False,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg

    input_width = text_embedding.shape[1]
    # Legacy behavior uses the full LLM width. CoCa-Lite explicitly requests a
    # bottleneck width and projects both pointcloud/text inputs into that space.
    multimodal_cfg.width = input_width if decoder_width is None else decoder_width

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        text_embedding=text_embedding,
        input_width=input_width,
        external_vocab_projection=external_vocab_projection,
    )

    return decoder


def _token_to_tensor(token_id, device: str = "cpu") -> torch.Tensor:
    if not isinstance(token_id, torch.Tensor):
        if isinstance(token_id, int):
            token_id = [token_id]
        token_id = torch.tensor(token_id, device=device)
    return token_id


class CoCa(nn.Module):
    def __init__(
            self,
            text_cfg: CLIPTextCfg,
            multimodal_cfg: MultimodalCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
            text_embedding: Optional[torch.dtype] = None,
            decoder_width: Optional[int] = None,
            external_vocab_projection: bool = False,
            original_contrastive: bool = False,
            contrastive_dim: int = 512,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.text_decoder = build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            text_embedding=text_embedding,
            decoder_width=decoder_width,
            external_vocab_projection=external_vocab_projection,
        )
        self.external_vocab_projection = external_vocab_projection
        self.original_contrastive = original_contrastive

        if original_contrastive:
            width = multimodal_cfg.width
            heads = multimodal_cfg.heads
            self.img_queries = nn.Parameter(
                torch.randn(multimodal_cfg.n_queries + 1, width) * (width ** -0.5))
            self.img_attn_pool = ResidualAttentionBlock(
                width, heads, mlp_ratio=multimodal_cfg.mlp_ratio,
                ls_init_value=multimodal_cfg.ls_init_value,
                is_cross_attention=True, batch_first=True)
            self.img_attn_pool_norm = LayerNorm(width)

            # Original CoCa obtains its contrastive text representation before
            # multimodal cross-attention. Keep that branch inside CoCa-Lite so
            # the text anchor never sees pointcloud tokens.
            self.contrastive_text_encoder = Transformer(
                width=width, layers=multimodal_cfg.layers, heads=heads,
                mlp_ratio=multimodal_cfg.mlp_ratio,
                ls_init_value=multimodal_cfg.ls_init_value,
                batch_first=True)
            self.contrastive_text_norm = LayerNorm(width)
            self.image_to_latent = nn.Linear(width, contrastive_dim, bias=False)
            self.text_to_latent = nn.Linear(width, contrastive_dim, bias=False)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(
            torch.ones(lshape) * init_logit_scale,
            requires_grad=original_contrastive)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias, requires_grad=False)
        else:
            self.logit_bias = None
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length


    def forward(
            self,
            image_embs: Optional[torch.Tensor] = None,
            token_embs: Optional[torch.Tensor] = None,
            objects_len: int = 0,
            vocab_projection: Optional[nn.Module] = None,
    ):
        if self.original_contrastive:
            image_embs, _ = self.encode_original_image(image_embs)
            text_mask = torch.ones(
                token_embs.shape[:2], dtype=torch.bool, device=token_embs.device)
            token_embs, _ = self.encode_original_text(token_embs, text_mask)
            decoded = self.text_decoder(
                image_embs, token_embs, objects_len=objects_len,
                image_embs_projected=True, text_embs_projected=True)
        else:
            decoded = self.text_decoder(image_embs, token_embs, objects_len=objects_len)
        if self.external_vocab_projection:
            if vocab_projection is None:
                raise ValueError("CoCa-Lite requires the frozen Qwen vocab projection")
            return vocab_projection(decoded)
        return decoded

    def encode_original_image(self, image_embs: torch.Tensor):
        """Return pooled decoder tokens and a pointcloud-only contrastive latent."""
        if not self.original_contrastive:
            raise RuntimeError("encode_original_image requires original_contrastive=True")
        image_embs = self.text_decoder.image_input_projection(image_embs)
        queries = self.img_queries.unsqueeze(0).expand(image_embs.shape[0], -1, -1)
        pooled = self.img_attn_pool(queries, k_x=image_embs, v_x=image_embs)
        pooled = self.img_attn_pool_norm(pooled)
        image_latent = F.normalize(
            self.image_to_latent(pooled[:, 0]).float(), dim=-1)
        return pooled[:, 1:], image_latent

    def encode_original_text(
            self, text_embs: torch.Tensor,
            text_pool_mask: torch.Tensor):
        """Return shared unimodal text states and a text-only latent.

        The states are consumed directly by the multimodal decoder for caption
        CE, while their masked pool supplies the InfoNCE text representation.
        """
        if not self.original_contrastive:
            raise RuntimeError("encode_original_text requires original_contrastive=True")
        seq_len = text_embs.shape[1]
        text_embs = self.text_decoder.text_input_projection(text_embs)
        text_embs = text_embs + self.text_decoder.positional_embedding[:seq_len]
        text_hidden = self.contrastive_text_encoder(
            text_embs, attn_mask=self.text_decoder.attn_mask[:seq_len, :seq_len])
        normalized_hidden = self.contrastive_text_norm(text_hidden)
        valid = text_pool_mask.to(device=text_hidden.device, dtype=torch.bool)
        pooled = (
            normalized_hidden.float() * valid.unsqueeze(-1)
        ).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        text_latent = F.normalize(self.text_to_latent(pooled).float(), dim=-1)
        return text_hidden, text_latent

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"
        device = image.device

        with torch.no_grad():
            sot_token_id = _token_to_tensor(49406 if sot_token_id is None else sot_token_id, device=device)
            eos_token_id = _token_to_tensor(49407 if eos_token_id is None else eos_token_id, device=device)
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]
            stopping_criteria = StoppingCriteriaList(stopping_criteria)

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs=image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    pad_len = seq_len - output.shape[1]
                    return torch.cat((
                            output,
                            torch.ones(output.shape[0], pad_len, device=device, dtype=output.dtype) * pad_token_id
                        ),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(
                    image,
                    x,
                    image_latent=image_latent,
                    image_embs=image_embs,
                    output_labels=False,
                )["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if all(stopping_criteria(out, None)):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def _generate_beamsearch(
            self,
            image_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=6,
            num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
        image_latent, image_embs = self._encode_image(image_inputs)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(
                model_inputs['images'],
                model_inputs['text'],
                image_latent=image_latent,
                image_embs=image_embs,
                output_labels=False,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
