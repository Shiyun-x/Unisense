import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import re

import torch
import numpy as np
import torchaudio
from torchaudio.transforms import Resample

import transformers
import whisper
import tokenizers

from qwen_vl_utils import process_vision_info

from unisense.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_SPEECH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from torch.utils.data import Dataset
from unisense.train.unisense_trainer import (
    UnisenseTrainer,
    get_mm_adapter_state_maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    maybe_zero_3,
)

from unisense import conversation as conversation_lib
from unisense.model import *
from unisense.mm_utils import (tokenizer_image_token, tokenizer_image_speech_token,
                            tokenizer_speech_token, tokenizer_pointcloud_token,
                            tokenizer_imu_token, tokenizer_multimodal,
                            process_highres_image, process_anyres_image,
                            process_highres_image_crop_split)

from PIL import Image

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_speech_projector_only: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_encode_chunk_size: int = field(
        default=1,
        metadata={"help": "Number of packed Qwen-VL image/depth items per frozen ViT "
                          "forward. 1 preserves the serial path; larger values improve "
                          "throughput at the cost of transient vision-forward memory."},
    )
    speech_tower: Optional[str] = field(default=None)
    optimize_vision_tower: bool = field(default=False) # whether to optimize vision tower
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_speech_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_speech_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    down_factor_1: int = field(default=1)
    down_factor_2: int = field(default=1)
    train_modality: Optional[str] = field(default='text_image')
    asr_align: bool = field(default=False)
    check_data_modality: bool = field(default=False)
    align_temperature: float = field(default=1.0)
    weight_lambda: float = field(default=0.5)
    # Shared auxiliary alignment for every non-speech added modality.  None keeps
    # the legacy per-modality flags below fully backward compatible; an explicit
    # value applies the same strategy to every enabled modality projector.
    modality_alignment_type: Optional[str] = field(
        default=None,
        metadata={"help": "Shared alignment strategy for enabled depth/pointcloud/IMU/"
                          "tactile/thermal/radar/smell/motion branches: "
                          "none, coca_lite, coca_original, coca_qwen_dual, "
                          "coca_encoder_embed, coca_staged_qwen, frozen_qwen, "
                          "semantic_anchor, dual_path, or dual_path_contrastive. "
                          "None uses legacy flags."},
    )
    modality_alignment_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Shared auxiliary-loss weight. None preserves each legacy "
                          "{modality}_caption_weight."},
    )
    modality_alignment_micro_batch_size: int = field(
        default=4,
        metadata={"help": "Micro-batch size for batched CoCa-Lite auxiliary decoding. "
                          "Larger values improve GPU utilization but increase peak memory."},
    )
    modality_alignment_coca_width: int = field(
        default=1024,
        metadata={"help": "Hidden width of CoCa modules created through the shared "
                          "modality-alignment interface."},
    )
    modality_alignment_coca_layers: int = field(
        default=6,
        metadata={"help": "Number of CoCa decoder layers. coca_original also uses this "
                          "depth for its independent contrastive text encoder."},
    )
    modality_alignment_coca_stage1_only: bool = field(
        default=False,
        metadata={"help": "For coca_staged_qwen, disable both CoCa caption CE and "
                          "InfoNCE after the staged MoE-unfreeze boundary. The modality "
                          "projector then receives only the main SFT CE gradient."},
    )
    modality_alignment_coca_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads in shared-interface CoCa modules."},
    )
    modality_semantic_temperature: float = field(
        default=0.07,
        metadata={"help": "InfoNCE temperature for shared semantic_anchor alignment."},
    )
    modality_semantic_nce_weight: float = field(
        default=0.1,
        metadata={"help": "Batch-local InfoNCE weight inside shared semantic_anchor loss."},
    )
    modality_alignment_shallow_layer: int = field(
        default=8,
        metadata={"help": "One-based frozen-Qwen layer used by dual_path contextual alignment."},
    )
    modality_alignment_shallow_weight: float = field(
        default=2.0,
        metadata={"help": "Shallow cosine coefficient relative to the deep caption CE inside "
                          "dual_path. The whole combined loss is subsequently multiplied by "
                          "modality_alignment_weight."},
    )
    modality_alignment_contrastive_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature of the symmetric in-batch InfoNCE term used by "
                          "dual_path_contrastive."},
    )
    modality_alignment_contrastive_weight: float = field(
        default=0.25,
        metadata={"help": "InfoNCE coefficient inside contrastive alignment modes."},
    )
    modality_alignment_caption_weight: float = field(
        default=0.25,
        metadata={"help": "CoCa-Lite caption CE coefficient inside coca_qwen_dual "
                          "and coca_encoder_embed."},
    )
    modality_alignment_contrastive_space: str = field(
        default="projected",
        metadata={"help": "InfoNCE space for coca_qwen_dual: projected uses a shared "
                          "metric head; raw uses the frozen-Qwen hidden space directly."},
    )
    modality_alignment_contrastive_dim: int = field(
        default=512,
        metadata={"help": "Output dimension of the contrastive projection heads. "
                          "coca_staged_qwen uses separate trainable modality and "
                          "text heads over frozen-Qwen features."},
    )
    modality_alignment_teacher_pooling: str = field(
        default="last",
        metadata={"help": "Caption pooling for coca_qwen_dual/coca_staged_qwen "
                          "frozen-Qwen teacher: last or mean."},
    )
    depth_caption_weight: float = field(default=0.01)
    depth_decorr_weight: float = field(default=0.0, metadata={"help": "Weight for depth-image decorrelation loss (relative to depth_caption_weight). 0 = disabled."})
    depth_decorr_margin: float = field(default=0.3, metadata={"help": "Cosine similarity margin for decorrelation hinge loss."})
    speech_norm: bool = field(default=True)
    speech_learn: bool = field(default=False)
    align_norm: bool = field(default=True)
    align_type: Optional[str] = field(default='dtw')
    speech_encoder_ds_rate: int = field(default=5)
    speech_encoder_hidden_size: int = field(default=1280)
    generate: bool = field(default=False)
    ctc_decoder_config: str = "(2,4096,32,11008)"
    ctc_upsample_factor: int = 25
    ctc_loss_weight: float = 1.0
    unit_vocab_size: int = 1000
    tune_speech_generator_only: bool = field(default=True)
    mm_depth_projector_type: str = field(default=None)
    depth_caption: bool = field(default=False)
    object_depth: bool = field(default=False,
                               metadata={"help": "Prepend objects description before caption in CoCa training; "
                                                 "caption tokens cross-attend to image, objects tokens do not."})
    coca_image_input: str = field(default="rgb_depth",
                                  metadata={"help": "Visual tokens fed into CoCa cross-attention. "
                                                    "'rgb_depth': concat(RGB, depth) [default]; "
                                                    "'depth_only': depth features only."})
    # ── Point cloud (scene-level) ──────────────────────────────────────────
    mm_pointcloud_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for pointcloud branch. None disables pointcloud. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    pointcloud_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for pointcloud."},
    )
    pointcloud_frozen_qwen_caption: bool = field(
        default=False,
        metadata={"help": "Use the frozen target Qwen backbone as the pointcloud caption decoder. "
                          "Mutually exclusive with pointcloud_caption and pointcloud_semantic_align."},
    )
    pointcloud_semantic_align: bool = field(
        default=False,
        metadata={"help": "Directly align pooled pointcloud projector features with frozen Qwen "
                          "caption-embedding anchors. Mutually exclusive with caption decoder losses."},
    )
    pointcloud_semantic_temperature: float = field(
        default=0.07,
        metadata={"help": "InfoNCE temperature for pointcloud semantic-anchor alignment."},
    )
    pointcloud_semantic_nce_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of batch-local InfoNCE inside pointcloud semantic loss; "
                          "positive cosine loss always has weight 1."},
    )
    pointcloud_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the selected pointcloud alignment loss."},
    )
    pointcloud_in_channels: int = field(
        default=6,
        metadata={"help": "Input feature dim of pointcloud (xyz + rgb = 6)."},
    )
    pointcloud_embed_dim: int = field(
        default=0,
        metadata={"help": "Tokenizer output dim. 0 = auto match ViT.hidden_size (1024 for Qwen3.5-2B)."},
    )
    pointcloud_use_relative_pos: bool = field(
        default=True,
        metadata={"help": "Concatenate relative xyz (per group) as extra tokenizer input."},
    )
    pointcloud_normalize_neighbors: bool = field(
        default=True,
        metadata={"help": "Use mean of group neighbors (vs first neighbor) as reference center."},
    )
    # ── IMU (motion sensor) ────────────────────────────────────────────────
    mm_imu_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for IMU branch. None disables IMU. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    imu_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for IMU."},
    )
    imu_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the IMU CoCa loss."},
    )
    imu_in_channels: int = field(
        default=6,
        metadata={"help": "Input channel dim of IMU (accel xyz + gyro xyz = 6)."},
    )
    imu_embed_dim: int = field(
        default=0,
        metadata={"help": "IMU tokenizer output dim. 0 = auto match ViT.hidden_size (1024)."},
    )
    imu_seq_len: int = field(
        default=120,
        metadata={"help": "Fixed IMU time-step count; input is interpolated to this length."},
    )
    imu_conv_kernel: int = field(
        default=10,
        metadata={"help": "Conv1D kernel size in IMUTokenizer (OneLLM uses 10)."},
    )
    # ── Radar (mmWave 多帧点云) ─────────────────────────────────────────────
    mm_radar_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for radar branch. None disables radar. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    radar_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for radar."},
    )
    radar_frozen_qwen_caption: bool = field(
        default=False,
        metadata={"help": "Use the frozen target Qwen backbone as the radar caption decoder. "
                          "Mutually exclusive with radar_caption and radar_semantic_align."},
    )
    radar_semantic_align: bool = field(
        default=False,
        metadata={"help": "Directly align pooled radar projector features with frozen Qwen "
                          "caption-embedding anchors. Mutually exclusive with caption decoder losses."},
    )
    radar_semantic_temperature: float = field(
        default=0.07,
        metadata={"help": "InfoNCE temperature for radar semantic-anchor alignment."},
    )
    radar_semantic_nce_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of batch-local InfoNCE inside radar semantic loss; "
                          "positive cosine loss always has weight 1."},
    )
    radar_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the selected radar alignment loss."},
    )
    radar_in_channels: int = field(
        default=5,
        metadata={"help": "Input channel dim per radar point ([x,y,z,Doppler,intensity] = 5)."},
    )
    radar_embed_dim: int = field(
        default=0,
        metadata={"help": "Radar tokenizer output dim. 0 = auto match ViT.hidden_size (1024)."},
    )
    radar_seq_len: int = field(
        default=16,
        metadata={"help": "Fixed radar frame count; per-frame features are interpolated to this length."},
    )
    radar_tokens_per_frame: int = field(
        default=4,
        metadata={"help": "Attention-pooled sub-tokens per frame. "
                          "seq_len * tokens_per_frame must be factorable for ViT grid "
                          "(16*4=64 -> grid 8x8 -> merger /4 -> 16 LLM tokens)."},
    )
    radar_count_dim: int = field(
        default=64,
        metadata={"help": "Intermediate dim of the per-frame point-count embedding."},
    )
    radar_pool_mode: str = field(
        default="attn",
        metadata={"help": "Radar tokenizer: 'attn' = query attention; "
                          "'max' = masked max + M bias copies; "
                          "'simple' = one max token per frame, interpolate to "
                          "seq_len*tokens_per_frame distinct time steps (no t/bias/count). "
                          "Default attn preserves old checkpoints."},
    )
    radar_grid_mode: str = field(
        default=None,
        metadata={"help": "ViT grid for radar tokens: 'square' | 'temporal1d'. "
                          "None = square for attn/max, temporal1d for simple."},
    )
    radar_bottleneck_dim: int = field(
        default=128,
        metadata={"help": "Bottleneck dim for original RadarTokenizer v_proj/out_proj. "
                          "Ignored when radar_pool_mode=max."},
    )
    # ── Tactile (触觉图像) ──────────────────────────────────────────────────
    mm_tactile_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for tactile branch. None disables tactile. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    tactile_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for tactile."},
    )
    tactile_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the tactile CoCa loss."},
    )
    # ── Thermal (热成像) ────────────────────────────────────────────────────
    mm_thermal_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for thermal branch. None disables thermal. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    thermal_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for thermal. "
                          "Vision tokens = cat(RGB[:512], thermal[:512])."},
    )
    thermal_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the thermal CoCa loss."},
    )
    # ── Smell (multichannel gas sensor) ─────────────────────────────────────
    mm_smell_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for smell branch. None disables smell. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    smell_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for smell."},
    )
    smell_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the smell CoCa loss."},
    )
    smell_in_channels: int = field(
        default=6,
        metadata={"help": "Input channel dim of smell sensor (NO2, C2H5OH, VOC, CO, Alcohol, LPG = 6)."},
    )
    smell_embed_dim: int = field(
        default=0,
        metadata={"help": "Smell tokenizer output dim. 0 = auto match ViT.hidden_size (1024)."},
    )
    smell_seq_len: int = field(
        default=100,
        metadata={"help": "Fixed smell time-step count; input is interpolated to this length "
                          "(matches SmellNet preprocessing window=100)."},
    )
    smell_conv_kernel: int = field(
        default=10,
        metadata={"help": "Conv1D kernel size in SmellTokenizer (OneLLM-style, same as IMU)."},
    )
    smell_tokenizer_depth: int = field(
        default=1,
        metadata={"help": "Number of stacked Conv1D layers in SmellTokenizer. Default 1 = "
                          "original single-layer OneLLM style (checkpoint-compatible). >1 "
                          "stacks extra same-padding Conv1D+GELU layers to widen the "
                          "temporal receptive field."},
    )
    smell_use_gru: bool = field(
        default=False,
        metadata={"help": "Add a bidirectional GRU after the Conv1D stack in SmellTokenizer "
                          "to model longer-range temporal dependencies. Default off."},
    )
    smell_tokenizer_arch: str = field(
        default="conv1d",
        metadata={"help": "SmellTokenizer architecture. 'conv1d' (default): original "
                          "OneLLM-style single/stacked Conv1D, no activation, checkpoint-"
                          "compatible. 'cnn': lightweight multi-layer Conv1D+GELU stack "
                          "(see docs/smell.md §11.6/§11.4) followed by a 1x1 Conv "
                          "projection to embed_dim; mutually exclusive with "
                          "smell_tokenizer_depth/smell_use_gru (conv1d-only knobs)."},
    )
    smell_cnn_channels: str = field(
        default="64,128",
        metadata={"help": "Comma-separated hidden channel sizes for smell_tokenizer_arch="
                          "'cnn', e.g. '64,128' for two Conv1D+GELU layers before the final "
                          "1x1 projection to embed_dim."},
    )
    smell_cnn_kernel: int = field(
        default=7,
        metadata={"help": "Conv1D kernel size used by each layer of smell_tokenizer_arch='cnn'."},
    )
    smell_grid_mode: str = field(
        default="square",
        metadata={"help": "Fake-2D grid shape passed to the frozen ViT for smell tokens. "
                          "'square' (default): closest-to-square (H,W) via _grid_thw_for_tokens "
                          "(e.g. 100 tokens -> 10x10), checkpoint-compatible with all existing "
                          "smell runs. 'temporal1d': H=spatial_merge_size, W=num_tokens/H, so "
                          "each merge block covers spatial_merge_size**2 consecutive raw time "
                          "steps with monotonically increasing position ids (no periodic row-wrap "
                          "seen with 'square'); validated via linear-probe ablation to recover "
                          "part of the frozen-ViT B-point loss (see docs/smell.md §11.10/§11.11)."},
    )
    smell_vit_depth: int = field(
        default=-1,
        metadata={"help": "Number of frozen ViT self-attention blocks to run for smell tokens "
                          "before the Patch Merger, out of the full stack (24 for Qwen3.5-2B ViT). "
                          "-1 (default): run all blocks, checkpoint-compatible with all existing "
                          "smell runs. 0: skip every self-attention block entirely — tokens only "
                          "get the absolute position embedding added, then go straight into the "
                          "Patch Merger; no gradient/mixing through the frozen photo-pretrained "
                          "attention stack at all. Motivated by docs/smell.md §11.11's depth sweep, "
                          "which found B-point accuracy decays monotonically with depth (55.58%% at "
                          "depth=0 down to 31.47%% at depth=24 on an eval-only ablation of an "
                          "already-trained checkpoint) — this flag makes depth=0 (or any "
                          "intermediate depth) a trainable Stage2/3 configuration instead of an "
                          "eval-only diagnostic. Values other than -1/0 run the first N blocks."},
    )
    smell_use_augment: bool = field(
        default=False,
        metadata={"help": "Enable data augmentation inside SmellTokenizer during training: "
                          "random global amplitude scaling (×U(0.7,1.3)), per-channel "
                          "proportional Gaussian noise (σ=5%%×ch_rms), and circular time "
                          "shift (±10 steps). Disabled at eval automatically (self.training=False)."},
    )
    smell_standardize: bool = field(
        default=False,
        metadata={"help": "Apply global per-channel (x-mean)/std standardization inside "
                          "SmellTokenizer, using train-only statistics from smell_channel_mean/std."},
    )
    smell_channel_mean: str = field(
        default="0,0,0,0,0,0",
        metadata={"help": "Comma-separated per-channel mean (train-only), used when smell_standardize=True."},
    )
    smell_channel_std: str = field(
        default="1,1,1,1,1,1",
        metadata={"help": "Comma-separated per-channel std (train-only), used when smell_standardize=True."},
    )
    # ── Motion (skeleton trajectory, QuAnTS) ────────────────────────────────
    mm_motion_projector_type: str = field(
        default=None,
        metadata={"help": "Projector type for motion branch. None disables motion. "
                          "'simple_mlp' (recommended; 0-init residual), 'identity', or 'linear'."},
    )
    motion_caption: bool = field(
        default=False,
        metadata={"help": "Enable CoCa caption-alignment loss for motion."},
    )
    motion_caption_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the motion CoCa loss."},
    )
    motion_num_joints: int = field(
        default=24,
        metadata={"help": "Skeleton joint count (QuAnTS trajectories are (T, 24, 3))."},
    )
    motion_coord_dim: int = field(
        default=3,
        metadata={"help": "Per-joint coordinate dim (xyz = 3). "
                          "MotionTokenizer's Conv1D in_channels = motion_num_joints * motion_coord_dim."},
    )
    motion_embed_dim: int = field(
        default=0,
        metadata={"help": "Motion tokenizer output dim. 0 = auto match ViT.hidden_size (1024)."},
    )
    motion_seq_len: int = field(
        default=320,
        metadata={"help": "Fixed motion frame count; input is interpolated to this length "
                          "(QuAnTS trajectories are fixed at 320 frames)."},
    )
    motion_conv_kernel: int = field(
        default=10,
        metadata={"help": "Conv1D kernel size in MotionTokenizer (OneLLM-style, same as IMU/Smell)."},
    )
    # ── Decoder-side MoE LoRA ──────────────────────────────────────────────
    moe_enable: bool = field(default=False)
    moe_shared_dense: bool = field(
        default=False,
        metadata={"help": (
            "MSC ablation: install SharedDenseFFN (single shared LoRA on all "
            "tokens) instead of ModalityAwareMoEFFN. Requires moe_enable=True. "
            "Ignores moe_modal_list / moe_num_experts / balance losses."
        )},
    )
    moe_shared_pool: bool = field(
        default=False,
        metadata={"help": (
            "Install one router and one expert pool shared by every modality "
            "in moe_modal_list. Mutually exclusive with moe_shared_dense."
        )},
    )
    moe_modal_list: str = field(default="depth",
                                metadata={"help": "Comma-separated modalities with experts, e.g. 'depth'"})
    moe_num_experts: int = field(default=2,
                                 metadata={"help": "Number of LoRA experts per modality"})
    moe_group_indices: str = field(default="0",
                                   metadata={"help": "Comma-separated group indices (0-based) to patch, "
                                                     "e.g. '0,2,5'. Qwen3.5-2B has 6 groups (0-5)."})
    moe_full_attn_only: bool = field(default=True,
                                     metadata={"help": "Only patch full_attention layers (not linear attention)"})
    moe_proj_mode: str = field(default="up_down",
                               metadata={"help": "'up_down': B(A(x)); 'gate_up_down': SwiGLU-style"})
    moe_lora_r: int = field(default=8, metadata={"help": "MoE expert LoRA rank"})
    moe_lora_alpha: int = field(default=16, metadata={"help": "MoE expert LoRA alpha"})
    moe_balance_loss_type: str = field(
        default="none",
        metadata={"help": (
            "Load-balancing auxiliary loss type. "
            "Options: 'none' | 'load_variance' | 'switch' | 'entropy' | 'z_loss' "
            "| 'switch+z_loss' | 'load_variance+z_loss'"
        )},
    )
    moe_balance_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for MoE load-balancing aux loss (0 = disabled)."},
    )
    moe_diversity_loss_weight: float = field(
        default=0.0,
        metadata={"help": (
            "Weight for MoE expert diversity loss (0 = disabled). "
            "Minimises mean pairwise cosine similarity between per-expert "
            "responsibility-weighted prototype vectors, pushing experts to "
            "specialise in different residual directions. "
            "Recommended starting value: 0.05."
        )},
    )
    moe_diversity_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": (
            "If > 0, linearly warm up the diversity-loss weight over this "
            "fraction of the MoE-active optimizer steps. At train start this "
            "is converted to forward passes using gradient_accumulation_steps, "
            "so it adapts to run length and gradient accumulation. Set to 0 "
            "to use the full diversity-loss weight as soon as MoE is active."
        )},
    )
    moe_diagnostic_interval: int = field(
        default=50,
        metadata={"help": (
            "Collect detached router/expert/delta diagnostics every N optimizer "
            "steps and add them to Trainer logs. 0 disables diagnostics; 50 or "
            "100 is recommended. Statistics are merged across DDP ranks."
        )},
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    is_speech_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    speech_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_resolution: int = field(default=2560,
        metadata={"help": "Max side length for qwen2vl/qwen25vl/qwen35vl image resize "
                          "(_preprocess_image_qwen2vl). Applies to RGB, depth, tactile, thermal."})
    speech_overlap: bool = False
    speech_overlap_time: int = field(default=30)
    image_grid_pinpoints: Optional[str] = field(default=None)
    overlap_ratio: float = field(default=0.0)
    image_grid: Optional[int] = field(default=1)
    image_global: Optional[bool] = field(default=False)
    datasets: Optional[str] = field(default=None)
    radar_max_points: Optional[int] = field(default=128,
        metadata={"help": "Per-frame point pad/truncate count N_max for radar."})
    radar_max_frames: Optional[int] = field(default=32,
        metadata={"help": "Max frames loaded per radar segment (tokenizer interpolates to seq_len)."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_vision_mlp_adapter: bool = field(default=True)
    freeze_mm_speech_mlp_adapter: bool = field(default=True)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    target_modules: str = 'v_proj,o_proj,down_proj,gate_proj,q_proj,k_proj,up_proj'
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_target_modality: bool = field(
        default=False,
        metadata={"help": (
            "Build globally synchronized homogeneous micro-batches for added "
            "modalities while preserving each modality's natural data volume."
        )},
    )
    lr_multi: Optional[str] = field(default=None)
    unfreeze_moe_at_ratio: float = field(
        default=0.0,
        metadata={"help": "Valid range: [0, 1]. 1 keeps MoE router/expert gradients zero "
                          "throughout the run and keeps staged contrastive alignment active. "
                          "For 0 < ratio < 1, keep trainable MoE gradients zero for the "
                          "first <ratio> fraction of optimizer steps, then unfreeze them for "
                          "joint training. Parameters remain in the optimizer from step 0 "
                          "(ZeRO-2 safe). For example, use 0.25 to freeze MoE for the first "
                          "0.5 epoch of a 2-epoch run. With coca_staged_qwen, 0 means its "
                          "contrastive stage was completed by a separate prior run: MoE starts "
                          "immediately and only CoCa caption alignment remains active."},
    )
    moe_lr_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Optional MoE-only LR warmup after the phased-unfreeze boundary, "
                          "expressed as a fraction of total optimizer steps. The global "
                          "scheduler still controls the target LR. Requires MoE parameters "
                          "to be isolated in lr_multi optimizer groups. 0 preserves the "
                          "legacy schedule."},
    )
    load_non_lora_weights_from: Optional[str] = field(default=None,
        metadata={"help": "Path to a checkpoint dir containing non_lora_trainables.bin "
                          "(speech/depth projector weights from a prior stage). "
                          "When set, pretrain_mm_speech_mlp_adapter is ignored to avoid double-loading."})
    load_non_lora_ignore_moe: bool = field(
        default=False,
        metadata={"help": (
            "When loading non_lora_trainables.bin, discard saved MoE router/expert "
            "state and initialize the current run's MoE from scratch. Use this for "
            "Stage-2 projector/alignment checkpoints whose MoE gradients were frozen "
            "but whose requires_grad=True initialization was still saved."
        )},
    )
    load_lora_weights_from: Optional[str] = field(default=None,
        metadata={"help": "Path to a checkpoint dir containing a LoRA adapter "
                          "(adapter_config.json + adapter_model.safetensors). "
                          "Only effective when lora_enable=True."})
    freeze_mm_depth_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze depth projector (modality_projectors.depth) and CoCa module. "
                          "Use this to train only MoE in Stage 3 while keeping Stage 2 weights fixed. "
                          "Frozen params are excluded from non_lora_trainables.bin at save time."})
    unfreeze_depth_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.7), use gradient hooks to keep depth projector + CoCa "
                          "frozen for the first <ratio> fraction of training steps, then unfreeze "
                          "for joint training. Incompatible with freeze_mm_depth_mlp_adapter=True. "
                          "Params remain in optimizer from step 0 (ZeRO-2 safe)."})
    freeze_mm_pointcloud_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze pointcloud tokenizer, projector (modality_projectors.pointcloud) "
                          "and pointcloud CoCa module. Use when training only MoE on top of a "
                          "pre-trained pointcloud projector."})
    unfreeze_pointcloud_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep pointcloud tokenizer + "
                          "projector + CoCa frozen for the first <ratio> fraction of training "
                          "steps, then unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_pointcloud_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})
    freeze_mm_imu_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze IMU tokenizer, projector (modality_projectors.imu) "
                          "and IMU CoCa module. Use when training only MoE on top of a "
                          "pre-trained IMU projector."})
    unfreeze_imu_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep IMU tokenizer + "
                          "projector + CoCa frozen for the first <ratio> fraction of training "
                          "steps, then unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_imu_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})
    freeze_mm_tactile_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze tactile projector (modality_projectors.tactile) and CoCa module. "
                          "Use when training only MoE on top of a pre-trained tactile projector."})
    unfreeze_tactile_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep tactile projector + "
                          "CoCa frozen for the first <ratio> fraction of training steps, then "
                          "unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_tactile_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})
    freeze_mm_thermal_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze thermal projector (modality_projectors.thermal) and CoCa module. "
                          "Use when training only MoE on top of a pre-trained thermal projector."})
    unfreeze_thermal_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep thermal projector + "
                          "CoCa frozen for the first <ratio> fraction of training steps, then "
                          "unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_thermal_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})
    freeze_mm_radar_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze radar tokenizer, projector (modality_projectors.radar) "
                          "and radar CoCa module. Use when training only MoE on top of a "
                          "pre-trained radar projector."})
    unfreeze_radar_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep radar tokenizer + "
                          "projector + CoCa frozen for the first <ratio> fraction of training "
                          "steps, then unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_radar_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})
    freeze_mm_smell_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze smell tokenizer, projector (modality_projectors.smell) "
                          "and smell CoCa module. Use when training only MoE on top of a "
                          "pre-trained smell projector."})
    unfreeze_smell_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep smell tokenizer + "
                          "projector + CoCa frozen for the first <ratio> fraction of training "
                          "steps, then unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_smell_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})
    freeze_mm_motion_mlp_adapter: bool = field(default=False,
        metadata={"help": "Freeze motion tokenizer, projector (modality_projectors.motion) "
                          "and motion CoCa module. Use when training only MoE on top of a "
                          "pre-trained motion projector."})
    unfreeze_motion_at_ratio: float = field(default=0.0,
        metadata={"help": "If > 0 (e.g. 0.5), use gradient hooks to keep motion tokenizer + "
                          "projector + CoCa frozen for the first <ratio> fraction of training "
                          "steps, then unfreeze for joint training with MoE. Incompatible with "
                          "freeze_mm_motion_mlp_adapter=True. Params remain in optimizer "
                          "from step 0 (ZeRO-2 safe)."})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_uni']
        # add vision tower
        keys_to_match.extend(['vision_tower'])
        
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    if getattr(trainer.args, "tune_mm_mlp_adapter", False) or getattr(trainer.args, "tune_speech_projector_only", False):
        # Only save Adapter
        keys_to_match = ['mm_speech_projector']
        
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_speech_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_speech_projector.bin'))
        # return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _mask_assistant_prefixed_think(target, input_id_row, assistant_prefix_ids, think_ids):
    if not assistant_prefix_ids or not think_ids:
        return
    prefix_len = len(assistant_prefix_ids)
    think_len = len(think_ids)
    input_list = input_id_row.tolist()
    max_start = len(input_list) - (prefix_len + think_len) + 1
    for start in range(max_start):
        prefix_end = start + prefix_len
        think_end = prefix_end + think_len
        if input_list[start:prefix_end] == assistant_prefix_ids and input_list[prefix_end:think_end] == think_ids:
            target[prefix_end:think_end] = IGNORE_INDEX


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # calculate the number of image tokens in sentence['value']
                image_token_len = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = sentence['value'].replace('\n', '').strip()
                sentence['value'] = (DEFAULT_IMAGE_TOKEN + '\n') * image_token_len + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            replace_token = '<|vision_start|>' + replace_token + '<|vision_end|>'
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1

        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            round_len += 1
            instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3_1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len += 2
                instruction_len += 2
                # # Hermes-3-Llama-3.1-8B
                # round_len += 1
                # instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def _qwen_think_closed_string() -> str:
    _topen, _tclose = "<" + "think" + ">", "</" + "think" + ">"
    return f"{_topen}\n\n{_tclose}\n\n"


def _tokenize_qwen2vl_text(text: str, tokenizer: transformers.PreTrainedTokenizer) -> List[int]:
    """Tokenize text containing any combination of multimodal special tokens."""
    return tokenizer_multimodal(text, tokenizer)


def _prefix_len_aligned_to_full(
    full_ids: List[int],
    prefix_text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int,
) -> int:
    pref = _tokenize_qwen2vl_text(prefix_text, tokenizer)
    m = min(len(pref), len(full_ids), seq_len)
    matched = 0
    for i in range(m):
        if full_ids[i] != pref[i]:
            break
        matched = i + 1
    return matched


def _assistant_content_char_spans(
    conversation: str,
    conv,
    sep_style,
) -> List[tuple]:
    assistant_header = conv.roles[1]
    turn_sep = conv.sep
    think_closed = _qwen_think_closed_string()

    positions: List[int] = []
    search_from = 0
    while True:
        p = conversation.find(assistant_header, search_from)
        if p < 0:
            break
        positions.append(p)
        search_from = p + len(assistant_header)

    spans: List[tuple] = []
    n_asst = len(positions)
    for k, p in enumerate(positions):
        cs = p + len(assistant_header)
        if sep_style == conversation_lib.SeparatorStyle.QWEN35ALLROUND:
            cs += len(think_closed)
        elif sep_style == conversation_lib.SeparatorStyle.QWEN35_LAST_THINK_ALLROUND and k == n_asst - 1:
            cs += len(think_closed)

        pe = conversation.find(turn_sep, cs)
        if pe < 0:
            pe = len(conversation)
        else:
            pe += len(turn_sep)  # include <|im_end|>\n in CE loss, matching preprocess_qwen35 behavior
        if pe > cs:
            spans.append((cs, pe))
    return spans


def _apply_qwen2vl_label_mask_from_aligned_spans(
    target: torch.Tensor,
    input_id_row: torch.Tensor,
    conversation: str,
    conv,
    tokenizer: transformers.PreTrainedTokenizer,
    sep_style,
) -> None:
    pad_id = tokenizer.pad_token_id
    full_list = input_id_row.tolist()
    if pad_id is not None:
        seq_len = int(input_id_row.ne(pad_id).sum().item())
    else:
        seq_len = len(full_list)

    target.fill_(IGNORE_INDEX)

    spans = _assistant_content_char_spans(conversation, conv, sep_style)
    for cs, ce in spans:
        ts = _prefix_len_aligned_to_full(full_list, conversation[:cs], tokenizer, seq_len)
        te = _prefix_len_aligned_to_full(full_list, conversation[:ce], tokenizer, seq_len)
        te = max(te, ts)
        if te > ts:
            target[ts:te] = input_id_row[ts:te]

    if pad_id is not None:
        target[input_id_row.eq(pad_id)] = IGNORE_INDEX


# QWEN2VL / all-round variants: label mask from assistant char spans aligned to full input_ids.

def preprocess_qwen2vl(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs,  # 保留兼容性，实际不再使用 has_* 参数
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize: 通用函数一次处理所有 special token，无需按模态分支
    input_ids = torch.stack(
        [tokenizer_multimodal(prompt, tokenizer, return_tensors='pt')
         for prompt in conversations], dim=0)

    targets = input_ids.clone()
    
    assert conv.sep_style in (
        conversation_lib.SeparatorStyle.QWEN2VL,
        conversation_lib.SeparatorStyle.QWEN35ALLROUND,
        conversation_lib.SeparatorStyle.QWEN35ALLROUND_NO_THINK,
        conversation_lib.SeparatorStyle.QWEN35_LAST_THINK_ALLROUND,
    )
    # assert conv.sep_style == conversation_lib.SeparatorStyle.CHATML
    think_token_ids = []
    assistant_prefix_ids = []
    if conv.sep_style in (
        conversation_lib.SeparatorStyle.QWEN35ALLROUND,
        conversation_lib.SeparatorStyle.QWEN35_LAST_THINK_ALLROUND,
    ):
        _topen, _tclose = "<" + "think" + ">", "</" + "think" + ">"
        think_closed = f"{_topen}\n\n{_tclose}\n\n"
        think_token_ids = tokenizer(think_closed, add_special_tokens=False).input_ids
        assistant_prefix = conv.sep + conv.roles[1]
        assistant_prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids

    # Mask targets: assistant spans from string, align token boundaries to full input_ids
    for sample_idx, (conversation, target) in enumerate(zip(conversations, targets)):
        input_id_row = input_ids[sample_idx]
        _apply_qwen2vl_label_mask_from_aligned_spans(
            target, input_id_row, conversation, conv, tokenizer, conv.sep_style,
        )
        if think_token_ids:
            _mask_assistant_prefixed_think(target, input_id_row, assistant_prefix_ids, think_token_ids)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen2vl_original(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and has_speech:
        input_ids = torch.stack([tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2VL_ORIGINAL

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt

        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image and has_speech:
                round_len = len(tokenizer_image_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_speech_token(parts[0], tokenizer)) - 2
            elif has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len += 3
                instruction_len += 3

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len += 1

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen35(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs,  # 保留兼容性，实际不再使用 has_* 参数
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN35

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    input_ids_list = []
    labels_list = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        assert len(source) >= 2, f"Invalid conversation length at sample {i}"
        assert source[-1]["from"] == "gpt", f"Last turn must be assistant at sample {i}"

        # 历史轮次：除了最后一条 assistant answer，其余照常进模板
        conv.messages = []
        for j, sentence in enumerate(source[:-1]):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

        # 最后一轮：只加 assistant 占位，让 QWEN35 模板自动生成
        # <|im_start|>assistant\n<think>\n\n</think>\n\n
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        answer = source[-1]["value"] + conv.sep

        full_text = prompt + answer

        # tokenize full sequence — 通用函数处理所有 special token
        full_ids = tokenizer_multimodal(full_text, tokenizer, return_tensors='pt')
        prompt_ids = tokenizer_multimodal(prompt, tokenizer, return_tensors='pt')

        labels = full_ids.clone()
        prompt_len = len(prompt_ids)

        labels[:prompt_len] = IGNORE_INDEX

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    # pad to longest
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=IGNORE_INDEX
    )

    return dict(
        input_ids=input_ids,
        labels=labels,
    )

def preprocess_qwen2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_speech: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and has_speech:
        input_ids = torch.stack([tokenizer_image_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2


    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        
        # include <bos> for all rounds
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            # include <bos> for all rounds
            if has_image and has_speech:
                round_len = len(tokenizer_image_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_speech_token(parts[0], tokenizer)) - 2
            elif has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_speech:
                round_len = len(tokenizer_speech_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # include <|eot_id|> for all rounds
            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len += 3
                instruction_len += 3

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len += 1

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # not included <|im_end|>
            if has_image: 
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # include <|im_end|> for all rounds
            # if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
            if getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def preprocess_speech_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_SPEECH_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_SPEECH_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_speech_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_speech: bool = False,
    has_pointcloud: bool = False,
    has_imu: bool = False,
    has_depth: bool = False,
    has_tactile: bool = False,
    has_thermal: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.SPEECH_PLAIN:
        return preprocess_speech_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1:
        return preprocess_llama_3_1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        return preprocess_qwen2(sources, tokenizer, has_image=has_image, has_speech=has_speech)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2VL:
        return preprocess_qwen2vl(sources, tokenizer, has_image=has_image, has_speech=has_speech, has_pointcloud=has_pointcloud, has_imu=has_imu, has_depth=has_depth, has_tactile=has_tactile)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2VL_ORIGINAL:
        return preprocess_qwen2vl_original(sources, tokenizer, has_image=has_image, has_speech=has_speech)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN35ALLROUND_NO_THINK:
        return preprocess_qwen2vl(sources, tokenizer, has_image=has_image, has_speech=has_speech, has_pointcloud=has_pointcloud, has_imu=has_imu, has_depth=has_depth, has_tactile=has_tactile)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN35ALLROUND:
        return preprocess_qwen2vl(sources, tokenizer, has_image=has_image, has_speech=has_speech, has_pointcloud=has_pointcloud, has_imu=has_imu, has_depth=has_depth, has_tactile=has_tactile)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN35_LAST_THINK_ALLROUND:
        return preprocess_qwen2vl(sources, tokenizer, has_image=has_image, has_speech=has_speech, has_pointcloud=has_pointcloud, has_imu=has_imu, has_depth=has_depth, has_tactile=has_tactile)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN35:
        return preprocess_qwen35(sources, tokenizer, has_image=has_image, has_speech=has_speech, has_pointcloud=has_pointcloud, has_imu=has_imu, has_depth=has_depth, has_tactile=has_tactile)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
