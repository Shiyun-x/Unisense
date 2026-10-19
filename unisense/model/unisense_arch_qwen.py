#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
#    Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
#    Copyright 2024 Zhisheng Zhong, Chengyao Wang
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import hashlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import transformers
import safetensors
import deepspeed
import torch.distributed as dist

# 兼容不同 transformers 版本的 DeepSpeed 集成
try:
    # 旧版本 transformers
    from transformers.deepspeed import is_deepspeed_zero3_enabled  # type: ignore
except ImportError:
    try:
        # 新版本 transformers 将 DeepSpeed 集成放在 integrations 下
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled  # type: ignore
    except ImportError:
        # 若 transformers 不提供该工具函数，则认为未启用 ZeRO-3
        def is_deepspeed_zero3_enabled():
            return False

from .multimodal_encoder.builder import build_vision_tower, build_speech_tower
from .multimodal_encoder.pointcloud_tokenizer import PointCloudTokenizer
from .multimodal_encoder.imu_tokenizer import IMUTokenizer
from .multimodal_encoder.radar_tokenizer import build_radar_tokenizer
from .multimodal_encoder.smell_tokenizer import SmellTokenizer
from .multimodal_encoder.motion_tokenizer import MotionTokenizer
from .multimodal_projector.builder import (
    build_vision_projector, build_speech_projector,
    build_depth_projector, build_pointcloud_projector, build_imu_projector,
    build_tactile_projector, build_thermal_projector, build_radar_projector,
    build_smell_projector, build_motion_projector,
)

from unisense.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEPTH_TOKEN_INDEX,
                             DEFAULT_IMAGE_PATCH_TOKEN, SPEECH_TOKEN_INDEX,
                             POINTCLOUD_TOKEN_INDEX, IMU_TOKEN_INDEX, TACTILE_TOKEN_INDEX,
                             THERMAL_TOKEN_INDEX, RADAR_TOKEN_INDEX, SMELL_TOKEN_INDEX,
                             MOTION_TOKEN_INDEX,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)

from unisense.model.soft_dtw_cuda import SoftDTW
from unisense.model.coca import CoCa
from unisense.model.coca import CLIPTextCfg, MultimodalCfg
from unisense.model.moe_ffn import (
    MODALITY_TEXT, MODALITY_IMAGE, MODALITY_SPEECH, MODALITY_DEPTH,
    MODALITY_POINTCLOUD, MODALITY_IMU, MODALITY_TACTILE, MODALITY_THERMAL,
    MODALITY_RADAR, MODALITY_SMELL, MODALITY_MOTION,
    clear_modality_ids,
)

_TARGET_MODALITY_ID_TO_NAME = {
    MODALITY_DEPTH: 'depth',
    MODALITY_POINTCLOUD: 'pointcloud',
    MODALITY_IMU: 'imu',
    MODALITY_TACTILE: 'tactile',
    MODALITY_THERMAL: 'thermal',
    MODALITY_RADAR: 'radar',
    MODALITY_SMELL: 'smell',
    MODALITY_MOTION: 'motion',
}


class _QwenTeacherEarlyExit(Exception):
    """Internal control flow used to stop a no-grad teacher at a target layer."""

    def __init__(self, hidden_state):
        super().__init__()
        self.hidden_state = hidden_state


def _grid_thw_for_tokens(num_tokens: int, spatial_merge_size: int = 2):
    """为非像素模态 token（点云 / IMU 等）选一对 (H, W) 满足 H*W==num_tokens
    且都被 spatial_merge_size 整除。

    优先选最接近方形的 (H, W)。例如：
        num_tokens=512, spatial_merge_size=2 -> (16, 32)
        num_tokens=256, spatial_merge_size=2 -> (16, 16)
        num_tokens=120, spatial_merge_size=2 -> (10, 12)
    """
    import math
    best = None
    for h in range(spatial_merge_size, num_tokens // spatial_merge_size + 1, spatial_merge_size):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        if w % spatial_merge_size != 0:
            continue
        if best is None or abs(h - w) < abs(best[0] - best[1]):
            best = (h, w)
    if best is None:
        raise ValueError(
            f"Cannot factor num_tokens={num_tokens} into (H, W) with "
            f"spatial_merge_size={spatial_merge_size} divisibility")
    return best


def _grid_thw_temporal1d(num_tokens: int, spatial_merge_size: int = 2):
    """真「1D」布局：H 固定为 spatial_merge_size，W = num_tokens // H。

    这样 merged_h==1，冻结 ViT 的 rot_pos_emb/fast_pos_embed_interpolate 的
    block-order 枚举退化为单纯按列扫描，每个 merge block 直接对应输入序列里
    连续的 spatial_merge_size**2 个原始 token（对 smell/IMU 等时序模态即连续
    的几个时间步），且相邻 block 的位置标号随时间单调递增，不会像"最接近
    方形"的 _grid_thw_for_tokens 那样每隔 H 步出现一次行跳变打断时序局部性。
    经线性探针验证可回收部分冻结 ViT 造成的信息损失（docs/smell.md §11.10）。

    要求 num_tokens 能被 spatial_merge_size**2 整除。
    """
    assert num_tokens % (spatial_merge_size ** 2) == 0, (
        f"num_tokens={num_tokens} must be divisible by spatial_merge_size**2={spatial_merge_size**2}"
    )
    return spatial_merge_size, num_tokens // spatial_merge_size


class UnisenseMetaModel:

    def __init__(self, config):
        super(UnisenseMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            _has_depth = getattr(config, 'mm_depth_projector_type', None) is not None
            _has_pc = getattr(config, 'mm_pointcloud_projector_type', None) is not None
            _has_imu = getattr(config, 'mm_imu_projector_type', None) is not None
            _has_tactile = getattr(config, 'mm_tactile_projector_type', None) is not None
            _has_thermal = getattr(config, 'mm_thermal_projector_type', None) is not None
            _has_radar = getattr(config, 'mm_radar_projector_type', None) is not None
            _has_smell = getattr(config, 'mm_smell_projector_type', None) is not None
            _has_motion = getattr(config, 'mm_motion_projector_type', None) is not None
            if _has_depth or _has_pc or _has_imu or _has_tactile or _has_thermal or _has_radar or _has_smell or _has_motion:
                self.modality_projectors = nn.ModuleDict()
            if _has_depth:
                print("Add depth projector...")
                self.modality_projectors['depth'] = build_depth_projector(config)
            if _has_pc:
                print("Add pointcloud tokenizer + projector...")
                self.pointcloud_tokenizer = PointCloudTokenizer(
                    in_channels=getattr(config, 'pointcloud_in_channels', 6),
                    embed_dim=getattr(config, 'pointcloud_embed_dim',
                                      getattr(config, 'mm_hidden_size', 1024)),
                    use_relative_pos=getattr(config, 'pointcloud_use_relative_pos', True),
                    normalize_neighbors=getattr(config, 'pointcloud_normalize_neighbors', True),
                )
                self.modality_projectors['pointcloud'] = build_pointcloud_projector(config)
            if _has_imu:
                print("Add imu tokenizer + projector...")
                self.imu_tokenizer = IMUTokenizer(
                    in_channels=getattr(config, 'imu_in_channels', 6),
                    embed_dim=(getattr(config, 'imu_embed_dim', 0)
                               or getattr(config, 'mm_hidden_size', 1024)),
                    seq_len=getattr(config, 'imu_seq_len', 120),
                    kernel_size=getattr(config, 'imu_conv_kernel', 10),
                )
                self.modality_projectors['imu'] = build_imu_projector(config)
            if _has_tactile:
                print("Add tactile projector...")
                self.modality_projectors['tactile'] = build_tactile_projector(config)
            if _has_thermal:
                print("Add thermal projector...")
                self.modality_projectors['thermal'] = build_thermal_projector(config)
            if _has_radar:
                print("Add radar tokenizer + projector...")
                self.radar_tokenizer = build_radar_tokenizer(
                    in_channels=getattr(config, 'radar_in_channels', 5),
                    embed_dim=(getattr(config, 'radar_embed_dim', 0)
                               or getattr(config, 'mm_hidden_size', 1024)),
                    seq_len=getattr(config, 'radar_seq_len', 16),
                    tokens_per_frame=getattr(config, 'radar_tokens_per_frame', 4),
                    count_dim=getattr(config, 'radar_count_dim', 64),
                    bottleneck_dim=getattr(config, 'radar_bottleneck_dim', 128),
                    pool_mode=getattr(config, 'radar_pool_mode', 'attn'),
                )
                self.modality_projectors['radar'] = build_radar_projector(config)
            if _has_smell:
                print("Add smell tokenizer + projector...")
                self.smell_tokenizer = SmellTokenizer(
                    in_channels=getattr(config, 'smell_in_channels', 6),
                    embed_dim=(getattr(config, 'smell_embed_dim', 0)
                               or getattr(config, 'mm_hidden_size', 1024)),
                    seq_len=getattr(config, 'smell_seq_len', 100),
                    kernel_size=getattr(config, 'smell_conv_kernel', 10),
                    standardize=getattr(config, 'smell_standardize', False),
                    channel_mean=[float(v) for v in getattr(config, 'smell_channel_mean', '0,0,0,0,0,0').split(',')],
                    channel_std=[float(v) for v in getattr(config, 'smell_channel_std', '1,1,1,1,1,1').split(',')],
                    temporal_depth=getattr(config, 'smell_tokenizer_depth', 1),
                    use_gru=getattr(config, 'smell_use_gru', False),
                    use_augment=getattr(config, 'smell_use_augment', False),
                    arch=getattr(config, 'smell_tokenizer_arch', 'conv1d'),
                    cnn_channels=[int(v) for v in getattr(config, 'smell_cnn_channels', '64,128').split(',')],
                    cnn_kernel_size=getattr(config, 'smell_cnn_kernel', 7),
                )
                self.modality_projectors['smell'] = build_smell_projector(config)
            if _has_motion:
                print("Add motion tokenizer + projector...")
                self.motion_tokenizer = MotionTokenizer(
                    num_joints=getattr(config, 'motion_num_joints', 24),
                    coord_dim=getattr(config, 'motion_coord_dim', 3),
                    embed_dim=(getattr(config, 'motion_embed_dim', 0)
                               or getattr(config, 'mm_hidden_size', 1024)),
                    seq_len=getattr(config, 'motion_seq_len', 320),
                    kernel_size=getattr(config, 'motion_conv_kernel', 10),
                )
                self.modality_projectors['motion'] = build_motion_projector(config)

        if hasattr(config, "mm_speech_tower"):
            self.speech_tower = build_speech_tower(config, delay_load=True)
            self.mm_speech_projector = build_speech_projector(config)
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_speech_tower(self):
        speech_tower = getattr(self, 'speech_tower', None)
        if type(speech_tower) is list:
            speech_tower = speech_tower[0]
        return speech_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # import pdb; pdb.set_trace()
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        vision_encode_chunk_size = int(getattr(
            model_args, 'vision_encode_chunk_size', 1))
        if vision_encode_chunk_size <= 0:
            raise ValueError("vision_encode_chunk_size must be positive")
        self.config.vision_encode_chunk_size = vision_encode_chunk_size
        
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_use_mrope = getattr(model_args, 'mm_use_mrope', False)
        self.config.mm_depth_projector_type = getattr(model_args, 'mm_depth_projector_type', 'linear')
        self.config.depth_caption = getattr(model_args, 'depth_caption', False)
        self.config.depth_caption_weight = getattr(model_args, 'depth_caption_weight', 0.01)
        self.config.object_depth = getattr(model_args, 'object_depth', False)
        # pointcloud config
        self.config.mm_pointcloud_projector_type = getattr(model_args, 'mm_pointcloud_projector_type', None)
        self.config.pointcloud_caption = getattr(model_args, 'pointcloud_caption', False)
        self.config.pointcloud_frozen_qwen_caption = getattr(
            model_args, 'pointcloud_frozen_qwen_caption', False)
        self.config.pointcloud_semantic_align = getattr(
            model_args, 'pointcloud_semantic_align', False)
        self.config.pointcloud_semantic_temperature = getattr(
            model_args, 'pointcloud_semantic_temperature', 0.07)
        self.config.pointcloud_semantic_nce_weight = getattr(
            model_args, 'pointcloud_semantic_nce_weight', 0.1)
        self.config.pointcloud_in_channels = getattr(model_args, 'pointcloud_in_channels', 6)
        self.config.pointcloud_embed_dim = getattr(model_args, 'pointcloud_embed_dim', None) \
                                            or self.config.mm_hidden_size
        self.config.pointcloud_use_relative_pos = getattr(model_args, 'pointcloud_use_relative_pos', True)
        self.config.pointcloud_normalize_neighbors = getattr(model_args, 'pointcloud_normalize_neighbors', True)
        self.config.pointcloud_caption_weight = getattr(model_args, 'pointcloud_caption_weight', 0.1)
        # imu config
        self.config.mm_imu_projector_type = getattr(model_args, 'mm_imu_projector_type', None)
        self.config.imu_caption = getattr(model_args, 'imu_caption', False)
        self.config.imu_in_channels = getattr(model_args, 'imu_in_channels', 6)
        self.config.imu_embed_dim = getattr(model_args, 'imu_embed_dim', None) \
                                    or self.config.mm_hidden_size
        self.config.imu_seq_len = getattr(model_args, 'imu_seq_len', 120)
        self.config.imu_conv_kernel = getattr(model_args, 'imu_conv_kernel', 10)
        self.config.imu_caption_weight = getattr(model_args, 'imu_caption_weight', 0.1)
        # tactile config
        self.config.mm_tactile_projector_type = getattr(model_args, 'mm_tactile_projector_type', None)
        self.config.tactile_caption = getattr(model_args, 'tactile_caption', False)
        self.config.tactile_caption_weight = getattr(model_args, 'tactile_caption_weight', 0.1)
        # thermal config
        self.config.mm_thermal_projector_type = getattr(model_args, 'mm_thermal_projector_type', None)
        self.config.thermal_caption = getattr(model_args, 'thermal_caption', False)
        self.config.thermal_caption_weight = getattr(model_args, 'thermal_caption_weight', 0.1)
        # radar config
        self.config.mm_radar_projector_type = getattr(model_args, 'mm_radar_projector_type', None)
        self.config.radar_caption = getattr(model_args, 'radar_caption', False)
        self.config.radar_frozen_qwen_caption = getattr(
            model_args, 'radar_frozen_qwen_caption', False)
        self.config.radar_semantic_align = getattr(
            model_args, 'radar_semantic_align', False)
        self.config.radar_semantic_temperature = getattr(
            model_args, 'radar_semantic_temperature', 0.07)
        self.config.radar_semantic_nce_weight = getattr(
            model_args, 'radar_semantic_nce_weight', 0.1)
        self.config.radar_in_channels = getattr(model_args, 'radar_in_channels', 5)
        self.config.radar_embed_dim = getattr(model_args, 'radar_embed_dim', None) \
                                      or self.config.mm_hidden_size
        self.config.radar_seq_len = getattr(model_args, 'radar_seq_len', 16)
        self.config.radar_tokens_per_frame = getattr(model_args, 'radar_tokens_per_frame', 4)
        self.config.radar_count_dim = getattr(model_args, 'radar_count_dim', 64)
        self.config.radar_pool_mode = getattr(model_args, 'radar_pool_mode', 'attn')
        self.config.radar_grid_mode = getattr(model_args, 'radar_grid_mode', None)
        self.config.radar_bottleneck_dim = getattr(model_args, 'radar_bottleneck_dim', 128)
        self.config.radar_caption_weight = getattr(model_args, 'radar_caption_weight', 0.1)
        # smell config
        self.config.mm_smell_projector_type = getattr(model_args, 'mm_smell_projector_type', None)
        self.config.smell_caption = getattr(model_args, 'smell_caption', False)
        self.config.smell_in_channels = getattr(model_args, 'smell_in_channels', 6)
        self.config.smell_embed_dim = getattr(model_args, 'smell_embed_dim', None) \
                                       or self.config.mm_hidden_size
        self.config.smell_seq_len = getattr(model_args, 'smell_seq_len', 100)
        self.config.smell_conv_kernel = getattr(model_args, 'smell_conv_kernel', 10)
        self.config.smell_tokenizer_depth = getattr(model_args, 'smell_tokenizer_depth', 1)
        self.config.smell_use_gru = getattr(model_args, 'smell_use_gru', False)
        self.config.smell_use_augment = getattr(model_args, 'smell_use_augment', False)
        self.config.smell_tokenizer_arch = getattr(model_args, 'smell_tokenizer_arch', 'conv1d')
        self.config.smell_cnn_channels = getattr(model_args, 'smell_cnn_channels', '64,128')
        self.config.smell_cnn_kernel = getattr(model_args, 'smell_cnn_kernel', 7)
        self.config.smell_grid_mode = getattr(model_args, 'smell_grid_mode', 'square')
        self.config.smell_vit_depth = getattr(model_args, 'smell_vit_depth', -1)
        self.config.smell_caption_weight = getattr(model_args, 'smell_caption_weight', 0.1)
        self.config.smell_standardize = getattr(model_args, 'smell_standardize', False)
        self.config.smell_channel_mean = getattr(model_args, 'smell_channel_mean', '0,0,0,0,0,0')
        self.config.smell_channel_std = getattr(model_args, 'smell_channel_std', '1,1,1,1,1,1')
        # motion config
        self.config.mm_motion_projector_type = getattr(model_args, 'mm_motion_projector_type', None)
        self.config.motion_caption = getattr(model_args, 'motion_caption', False)
        self.config.motion_num_joints = getattr(model_args, 'motion_num_joints', 24)
        self.config.motion_coord_dim = getattr(model_args, 'motion_coord_dim', 3)
        self.config.motion_embed_dim = getattr(model_args, 'motion_embed_dim', None) \
                                        or self.config.mm_hidden_size
        self.config.motion_seq_len = getattr(model_args, 'motion_seq_len', 320)
        self.config.motion_conv_kernel = getattr(model_args, 'motion_conv_kernel', 10)
        self.config.motion_caption_weight = getattr(model_args, 'motion_caption_weight', 0.1)

        # One alignment interface for every added modality.  An unspecified
        # shared mode deliberately preserves historical per-modality flags and
        # checkpoint shapes; an explicit mode overrides those flags uniformly.
        alignment_modalities = (
            'depth', 'pointcloud', 'imu', 'tactile', 'thermal',
            'radar', 'smell', 'motion',
        )
        shared_alignment = getattr(model_args, 'modality_alignment_type', None)
        if shared_alignment is not None:
            shared_alignment = shared_alignment.lower()
            aliases = {
                'ce': 'none', 'pure_ce': 'none', 'semantic': 'semantic_anchor',
                # Briefly exposed during development; retain as a harmless alias.
                'coca': 'coca_lite',
            }
            shared_alignment = aliases.get(shared_alignment, shared_alignment)
            valid_alignment_types = {
                'none', 'coca_lite', 'coca_original', 'coca_qwen_dual',
                'coca_encoder_embed', 'coca_staged_qwen', 'frozen_qwen',
                'semantic_anchor', 'dual_path',
                'dual_path_contrastive'}
            if shared_alignment not in valid_alignment_types:
                raise ValueError(
                    f"Unknown modality_alignment_type={shared_alignment!r}; expected one of "
                    f"{sorted(valid_alignment_types)}")

        shared_weight = getattr(model_args, 'modality_alignment_weight', None)
        self.config.modality_alignment_type = shared_alignment
        self.config.modality_alignment_weight = shared_weight
        alignment_micro_batch_size = int(getattr(
            model_args, 'modality_alignment_micro_batch_size', 4))
        if alignment_micro_batch_size <= 0:
            raise ValueError("modality_alignment_micro_batch_size must be positive")
        self.config.modality_alignment_micro_batch_size = alignment_micro_batch_size
        coca_width = int(getattr(
            model_args, 'modality_alignment_coca_width', 1024))
        coca_layers = int(getattr(
            model_args, 'modality_alignment_coca_layers', 12))
        coca_heads = int(getattr(
            model_args, 'modality_alignment_coca_heads', 16))
        if coca_width <= 0 or coca_layers <= 0 or coca_heads <= 0:
            raise ValueError(
                "CoCa width, layers, and heads must all be positive")
        if coca_width % coca_heads != 0:
            raise ValueError(
                "modality_alignment_coca_width must be divisible by "
                "modality_alignment_coca_heads")
        self.config.modality_alignment_coca_width = coca_width
        self.config.modality_alignment_coca_layers = coca_layers
        self.config.modality_alignment_coca_heads = coca_heads
        self.config.modality_alignment_coca_stage1_only = bool(getattr(
            model_args, 'modality_alignment_coca_stage1_only', False))
        self.config.modality_semantic_temperature = getattr(
            model_args, 'modality_semantic_temperature', 0.07)
        self.config.modality_semantic_nce_weight = getattr(
            model_args, 'modality_semantic_nce_weight', 0.1)
        shallow_layer = int(getattr(
            model_args, 'modality_alignment_shallow_layer', 8))
        if shallow_layer <= 0:
            raise ValueError("modality_alignment_shallow_layer must be positive")
        shallow_weight = float(getattr(
            model_args, 'modality_alignment_shallow_weight', 2.0))
        if shallow_weight < 0.0:
            raise ValueError("modality_alignment_shallow_weight must be non-negative")
        self.config.modality_alignment_shallow_layer = shallow_layer
        self.config.modality_alignment_shallow_weight = shallow_weight
        contrastive_temperature = float(getattr(
            model_args, 'modality_alignment_contrastive_temperature', 0.07))
        if contrastive_temperature <= 0.0:
            raise ValueError(
                "modality_alignment_contrastive_temperature must be positive")
        contrastive_weight = float(getattr(
            model_args, 'modality_alignment_contrastive_weight', 0.25))
        if contrastive_weight < 0.0:
            raise ValueError(
                "modality_alignment_contrastive_weight must be non-negative")
        self.config.modality_alignment_contrastive_temperature = contrastive_temperature
        self.config.modality_alignment_contrastive_weight = contrastive_weight
        caption_weight = float(getattr(
            model_args, 'modality_alignment_caption_weight', 0.25))
        if caption_weight < 0.0:
            raise ValueError(
                "modality_alignment_caption_weight must be non-negative")
        contrastive_space = str(getattr(
            model_args, 'modality_alignment_contrastive_space',
            'projected')).lower()
        if contrastive_space not in {'raw', 'projected'}:
            raise ValueError(
                "modality_alignment_contrastive_space must be raw or projected")
        contrastive_dim = int(getattr(
            model_args, 'modality_alignment_contrastive_dim', 512))
        if contrastive_dim <= 0:
            raise ValueError(
                "modality_alignment_contrastive_dim must be positive")
        teacher_pooling = str(getattr(
            model_args, 'modality_alignment_teacher_pooling', 'last')).lower()
        if teacher_pooling not in {'last', 'mean'}:
            raise ValueError(
                "modality_alignment_teacher_pooling must be last or mean")
        self.config.modality_alignment_caption_weight = caption_weight
        self.config.modality_alignment_contrastive_space = contrastive_space
        self.config.modality_alignment_contrastive_dim = contrastive_dim
        self.config.modality_alignment_teacher_pooling = teacher_pooling

        for modality in alignment_modalities:
            legacy_modes = []
            if getattr(self.config, f'{modality}_caption', False):
                legacy_modes.append('coca')
            if getattr(self.config, f'{modality}_frozen_qwen_caption', False):
                legacy_modes.append('frozen_qwen')
            if getattr(self.config, f'{modality}_semantic_align', False):
                legacy_modes.append('semantic_anchor')
            if shared_alignment is None:
                if len(legacy_modes) > 1:
                    raise ValueError(
                        f"Legacy alignment flags for {modality} are mutually exclusive")
                alignment_type = legacy_modes[0] if legacy_modes else 'none'
            else:
                alignment_type = shared_alignment
                setattr(self.config, f'{modality}_caption', alignment_type in {
                    'coca_lite', 'coca_qwen_dual', 'coca_encoder_embed',
                    'coca_staged_qwen'})
                setattr(self.config, f'{modality}_frozen_qwen_caption',
                        alignment_type == 'frozen_qwen')
                setattr(self.config, f'{modality}_semantic_align',
                        alignment_type == 'semantic_anchor')
            setattr(self.config, f'{modality}_alignment_type', alignment_type)
            if shared_weight is not None:
                setattr(self.config, f'{modality}_caption_weight', shared_weight)
            if shared_alignment is not None:
                setattr(self.config, f'{modality}_semantic_temperature',
                        self.config.modality_semantic_temperature)
                setattr(self.config, f'{modality}_semantic_nce_weight',
                        self.config.modality_semantic_nce_weight)
            elif not hasattr(self.config, f'{modality}_semantic_temperature'):
                setattr(self.config, f'{modality}_semantic_temperature', 0.07)
                setattr(self.config, f'{modality}_semantic_nce_weight', 0.1)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # depth / pointcloud / imu / tactile / thermal projector 注册（ModuleDict 复用）
        _has_depth = self.config.mm_depth_projector_type is not None
        _has_pc = self.config.mm_pointcloud_projector_type is not None
        _has_imu = self.config.mm_imu_projector_type is not None
        _has_tactile = self.config.mm_tactile_projector_type is not None
        _has_thermal = self.config.mm_thermal_projector_type is not None
        _has_radar = self.config.mm_radar_projector_type is not None
        _has_smell = self.config.mm_smell_projector_type is not None
        _has_motion = self.config.mm_motion_projector_type is not None
        if (_has_depth or _has_pc or _has_imu or _has_tactile or _has_thermal or _has_radar or _has_smell or _has_motion) and getattr(self, 'modality_projectors', None) is None:
            self.modality_projectors = nn.ModuleDict()
        if _has_depth and 'depth' not in self.modality_projectors:
            print("Add depth projector...")
            self.modality_projectors['depth'] = build_depth_projector(self.config).to(device='cuda')
        if _has_pc:
            print("Add pointcloud tokenizer + projector...")
            if getattr(self, 'pointcloud_tokenizer', None) is None:
                self.pointcloud_tokenizer = PointCloudTokenizer(
                    in_channels=self.config.pointcloud_in_channels,
                    embed_dim=self.config.pointcloud_embed_dim,
                    use_relative_pos=self.config.pointcloud_use_relative_pos,
                    normalize_neighbors=self.config.pointcloud_normalize_neighbors,
                ).to(device='cuda')
            if 'pointcloud' not in self.modality_projectors:
                self.modality_projectors['pointcloud'] = build_pointcloud_projector(self.config).to(device='cuda')
        if _has_imu:
            print("Add imu tokenizer + projector...")
            if getattr(self, 'imu_tokenizer', None) is None:
                self.imu_tokenizer = IMUTokenizer(
                    in_channels=self.config.imu_in_channels,
                    embed_dim=self.config.imu_embed_dim,
                    seq_len=self.config.imu_seq_len,
                    kernel_size=self.config.imu_conv_kernel,
                ).to(device='cuda')
            if 'imu' not in self.modality_projectors:
                self.modality_projectors['imu'] = build_imu_projector(self.config).to(device='cuda')
        if _has_tactile and 'tactile' not in self.modality_projectors:
            print("Add tactile projector...")
            self.modality_projectors['tactile'] = build_tactile_projector(self.config).to(device='cuda')
        if _has_thermal and 'thermal' not in self.modality_projectors:
            print("Add thermal projector...")
            self.modality_projectors['thermal'] = build_thermal_projector(self.config).to(device='cuda')
        if _has_radar:
            print("Add radar tokenizer + projector...")
            if getattr(self, 'radar_tokenizer', None) is None:
                self.radar_tokenizer = build_radar_tokenizer(
                    in_channels=self.config.radar_in_channels,
                    embed_dim=self.config.radar_embed_dim,
                    seq_len=self.config.radar_seq_len,
                    tokens_per_frame=self.config.radar_tokens_per_frame,
                    count_dim=self.config.radar_count_dim,
                    bottleneck_dim=getattr(self.config, 'radar_bottleneck_dim', 128),
                    pool_mode=getattr(self.config, 'radar_pool_mode', 'attn'),
                ).to(device='cuda')
            if 'radar' not in self.modality_projectors:
                self.modality_projectors['radar'] = build_radar_projector(self.config).to(device='cuda')
        if _has_smell:
            print("Add smell tokenizer + projector...")
            if getattr(self, 'smell_tokenizer', None) is None:
                self.smell_tokenizer = SmellTokenizer(
                    in_channels=self.config.smell_in_channels,
                    embed_dim=self.config.smell_embed_dim,
                    seq_len=self.config.smell_seq_len,
                    kernel_size=self.config.smell_conv_kernel,
                    standardize=self.config.smell_standardize,
                    channel_mean=[float(v) for v in self.config.smell_channel_mean.split(',')],
                    channel_std=[float(v) for v in self.config.smell_channel_std.split(',')],
                    temporal_depth=self.config.smell_tokenizer_depth,
                    use_gru=self.config.smell_use_gru,
                    use_augment=getattr(self.config, 'smell_use_augment', False),
                    arch=getattr(self.config, 'smell_tokenizer_arch', 'conv1d'),
                    cnn_channels=[int(v) for v in getattr(self.config, 'smell_cnn_channels', '64,128').split(',')],
                    cnn_kernel_size=getattr(self.config, 'smell_cnn_kernel', 7),
                ).to(device='cuda')
            if 'smell' not in self.modality_projectors:
                self.modality_projectors['smell'] = build_smell_projector(self.config).to(device='cuda')
        if _has_motion:
            print("Add motion tokenizer + projector...")
            if getattr(self, 'motion_tokenizer', None) is None:
                self.motion_tokenizer = MotionTokenizer(
                    num_joints=self.config.motion_num_joints,
                    coord_dim=self.config.motion_coord_dim,
                    embed_dim=self.config.motion_embed_dim,
                    seq_len=self.config.motion_seq_len,
                    kernel_size=self.config.motion_conv_kernel,
                ).to(device='cuda')
            if 'motion' not in self.modality_projectors:
                self.modality_projectors['motion'] = build_motion_projector(self.config).to(device='cuda')

        def build_alignment_coca(modality):
            """Build one modality decoder from the shared CoCa configuration.

            Legacy flags retain the old full-width/12-layer shape.
            """
            using_shared_interface = self.config.modality_alignment_type is not None
            if using_shared_interface:
                width = self.config.modality_alignment_coca_width
                layers = self.config.modality_alignment_coca_layers
                heads = self.config.modality_alignment_coca_heads
                external_vocab_projection = True
            else:
                legacy_cfg = MultimodalCfg()
                width = self.config.mm_hidden_size
                layers = legacy_cfg.layers
                heads = legacy_cfg.heads
                external_vocab_projection = False
            if width <= 0 or layers <= 0 or heads <= 0 or width % heads != 0:
                raise ValueError(
                    f"Invalid CoCa config: width={width}, layers={layers}, heads={heads}; "
                    "values must be positive and width must be divisible by heads")
            multimodal_cfg = MultimodalCfg(
                width=width, layers=layers, heads=heads, dim_head=width // heads)
            text_cfg = CLIPTextCfg(
                vocab_size=self.config.vocab_size, width=self.config.mm_hidden_size)
            alignment_type = getattr(self.config, f'{modality}_alignment_type')
            original_contrastive = alignment_type == 'coca_original'
            temperature = max(float(getattr(
                self.config, 'modality_alignment_contrastive_temperature', 0.07)), 1e-6)
            return CoCa(
                text_cfg, multimodal_cfg,
                text_embedding=getattr(model_args, 'text_embedding', None),
                decoder_width=width if external_vocab_projection else None,
                external_vocab_projection=external_vocab_projection,
                original_contrastive=original_contrastive,
                init_logit_scale=math.log(1.0 / temperature),
            )

        for modality in alignment_modalities:
            modality_enabled = getattr(
                self.config, f'mm_{modality}_projector_type', None) is not None
            if (modality_enabled
                    and getattr(self.config, f'{modality}_alignment_type')
                    in {'coca', 'coca_lite', 'coca_original', 'coca_qwen_dual',
                        'coca_encoder_embed', 'coca_staged_qwen'}):
                setattr(self, f'{modality}_caption_module', build_alignment_coca(modality))
                if (getattr(self.config, f'{modality}_alignment_type')
                        == 'coca_qwen_dual'
                        and self.config.modality_alignment_contrastive_space == 'projected'
                        and self.config.modality_alignment_contrastive_weight > 0.0):
                    hidden_size = int(self.config.hidden_size)
                    metric_head = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(
                            hidden_size,
                            self.config.modality_alignment_contrastive_dim,
                            bias=False),
                    ).to(device='cuda')
                    setattr(self, f'{modality}_alignment_metric_head', metric_head)
                if (getattr(self.config, f'{modality}_alignment_type')
                        in {'coca_encoder_embed', 'coca_staged_qwen'}):
                    hidden_size = int(self.config.hidden_size)
                    staged_qwen = (
                        getattr(self.config, f'{modality}_alignment_type')
                        == 'coca_staged_qwen')
                    for branch in ('student', 'text'):
                        if staged_qwen:
                            # These heads are created after from_pretrained and
                            # before Trainer seeds its runtime. Give each branch
                            # an independent, reproducible initialization without
                            # consuming the caller's global RNG stream.
                            seed_material = (
                                f'coca_staged_qwen:{modality}:{branch}'
                                .encode('utf-8'))
                            head_seed = int.from_bytes(
                                hashlib.sha256(seed_material).digest()[:8],
                                byteorder='little') % (2 ** 31)
                            with torch.random.fork_rng(devices=[]):
                                torch.manual_seed(head_seed)
                                metric_head = nn.Sequential(
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(
                                        hidden_size,
                                        self.config.modality_alignment_contrastive_dim,
                                        bias=False),
                                )
                        else:
                            metric_head = nn.Sequential(
                                nn.LayerNorm(hidden_size),
                                nn.Linear(
                                    hidden_size,
                                    self.config.modality_alignment_contrastive_dim,
                                    bias=False),
                            )
                        metric_head = metric_head.to(device='cuda')
                        setattr(
                            self, f'{modality}_alignment_{branch}_head', metric_head)
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            if 'model' in mm_projector_weights.keys():
                mm_projector_weights = mm_projector_weights['model']
                if is_deepspeed_zero3_enabled():
                    if len(mm_projector_weights) > 0:
                        with deepspeed.zero.GatheredParameters(mm_projector_weights, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_projector.load_state_dict(mm_projector_weights)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_projector.load_state_dict(mm_projector_weights, strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
            else:
                if is_deepspeed_zero3_enabled():
                    named_parameters = get_w(mm_projector_weights, 'mm_projector')
                    if len(named_parameters) > 0:
                        with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_projector.load_state_dict(named_parameters)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_mlp_adapter, status.missing_keys))
            self.mm_projector = self.mm_projector.to(device='cuda')
            
    def initialize_speech_modules(self, model_args, fsdp=None):  
        speech_tower = model_args.speech_tower
        pretrain_mm_speech_mlp_adapter = model_args.pretrain_mm_speech_mlp_adapter
        self.config.mm_speech_tower = speech_tower
        self.config.speech_encoder_ds_rate = model_args.speech_encoder_ds_rate
        self.config.speech_encoder_hidden_size = model_args.speech_encoder_hidden_size
        if self.get_speech_tower() is None:
            speech_tower = build_speech_tower(model_args)
            speech_tower.requires_grad_(False)

            if fsdp is not None and len(fsdp) > 0:
                self.speech_tower = [speech_tower]
            else:
                self.speech_tower = speech_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                speech_tower = self.speech_tower[0]
            else:
                speech_tower = self.speech_tower
            speech_tower.load_model()

        self.config.mm_speech_projector_type = getattr(model_args, 'mm_speech_projector_type', 'linear')
        if 'encodec' in model_args.speech_tower.lower():
            self.config.mm_speech_hidden_size = speech_tower.config.codebook_dim
        elif 'whisper' in model_args.speech_tower.lower():
            self.config.mm_speech_hidden_size = speech_tower.config.d_model # d_model for whisper and codebook_dim for encodec
        else:
            raise NotImplementedError

        self.soft_dtw = SoftDTW(use_cuda=True, gamma=0.1, contrastive_temp=model_args.align_temperature)
        
        if getattr(self, 'mm_speech_projector', None) is None:
            self.mm_speech_projector = build_speech_projector(self.config)

        if pretrain_mm_speech_mlp_adapter is not None:
            mm_speech_projector_weights = torch.load(pretrain_mm_speech_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword + '.' in k}

            if 'model' in mm_speech_projector_weights.keys():
                mm_speech_projector_weights = mm_speech_projector_weights['model']
                if is_deepspeed_zero3_enabled():
                    if len(mm_speech_projector_weights) > 0:
                        with deepspeed.zero.GatheredParameters(mm_speech_projector_weights, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_speech_projector.load_state_dict(mm_speech_projector_weights)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_speech_projector.load_state_dict(mm_speech_projector_weights, strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
            else:
                if is_deepspeed_zero3_enabled():
                    named_parameters = get_w(mm_speech_projector_weights, 'mm_speech_projector')
                    if len(named_parameters) > 0:
                        with deepspeed.zero.GatheredParameters(named_parameters, modifier_rank=0):
                            if torch.distributed.get_rank() == 0:
                                status = self.mm_speech_projector.load_state_dict(named_parameters)
                                print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
                else:
                    status = self.mm_speech_projector.load_state_dict(get_w(mm_speech_projector_weights, 'mm_speech_projector'), strict=False)
                    print('load pretrain_mm_mlp_adapter from {}, missing_keys:{}'.format(pretrain_mm_speech_mlp_adapter, status.missing_keys))
            self.mm_speech_projector = self.mm_speech_projector.to(device='cuda')
    
class UnisenseMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_speech_tower(self):
        return self.get_model().get_speech_tower()

    def encode_images(self, images, is_video=False):
        # import pdb; pdb.set_trace()
        if "qwen2vl" in self.config.mm_vision_tower.lower() or "qwen25" in self.config.mm_vision_tower.lower() or "qwen35" in self.config.mm_vision_tower.lower():
            image_feature = self.get_model().get_vision_tower()(images)
            return image_feature
        
        if isinstance(images, list):
            image_features = []
            for img in images:
                if len(img.shape) == 3:
                    image_feature = self.get_model().get_vision_tower()(img[None, :])
                else:
                    image_feature = self.get_model().get_vision_tower()(img)
                image_feature = self.get_model().mm_projector(image_feature)
                single_img_token_num = image_feature.shape[-2]
                llm_dim = image_feature.shape[-1]
                # [image_original_resize] + patches
                if getattr(self.config, 'video_inference', False) is False:
                    global_image_feature = image_feature[0, :]
                    grid_image_feature = image_feature[1:, :]
                
                    grid_image_feature = grid_image_feature.reshape(-1, int(single_img_token_num ** 0.5), int(single_img_token_num ** 0.5), llm_dim)
                    downsample = torch.nn.Upsample(scale_factor=self.config.down_factor_1 / self.config.down_factor_2, mode='bilinear')
                    grid_image_feature = grid_image_feature.permute(0, 3, 1, 2)
                    grid_image_feature = downsample(grid_image_feature)
                    grid_image_feature = grid_image_feature.flatten(-2, -1).contiguous()
                    grid_image_feature = grid_image_feature.permute(0, -1, 1).flatten(0, 1).contiguous()
                    image_feature = torch.cat([global_image_feature, grid_image_feature], dim=0)
                else:
                    grid_image_feature = image_feature[0:, :]
                    grid_image_feature = grid_image_feature.reshape(-1, int(single_img_token_num ** 0.5), int(single_img_token_num ** 0.5), llm_dim)
                    downsample = torch.nn.Upsample(scale_factor=self.config.down_factor_1 / self.config.down_factor_2, mode='bilinear')
                    grid_image_feature = grid_image_feature.permute(0, 3, 1, 2)
                    grid_image_feature = downsample(grid_image_feature)
                    grid_image_feature = grid_image_feature.flatten(-2, -1).contiguous()
                    grid_image_feature = grid_image_feature.permute(0, -1, 1).flatten(0, 1).contiguous()
                    image_feature = torch.cat([grid_image_feature], dim=0)

                image_features.append(image_feature)
        
        else:
            batch_size = images.shape[0]
            images = images.flatten(0, 1).contiguous()
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
            single_img_token_num = image_features.shape[-2]
            llm_dim = image_features.shape[-1]
            image_features = image_features.reshape(batch_size, -1, single_img_token_num, llm_dim)
            image_features = image_features.flatten(1, 2).contiguous()
        return image_features

    def encode_image_batches(self, image_batches):
        """Encode a dataloader batch while preserving list[B][num_images].

        Qwen-VL represents variable-resolution inputs as packed patch rows plus
        ``grid_thw`` boundaries, so its frozen ViT can process several image
        items without spatial padding. Other vision towers retain the historical
        per-sample path.
        """
        chunk_size = int(getattr(
            self.config, 'vision_encode_chunk_size', 1))
        vision_tower = self.get_vision_tower()
        if chunk_size > 1 and hasattr(vision_tower, 'forward_packed'):
            return vision_tower.forward_packed(
                image_batches, chunk_size=chunk_size)
        return [self.encode_images(images) for images in image_batches]
 
    def encode_pointclouds(self, pointclouds):
        """点云 → ViT-merged token sequence.

        Args:
            pointclouds: list of length B, 每个元素是 dict 或 list[dict]，
                每个 dict 包含：
                    'sampled_features': (P, 6) tensor — FPS 后的点（xyz_normalized + rgb/255）
                    'group_idx':        (G, K) int tensor — KNN 索引
                目前每个 batch 只支持 1 个点云，所以 pointclouds[i] 直接是 dict
                也接受 list[dict] 形式（取第 0 个）。

        Returns:
            pc_features: list of length B，每个元素是 list[Tensor]，
                内层每个 tensor shape (G/spatial_merge_size**2, llm_hidden_size)。
                外层 list 与 batch 对齐；内层 list 与每 batch 多个点云对齐（当前固定 1 个）。
        """
        vision_tower = self.get_vision_tower()
        spatial_merge_size = vision_tower.config.spatial_merge_size
        target_dtype = vision_tower.dtype
        target_device = vision_tower.device

        pc_tokenizer = self.get_model().pointcloud_tokenizer
        pc_projector = self.get_model().modality_projectors['pointcloud']

        out_features = []
        for batch_pc in pointclouds:
            if isinstance(batch_pc, dict):
                batch_pc = [batch_pc]
            per_pc_features = []
            for pc in batch_pc:
                sampled = pc['sampled_features']
                group_idx = pc['group_idx']
                # Ensure tensors on correct device/dtype
                sampled = sampled.to(device=target_device, dtype=target_dtype)
                group_idx = group_idx.to(device=target_device, dtype=torch.long)

                # 1. Tokenizer: (P,6) -> (G, embed_dim)
                tokens = pc_tokenizer(sampled, group_idx)              # (1, G, embed_dim)
                tokens_flat = tokens.squeeze(0)                        # (G, embed_dim)

                # 2. grid_thw for ViT
                G = tokens_flat.shape[0]
                H, W = _grid_thw_for_tokens(G, spatial_merge_size=spatial_merge_size)
                grid_thw = torch.tensor([[1, H, W]], dtype=torch.long, device=target_device)

                # 3. ViT (frozen): skip patch_embed, run blocks + merger
                merged = vision_tower.encode_pre_embedded(tokens_flat, grid_thw)
                # merged: (G/spatial_merge_size**2, llm_hidden_size)

                # 4. Projector to LLM space
                merged = pc_projector(merged)
                per_pc_features.append(merged)
            out_features.append(per_pc_features)
        return out_features

    def align_scene_with_caption(self, pc_embed, caption_ids, objects_len: int = 0):
        """点云场景级 CoCa caption loss（与 align_rgbd_with_caption 对称）。

        Args:
            pc_embed:    (pc_tokens, hidden) — ViT merger 后 + projector 输出
            caption_ids: (1, total_len) — caption token ids，已包含可能的 objects_prefix
            objects_len: 前面 object+sep 的长度（与 depth 路径同语义）

        Returns:
            scalar loss
        """
        # 截断 caption 到 CoCa context_length，防止超出 positional embedding 范围
        context_length = self.get_model().pointcloud_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)        # (1, T, H)
        # 截断到 1024 tokens，保持 K/V 预算
        if pc_embed.shape[0] > 1024:
            pc_embed = pc_embed[:1024]
        image_embed = pc_embed.unsqueeze(0)                                # (1, pc_tok, H)

        logit = self.get_model().pointcloud_caption_module(
            image_embed, caption_embed, objects_len=objects_len,
            vocab_projection=self.get_output_embeddings())                # (1, T, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)    # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                       # (1, cap_len)
            loss = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])
        return loss

    def align_with_coca(
            self, modality, alignment_items, return_sum_count=False):
        """Run CoCa alignment while preserving equal per-sample loss weighting.

        CoCa-Lite items with equal visual length and ``objects_len`` are decoded
        in micro-batches. Caption padding is appended on the right, so causal
        self-attention makes it invisible to every valid prediction. Valid
        decoder states are packed before the frozen Qwen vocabulary projection;
        this avoids both one lm-head call per sample and padded vocabulary logits.

        Legacy CoCa owns its vocabulary projection inside the decoder and keeps
        the historical serial path for checkpoint compatibility.
        """
        caption_module = getattr(self.get_model(), f'{modality}_caption_module')
        if not alignment_items:
            if return_sum_count:
                reference = next(caption_module.parameters())
                return reference.float().sum() * 0.0, 0
            return None

        # Legacy CoCa returns vocabulary logits directly from text_decoder. Its
        # checkpoint topology differs from CoCa-Lite, so retain the old path.
        if not getattr(caption_module, 'external_vocab_projection', False):
            losses = []
            for item in alignment_items:
                modality_embed, caption_ids, objects_len, _ = (
                    self._unpack_alignment_item(item))
                caption_ids = caption_ids.reshape(1, -1).to(device=modality_embed.device)
                caption_ids = caption_ids[:, :caption_module.context_length]
                if caption_ids.shape[1] < 2:
                    continue
                modality_embed = modality_embed[:1024]
                caption_embed = self.get_model().embed_tokens(caption_ids)
                logits = caption_module(
                    modality_embed.unsqueeze(0), caption_embed,
                    objects_len=int(objects_len),
                    vocab_projection=self.get_output_embeddings(),
                )
                objects_len = min(int(objects_len), caption_ids.shape[1] - 1)
                if objects_len > 0:
                    pred = logits[:, objects_len - 1:-1].transpose(1, 2)
                    target = caption_ids[:, objects_len:]
                else:
                    pred = logits[:, :-1].transpose(1, 2)
                    target = caption_ids[:, 1:]
                if target.numel() > 0:
                    losses.append(F.cross_entropy(pred, target))
            if return_sum_count:
                if losses:
                    return torch.stack(losses).sum(), len(losses)
                reference = next(caption_module.parameters())
                return reference.float().sum() * 0.0, 0
            return torch.stack(losses).mean() if losses else None

        # A scalar objects_len is consumed by CoCa's cross-attention path, and
        # visual padding has no key-padding mask. Group on both properties so a
        # batched call is exactly equivalent to the former per-item calls.
        grouped_items = {}
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, _ = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=modality_embed.device)
            caption_ids = caption_ids[:caption_module.context_length]
            if caption_ids.numel() < 2:
                continue
            objects_len = min(int(objects_len), caption_ids.numel() - 1)
            key = (objects_len, modality_embed.shape[0],
                   modality_embed.device, modality_embed.dtype)
            grouped_items.setdefault(key, []).append((modality_embed, caption_ids))

        losses = []
        micro_batch_size = int(getattr(
            self.config, 'modality_alignment_micro_batch_size', 4))
        vocab_projection = self.get_output_embeddings()
        for (objects_len, _, _, _), items in grouped_items.items():
            for chunk_start in range(0, len(items), micro_batch_size):
                chunk = items[chunk_start:chunk_start + micro_batch_size]
                max_caption_len = max(ids.numel() for _, ids in chunk)
                padded_ids = torch.zeros(
                    len(chunk), max_caption_len, dtype=torch.long,
                    device=chunk[0][1].device,
                )
                caption_lengths = []
                for batch_idx, (_, caption_ids) in enumerate(chunk):
                    caption_len = caption_ids.numel()
                    padded_ids[batch_idx, :caption_len] = caption_ids
                    caption_lengths.append(caption_len)

                modality_batch = torch.stack([embed for embed, _ in chunk], dim=0)
                caption_embed = self.get_model().embed_tokens(padded_ids)
                decoded = caption_module.text_decoder(
                    modality_batch, caption_embed, objects_len=objects_len)

                # Project only valid next-token states. Keeping each sample's
                # token losses separate preserves the old sample-balanced mean.
                prediction_states = []
                prediction_targets = []
                prediction_lengths = []
                for batch_idx, caption_len in enumerate(caption_lengths):
                    pred_start = objects_len - 1 if objects_len > 0 else 0
                    target_start = objects_len if objects_len > 0 else 1
                    states = decoded[batch_idx, pred_start:caption_len - 1]
                    targets = padded_ids[batch_idx, target_start:caption_len]
                    if targets.numel() == 0:
                        continue
                    prediction_states.append(states)
                    prediction_targets.append(targets)
                    prediction_lengths.append(targets.numel())

                if not prediction_states:
                    continue
                packed_states = torch.cat(prediction_states, dim=0)
                packed_targets = torch.cat(prediction_targets, dim=0)
                packed_logits = vocab_projection(packed_states)
                token_losses = F.cross_entropy(
                    packed_logits, packed_targets, reduction='none')
                offset = 0
                for prediction_len in prediction_lengths:
                    losses.append(token_losses[offset:offset + prediction_len].mean())
                    offset += prediction_len
        if return_sum_count:
            if losses:
                return torch.stack(losses).sum(), len(losses)
            reference = next(caption_module.parameters())
            return reference.float().sum() * 0.0, 0
        return torch.stack(losses).mean() if losses else None

    def _forward_frozen_qwen_to_layer(
            self, base_model, inputs_embeds, attention_mask, position_ids,
            shallow_layer, track_input_grad):
        """Run frozen Qwen only through one requested layer.

        ``track_input_grad`` keeps the student graph back to modality inputs while
        the caption teacher runs without autograd.  Qwen parameters themselves
        remain frozen in both cases.
        """
        layers = self._qwen_text_layers(base_model)
        if int(shallow_layer) > len(layers):
            raise ValueError(
                f"coca_qwen_dual shallow layer {shallow_layer} exceeds "
                f"Qwen depth {len(layers)}")
        captured = []
        hook = layers[int(shallow_layer) - 1].register_forward_hook(
            self._early_exit_teacher_hook)
        clear_modality_ids()
        checkpointed_modules = [
            (module, module.gradient_checkpointing)
            for module in base_model.modules()
            if hasattr(module, 'gradient_checkpointing')
        ]
        for module, _ in checkpointed_modules:
            module.gradient_checkpointing = False
        try:
            with torch.set_grad_enabled(bool(track_input_grad)):
                try:
                    base_model(
                        input_ids=None,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                except _QwenTeacherEarlyExit as early_exit:
                    captured.append(early_exit.hidden_state)
        finally:
            hook.remove()
            for module, original_value in checkpointed_modules:
                module.gradient_checkpointing = original_value
        if len(captured) != 1:
            raise RuntimeError(
                "Frozen Qwen did not exit at the requested shallow layer; "
                f"captured {len(captured)} activations")
        return captured[0]

    def align_with_coca_qwen_dual(self, modality, alignment_items):
        """CoCa-Lite caption CE plus frozen-Qwen contextual alignment.

        The lightweight CoCa decoder supplies token-level caption supervision.
        A separate shallow path aligns modality states to a caption-only Qwen
        teacher in raw Qwen space; optional InfoNCE is computed either there or
        through one shared learned metric head.
        """
        local_caption_sum, local_caption_count = self.align_with_coca(
            modality, alignment_items, return_sum_count=True)
        caption_loss, global_caption_count = self._distributed_global_mean(
            local_caption_sum, local_caption_count)

        base_model = self.get_model()
        embed_weight = base_model.get_input_embeddings().weight
        hidden_size = embed_weight.shape[-1]
        device = embed_weight.device
        dtype = embed_weight.dtype
        shallow_layer = int(getattr(
            self.config, 'modality_alignment_shallow_layer', 8))

        valid_items = []
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, instance_id = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=device)[:512]
            caption_start = min(int(objects_len), caption_ids.numel())
            if modality_embed.shape[0] == 0 or caption_start >= caption_ids.numel():
                continue
            valid_items.append((
                modality_embed, caption_ids, caption_start, instance_id,
                self._caption_token_key(caption_ids, caption_start)))

        student_anchors = []
        teacher_anchors = []
        instance_ids = []
        caption_keys = []
        if valid_items:
            batch_size = len(valid_items)
            max_modality_len = max(item[0].shape[0] for item in valid_items)
            modality_inputs = torch.zeros(
                batch_size, max_modality_len, hidden_size,
                dtype=dtype, device=device)
            modality_mask = torch.zeros(
                batch_size, max_modality_len, dtype=torch.bool, device=device)
            modality_positions = torch.zeros(
                3, batch_size, max_modality_len,
                dtype=torch.long, device=device)
            for batch_idx, (modality_embed, _, _, _, _) in enumerate(valid_items):
                modality_len = modality_embed.shape[0]
                modality_inputs[batch_idx, :modality_len] = modality_embed
                modality_mask[batch_idx, :modality_len] = True
                positions = torch.arange(
                    modality_len, dtype=torch.long, device=device)
                modality_positions[:, batch_idx, :modality_len] = (
                    positions.unsqueeze(0).expand(3, -1))
            student_hidden = self._forward_frozen_qwen_to_layer(
                base_model, modality_inputs, modality_mask,
                modality_positions, shallow_layer, track_input_grad=True)

            max_caption_len = max(item[1].numel() for item in valid_items)
            teacher_inputs = torch.zeros(
                batch_size, max_caption_len, hidden_size,
                dtype=dtype, device=device)
            teacher_mask = torch.zeros(
                batch_size, max_caption_len, dtype=torch.bool, device=device)
            teacher_positions = torch.zeros(
                3, batch_size, max_caption_len,
                dtype=torch.long, device=device)
            with torch.no_grad():
                for batch_idx, (_, caption_ids, _, _, _) in enumerate(valid_items):
                    caption_len = caption_ids.numel()
                    teacher_inputs[batch_idx, :caption_len] = (
                        base_model.embed_tokens(
                            caption_ids.unsqueeze(0)).squeeze(0))
                    teacher_mask[batch_idx, :caption_len] = True
                    positions = torch.arange(
                        caption_len, dtype=torch.long, device=device)
                    teacher_positions[:, batch_idx, :caption_len] = (
                        positions.unsqueeze(0).expand(3, -1))
            teacher_hidden = self._forward_frozen_qwen_to_layer(
                base_model, teacher_inputs, teacher_mask,
                teacher_positions, shallow_layer, track_input_grad=False)

            teacher_pooling = getattr(
                self.config, 'modality_alignment_teacher_pooling', 'last')
            for batch_idx, (modality_embed, caption_ids, caption_start,
                            instance_id, caption_key) in enumerate(valid_items):
                modality_len = modality_embed.shape[0]
                caption_len = caption_ids.numel()
                student_anchors.append(F.normalize(
                    student_hidden[batch_idx, :modality_len]
                    .float().mean(dim=0), dim=-1))
                if teacher_pooling == 'last':
                    pooled_teacher = teacher_hidden[batch_idx, caption_len - 1].float()
                else:
                    pooled_teacher = teacher_hidden[
                        batch_idx, caption_start:caption_len].float().mean(dim=0)
                teacher_anchors.append(F.normalize(pooled_teacher, dim=-1))
                instance_ids.append(instance_id)
                caption_keys.append(caption_key)

        local_student = (
            torch.stack(student_anchors, dim=0) if student_anchors
            else torch.empty(0, hidden_size, dtype=torch.float32, device=device))
        local_teacher = (
            torch.stack(teacher_anchors, dim=0).detach() if teacher_anchors
            else torch.empty(0, hidden_size, dtype=torch.float32, device=device))
        local_positive_sum = (
            1.0 - (local_student * local_teacher).sum(dim=-1)).sum()
        positive_loss, global_pair_count = self._distributed_global_mean(
            local_positive_sum, local_student.shape[0])

        contrastive_weight = float(getattr(
            self.config, 'modality_alignment_contrastive_weight', 0.25))
        contrastive_space = getattr(
            self.config, 'modality_alignment_contrastive_space', 'projected')
        nce_loss = positive_loss.new_zeros(())
        contrastive_accuracy = positive_loss.new_ones(())
        masked_same_instance = 0
        masked_same_caption = 0
        if contrastive_weight > 0.0:
            if contrastive_space == 'projected':
                metric_head = getattr(
                    base_model, f'{modality}_alignment_metric_head')
                metric_dtype = next(metric_head.parameters()).dtype
                nce_left = F.normalize(
                    metric_head(local_student.to(metric_dtype)).float(), dim=-1)
                nce_right = F.normalize(
                    metric_head(local_teacher.to(metric_dtype)).float(), dim=-1)
                right_differentiable = True
            else:
                nce_left = local_student
                nce_right = local_teacher
                right_differentiable = False
            temperature = max(float(getattr(
                self.config,
                'modality_alignment_contrastive_temperature', 0.07)), 1e-6)
            (nce_loss, contrastive_accuracy, nce_pair_count,
             masked_same_instance, masked_same_caption) = (
                self._distributed_symmetric_nce(
                    nce_left, nce_right, instance_ids, caption_keys,
                    1.0 / temperature, left_differentiable=True,
                    right_differentiable=right_differentiable))
            if nce_pair_count != global_pair_count:
                raise RuntimeError(
                    "CoCa-Qwen positive/NCE pair counts diverged: "
                    f"positive={global_pair_count}, nce={nce_pair_count}")

        caption_weight = float(getattr(
            self.config, 'modality_alignment_caption_weight', 0.25))
        shallow_weight = float(getattr(
            self.config, 'modality_alignment_shallow_weight', 2.0))
        combined_loss = (
            caption_weight * caption_loss
            + shallow_weight * (
                positive_loss + contrastive_weight * nce_loss))
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "CoCa-Qwen dual alignment: caption_ce={:.4f}, "
                "caption_weight={}, shallow_layer={}, teacher_pooling={}, "
                "raw_pos={:.4f}, shallow_weight={}, nce={:.4f}, "
                "nce_weight={}, nce_space={}, contrastive_top1={:.3f}, "
                "local_pair_count={}, global_pair_count={}, "
                "global_caption_count={}, masked_same_instance_negatives={}, "
                "masked_same_caption_negatives={}, combined={:.4f}"
                .format(
                    caption_loss.detach().item(), caption_weight,
                    shallow_layer,
                    self.config.modality_alignment_teacher_pooling,
                    positive_loss.detach().item(), shallow_weight,
                    nce_loss.detach().item(), contrastive_weight,
                    contrastive_space,
                    contrastive_accuracy.detach().item(),
                    local_student.shape[0], global_pair_count,
                    global_caption_count, masked_same_instance,
                    masked_same_caption, combined_loss.detach().item()))
        return combined_loss

    def align_with_coca_encoder_embed(self, modality, alignment_items):
        """CoCa caption CE plus direct encoder-to-embedding alignment.

        The modality student is pooled directly from tokenizer/projector output;
        it does not run through Qwen.  The caption side mean-pools frozen Qwen
        token embeddings and likewise enters no decoder layer. Independent
        learned heads map both representations into one normalized metric space,
        where positive cosine and distributed symmetric InfoNCE are computed.
        """
        local_caption_sum, local_caption_count = self.align_with_coca(
            modality, alignment_items, return_sum_count=True)
        caption_loss, global_caption_count = self._distributed_global_mean(
            local_caption_sum, local_caption_count)

        base_model = self.get_model()
        embed_weight = base_model.get_input_embeddings().weight
        hidden_size = embed_weight.shape[-1]
        device = embed_weight.device

        valid_items = []
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, instance_id = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=device)[:512]
            caption_start = min(int(objects_len), caption_ids.numel())
            if modality_embed.shape[0] == 0 or caption_start >= caption_ids.numel():
                continue
            valid_items.append((
                modality_embed, caption_ids, caption_start, instance_id,
                self._caption_token_key(caption_ids, caption_start)))

        local_student_raw = torch.empty(
            0, hidden_size, dtype=torch.float32, device=device)
        local_text_raw = torch.empty(
            0, hidden_size, dtype=torch.float32, device=device)
        instance_ids = []
        caption_keys = []
        if valid_items:
            student_raw = []
            text_raw = []
            for (modality_embed, caption_ids, caption_start,
                 instance_id, caption_key) in valid_items:
                student_raw.append(modality_embed.float().mean(dim=0))
                with torch.no_grad():
                    caption_embed = base_model.embed_tokens(
                        caption_ids.unsqueeze(0)).squeeze(0)
                text_raw.append(
                    caption_embed[caption_start:].float().mean(dim=0))
                instance_ids.append(instance_id)
                caption_keys.append(caption_key)
            local_student_raw = torch.stack(student_raw, dim=0)
            local_text_raw = torch.stack(text_raw, dim=0)

        student_head = getattr(
            base_model, f'{modality}_alignment_student_head')
        text_head = getattr(base_model, f'{modality}_alignment_text_head')
        head_dtype = next(student_head.parameters()).dtype
        local_student = F.normalize(
            student_head(local_student_raw.to(head_dtype)).float(), dim=-1)
        local_text = F.normalize(
            text_head(local_text_raw.to(head_dtype)).float(), dim=-1)

        local_positive_sum = (
            1.0 - (local_student * local_text).sum(dim=-1)).sum()
        positive_loss, global_pair_count = self._distributed_global_mean(
            local_positive_sum, local_student.shape[0])

        contrastive_weight = float(getattr(
            self.config, 'modality_alignment_contrastive_weight', 0.25))
        nce_loss = positive_loss.new_zeros(())
        contrastive_accuracy = positive_loss.new_ones(())
        masked_same_instance = 0
        masked_same_caption = 0
        if contrastive_weight > 0.0:
            temperature = max(float(getattr(
                self.config,
                'modality_alignment_contrastive_temperature', 0.07)), 1e-6)
            (nce_loss, contrastive_accuracy, nce_pair_count,
             masked_same_instance, masked_same_caption) = (
                self._distributed_symmetric_nce(
                    local_student, local_text, instance_ids, caption_keys,
                    1.0 / temperature, left_differentiable=True,
                    right_differentiable=True))
            if nce_pair_count != global_pair_count:
                raise RuntimeError(
                    "Encoder-embedding positive/NCE pair counts diverged: "
                    f"positive={global_pair_count}, nce={nce_pair_count}")

        caption_weight = float(getattr(
            self.config, 'modality_alignment_caption_weight', 0.25))
        alignment_weight = float(getattr(
            self.config, 'modality_alignment_shallow_weight', 2.0))
        combined_loss = (
            caption_weight * caption_loss
            + alignment_weight * (
                positive_loss + contrastive_weight * nce_loss))
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "CoCa encoder-embedding alignment: caption_ce={:.4f}, "
                "caption_weight={}, text_pooling=mean, "
                "projected_pos={:.4f}, alignment_weight={}, nce={:.4f}, "
                "nce_weight={}, contrastive_top1={:.3f}, "
                "local_pair_count={}, global_pair_count={}, "
                "global_caption_count={}, masked_same_instance_negatives={}, "
                "masked_same_caption_negatives={}, combined={:.4f}"
                .format(
                    caption_loss.detach().item(), caption_weight,
                    positive_loss.detach().item(), alignment_weight,
                    nce_loss.detach().item(), contrastive_weight,
                    contrastive_accuracy.detach().item(),
                    local_student.shape[0], global_pair_count,
                    global_caption_count, masked_same_instance,
                    masked_same_caption, combined_loss.detach().item()))
        return combined_loss

    def align_with_coca_staged_qwen(self, modality, alignment_items):
        """CoCa caption CE plus stage-1-only projected Qwen InfoNCE.

        During the projector warm-up, modality tokens and caption tokens run
        through the same frozen Qwen prefix. Two independent small projection
        heads map their pooled hidden states into the contrastive space. Both
        heads are trainable, while the Qwen backbone that produces their inputs
        remains frozen. Once MoE unfreezes,
        ``modality_alignment_contrastive_end_forward`` disables this branch and
        its extra Qwen forwards. ``modality_alignment_coca_stage1_only`` also
        disables CoCa caption CE at that boundary when requested.
        """
        end_forward = int(getattr(
            self.config, 'modality_alignment_contrastive_end_forward', 0))
        upcoming_forward = int(self._moe_forward_count.item()) + 1
        if (bool(getattr(
                self.config, 'modality_alignment_coca_stage1_only', False))
                and end_forward > 0 and upcoming_forward > end_forward):
            # Returning None removes the auxiliary objective entirely. In
            # particular, no zero-valued surrogate is attached to the pointcloud
            # features, so the projector's only remaining gradient source is the
            # main SFT CE path.
            return None

        local_caption_sum, local_caption_count = self.align_with_coca(
            modality, alignment_items, return_sum_count=True)
        caption_loss, global_caption_count = self._distributed_global_mean(
            local_caption_sum, local_caption_count)

        if end_forward <= 0 or upcoming_forward > end_forward:
            return caption_loss

        base_model = self.get_model()
        embed_weight = base_model.get_input_embeddings().weight
        hidden_size = embed_weight.shape[-1]
        device = embed_weight.device
        dtype = embed_weight.dtype
        shallow_layer = int(getattr(
            self.config, 'modality_alignment_shallow_layer', 8))

        valid_items = []
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, instance_id = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=device)[:512]
            caption_start = min(int(objects_len), caption_ids.numel())
            if modality_embed.shape[0] == 0 or caption_start >= caption_ids.numel():
                continue
            valid_items.append((
                modality_embed, caption_ids, caption_start, instance_id,
                self._caption_token_key(caption_ids, caption_start)))

        modality_anchors = []
        caption_anchors = []
        instance_ids = []
        caption_keys = []
        if valid_items:
            batch_size = len(valid_items)
            max_modality_len = max(item[0].shape[0] for item in valid_items)
            modality_inputs = torch.zeros(
                batch_size, max_modality_len, hidden_size,
                dtype=dtype, device=device)
            modality_mask = torch.zeros(
                batch_size, max_modality_len, dtype=torch.bool, device=device)
            modality_positions = torch.zeros(
                3, batch_size, max_modality_len,
                dtype=torch.long, device=device)
            for batch_idx, (modality_embed, _, _, _, _) in enumerate(valid_items):
                modality_len = modality_embed.shape[0]
                modality_inputs[batch_idx, :modality_len] = modality_embed
                modality_mask[batch_idx, :modality_len] = True
                positions = torch.arange(
                    modality_len, dtype=torch.long, device=device)
                modality_positions[:, batch_idx, :modality_len] = (
                    positions.unsqueeze(0).expand(3, -1))
            modality_hidden = self._forward_frozen_qwen_to_layer(
                base_model, modality_inputs, modality_mask,
                modality_positions, shallow_layer, track_input_grad=True)

            max_caption_len = max(item[1].numel() for item in valid_items)
            caption_inputs = torch.zeros(
                batch_size, max_caption_len, hidden_size,
                dtype=dtype, device=device)
            caption_mask = torch.zeros(
                batch_size, max_caption_len, dtype=torch.bool, device=device)
            caption_positions = torch.zeros(
                3, batch_size, max_caption_len,
                dtype=torch.long, device=device)
            with torch.no_grad():
                for batch_idx, (_, caption_ids, _, _, _) in enumerate(valid_items):
                    caption_len = caption_ids.numel()
                    caption_inputs[batch_idx, :caption_len] = (
                        base_model.embed_tokens(
                            caption_ids.unsqueeze(0)).squeeze(0))
                    caption_mask[batch_idx, :caption_len] = True
                    positions = torch.arange(
                        caption_len, dtype=torch.long, device=device)
                    caption_positions[:, batch_idx, :caption_len] = (
                        positions.unsqueeze(0).expand(3, -1))
            caption_hidden = self._forward_frozen_qwen_to_layer(
                base_model, caption_inputs, caption_mask,
                caption_positions, shallow_layer, track_input_grad=False)

            text_pooling = getattr(
                self.config, 'modality_alignment_teacher_pooling', 'last')
            for batch_idx, (modality_embed, caption_ids, caption_start,
                            instance_id, caption_key) in enumerate(valid_items):
                modality_len = modality_embed.shape[0]
                caption_len = caption_ids.numel()
                modality_anchors.append(
                    modality_hidden[batch_idx, :modality_len]
                    .float().mean(dim=0))
                if text_pooling == 'last':
                    caption_global = caption_hidden[
                        batch_idx, caption_len - 1].float()
                else:
                    caption_global = caption_hidden[
                        batch_idx, caption_start:caption_len].float().mean(dim=0)
                caption_anchors.append(caption_global)
                instance_ids.append(instance_id)
                caption_keys.append(caption_key)

        local_modality_raw = (
            torch.stack(modality_anchors, dim=0) if modality_anchors
            else torch.empty(
                0, hidden_size, dtype=torch.float32, device=device))
        local_caption_raw = (
            torch.stack(caption_anchors, dim=0) if caption_anchors
            else torch.empty(
                0, hidden_size, dtype=torch.float32, device=device))
        student_head = getattr(
            base_model, f'{modality}_alignment_student_head')
        text_head = getattr(base_model, f'{modality}_alignment_text_head')
        student_dtype = next(student_head.parameters()).dtype
        text_dtype = next(text_head.parameters()).dtype
        local_modality = F.normalize(
            student_head(local_modality_raw.to(student_dtype)).float(), dim=-1)
        local_caption = F.normalize(
            text_head(local_caption_raw.to(text_dtype)).float(), dim=-1)
        temperature = max(float(getattr(
            self.config,
            'modality_alignment_contrastive_temperature', 0.07)), 1e-6)
        self._diagnose_c2_contrastive(
            local_modality, local_caption, upcoming_forward, temperature)
        (nce_loss, contrastive_accuracy, global_pair_count,
         masked_same_instance, masked_same_caption) = (
            self._distributed_symmetric_nce(
                local_modality, local_caption, instance_ids, caption_keys,
                1.0 / temperature, left_differentiable=True,
                right_differentiable=True))
        if global_caption_count != global_pair_count:
            raise RuntimeError(
                "Staged-Qwen caption/pair counts diverged: "
                f"caption={global_caption_count}, pair={global_pair_count}")

        contrastive_weight = float(getattr(
            self.config, 'modality_alignment_contrastive_weight', 0.25))
        combined_loss = caption_loss + contrastive_weight * nce_loss
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "C2 staged-Qwen alignment: caption_ce={:.4f}, layer={}, "
                "projection_dim={}, projection_heads=trainable, "
                "nce={:.4f}, nce_weight={}, contrastive_top1={:.3f}, "
                "active_forward={}/{}, global_pair_count={}, "
                "masked_same_instance_negatives={}, "
                "masked_same_caption_negatives={}, combined={:.4f}"
                .format(
                    caption_loss.detach().item(), shallow_layer,
                    local_modality.shape[-1],
                    nce_loss.detach().item(), contrastive_weight,
                    contrastive_accuracy.detach().item(), upcoming_forward,
                    end_forward, global_pair_count, masked_same_instance,
                    masked_same_caption, combined_loss.detach().item()))
        return combined_loss

    def _diagnose_c2_contrastive(
            self, local_modality, local_caption, active_forward, temperature):
        """Opt-in diagnostics for the real C2 anchors and backward graph.

        Set ``UNISENSE_C2_DEBUG_STEPS=N`` to inspect the first N C2 micro-forwards.
        The default is disabled and therefore adds no collectives or hooks to a
        normal training run.
        """
        debug_steps = int(os.environ.get('UNISENSE_C2_DEBUG_STEPS', '0'))
        if debug_steps <= 0 or active_forward > debug_steps:
            return

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if local_modality.requires_grad:
            def report_anchor_gradient(gradient, step=active_forward,
                                       process_rank=rank):
                if process_rank == 0:
                    finite = torch.isfinite(gradient)
                    safe_gradient = gradient[finite]
                    grad_norm = gradient.float().norm().item()
                    grad_abs_mean = (
                        safe_gradient.float().abs().mean().item()
                        if safe_gradient.numel() else float('nan'))
                    print(
                        "C2 diagnostics backward: active_forward={}, "
                        "local_anchor_grad_norm={:.8e}, "
                        "local_anchor_grad_abs_mean={:.8e}, finite={}/{}"
                        .format(
                            step, grad_norm, grad_abs_mean,
                            int(finite.sum().item()), gradient.numel()))
                return gradient

            local_modality.register_hook(report_anchor_gradient)

        if distributed:
            local_count_tensor = torch.tensor(
                local_modality.shape[0], dtype=torch.long,
                device=local_modality.device)
            gathered_counts = [
                torch.zeros_like(local_count_tensor)
                for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_counts, local_count_tensor)
            counts = [int(value.item()) for value in gathered_counts]
            max_count = max(1, max(counts))
            gathered_modality = self._pad_and_gather_embeddings(
                local_modality.detach(), max_count, differentiable=False)
            gathered_caption = self._pad_and_gather_embeddings(
                local_caption.detach(), max_count, differentiable=False)
            global_modality = torch.cat([
                value[:count]
                for value, count in zip(gathered_modality, counts)], dim=0)
            global_caption = torch.cat([
                value[:count]
                for value, count in zip(gathered_caption, counts)], dim=0)
        else:
            global_modality = local_modality.detach()
            global_caption = local_caption.detach()

        if rank != 0 or global_modality.shape[0] < 2:
            return

        with torch.no_grad():
            pair_count = global_modality.shape[0]
            eye = torch.eye(
                pair_count, dtype=torch.bool, device=global_modality.device)
            cross_cosine = global_modality @ global_caption.T
            positive = cross_cosine.diagonal()
            negative = cross_cosine[~eye]
            modality_between = (global_modality @ global_modality.T)[~eye]
            caption_between = (global_caption @ global_caption.T)[~eye]
            logits = cross_cosine / temperature
            print(
                "C2 diagnostics forward: active_forward={}, pairs={}, "
                "student_requires_grad={}, positive_cos={:.6f}+/-{:.6f}, "
                "negative_cos={:.6f}+/-{:.6f}, positive_negative_gap={:.6f}, "
                "student_between_cos={:.6f}, teacher_between_cos={:.6f}, "
                "student_dim_variance={:.8e}, teacher_dim_variance={:.8e}, "
                "logit_std={:.6f}"
                .format(
                    active_forward, pair_count, local_modality.requires_grad,
                    positive.mean().item(), positive.std(unbiased=False).item(),
                    negative.mean().item(), negative.std(unbiased=False).item(),
                    (positive.mean() - negative.mean()).item(),
                    modality_between.mean().item(),
                    caption_between.mean().item(),
                    global_modality.var(dim=0, unbiased=False).mean().item(),
                    global_caption.var(dim=0, unbiased=False).mean().item(),
                    logits.std(unbiased=False).item()))

    def align_with_coca_original(self, modality, alignment_items):
        """Original CoCa objective: caption CE plus symmetric image-text InfoNCE.

        Learned image queries attention-pool pointcloud tokens. Query zero is the
        pointcloud-only contrastive representation; the remaining queries are
        the K/V memory of the caption decoder. A separate CoCa-internal
        text-only transformer produces caption latents without seeing pointcloud.
        """
        caption_module = getattr(self.get_model(), f'{modality}_caption_module')
        if not getattr(caption_module, 'original_contrastive', False):
            raise RuntimeError("coca_original requires an original-contrastive CoCa module")

        grouped_items = {}
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, instance_id = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=modality_embed.device)
            caption_ids = caption_ids[:caption_module.context_length]
            if caption_ids.numel() < 2 or modality_embed.shape[0] == 0:
                continue
            objects_len = min(int(objects_len), caption_ids.numel() - 1)
            caption_key = self._caption_token_key(caption_ids, objects_len)
            key = (objects_len, modality_embed.shape[0],
                   modality_embed.device, modality_embed.dtype)
            grouped_items.setdefault(key, []).append(
                (modality_embed, caption_ids, instance_id, caption_key))

        sample_caption_losses = []
        image_latents = []
        text_latents = []
        instance_ids = []
        caption_keys = []
        micro_batch_size = int(getattr(
            self.config, 'modality_alignment_micro_batch_size', 4))
        vocab_projection = self.get_output_embeddings()

        for (objects_len, _, _, _), items in grouped_items.items():
            for chunk_start in range(0, len(items), micro_batch_size):
                chunk = items[chunk_start:chunk_start + micro_batch_size]
                max_caption_len = max(ids.numel() for _, ids, _, _ in chunk)
                caption_ids_batch = torch.zeros(
                    len(chunk), max_caption_len, dtype=torch.long,
                    device=chunk[0][1].device)
                caption_lengths = []
                for batch_idx, (_, caption_ids, _, _) in enumerate(chunk):
                    caption_len = caption_ids.numel()
                    caption_ids_batch[batch_idx, :caption_len] = caption_ids
                    caption_lengths.append(caption_len)

                text_pool_mask = torch.zeros(
                    len(chunk), max_caption_len, dtype=torch.bool,
                    device=caption_ids_batch.device)
                for batch_idx, caption_len in enumerate(caption_lengths):
                    text_pool_mask[batch_idx, objects_len:caption_len] = True

                modality_batch = torch.stack(
                    [embed for embed, _, _, _ in chunk], dim=0)
                caption_embs = self.get_model().embed_tokens(caption_ids_batch)
                decoder_image_tokens, chunk_image_latents = (
                    caption_module.encode_original_image(modality_batch))
                unimodal_text_tokens, chunk_text_latents = (
                    caption_module.encode_original_text(
                        caption_embs, text_pool_mask))
                decoded = caption_module.text_decoder(
                    decoder_image_tokens, unimodal_text_tokens,
                    objects_len=objects_len, image_embs_projected=True,
                    text_embs_projected=True)

                prediction_states = []
                prediction_targets = []
                prediction_lengths = []
                for batch_idx, caption_len in enumerate(caption_lengths):
                    pred_start = objects_len - 1 if objects_len > 0 else 0
                    target_start = objects_len if objects_len > 0 else 1
                    states = decoded[batch_idx, pred_start:caption_len - 1]
                    targets = caption_ids_batch[batch_idx, target_start:caption_len]
                    if targets.numel() == 0:
                        continue
                    prediction_states.append(states)
                    prediction_targets.append(targets)
                    prediction_lengths.append(targets.numel())

                if prediction_states:
                    packed_states = torch.cat(prediction_states, dim=0)
                    packed_targets = torch.cat(prediction_targets, dim=0)
                    token_losses = F.cross_entropy(
                        vocab_projection(packed_states).float(), packed_targets,
                        reduction='none')
                    offset = 0
                    for prediction_len in prediction_lengths:
                        sample_caption_losses.append(
                            token_losses[offset:offset + prediction_len].mean())
                        offset += prediction_len

                image_latents.append(chunk_image_latents)
                text_latents.append(chunk_text_latents)
                instance_ids.extend(
                    instance_id for _, _, instance_id, _ in chunk)
                caption_keys.extend(
                    caption_key for _, _, _, caption_key in chunk)

        reference = caption_module.logit_scale
        local_caption_sum = (
            torch.stack(sample_caption_losses).sum()
            if sample_caption_losses else reference.float() * 0.0)
        caption_loss, global_caption_count = self._distributed_global_mean(
            local_caption_sum, len(sample_caption_losses))

        contrastive_dim = caption_module.image_to_latent.out_features
        image_latents = (
            torch.cat(image_latents, dim=0) if image_latents
            else torch.empty(
                0, contrastive_dim, dtype=torch.float32,
                device=reference.device))
        text_latents = (
            torch.cat(text_latents, dim=0) if text_latents
            else torch.empty(
                0, contrastive_dim, dtype=torch.float32,
                device=reference.device))

        contrastive_weight = float(getattr(
            self.config, 'modality_alignment_contrastive_weight', 0.25))
        if contrastive_weight > 0.0:
            logit_scale = caption_module.logit_scale.exp().clamp(max=100.0)
            (nce_loss, contrastive_accuracy, global_pair_count,
             masked_same_instance, masked_same_caption) = (
                self._distributed_symmetric_nce(
                    image_latents, text_latents, instance_ids, caption_keys,
                    logit_scale, left_differentiable=True,
                    right_differentiable=True))
        else:
            nce_loss = caption_loss.new_zeros(())
            contrastive_accuracy = caption_loss.new_ones(())
            global_pair_count = global_caption_count
            masked_same_instance = 0
            masked_same_caption = 0
        if global_caption_count != global_pair_count:
            raise RuntimeError(
                "CoCa caption/pair counts diverged: "
                f"caption={global_caption_count}, pair={global_pair_count}")
        combined_loss = caption_loss + contrastive_weight * nce_loss
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "CoCa-original alignment: caption_ce={:.4f}, nce={:.4f}, "
                "nce_weight={}, contrastive_top1={:.3f}, global_pair_count={}, "
                "masked_same_instance_negatives={}, "
                "masked_same_caption_negatives={}, combined={:.4f}"
                .format(
                    caption_loss.detach().item(), nce_loss.detach().item(),
                    contrastive_weight, contrastive_accuracy.detach().item(),
                    global_pair_count, masked_same_instance,
                    masked_same_caption,
                    combined_loss.detach().item()))
        return combined_loss

    @staticmethod
    def _unpack_alignment_item(item):
        """Accept current 4-field items and legacy 3-field direct callers."""
        if len(item) == 4:
            return item
        if len(item) == 3:
            modality_embed, caption_ids, objects_len = item
            return modality_embed, caption_ids, objects_len, None
        raise ValueError(
            "alignment item must contain modality tokens, caption ids, "
            "objects_len, and optionally instance_id")

    @staticmethod
    def _caption_token_key(caption_ids, objects_len=0):
        """Exact identity of the caption tokens used by a contrastive target."""
        caption_ids = caption_ids.reshape(-1)
        start = min(max(int(objects_len), 0), caption_ids.numel())
        if start >= caption_ids.numel():
            start = 0
        return tuple(caption_ids[start:].detach().to(
            device='cpu', dtype=torch.long).tolist())

    @staticmethod
    def _instance_aware_symmetric_nce(
            similarity, instance_ids, caption_keys=None):
        """Symmetric paired InfoNCE with known false negatives masked.

        The diagonal remains the sole positive for each row. Off-diagonal
        entries that share either the underlying modality instance or the exact
        target-caption tokens are removed from the denominator rather than
        promoted to positives, so repeated inputs/labels do not create an
        impossible classification target or an artificial frequency reward.
        Missing identities retain the legacy diagonal-only behavior.
        """
        if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
            raise ValueError(
                "instance-aware local InfoNCE requires a square similarity matrix")
        batch_size = similarity.shape[0]
        if instance_ids is None:
            instance_ids = [None] * batch_size
        if len(instance_ids) != batch_size:
            raise ValueError(
                f"instance id count {len(instance_ids)} does not match "
                f"contrastive batch {batch_size}")
        if caption_keys is None:
            caption_keys = [None] * batch_size
        if len(caption_keys) != batch_size:
            raise ValueError(
                f"caption key count {len(caption_keys)} does not match "
                f"contrastive batch {batch_size}")

        same_instance_mask = torch.zeros(
            batch_size, batch_size, dtype=torch.bool,
            device=similarity.device)
        same_caption_mask = torch.zeros(
            batch_size, batch_size, dtype=torch.bool,
            device=similarity.device)
        for row in range(batch_size):
            for col in range(batch_size):
                if row == col:
                    continue
                row_id, col_id = instance_ids[row], instance_ids[col]
                if row_id is not None and col_id is not None and row_id == col_id:
                    same_instance_mask[row, col] = True
                row_caption, col_caption = caption_keys[row], caption_keys[col]
                if (row_caption is not None and col_caption is not None
                        and row_caption == col_caption):
                    same_caption_mask[row, col] = True

        false_negative_mask = same_instance_mask | same_caption_mask

        masked_similarity = similarity.masked_fill(
            false_negative_mask, float('-inf'))
        labels = torch.arange(batch_size, device=similarity.device)
        nce_loss = 0.5 * (
            F.cross_entropy(masked_similarity, labels)
            + F.cross_entropy(masked_similarity.T, labels))
        with torch.no_grad():
            contrastive_accuracy = 0.5 * (
                (masked_similarity.argmax(dim=1) == labels).float().mean()
                + (masked_similarity.argmax(dim=0) == labels).float().mean())
        return (
            nce_loss,
            contrastive_accuracy,
            int(same_instance_mask.sum().item()),
            int(same_caption_mask.sum().item()),
        )

    @staticmethod
    def _stable_contrastive_hash(value):
        """Return a deterministic signed int64 hash, plus a validity flag."""
        if value is None:
            return 0, False
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode('utf-8')
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder='little', signed=True), True

    @staticmethod
    def _distributed_global_mean(local_sum, local_count):
        """DDP-correct mean for a variable number of local valid samples.

        DDP averages parameter gradients across ranks, so the differentiable
        local sum is scaled by ``world_size / global_count``. A detached global
        reduction supplies the same reported scalar value on every rank.
        """
        local_count = int(local_count)
        if not dist.is_available() or not dist.is_initialized():
            if local_count == 0:
                return local_sum * 0.0, 0
            return local_sum / local_count, local_count

        world_size = dist.get_world_size()
        count_tensor = torch.tensor(
            local_count, dtype=torch.long, device=local_sum.device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_count = int(count_tensor.item())

        reported_sum = local_sum.detach().clone()
        dist.all_reduce(reported_sum, op=dist.ReduceOp.SUM)
        if global_count == 0:
            return local_sum * 0.0, 0

        gradient_value = local_sum * (world_size / global_count)
        reported_value = reported_sum / global_count
        return (
            gradient_value + (reported_value - gradient_value.detach()),
            global_count,
        )

    @staticmethod
    def _pad_and_gather_embeddings(local_embeddings, max_count, differentiable):
        """Gather an equal-shaped padded tensor, preserving gradients if asked."""
        local_count, width = local_embeddings.shape
        if local_count > max_count:
            raise ValueError(
                f"local contrastive count {local_count} exceeds pad size {max_count}")
        if local_count < max_count:
            padding = local_embeddings.new_zeros(
                max_count - local_count, width)
            padded = torch.cat((local_embeddings, padding), dim=0)
        else:
            padded = local_embeddings

        if differentiable:
            # Empty/frozen local branches still need an autograd edge so every
            # rank participates in the all-gather backward collective.
            if not padded.requires_grad:
                padded = padded.detach().requires_grad_(True)
            from torch.distributed.nn.functional import all_gather
            return list(all_gather(padded))

        gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded)
        return gathered

    def _distributed_symmetric_nce(
            self, local_left, local_right, instance_ids, caption_keys,
            logit_scale, left_differentiable=True,
            right_differentiable=False):
        """Variable-length, instance-aware symmetric InfoNCE across DDP ranks."""
        if (local_left.ndim != 2 or local_right.ndim != 2
                or local_left.shape != local_right.shape):
            raise ValueError(
                "paired contrastive embeddings must have matching [N, D] shapes")
        local_count = local_left.shape[0]
        if len(instance_ids) != local_count or len(caption_keys) != local_count:
            raise ValueError(
                "contrastive metadata count must match local embedding count")

        if not dist.is_available() or not dist.is_initialized():
            if local_count < 2:
                graph_zero = (local_left.sum() + local_right.sum()) * 0.0
                return graph_zero, graph_zero.detach() + 1.0, local_count, 0, 0
            similarity = local_left @ local_right.T * logit_scale
            loss, accuracy, masked_instance, masked_caption = (
                self._instance_aware_symmetric_nce(
                    similarity, instance_ids, caption_keys))
            return (
                loss, accuracy, local_count,
                masked_instance, masked_caption,
            )

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        count_tensor = torch.tensor(
            local_count, dtype=torch.long, device=local_left.device)
        gathered_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_counts, count_tensor)
        counts = [int(value.item()) for value in gathered_counts]
        global_count = sum(counts)
        max_count = max(1, max(counts))

        gathered_left = self._pad_and_gather_embeddings(
            local_left, max_count, left_differentiable)
        gathered_right = self._pad_and_gather_embeddings(
            local_right, max_count, right_differentiable)
        global_left = torch.cat(
            [value[:count] for value, count in zip(gathered_left, counts)], dim=0)
        global_right = torch.cat(
            [value[:count] for value, count in zip(gathered_right, counts)], dim=0)

        local_metadata = torch.zeros(
            max_count, 4, dtype=torch.long, device=local_left.device)
        for index, (instance_id, caption_key) in enumerate(
                zip(instance_ids, caption_keys)):
            instance_hash, instance_valid = self._stable_contrastive_hash(
                instance_id)
            caption_hash, caption_valid = self._stable_contrastive_hash(
                caption_key)
            local_metadata[index] = torch.tensor(
                [instance_hash, int(instance_valid),
                 caption_hash, int(caption_valid)],
                dtype=torch.long, device=local_left.device)
        gathered_metadata = [
            torch.empty_like(local_metadata) for _ in range(world_size)]
        dist.all_gather(gathered_metadata, local_metadata)
        global_metadata = torch.cat(
            [value[:count] for value, count in zip(gathered_metadata, counts)],
            dim=0)

        # Make the zero-pair path depend on differentiable gathers so their
        # backward collectives run on every rank, including ranks with N_local=0.
        graph_zero = (global_left.sum() + global_right.sum()) * 0.0
        if global_count < 2:
            return graph_zero, graph_zero.detach() + 1.0, global_count, 0, 0

        offset = sum(counts[:rank])
        labels = torch.arange(
            local_count, device=local_left.device, dtype=torch.long) + offset
        if local_count > 0:
            left_logits = local_left @ global_right.T * logit_scale
            right_logits = local_right @ global_left.T * logit_scale
            local_instance_hash = global_metadata[offset:offset + local_count, 0]
            local_instance_valid = global_metadata[offset:offset + local_count, 1].bool()
            local_caption_hash = global_metadata[offset:offset + local_count, 2]
            local_caption_valid = global_metadata[offset:offset + local_count, 3].bool()

            same_instance = (
                local_instance_hash[:, None] == global_metadata[None, :, 0])
            same_instance &= (
                local_instance_valid[:, None]
                & global_metadata[None, :, 1].bool())
            same_caption = (
                local_caption_hash[:, None] == global_metadata[None, :, 2])
            same_caption &= (
                local_caption_valid[:, None]
                & global_metadata[None, :, 3].bool())
            false_negative_mask = same_instance | same_caption
            false_negative_mask.scatter_(1, labels[:, None], False)

            left_logits = left_logits.masked_fill(
                false_negative_mask, float('-inf'))
            right_logits = right_logits.masked_fill(
                false_negative_mask, float('-inf'))
            local_loss_sum = 0.5 * (
                F.cross_entropy(left_logits, labels, reduction='sum')
                + F.cross_entropy(right_logits, labels, reduction='sum'))
            with torch.no_grad():
                local_correct = 0.5 * (
                    (left_logits.argmax(dim=1) == labels).float().sum()
                    + (right_logits.argmax(dim=1) == labels).float().sum())
            local_masked_instance = same_instance.sum(dtype=torch.long)
            local_masked_caption = same_caption.sum(dtype=torch.long)
            # The paired diagonal was never a negative and is excluded from the
            # reported mask counts as well.
            local_masked_instance -= same_instance.gather(
                1, labels[:, None]).sum(dtype=torch.long)
            local_masked_caption -= same_caption.gather(
                1, labels[:, None]).sum(dtype=torch.long)
        else:
            local_loss_sum = graph_zero
            local_correct = torch.zeros((), device=local_left.device)
            local_masked_instance = torch.zeros(
                (), dtype=torch.long, device=local_left.device)
            local_masked_caption = torch.zeros(
                (), dtype=torch.long, device=local_left.device)

        reported_loss_sum = local_loss_sum.detach().clone()
        dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_masked_instance, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_masked_caption, op=dist.ReduceOp.SUM)

        gradient_loss = local_loss_sum * (world_size / global_count)
        reported_loss = reported_loss_sum / global_count
        loss = gradient_loss + (reported_loss - gradient_loss.detach())
        accuracy = local_correct / global_count
        return (
            loss, accuracy, global_count,
            int(local_masked_instance.item()),
            int(local_masked_caption.item()),
        )

    def compute_modality_alignment(self, modality, alignment_items):
        """Route common alignment items through CoCa, frozen Qwen, or anchors."""
        alignment_type = getattr(
            self.config, f'{modality}_alignment_type', 'none')
        if alignment_type == 'none':
            return None
        if alignment_type in {'coca', 'coca_lite'}:
            if not alignment_items:
                return None
            return self.align_with_coca(modality, alignment_items)
        if alignment_type == 'coca_qwen_dual':
            return self.align_with_coca_qwen_dual(modality, alignment_items)
        if alignment_type == 'coca_encoder_embed':
            return self.align_with_coca_encoder_embed(modality, alignment_items)
        if alignment_type == 'coca_staged_qwen':
            return self.align_with_coca_staged_qwen(modality, alignment_items)
        if alignment_type == 'coca_original':
            return self.align_with_coca_original(modality, alignment_items)
        if alignment_type == 'frozen_qwen':
            return self.align_with_frozen_qwen(alignment_items)
        if alignment_type == 'semantic_anchor':
            return self.align_with_semantic_anchors(
                alignment_items,
                temperature=getattr(
                    self.config, f'{modality}_semantic_temperature', 0.07),
                nce_weight=getattr(
                    self.config, f'{modality}_semantic_nce_weight', 0.1),
            )
        if alignment_type == 'dual_path':
            return self.align_with_frozen_qwen(
                alignment_items,
                shallow_layer=getattr(
                    self.config, 'modality_alignment_shallow_layer', 8),
                shallow_weight=getattr(
                    self.config, 'modality_alignment_shallow_weight', 2.0),
            )
        if alignment_type == 'dual_path_contrastive':
            return self.align_with_frozen_qwen(
                alignment_items,
                shallow_layer=getattr(
                    self.config, 'modality_alignment_shallow_layer', 8),
                shallow_weight=getattr(
                    self.config, 'modality_alignment_shallow_weight', 2.0),
                contrastive_temperature=getattr(
                    self.config, 'modality_alignment_contrastive_temperature', 0.07),
                contrastive_weight=getattr(
                    self.config, 'modality_alignment_contrastive_weight', 0.25),
            )
        raise ValueError(f"Unknown {modality} alignment type: {alignment_type}")

    @staticmethod
    def _qwen_text_layers(base_model):
        """Return the decoder layer list across supported Qwen wrappers."""
        candidates = (
            getattr(base_model, 'layers', None),
            getattr(getattr(base_model, 'language_model', None), 'layers', None),
            getattr(getattr(base_model, 'model', None), 'layers', None),
        )
        for layers in candidates:
            if layers is not None:
                return layers
        raise AttributeError("Cannot locate frozen Qwen text decoder layers")

    @staticmethod
    def _hook_hidden_tensor(module, inputs, output, destination):
        """Save one decoder layer output without retaining every hidden layer."""
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden):
            raise TypeError(
                f"Unexpected Qwen decoder layer output type: {type(output)!r}")
        destination.append(hidden)

    @staticmethod
    def _early_exit_teacher_hook(module, inputs, output):
        """Capture one frozen-teacher layer and skip all subsequent layers."""
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden):
            raise TypeError(
                f"Unexpected Qwen decoder layer output type: {type(output)!r}")
        raise _QwenTeacherEarlyExit(hidden)

    def align_with_frozen_qwen(
            self, alignment_items, shallow_layer=None, shallow_weight=0.0,
            contrastive_temperature=None, contrastive_weight=0.0):
        """Caption loss through the frozen target Qwen instead of a separate CoCa.

        ``alignment_items`` contains ``(modality_tokens, caption_ids,
        objects_len, instance_id)``. Legacy three-field items remain supported.
        Modality tokens are a causal prefix; labels before the actual caption are
        ignored.  Qwen parameters stay frozen, but autograd through this forward is
        intentionally retained so the loss reaches the modality projector.

        The auxiliary forward temporarily disables gradient checkpointing.  MoE
        routing state is process-global and the main SFT forward follows this call;
        allowing this auxiliary graph to checkpoint would make its backward
        recomputation observe the main forward's modality ids.
        """
        base_model = self.get_model()
        # Qwen3.5 exposes ``embed_tokens`` as a compatibility method on Unisense's
        # model wrapper, so the actual nn.Embedding (and its weight/device/dtype)
        # must be obtained through the standard Transformers accessor.
        embed_weight = base_model.get_input_embeddings().weight
        hidden_size = embed_weight.shape[-1]
        device = embed_weight.device
        dtype = embed_weight.dtype
        reference = embed_weight.reshape(-1)[0]
        embed_sequences = []
        label_sequences = []
        valid_alignment_items = []
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, instance_id = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=modality_embed.device)
            if caption_ids.numel() == 0:
                continue
            caption_ids = caption_ids[:512]
            caption_embed = base_model.embed_tokens(caption_ids.unsqueeze(0)).squeeze(0)
            combined = torch.cat((modality_embed, caption_embed), dim=0)
            combined_labels = torch.full(
                (combined.shape[0],), IGNORE_INDEX,
                dtype=torch.long, device=combined.device,
            )
            caption_start = modality_embed.shape[0] + min(int(objects_len), caption_ids.numel())
            combined_labels[caption_start:] = caption_ids[caption_start - modality_embed.shape[0]:]
            embed_sequences.append(combined)
            label_sequences.append(combined_labels)
            valid_alignment_items.append(
                (modality_embed, caption_ids, objects_len, instance_id))

        batch_size = len(embed_sequences)
        captured_student = []
        local_deep_sum = reference.float() * 0.0
        local_deep_count = 0
        if batch_size > 0:
            max_len = max(x.shape[0] for x in embed_sequences)
            inputs_embeds = torch.zeros(
                batch_size, max_len, hidden_size, dtype=dtype, device=device)
            labels = torch.full(
                (batch_size, max_len), IGNORE_INDEX,
                dtype=torch.long, device=device)
            attention_mask = torch.zeros(
                batch_size, max_len, dtype=torch.bool, device=device)
            position_ids = torch.zeros(
                3, batch_size, max_len, dtype=torch.long, device=device)

            for batch_idx, (embeds, item_labels) in enumerate(
                    zip(embed_sequences, label_sequences)):
                seq_len = embeds.shape[0]
                inputs_embeds[batch_idx, :seq_len] = embeds
                labels[batch_idx, :seq_len] = item_labels
                attention_mask[batch_idx, :seq_len] = True
                positions = torch.arange(
                    seq_len, dtype=torch.long, device=device)
                position_ids[:, batch_idx, :seq_len] = (
                    positions.unsqueeze(0).expand(3, -1))

            # Use only the frozen base FFNs for the alignment teacher. The main
            # forward below restores the correct modality ids for MoE training.
            clear_modality_ids()
            checkpointed_modules = [
                (module, module.gradient_checkpointing)
                for module in base_model.modules()
                if hasattr(module, 'gradient_checkpointing')
            ]
            for module, _ in checkpointed_modules:
                module.gradient_checkpointing = False
            hook_handle = None
            if shallow_layer is not None and float(shallow_weight) > 0.0:
                layers = self._qwen_text_layers(base_model)
                if int(shallow_layer) > len(layers):
                    raise ValueError(
                        f"dual_path shallow layer {shallow_layer} exceeds "
                        f"Qwen depth {len(layers)}")
                hook_handle = layers[int(shallow_layer) - 1].register_forward_hook(
                    lambda module, inputs, output: self._hook_hidden_tensor(
                        module, inputs, output, captured_student))
            try:
                outputs = base_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
                for module, original_value in checkpointed_modules:
                    module.gradient_checkpointing = original_value

            hidden_states = outputs[0]
            shift_hidden = hidden_states[:, :-1]
            shift_labels = labels[:, 1:]
            valid = shift_labels != IGNORE_INDEX
            if valid.any():
                logits = self.lm_head(shift_hidden[valid]).float()
                token_losses = F.cross_entropy(
                    logits, shift_labels[valid], reduction='none')
                # Give every alignment sample equal weight regardless of caption
                # length: mean over tokens inside each sample, then sum locally.
                sample_indices = valid.nonzero(as_tuple=False)[:, 0]
                sample_loss_sums = token_losses.new_zeros(batch_size).scatter_add(
                    0, sample_indices, token_losses)
                sample_token_counts = valid.sum(dim=1)
                samples_with_targets = sample_token_counts > 0
                sample_losses = (
                    sample_loss_sums[samples_with_targets]
                    / sample_token_counts[samples_with_targets].to(
                        token_losses.dtype))
                local_deep_sum = sample_losses.sum()
                local_deep_count = int(samples_with_targets.sum().item())

        deep_loss, global_deep_count = self._distributed_global_mean(
            local_deep_sum, local_deep_count)

        if shallow_layer is None or float(shallow_weight) <= 0.0:
            return deep_loss
        if batch_size > 0 and len(captured_student) != 1:
            raise RuntimeError(
                f"Expected one shallow student activation, got {len(captured_student)}")

        student_hidden = captured_student[0] if captured_student else None
        student_anchors = []
        caption_sequences = []
        caption_starts = []
        instance_ids = []
        caption_keys = []
        for batch_idx, item in enumerate(valid_alignment_items):
            modality_embed, caption_ids, objects_len, instance_id = item
            modality_len = min(modality_embed.shape[0], 1024)
            caption_ids = caption_ids.reshape(-1).to(device=device)[:512]
            if modality_len == 0 or caption_ids.numel() == 0:
                continue
            student_anchors.append(F.normalize(
                student_hidden[batch_idx, :modality_len]
                .float().mean(dim=0), dim=-1))
            caption_sequences.append(caption_ids)
            caption_start = min(int(objects_len), caption_ids.numel())
            caption_starts.append(caption_start)
            instance_ids.append(instance_id)
            caption_keys.append(
                self._caption_token_key(caption_ids, caption_start))

        # Recompute the frozen caption teacher every step. This keeps no
        # per-process host-memory cache while still batching all valid captions
        # and exiting Qwen immediately after the requested shallow layer.
        teacher_anchors = []
        if student_anchors:
            teacher_max_len = max(ids.numel() for ids in caption_sequences)
            teacher_embeds = torch.zeros(
                len(caption_sequences), teacher_max_len, hidden_size,
                dtype=dtype, device=device)
            teacher_attention_mask = torch.zeros(
                len(caption_sequences), teacher_max_len,
                dtype=torch.bool, device=device)
            teacher_position_ids = torch.zeros(
                3, len(caption_sequences), teacher_max_len,
                dtype=torch.long, device=device)
            with torch.no_grad():
                for batch_idx, caption_ids in enumerate(caption_sequences):
                    caption_len = caption_ids.numel()
                    teacher_embeds[batch_idx, :caption_len] = (
                        base_model.embed_tokens(
                            caption_ids.unsqueeze(0)).squeeze(0))
                    teacher_attention_mask[batch_idx, :caption_len] = True
                    positions = torch.arange(
                        caption_len, dtype=torch.long, device=device)
                    teacher_position_ids[:, batch_idx, :caption_len] = (
                        positions.unsqueeze(0).expand(3, -1))

            captured_teacher = []
            teacher_layers = self._qwen_text_layers(base_model)
            teacher_hook = teacher_layers[
                int(shallow_layer) - 1].register_forward_hook(
                    self._early_exit_teacher_hook)
            clear_modality_ids()
            checkpointed_modules = [
                (module, module.gradient_checkpointing)
                for module in base_model.modules()
                if hasattr(module, 'gradient_checkpointing')
            ]
            for module, _ in checkpointed_modules:
                module.gradient_checkpointing = False
            try:
                with torch.no_grad():
                    try:
                        base_model(
                            input_ids=None,
                            inputs_embeds=teacher_embeds,
                            attention_mask=teacher_attention_mask,
                            position_ids=teacher_position_ids,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    except _QwenTeacherEarlyExit as early_exit:
                        captured_teacher.append(early_exit.hidden_state)
            finally:
                teacher_hook.remove()
                for module, original_value in checkpointed_modules:
                    module.gradient_checkpointing = original_value

            if len(captured_teacher) != 1:
                raise RuntimeError(
                    "Frozen Qwen teacher did not exit at the requested shallow "
                    f"layer; captured {len(captured_teacher)} activations")
            teacher_hidden = captured_teacher[0]
            for batch_idx, (caption_ids, caption_start) in enumerate(
                    zip(caption_sequences, caption_starts)):
                caption_len = caption_ids.numel()
                if caption_start >= caption_len:
                    # Preserve a valid target for malformed all-prefix samples.
                    caption_start = 0
                teacher_anchors.append(F.normalize(
                    teacher_hidden[batch_idx, caption_start:caption_len]
                    .float().mean(dim=0), dim=-1))

        local_student = (
            torch.stack(student_anchors, dim=0) if student_anchors
            else torch.empty(
                0, hidden_size, dtype=torch.float32, device=device))
        local_teacher = (
            torch.stack(teacher_anchors, dim=0).detach() if teacher_anchors
            else torch.empty(
                0, hidden_size, dtype=torch.float32, device=device))
        local_positive_sum = (
            1.0 - (local_student * local_teacher).sum(dim=-1)).sum()
        positive_loss, global_pair_count = self._distributed_global_mean(
            local_positive_sum, local_student.shape[0])

        # Each row has exactly one paired positive. Other globally gathered
        # captions are negatives except known same-instance/same-caption pairs.
        nce_loss = positive_loss.new_zeros(())
        contrastive_accuracy = positive_loss.new_ones(())
        contrastive_weight = float(contrastive_weight)
        if contrastive_weight > 0.0:
            temperature = max(float(contrastive_temperature), 1e-6)
            (nce_loss, contrastive_accuracy, nce_global_pair_count,
             masked_same_instance, masked_same_caption) = (
                self._distributed_symmetric_nce(
                    local_student, local_teacher, instance_ids, caption_keys,
                    1.0 / temperature, left_differentiable=True,
                    right_differentiable=False))
            if nce_global_pair_count != global_pair_count:
                raise RuntimeError(
                    "Dual-path positive/NCE pair counts diverged: "
                    f"positive={global_pair_count}, nce={nce_global_pair_count}")
        else:
            masked_same_instance = 0
            masked_same_caption = 0
        shallow_loss = positive_loss + contrastive_weight * nce_loss
        combined_loss = deep_loss + float(shallow_weight) * shallow_loss
        if not dist.is_initialized() or dist.get_rank() == 0:
            if contrastive_weight > 0.0:
                print(
                    "Dual-path contrastive alignment: deep_ce={:.4f}, "
                    "deep_ce_reduction=per_sample, "
                    "shallow_teacher=recompute, "
                    "shallow_pos={:.4f}, shallow_nce={:.4f}, nce_weight={}, "
                    "contrastive_top1={:.3f}, local_pair_count={}, "
                    "global_pair_count={}, global_deep_ce_count={}, "
                    "masked_same_instance_negatives={}, "
                    "masked_same_caption_negatives={}, "
                    "shallow_relative_weight={}, combined={:.4f}".format(
                        deep_loss.detach().item(), positive_loss.detach().item(),
                        nce_loss.detach().item(), contrastive_weight,
                        contrastive_accuracy.detach().item(),
                        local_student.shape[0], global_pair_count,
                        global_deep_count,
                        masked_same_instance,
                        masked_same_caption,
                        float(shallow_weight), combined_loss.detach().item()))
            else:
                print(
                    "Dual-path alignment: deep_ce={:.4f}, "
                    "deep_ce_reduction=per_sample, shallow_teacher=recompute, "
                    "shallow_cos={:.4f}, global_pair_count={}, "
                    "global_deep_ce_count={}, "
                    "shallow_relative_weight={}, combined={:.4f}".format(
                        deep_loss.detach().item(), positive_loss.detach().item(),
                        global_pair_count, global_deep_count,
                        float(shallow_weight), combined_loss.detach().item()))
        return combined_loss

    def align_pointclouds_with_frozen_qwen(self, alignment_items):
        """Backward-compatible alias for :meth:`align_with_frozen_qwen`."""
        return self.align_with_frozen_qwen(alignment_items)

    def align_with_semantic_anchors(
            self, alignment_items, temperature=None, nce_weight=None):
        """Directly align projector outputs to frozen Qwen caption embeddings.

        The positive cosine term is valid even when only one global pair exists.
        The optional symmetric InfoNCE term uses variable-length globally
        gathered negatives while excluding known false negatives.
        """
        embed_weight = self.get_model().get_input_embeddings().weight
        hidden_size = embed_weight.shape[-1]
        device = embed_weight.device
        signal_anchors = []
        text_anchors = []
        instance_ids = []
        caption_keys = []
        for item in alignment_items:
            modality_embed, caption_ids, objects_len, instance_id = (
                self._unpack_alignment_item(item))
            modality_embed = modality_embed[:1024]
            caption_ids = caption_ids.reshape(-1).to(device=modality_embed.device)
            caption_ids = caption_ids[min(int(objects_len), caption_ids.numel()):512]
            if modality_embed.numel() == 0 or caption_ids.numel() == 0:
                continue

            # No trainable projection head: the loss acts directly on the LLM-space
            # output of the modality projector.
            signal_anchor = F.normalize(modality_embed.float().mean(dim=0), dim=-1)
            with torch.no_grad():
                caption_embed = self.get_model().embed_tokens(caption_ids.unsqueeze(0))
                text_anchor = F.normalize(
                    caption_embed.float().mean(dim=1).squeeze(0), dim=-1)
            signal_anchors.append(signal_anchor)
            text_anchors.append(text_anchor)
            instance_ids.append(instance_id)
            caption_keys.append(self._caption_token_key(caption_ids))

        local_signal = (
            torch.stack(signal_anchors, dim=0) if signal_anchors
            else torch.empty(
                0, hidden_size, dtype=torch.float32, device=device))
        local_text = (
            torch.stack(text_anchors, dim=0).detach() if text_anchors
            else torch.empty(
                0, hidden_size, dtype=torch.float32, device=device))
        local_positive_sum = (
            1.0 - (local_signal * local_text).sum(dim=-1)).sum()
        positive_loss, global_pair_count = self._distributed_global_mean(
            local_positive_sum, local_signal.shape[0])

        if temperature is None:
            temperature = getattr(self.config, 'pointcloud_semantic_temperature', 0.07)
        if nce_weight is None:
            nce_weight = getattr(self.config, 'pointcloud_semantic_nce_weight', 0.1)
        temperature = max(float(temperature), 1e-6)
        nce_weight = float(nce_weight)
        if nce_weight > 0.0:
            (nce_loss, _, nce_global_pair_count,
             _, _) = self._distributed_symmetric_nce(
                local_signal, local_text, instance_ids, caption_keys,
                1.0 / temperature, left_differentiable=True,
                right_differentiable=False)
            if nce_global_pair_count != global_pair_count:
                raise RuntimeError(
                    "Semantic positive/NCE pair counts diverged: "
                    f"positive={global_pair_count}, nce={nce_global_pair_count}")
        else:
            nce_loss = positive_loss.new_zeros(())
        return positive_loss + nce_weight * nce_loss

    def align_pointclouds_with_semantic_anchors(self, alignment_items):
        """Backward-compatible alias for :meth:`align_with_semantic_anchors`."""
        return self.align_with_semantic_anchors(alignment_items)

    def encode_imus(self, imus):
        """IMU 时间序列 → ViT-merged token sequence.

        Args:
            imus: list of length B，每个元素是 IMU tensor (T, 6)
                （或 list[tensor]，取每个元素）。当前每个 batch 支持 1 个 IMU。

        Returns:
            imu_features: list of length B，每个元素是 list[Tensor]，
                内层每个 tensor shape (seq_len/spatial_merge_size**2, llm_hidden_size)。
                外层 list 与 batch 对齐；内层 list 与每 batch 多个 IMU 对齐（当前固定 1 个）。
        """
        vision_tower = self.get_vision_tower()
        spatial_merge_size = vision_tower.config.spatial_merge_size
        target_dtype = vision_tower.dtype
        target_device = vision_tower.device

        imu_tokenizer = self.get_model().imu_tokenizer
        imu_projector = self.get_model().modality_projectors['imu']

        out_features = []
        for batch_imu in imus:
            if torch.is_tensor(batch_imu):
                batch_imu = [batch_imu]
            per_imu_features = []
            for imu in batch_imu:
                imu = imu.to(device=target_device, dtype=target_dtype)

                # 1. Tokenizer: (T, 6) -> (1, seq_len, embed_dim)
                tokens = imu_tokenizer(imu)
                tokens_flat = tokens.squeeze(0)                    # (seq_len, embed_dim)

                # 2. grid_thw for ViT
                L = tokens_flat.shape[0]
                H, W = _grid_thw_for_tokens(L, spatial_merge_size=spatial_merge_size)
                grid_thw = torch.tensor([[1, H, W]], dtype=torch.long, device=target_device)

                # 3. ViT (frozen): skip patch_embed, run blocks + merger
                merged = vision_tower.encode_pre_embedded(tokens_flat, grid_thw)
                # merged: (seq_len/spatial_merge_size**2, llm_hidden_size)

                # 4. Projector to LLM space
                merged = imu_projector(merged)
                per_imu_features.append(merged)
            out_features.append(per_imu_features)
        return out_features

    def align_imu_with_caption(self, imu_embed, caption_ids, objects_len: int = 0):
        """IMU 动作级 CoCa caption loss（与 align_scene_with_caption 对称）。

        Args:
            imu_embed:   (imu_tokens, hidden) — ViT merger 后 + projector 输出
            caption_ids: (1, total_len) — caption token ids
            objects_len: 前缀长度（IMU 数据无 objects 字段，恒为 0）

        Returns:
            scalar loss
        """
        # 截断 caption 到 CoCa context_length，防止超出 positional embedding 范围
        context_length = self.get_model().imu_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)        # (1, T, H)
        # 截断到 1024 tokens，保持 K/V 预算（IMU token 数远小于此，仅作保险）
        if imu_embed.shape[0] > 1024:
            imu_embed = imu_embed[:1024]
        image_embed = imu_embed.unsqueeze(0)                               # (1, imu_tok, H)

        logit = self.get_model().imu_caption_module(
            image_embed, caption_embed, objects_len=objects_len)           # (1, T, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)    # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                       # (1, cap_len)
            loss = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])
        return loss

    def encode_smells(self, smells):
        """Smell 传感器时间序列 → ViT-merged token sequence.

        与 ``encode_imus`` 完全对称（smell 数据结构与 IMU 一致：定长 (T, 6)
        时间序列，无需 padding/mask）。

        Args:
            smells: list of length B，每个元素是 smell tensor (T, 6)
                （或 list[tensor]，取每个元素）。当前每个 batch 支持 1 个 smell。

        Returns:
            smell_features: list of length B，每个元素是 list[Tensor]，
                内层每个 tensor shape (seq_len/spatial_merge_size**2, llm_hidden_size)。
                外层 list 与 batch 对齐；内层 list 与每 batch 多个 smell 对齐（当前固定 1 个）。
        """
        vision_tower = self.get_vision_tower()
        spatial_merge_size = vision_tower.config.spatial_merge_size
        target_dtype = vision_tower.dtype
        target_device = vision_tower.device

        smell_tokenizer = self.get_model().smell_tokenizer
        smell_projector = self.get_model().modality_projectors['smell']

        out_features = []
        for batch_smell in smells:
            if torch.is_tensor(batch_smell):
                batch_smell = [batch_smell]
            per_smell_features = []
            for smell in batch_smell:
                smell = smell.to(device=target_device, dtype=target_dtype)

                # 1. Tokenizer: (T, 6) -> (1, seq_len, embed_dim)
                tokens = smell_tokenizer(smell)
                tokens_flat = tokens.squeeze(0)                    # (seq_len, embed_dim)

                # 2. grid_thw for ViT
                L = tokens_flat.shape[0]
                if getattr(self.config, 'smell_grid_mode', 'square') == 'temporal1d':
                    H, W = _grid_thw_temporal1d(L, spatial_merge_size=spatial_merge_size)
                else:
                    H, W = _grid_thw_for_tokens(L, spatial_merge_size=spatial_merge_size)
                grid_thw = torch.tensor([[1, H, W]], dtype=torch.long, device=target_device)

                # 3. ViT (frozen): skip patch_embed, run blocks + merger.
                # smell_vit_depth<0 (default) runs all blocks; 0 skips every self-attention
                # block entirely (tokens only get abs pos + merger, see docs/smell.md §11.11/§12).
                _vit_depth = getattr(self.config, 'smell_vit_depth', -1)
                num_blocks = None if _vit_depth is None or _vit_depth < 0 else _vit_depth
                merged = vision_tower.encode_pre_embedded(tokens_flat, grid_thw, num_blocks=num_blocks)
                # merged: (seq_len/spatial_merge_size**2, llm_hidden_size)

                # 4. Projector to LLM space
                merged = smell_projector(merged)
                per_smell_features.append(merged)
            out_features.append(per_smell_features)
        return out_features

    def align_smell_with_caption(self, smell_embed, caption_ids, objects_len: int = 0):
        """Smell 气味级 CoCa caption loss（与 align_motion_with_caption 对称）。

        Args:
            smell_embed: (smell_tokens, hidden) — ViT merger 后 + projector 输出
            caption_ids: (1, total_len) — caption token ids
            objects_len: 前缀长度（smell 数据无 objects 字段，恒为 0）

        Returns:
            scalar loss
        """
        context_length = self.get_model().smell_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)        # (1, T, H)
        if smell_embed.shape[0] > 1024:
            smell_embed = smell_embed[:1024]
        image_embed = smell_embed.unsqueeze(0)                            # (1, smell_tok, H)

        logit = self.get_model().smell_caption_module(
            image_embed, caption_embed, objects_len=objects_len)          # (1, T, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)   # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                      # (1, cap_len)
            loss = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])
        return loss

    def encode_motions(self, motions):
        """骨骼轨迹时间序列 → ViT-merged token sequence.

        与 ``encode_smells``/``encode_imus`` 完全对称（motion 数据结构与 IMU
        一致：定长 (T, J, 3) 时间序列，无需 padding/mask）。

        Args:
            motions: list of length B，每个元素是 motion tensor (T, J, 3)
                （或 list[tensor]，取每个元素）。当前每个 batch 支持 1 个 motion。

        Returns:
            motion_features: list of length B，每个元素是 list[Tensor]，
                内层每个 tensor shape (seq_len/spatial_merge_size**2, llm_hidden_size)。
                外层 list 与 batch 对齐；内层 list 与每 batch 多个 motion 对齐（当前固定 1 个）。
        """
        vision_tower = self.get_vision_tower()
        spatial_merge_size = vision_tower.config.spatial_merge_size
        target_dtype = vision_tower.dtype
        target_device = vision_tower.device

        motion_tokenizer = self.get_model().motion_tokenizer
        motion_projector = self.get_model().modality_projectors['motion']

        out_features = []
        for batch_motion in motions:
            if torch.is_tensor(batch_motion):
                batch_motion = [batch_motion]
            per_motion_features = []
            for motion in batch_motion:
                motion = motion.to(device=target_device, dtype=target_dtype)

                # 1. Tokenizer: (T, J, 3) -> (1, seq_len, embed_dim)
                tokens = motion_tokenizer(motion)
                tokens_flat = tokens.squeeze(0)                    # (seq_len, embed_dim)

                # 2. grid_thw for ViT
                L = tokens_flat.shape[0]
                H, W = _grid_thw_for_tokens(L, spatial_merge_size=spatial_merge_size)
                grid_thw = torch.tensor([[1, H, W]], dtype=torch.long, device=target_device)

                # 3. ViT (frozen): skip patch_embed, run blocks + merger
                merged = vision_tower.encode_pre_embedded(tokens_flat, grid_thw)
                # merged: (seq_len/spatial_merge_size**2, llm_hidden_size)

                # 4. Projector to LLM space
                merged = motion_projector(merged)
                per_motion_features.append(merged)
            out_features.append(per_motion_features)
        return out_features

    def align_motion_with_caption(self, motion_embed, caption_ids, objects_len: int = 0):
        """骨骼轨迹 CoCa caption loss（与 align_smell_with_caption 对称）。

        Args:
            motion_embed: (motion_tokens, hidden) — ViT merger 后 + projector 输出
            caption_ids: (1, total_len) — caption token ids
            objects_len: 前缀长度（motion 数据无 objects 字段，恒为 0）

        Returns:
            scalar loss
        """
        context_length = self.get_model().motion_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)        # (1, T, H)
        if motion_embed.shape[0] > 1024:
            motion_embed = motion_embed[:1024]
        image_embed = motion_embed.unsqueeze(0)                           # (1, motion_tok, H)

        logit = self.get_model().motion_caption_module(
            image_embed, caption_embed, objects_len=objects_len)          # (1, T, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)   # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                      # (1, cap_len)
            loss = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])
        return loss

    def encode_radars(self, radars):
        """多帧 mmWave 雷达点云 → ViT-merged token sequence.

        与 ``encode_imus`` / ``encode_pointclouds`` 完全对称：先用 RadarTokenizer
        把 (T, N_max, 5) + point_mask 编码到 (seq_len*tokens_per_frame, embed_dim)，
        再走 ViT bypass（skip patch_embed）+ merger 注入 LLM 空间。

        Args:
            radars: list of length B，每个元素是 dict {'points': (T,N,5),
                'point_mask': (T,N) bool} 或 list[dict]。当前每 batch 支持 1 个 radar。

        Returns:
            radar_features: list of length B，每个元素是 list[Tensor]，
                内层每个 tensor shape (radar_tok/spatial_merge_size**2, llm_hidden_size)。
        """
        vision_tower = self.get_vision_tower()
        spatial_merge_size = vision_tower.config.spatial_merge_size
        target_dtype = vision_tower.dtype
        target_device = vision_tower.device

        radar_tokenizer = self.get_model().radar_tokenizer
        radar_projector = self.get_model().modality_projectors['radar']

        out_features = []
        for batch_radar in radars:
            if isinstance(batch_radar, dict):
                batch_radar = [batch_radar]
            per_radar_features = []
            for radar in batch_radar:
                points = radar['points'].to(device=target_device, dtype=target_dtype)
                point_mask = radar['point_mask'].to(device=target_device)

                # 1. Tokenizer: (T,N,5) -> (1, L, embed_dim)
                tokens = radar_tokenizer(points, point_mask)
                tokens_flat = tokens.squeeze(0)                    # (radar_tok, embed_dim)

                # 2. grid_thw for ViT
                # simple：时间序列用 1D 布局，避免 8×8 假图像把相邻帧拆到不同行
                L = tokens_flat.shape[0]
                pool_mode = getattr(radar_tokenizer, 'pool_mode', 'attn')
                grid_mode = getattr(self.config, 'radar_grid_mode', None)
                if grid_mode is None:
                    grid_mode = 'temporal1d' if pool_mode == 'simple' else 'square'
                if grid_mode == 'temporal1d':
                    H, W = _grid_thw_temporal1d(L, spatial_merge_size=spatial_merge_size)
                else:
                    H, W = _grid_thw_for_tokens(L, spatial_merge_size=spatial_merge_size)
                grid_thw = torch.tensor([[1, H, W]], dtype=torch.long, device=target_device)

                # 3. ViT (frozen): skip patch_embed, run blocks + merger
                merged = vision_tower.encode_pre_embedded(tokens_flat, grid_thw)
                # merged: (radar_tok/spatial_merge_size**2, llm_hidden_size)

                # 4. Projector to LLM space
                merged = radar_projector(merged)
                per_radar_features.append(merged)
            out_features.append(per_radar_features)
        return out_features

    def align_radar_with_caption(self, radar_embed, caption_ids, objects_len: int = 0):
        """雷达动作级 CoCa caption loss（与 align_motion_with_caption 对称）。

        Args:
            radar_embed: (radar_tokens, hidden) — ViT merger 后 + projector 输出
            caption_ids: (1, total_len) — caption token ids
            objects_len: 前缀长度（radar 数据无 objects 字段，恒为 0）

        Returns:
            scalar loss
        """
        context_length = self.get_model().radar_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)        # (1, T, H)
        if radar_embed.shape[0] > 1024:
            radar_embed = radar_embed[:1024]
        image_embed = radar_embed.unsqueeze(0)                            # (1, radar_tok, H)

        logit = self.get_model().radar_caption_module(
            image_embed, caption_embed, objects_len=objects_len)          # (1, T, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)   # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                      # (1, cap_len)
            loss = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])
        return loss

    def align_tactile_with_caption(self, tactile_embed, caption_ids, objects_len: int = 0):
        """触觉图像级 CoCa caption loss（与 align_rgbd_with_caption depth_only 模式对称）。

        Args:
            tactile_embed: (tactile_tokens, hidden) — ViT 编码 + projector 输出
            caption_ids:   (1, total_len) — caption token ids
            objects_len:   前缀长度（默认为 0）

        Returns:
            scalar loss
        """
        context_length = self.get_model().tactile_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)         # (1, T, H)
        if tactile_embed.shape[0] > 1024:
            tactile_embed = tactile_embed[:1024]
        image_embed = tactile_embed.unsqueeze(0)                           # (1, tac_tok, H)

        logit = self.get_model().tactile_caption_module(
            image_embed, caption_embed, objects_len=objects_len)           # (1, T, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)   # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                      # (1, cap_len)
            loss = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])
        return loss

    def encode_speeches(self, speeches):
        # import pdb; pdb.set_trace()
        if isinstance(speeches, list):
            if len(speeches[0].shape) == 2:
                speeches = torch.stack(speeches, dim=0)
                speech_features = self.get_model().get_speech_tower()(speeches) # e.g., [1500, 1280]
                speech_features = self.get_model().mm_speech_projector(speech_features) #e.g., [300,1536]
                if self.config.speech_norm:
                    speech_features = torch.nn.functional.normalize(speech_features, dim=-1)
                return speech_features
            
            # long speech case
            if len(speeches[0].shape) == 3:
                speech_features = []
                for speech in speeches:
                    speech_feature = self.get_model().get_speech_tower()(speech)
                    speech_feature = self.get_model().mm_speech_projector(speech_feature)
                    if self.config.speech_norm:
                        speech_feature = torch.nn.functional.normalize(speech_feature, dim=-1)
                    speech_features.append(speech_feature.flatten(0, 1).contiguous())
                return speech_features
        else:
            speech_features = self.get_model().get_speech_tower()(speeches)
            speech_features = self.get_model().mm_speech_projector(speech_features)
            if self.config.speech_norm:
                speech_features = torch.nn.functional.normalize(speech_features, dim=-1)
            return speech_features

    
    def align_speech_with_asr_embed(self, speech_features, speeches_asr):
        loss_align = 0.
        if self.config.speech_learn:
            speeches_asr_embeds = [self.get_model().embed_tokens(speech_asr) for speech_asr in speeches_asr]
        else:
            with torch.no_grad():
                speeches_asr_embeds = [self.get_model().embed_tokens(speech_asr) for speech_asr in speeches_asr]
        
        count = 0
        for idx in range(len(speeches_asr_embeds)):            
            if self.config.align_type == 'dtw':
                speech_feature = speech_features[idx]
                speeches_asr_embed = speeches_asr_embeds[idx]
                if self.config.align_norm:
                    speech_feature = torch.nn.functional.normalize(speech_feature, dim=-1)
                    speeches_asr_embed = torch.nn.functional.normalize(speeches_asr_embed, dim=-1)
                loss_align_item = (self.get_model().soft_dtw(speech_feature[None, :].to(torch.float32), speeches_asr_embed[None, :].to(torch.float32))[0] / speech_feature.shape[0]).to(dtype=speech_feature.dtype)
                
                if torch.all(speeches_asr[idx] == 0).item():
                    loss_align += loss_align_item * 0.
                else:
                    loss_align += loss_align_item
                    count += 1
                    
            elif self.config.align_type == 'truncated':
                speech_features_align = speech_features[idx, -speeches_asr_embeds[idx].shape[0]:]
                if self.config.align_norm:
                    logit = torch.nn.functional.normalize(speech_features_align, dim=-1) @ torch.nn.functional.normalize(speeches_asr_embeds[idx], dim=-1).T
                else:
                    logit = speech_features_align @ speeches_asr_embeds[idx].T
                logit /= self.config.align_temperature
                label = torch.arange(0, logit.shape[0], device=logit.device, dtype=torch.long)
                loss_align_item = nn.functional.cross_entropy(logit, label)
                
                if torch.all(speeches_asr[idx] == 0).item():
                    loss_align += loss_align_item * 0.
                else:
                    loss_align += loss_align_item
                    count += 1
                    
        if count > 0:    
            loss_align /= count

        return loss_align
    

    def align_rgbd_with_caption(self, image_embed, depth_embed, caption_ids, objects_len: int = 0,
                                coca_image_input: str = "rgb_depth"):
        """Compute CoCa captioning loss.

        Args:
            image_embed:       (rgb_tokens, hidden) — RGB visual features (already truncated to ≤512)
            depth_embed:       (dep_tokens, hidden) — depth visual features (already truncated to ≤512)
            caption_ids:       (1, total_len) token ids.
                               When objects_len == 0: pure caption ids, teacher-forcing from token 1.
                               When objects_len  > 0: [objects_ids... | sep | caption_ids...],
                               objects_len = len(objects) + 1 (sep).  Loss computed only over the
                               caption portion so that object tokens are never prediction targets.
            objects_len:       number of leading object+sep tokens. 0 means no objects prefix.
            coca_image_input:  'rgb_depth'  — concat(RGB, depth) as visual tokens [default],
                                              caller should pre-truncate each to ≤512 tokens;
                               'depth_only' — only depth features as visual tokens,
                                              caller should pre-truncate depth to ≤1024 tokens.
        """
        context_length = self.get_model().depth_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)       # (1, total_len, hidden)
        if coca_image_input == "depth_only":
            image_embed = depth_embed.unsqueeze(0)                       # (1, dep_tokens, hidden)
        else:  # "rgb_depth"
            image_embed = torch.cat([image_embed, depth_embed], dim=0).unsqueeze(0)  # (1, img_tok, hidden)

        logit = self.get_model().depth_caption_module(
            image_embed, caption_embed, objects_len=objects_len)          # (1, total_len, vocab)

        if objects_len > 0:
            # Only supervise caption tokens.
            # Position (objects_len-1) predicts caption_ids[:, objects_len] (first caption token).
            # Slice: logit[:, objects_len-1 : -1]  →  (1, cap_len, vocab)
            #        caption_ids[:, objects_len:]   →  (1, cap_len)
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)  # (1, vocab, cap_len)
            cap_labels = caption_ids[:, objects_len:]                      # (1, cap_len)
            loss_align = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss_align = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])

        return loss_align

    def align_rgb_thermal_with_caption(self, image_embed, thermal_embed, caption_ids, objects_len: int = 0):
        """Compute CoCa captioning loss for thermal modality.

        Args:
            image_embed:    (rgb_tokens, hidden) — RGB visual features, pre-truncated to ≤512.
            thermal_embed:  (thr_tokens, hidden) — thermal visual features, pre-truncated to ≤512.
            caption_ids:    (1, total_len) token ids. When objects_len == 0: pure caption ids.
                            When objects_len > 0: [objects_ids... | sep | caption_ids...], loss
                            computed only over the caption portion.
            objects_len:    number of leading object+sep tokens. 0 means no objects prefix.

        Unlike depth (which supports both rgb_depth and depth_only modes), thermal always uses
        rgb_thermal concat: vision_tokens = cat(RGB[:512], thermal[:512]).
        """
        context_length = self.get_model().thermal_caption_module.context_length
        if caption_ids.shape[1] > context_length:
            caption_ids = caption_ids[:, :context_length]
        caption_embed = self.get_model().embed_tokens(caption_ids)                        # (1, total_len, hidden)
        vision_embed = torch.cat([image_embed, thermal_embed], dim=0).unsqueeze(0)        # (1, rgb_tok+thr_tok, hidden)

        logit = self.get_model().thermal_caption_module(
            vision_embed, caption_embed, objects_len=objects_len)                          # (1, total_len, vocab)

        if objects_len > 0:
            cap_logits = logit[:, objects_len - 1 : -1].transpose(1, 2)
            cap_labels = caption_ids[:, objects_len:]
            loss_align = nn.functional.cross_entropy(cap_logits, cap_labels)
        else:
            loss_align = nn.functional.cross_entropy(
                logit[:, :-1].transpose(1, 2), caption_ids[:, 1:])

        return loss_align

    def _ensure_trainable_params_in_graph(self, loss_container):
        """Legacy mixed-batch fallback that attaches every trainable non-LoRA
        parameter to the backward graph via ``0.0 * param.sum()``.

        Globally homogeneous target-modality batches do not use this fallback:
        every rank has the same active/unused parameter set and ZeRO-2's
        ``ignore_unused_parameters`` path can safely skip inactive modalities.

        Instead of tracking individual projectors and their dummy input shapes,
        this iterates ALL non-LoRA ``requires_grad=True`` parameters.  The
        0.0 multiplier makes the actual gradient contribution zero while keeping
        the parameter in the autograd graph.

        Typical non-LoRA trainable params in Stage 2:
          - mm_speech_projector.*
          - modality_projectors.depth.*
          - depth_caption_module.*  (including logit_scale which is never
            used in CoCa.forward but is still an nn.Parameter)
        """
        if not self.training:
            return loss_container
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad and 'lora_' not in name:
                d = 0.0 * param.sum()
                loss_container = (loss_container + d) if loss_container is not None else d
                count += 1
        if hasattr(self, 'device') and getattr(self.device, 'index', -1) == 0:
            print(f"[_ensure_trainable_params_in_graph] covered {count} params, loss_container is {'None' if loss_container is None else 'tensor'}")
        return loss_container

    def prepare_inputs_labels_for_text_image_speech_qwen2vl(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images=None, speeches=None, speeches_asr=None, rope_deltas=None,
        caption=None, objects_len=None, pointclouds=None, imus=None,
        depth_images=None, tactile_images=None, thermal_images=None,
        radars=None, smell_signals=None, motions=None,
        modality_instance_ids=None,
        target_modality_id=None,
    ):
        # import pdb; pdb.set_trace()
        vision_tower = self.get_vision_tower()
        speech_tower = self.get_speech_tower()

        alignment_modalities = (
            'depth', 'pointcloud', 'imu', 'tactile', 'thermal',
            'radar', 'smell', 'motion',
        )
        alignment_items_by_modality = {
            modality: [] for modality in alignment_modalities}
        alignment_losses = {
            modality: None for modality in alignment_modalities}
        active_alignment_modality = None
        if target_modality_id is not None:
            if not torch.is_tensor(target_modality_id):
                target_modality_id = torch.as_tensor(
                    target_modality_id, dtype=torch.long,
                    device=input_ids.device)
            target_modality_id = target_modality_id.reshape(-1)
            if target_modality_id.numel() != 1:
                raise RuntimeError(
                    "homogeneous target_modality_id must be a scalar")
            local_id = target_modality_id.to(
                device=input_ids.device, dtype=torch.long)
            if dist.is_available() and dist.is_initialized():
                extrema = torch.stack((local_id[0], -local_id[0]))
                dist.all_reduce(extrema, op=dist.ReduceOp.MAX)
                global_max = int(extrema[0].item())
                global_min = int((-extrema[1]).item())
                if global_min != global_max:
                    raise RuntimeError(
                        "target modality differs across ranks: "
                        f"min_id={global_min}, max_id={global_max}")
                active_id = global_max
            else:
                active_id = int(local_id[0].item())
            active_alignment_modality = _TARGET_MODALITY_ID_TO_NAME.get(active_id)
            if active_alignment_modality is None:
                raise RuntimeError(
                    f"unsupported homogeneous target modality id: {active_id}")
        (loss_depth_caption, loss_pointcloud_caption, loss_imu_caption,
         loss_tactile_caption, loss_thermal_caption, loss_radar_caption,
         loss_smell_caption, loss_motion_caption) = (
            None, None, None, None, None, None, None, None)
        loss_depth_decorr = None

        distributed_alignment_types = {
            'coca_original', 'coca_qwen_dual', 'coca_encoder_embed',
            'coca_staged_qwen',
            'frozen_qwen', 'semantic_anchor',
            'dual_path', 'dual_path_contrastive',
        }
        needs_alignment_collectives = (
            self.training and dist.is_available() and dist.is_initialized()
            and any(
                getattr(self.config, f'mm_{modality}_projector_type', None)
                is not None
                and getattr(
                    self.config, f'{modality}_alignment_type', 'none')
                in distributed_alignment_types
                for modality in alignment_modalities
            )
        )
        early_return_condition = (((vision_tower is None or (images is None and depth_images is None and tactile_images is None and thermal_images is None)) and
            (speech_tower is None or speeches is None) and
            pointclouds is None and
            imus is None and
            radars is None and
            smell_signals is None and
            motions is None
            ) or input_ids.shape[1] == 1)
        is_decode_step = input_ids.shape[1] == 1 and past_key_values is not None
        if (early_return_condition
                and (not needs_alignment_collectives or is_decode_step)):
            if past_key_values is not None and vision_tower is not None and input_ids.shape[1] == 1:

                if hasattr(past_key_values, 'get_seq_length'):
                    target_shape = past_key_values.get_seq_length() + 1
                else:
                    target_shape = past_key_values[0][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

                if rope_deltas is not None:
                    for i in range(len(rope_deltas)):
                        position_ids[i] += rope_deltas[i]
                position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)

            # Ensure all trainable non-LoRA params are in the backward graph
            # even when no multimodal data is present in this batch.
            loss_depth_caption = (
                None if active_alignment_modality is not None
                else self._ensure_trainable_params_in_graph(None))
            # No modality_ids needed for early-return path (decode step / text-only).
            self._pending_modality_ids = None
            return (input_ids, position_ids, attention_mask, past_key_values,
                    None, labels, None, loss_depth_caption, loss_pointcloud_caption,
                    loss_imu_caption, loss_tactile_caption, loss_thermal_caption,
                    loss_radar_caption, loss_smell_caption,
                    loss_motion_caption, rope_deltas)

        # import pdb; pdb.set_trace()
        if vision_tower is not None and images is not None:
            image_features = self.encode_image_batches(images)
        else:
            image_features = None

        # ── Depth 编码：复用 ViT，再过 depth projector，独立于 RGB image_features ──
        depth_features = None
        if (vision_tower is not None and depth_images is not None
                and getattr(self.config, 'mm_depth_projector_type', None) is not None):
            modality_projectors = self.get_model().modality_projectors
            raw_depth = self.encode_image_batches(depth_images)
            # raw_depth[b] 是 list of Tensor，每个对应一张 depth 图的 ViT 输出
            depth_features = [
                [modality_projectors['depth'](f) for f in batch_d]
                for batch_d in raw_depth
            ]

            # Shared depth alignment. CoCa retains the historical optional RGB
            # context; frozen-Qwen/semantic-anchor see depth tokens only.
            depth_alignment_items = alignment_items_by_modality['depth']
            depth_alignment_type = getattr(
                self.config, 'depth_alignment_type', 'none')
            if depth_alignment_type != 'none' and self.training:
                for batch_i, depth_feat_list in enumerate(depth_features):
                    if not depth_feat_list:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    dep_feat = depth_feat_list[0][:1024]
                    alignment_feat = dep_feat
                    if depth_alignment_type in {'coca', 'coca_lite'} and getattr(
                            self.config, 'coca_image_input', 'rgb_depth') == 'rgb_depth':
                        dep_feat = dep_feat[:512]
                        rgb_feat = (image_features[batch_i][0][:512]
                                    if image_features else dep_feat)
                        alignment_feat = torch.cat((rgb_feat, dep_feat), dim=0)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('depth')
                                   if modality_instance_ids is not None else None)
                    depth_alignment_items.append(
                        (alignment_feat, caption[batch_i], obj_len, instance_id))

            # ── Depth-Image 解相关损失 ──
            # 惩罚 depth feature 和 image feature 的余弦相似度超过 margin，
            # 强制 depth projector 学到与 RGB 互补的表示而非冗余表示。
            # 有效权重 = depth_caption_weight * depth_decorr_weight（在 unisense_qwen35.py 里乘）
            decorr_weight = getattr(self.config, 'depth_decorr_weight', 0.0)
            if decorr_weight > 0 and self.training and image_features is not None:
                decorr_margin = getattr(self.config, 'depth_decorr_margin', 0.3)
                decorr_list = []
                for b in range(len(depth_features)):
                    if not depth_features[b]:
                        continue
                    if not image_features[b]:
                        continue
                    img_pool = F.normalize(image_features[b][0].mean(0), dim=-1)
                    dep_pool = F.normalize(depth_features[b][0].mean(0), dim=-1)
                    cos_sim = (img_pool * dep_pool).sum()
                    decorr_list.append(F.relu(cos_sim - decorr_margin))
                if decorr_list:
                    loss_depth_decorr = (
                        decorr_weight * torch.stack(decorr_list).mean())

        # ── Tactile 编码：复用 ViT，再过 tactile projector，与 depth 完全对称 ──
        tactile_features = None
        if (vision_tower is not None and tactile_images is not None
                and getattr(self.config, 'mm_tactile_projector_type', None) is not None):
            modality_projectors = self.get_model().modality_projectors
            raw_tactile = self.encode_image_batches(tactile_images)
            # raw_tactile[b] 是 list of Tensor，每个对应一张 tactile 图的 ViT 输出
            tactile_features = [
                [modality_projectors['tactile'](f) for f in batch_t]
                for batch_t in raw_tactile
            ]

            tactile_alignment_items = alignment_items_by_modality['tactile']
            if (getattr(self.config, 'tactile_alignment_type', 'none') != 'none'
                    and self.training):
                for batch_i, tactile_feat_list in enumerate(tactile_features):
                    if not tactile_feat_list:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    tac_feat = torch.cat(tactile_feat_list, dim=0)[:1024]
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('tactile')
                                   if modality_instance_ids is not None else None)
                    tactile_alignment_items.append(
                        (tac_feat, caption[batch_i], obj_len, instance_id))

        # ── Thermal 编码：复用 ViT，再过 thermal projector；CoCa 接 RGB+thermal 拼接 ──
        thermal_features = None
        if (vision_tower is not None and thermal_images is not None
                and getattr(self.config, 'mm_thermal_projector_type', None) is not None):
            modality_projectors = self.get_model().modality_projectors
            raw_thermal = self.encode_image_batches(thermal_images)
            thermal_features = [
                [modality_projectors['thermal'](f) for f in batch_t]
                for batch_t in raw_thermal
            ]

            thermal_alignment_items = alignment_items_by_modality['thermal']
            thermal_alignment_type = getattr(
                self.config, 'thermal_alignment_type', 'none')
            if thermal_alignment_type != 'none' and self.training:
                for batch_i, thermal_feat_list in enumerate(thermal_features):
                    if not thermal_feat_list:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    thr_feat = thermal_feat_list[0][:1024]
                    alignment_feat = thr_feat
                    if thermal_alignment_type in {'coca', 'coca_lite'}:
                        thr_feat = thr_feat[:512]
                        rgb_feat = (image_features[batch_i][0][:512]
                                    if image_features else thr_feat)
                        alignment_feat = torch.cat((rgb_feat, thr_feat), dim=0)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('thermal')
                                   if modality_instance_ids is not None else None)
                    thermal_alignment_items.append(
                        (alignment_feat, caption[batch_i], obj_len, instance_id))

        # ── Speech encoding ────────────────────────────────────────────────────────
        loss_align = None
        speech_features = None
        if speech_tower is not None and speeches is not None:
            speech_features = self.encode_speeches(speeches) # e.g., [B, 300, 1536]
            if self.config.asr_align and self.training:
                loss_align = self.align_speech_with_asr_embed(speech_features, speeches_asr) # e.g., speech_features: [300, 1536]; speeches_asr: 32

        # ── Point cloud encoding ───────────────────────────────────────────────────
        pc_features = None
        if pointclouds is not None and getattr(self.config, 'mm_pointcloud_projector_type', None) is not None:
            pc_features = self.encode_pointclouds(pointclouds)
            # pc_features: list[B] of list[per_pc] of (pc_tok, hidden)

            pointcloud_alignment_items = alignment_items_by_modality['pointcloud']
            if (getattr(self.config, 'pointcloud_alignment_type', 'none') != 'none'
                    and self.training):
                for batch_i, batch_pc_feats in enumerate(pc_features):
                    if not batch_pc_feats:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    pc_feat = batch_pc_feats[0]              # (pc_tok, hidden)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('pointcloud')
                                   if modality_instance_ids is not None else None)
                    pointcloud_alignment_items.append(
                        (pc_feat, caption[batch_i], obj_len, instance_id))


        # ── IMU encoding ───────────────────────────────────────────────────────────
        imu_features = None
        if imus is not None and getattr(self.config, 'mm_imu_projector_type', None) is not None:
            imu_features = self.encode_imus(imus)
            # imu_features: list[B] of list[per_imu] of (imu_tok, hidden)

            imu_alignment_items = alignment_items_by_modality['imu']
            if (getattr(self.config, 'imu_alignment_type', 'none') != 'none'
                    and self.training):
                for batch_i, batch_imu_feats in enumerate(imu_features):
                    if not batch_imu_feats:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    imu_feat = batch_imu_feats[0]            # (imu_tok, hidden)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('imu')
                                   if modality_instance_ids is not None else None)
                    imu_alignment_items.append(
                        (imu_feat, caption[batch_i], obj_len, instance_id))

        # ── Radar encoding ─────────────────────────────────────────────────────────
        radar_features = None
        if radars is not None and getattr(self.config, 'mm_radar_projector_type', None) is not None:
            radar_features = self.encode_radars(radars)
            # radar_features: list[B] of list[per_radar] of (radar_tok, hidden)

            radar_alignment_items = alignment_items_by_modality['radar']
            if (getattr(self.config, 'radar_alignment_type', 'none') != 'none'
                    and self.training):
                for batch_i, batch_radar_feats in enumerate(radar_features):
                    if not batch_radar_feats:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    radar_feat = batch_radar_feats[0]           # (radar_tok, hidden)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('radar')
                                   if modality_instance_ids is not None else None)
                    radar_alignment_items.append(
                        (radar_feat, caption[batch_i], obj_len, instance_id))


        # ── Smell encoding ─────────────────────────────────────────────────────────
        smell_features = None
        if smell_signals is not None and getattr(self.config, 'mm_smell_projector_type', None) is not None:
            smell_features = self.encode_smells(smell_signals)
            # smell_features: list[B] of list[per_smell] of (smell_tok, hidden)

            smell_alignment_items = alignment_items_by_modality['smell']
            if (getattr(self.config, 'smell_alignment_type', 'none') != 'none'
                    and self.training):
                for batch_i, batch_smell_feats in enumerate(smell_features):
                    if not batch_smell_feats:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    smell_feat = batch_smell_feats[0]           # (smell_tok, hidden)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('smell')
                                   if modality_instance_ids is not None else None)
                    smell_alignment_items.append(
                        (smell_feat, caption[batch_i], obj_len, instance_id))

        # ── Motion encoding ────────────────────────────────────────────────────────
        motion_features = None
        if motions is not None and getattr(self.config, 'mm_motion_projector_type', None) is not None:
            motion_features = self.encode_motions(motions)
            # motion_features: list[B] of list[per_motion] of (motion_tok, hidden)

            motion_alignment_items = alignment_items_by_modality['motion']
            if (getattr(self.config, 'motion_alignment_type', 'none') != 'none'
                    and self.training):
                for batch_i, batch_motion_feats in enumerate(motion_features):
                    if not batch_motion_feats:
                        continue
                    if caption is None or batch_i >= len(caption) or caption[batch_i] is None:
                        continue
                    motion_feat = batch_motion_feats[0]           # (motion_tok, hidden)
                    obj_len = objects_len[batch_i] if objects_len is not None else 0
                    instance_id = (modality_instance_ids[batch_i].get('motion')
                                   if modality_instance_ids is not None else None)
                    motion_alignment_items.append(
                        (motion_feat, caption[batch_i], obj_len, instance_id))

        # Homogeneous training executes only the globally verified active
        # modality. Legacy/mixed training retains the all-modality ordering so
        # ranks with zero local captions still enter matching collectives.
        if self.training:
            modalities_to_align = (
                (active_alignment_modality,)
                if active_alignment_modality is not None
                else alignment_modalities)
            for modality in modalities_to_align:
                if (getattr(
                        self.config, f'mm_{modality}_projector_type', None)
                        is None):
                    continue
                if getattr(
                        self.config, f'{modality}_alignment_type', 'none') == 'none':
                    continue
                alignment_losses[modality] = self.compute_modality_alignment(
                    modality, alignment_items_by_modality[modality])

        loss_depth_caption = alignment_losses['depth']
        if loss_depth_decorr is not None:
            loss_depth_caption = (
                loss_depth_decorr if loss_depth_caption is None
                else loss_depth_caption + loss_depth_decorr)
        loss_pointcloud_caption = alignment_losses['pointcloud']
        loss_imu_caption = alignment_losses['imu']
        loss_tactile_caption = alignment_losses['tactile']
        loss_thermal_caption = alignment_losses['thermal']
        loss_radar_caption = alignment_losses['radar']
        loss_smell_caption = alignment_losses['smell']
        loss_motion_caption = alignment_losses['motion']

        # Legacy mixed-batch fallback. Homogeneous mode leaves inactive modality
        # parameters unused so ZeRO-2 can skip their gradient communication.
        if active_alignment_modality is None:
            loss_depth_caption = self._ensure_trainable_params_in_graph(
                loss_depth_caption)

        if self.config.check_data_modality:
            all_have = True
            all_not_have = True
            for input_id in input_ids:
                all_have = all_have and (IMAGE_TOKEN_INDEX in input_id)
                all_not_have = all_not_have and (IMAGE_TOKEN_INDEX not in input_id)
            print("image: all have", all_have, ", all not have", all_not_have)
            same_modality = all_have or all_not_have
            
            all_have = True
            all_not_have = True
            for input_id in input_ids:
                all_have = all_have and (SPEECH_TOKEN_INDEX in input_id)
                all_not_have = all_not_have and (SPEECH_TOKEN_INDEX not in input_id)
            print("speech: all have ", all_have, ", all not have", all_not_have)
            
            same_modality = same_modality and (all_have or all_not_have)
            print("same modality?", same_modality)
            if not same_modality:
                print("!!!!!!!!!!!!!!!!  not same modality  !!!!!!!!!!!!!!!!")
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_position_ids = []
        new_modality_ids = []
        rope_deltas = []
        cur_device = input_ids[0].device
        # import pdb; pdb.set_trace()
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_im_idx = 0
            cur_depth_idx = 0
            cur_sp_idx = 0
            cur_pc_idx = 0
            cur_imu_idx = 0
            cur_tactile_idx = 0
            cur_thermal_idx = 0
            cur_radar_idx = 0
            cur_smell_idx = 0
            cur_motion_idx = 0
            num_mm = (
                (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                + (cur_input_ids == DEPTH_TOKEN_INDEX).sum()
                + (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
                + (cur_input_ids == POINTCLOUD_TOKEN_INDEX).sum()
                + (cur_input_ids == IMU_TOKEN_INDEX).sum()
                + (cur_input_ids == TACTILE_TOKEN_INDEX).sum()
                + (cur_input_ids == THERMAL_TOKEN_INDEX).sum()
                + (cur_input_ids == RADAR_TOKEN_INDEX).sum()
                + (cur_input_ids == SMELL_TOKEN_INDEX).sum()
                + (cur_input_ids == MOTION_TOKEN_INDEX).sum()
            )
            if num_mm == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)

                if image_features is not None and image_features[batch_idx]:
                    cur_image_features = image_features[batch_idx][cur_im_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0]], dim=0)
                    cur_im_idx += 1
                if depth_features is not None and depth_features[batch_idx]:
                    cur_depth_dummy = depth_features[batch_idx][cur_depth_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds, cur_depth_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_depth_idx += 1
                if speeches is not None:
                    # speech_features is batch-major for both short speech
                    # (Tensor[B, T, H]) and long speech (list[Tensor[T, H]]).
                    cur_sp_features = speech_features[batch_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds, cur_sp_features[0:0]], dim=0)
                    cur_sp_idx += 1
                if pc_features is not None and pc_features[batch_idx]:
                    cur_pc_dummy = pc_features[batch_idx][cur_pc_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_pc_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_pc_idx += 1
                if imu_features is not None and imu_features[batch_idx]:
                    cur_imu_dummy = imu_features[batch_idx][cur_imu_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_imu_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_imu_idx += 1
                if tactile_features is not None and tactile_features[batch_idx]:
                    cur_tac_dummy = tactile_features[batch_idx][cur_tactile_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_tac_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_tactile_idx += 1
                if thermal_features is not None and thermal_features[batch_idx]:
                    cur_thr_dummy = thermal_features[batch_idx][cur_thermal_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_thr_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_thermal_idx += 1
                if radar_features is not None and radar_features[batch_idx]:
                    cur_radar_dummy = radar_features[batch_idx][cur_radar_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_radar_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_radar_idx += 1
                if smell_features is not None and smell_features[batch_idx]:
                    cur_smell_dummy = smell_features[batch_idx][cur_smell_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_smell_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_smell_idx += 1
                if motion_features is not None and motion_features[batch_idx]:
                    cur_motion_dummy = motion_features[batch_idx][cur_motion_idx]
                    cur_input_embeds = torch.cat(
                        [cur_input_embeds, cur_motion_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
                    cur_motion_idx += 1

                cur_position_ids = torch.arange(len(cur_input_ids), dtype=torch.long, device=cur_device)
                cur_position_ids = cur_position_ids.unsqueeze(0).repeat(3, 1)
                cur_modality_ids = torch.full(
                    (len(cur_input_ids),), MODALITY_TEXT,
                    dtype=torch.long, device=cur_device,
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_position_ids.append(cur_position_ids)
                new_modality_ids.append(cur_modality_ids)
                continue

            # 把所有 mm token 位置合并、按位置排序（任意顺序出现均可）
            _img_pos     = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            _depth_pos   = torch.where(cur_input_ids == DEPTH_TOKEN_INDEX)[0].tolist()
            _sp_pos      = torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist()
            _pc_pos      = torch.where(cur_input_ids == POINTCLOUD_TOKEN_INDEX)[0].tolist()
            _imu_pos     = torch.where(cur_input_ids == IMU_TOKEN_INDEX)[0].tolist()
            _tactile_pos = torch.where(cur_input_ids == TACTILE_TOKEN_INDEX)[0].tolist()
            _thermal_pos = torch.where(cur_input_ids == THERMAL_TOKEN_INDEX)[0].tolist()
            _radar_pos   = torch.where(cur_input_ids == RADAR_TOKEN_INDEX)[0].tolist()
            _smell_pos   = torch.where(cur_input_ids == SMELL_TOKEN_INDEX)[0].tolist()
            _motion_pos  = torch.where(cur_input_ids == MOTION_TOKEN_INDEX)[0].tolist()
            mm_token_indices = [-1] + sorted(_img_pos + _depth_pos + _sp_pos + _pc_pos + _imu_pos + _tactile_pos + _thermal_pos + _radar_pos + _smell_pos + _motion_pos) + [cur_input_ids.shape[0]]
            cur_input_ids_nomm = []
            cur_labels = labels[batch_idx]
            cur_labels_nomm = []
            for i in range(len(mm_token_indices) - 1):
                cur_input_ids_nomm.append(cur_input_ids[mm_token_indices[i]+1:mm_token_indices[i+1]])
                cur_labels_nomm.append(cur_labels[mm_token_indices[i]+1:mm_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nomm]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nomm))

            # gradient anchors — 0-length slices keep all encoder params in the graph
            if image_features is not None and image_features[batch_idx]:
                cur_image_features = image_features[batch_idx][cur_im_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0].to(device=cur_input_embeds.device)], dim=0)
            if depth_features is not None and depth_features[batch_idx]:
                cur_depth_dummy = depth_features[batch_idx][cur_depth_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_depth_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
            if speeches is not None:
                cur_sp_features = speech_features[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_sp_features[0:0].to(device=cur_input_embeds.device)], dim=0)
            if pc_features is not None and pc_features[batch_idx]:
                cur_pc_dummy = pc_features[batch_idx][cur_pc_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_pc_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
            if imu_features is not None and imu_features[batch_idx]:
                cur_imu_dummy = imu_features[batch_idx][cur_imu_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_imu_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
            if tactile_features is not None and tactile_features[batch_idx]:
                cur_tac_dummy = tactile_features[batch_idx][cur_tactile_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_tac_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
            if radar_features is not None and radar_features[batch_idx]:
                cur_radar_dummy = radar_features[batch_idx][cur_radar_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_radar_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
            if smell_features is not None and smell_features[batch_idx]:
                cur_smell_dummy = smell_features[batch_idx][cur_smell_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_smell_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)
            if motion_features is not None and motion_features[batch_idx]:
                cur_motion_dummy = motion_features[batch_idx][cur_motion_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_motion_dummy[0:0].to(device=cur_input_embeds.device)], dim=0)

            cur_input_embeds_no_mm = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_modality_ids = []
            cur_position_ids = []

            max_pos_id = 0
            mrope_delta = 0
            # import pdb; pdb.set_trace()
            for i in range(num_mm + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_mm[i])
                cur_new_labels.append(cur_labels_nomm[i])
                cur_new_modality_ids.append(torch.full(
                    (split_sizes[i],), MODALITY_TEXT,
                    dtype=torch.long, device=cur_device,
                ))
                _pos_ids = torch.arange(split_sizes[i], dtype=torch.long, device=cur_device) + max_pos_id
                _pos_ids = _pos_ids.unsqueeze(0).repeat(3, 1)
                cur_position_ids.append(_pos_ids)
                max_pos_id += cur_input_embeds_no_mm[i].shape[0]
                # import pdb; pdb.set_trace()
                if i < num_mm:
                    if cur_input_ids[mm_token_indices[i+1]] == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[batch_idx][cur_im_idx]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_image_features.shape[0],), MODALITY_IMAGE,
                            dtype=torch.long, device=cur_device,
                        ))

                        if hasattr(images[batch_idx][cur_im_idx], 'image_grid_thw'):
                            if self.training:
                                t, h, w = images[batch_idx][cur_im_idx]['image_grid_thw'][0]
                            else:
                                if len(images[batch_idx][cur_im_idx]['image_grid_thw'].shape) == 3:
                                    t = images[batch_idx][cur_im_idx]['image_grid_thw'][0, 0, 0].item()
                                    h = images[batch_idx][cur_im_idx]['image_grid_thw'][0, 0, 1].item()
                                    w = images[batch_idx][cur_im_idx]['image_grid_thw'][0, 0, 2].item()
                                elif len(images[batch_idx][cur_im_idx]['image_grid_thw'].shape) == 2:
                                    t = images[batch_idx][cur_im_idx]['image_grid_thw'][0, 0].item()
                                    h = images[batch_idx][cur_im_idx]['image_grid_thw'][0, 1].item()
                                    w = images[batch_idx][cur_im_idx]['image_grid_thw'][0, 2].item()
                                else:
                                    t, h, w = images[batch_idx][cur_im_idx]['image_grid_thw'][0]
                        else:
                            if self.training:
                                t, h, w = images[batch_idx][cur_im_idx]['video_grid_thw'][0]
                            else:
                                if len(images[batch_idx][cur_im_idx]['video_grid_thw'].shape) == 3:
                                    t = images[batch_idx][cur_im_idx]['video_grid_thw'][0, 0, 0].item()
                                    h = images[batch_idx][cur_im_idx]['video_grid_thw'][0, 0, 1].item()
                                    w = images[batch_idx][cur_im_idx]['video_grid_thw'][0, 0, 2].item()
                                elif len(images[batch_idx][cur_im_idx]['video_grid_thw'].shape) == 2:
                                    t = images[batch_idx][cur_im_idx]['video_grid_thw'][0, 0].item()
                                    h = images[batch_idx][cur_im_idx]['video_grid_thw'][0, 1].item()
                                    w = images[batch_idx][cur_im_idx]['video_grid_thw'][0, 2].item()
                                else:
                                    t, h, w = images[batch_idx][cur_im_idx]['video_grid_thw'][0]

                        h = h // 2
                        w = w // 2
                        t_index = torch.arange(t, device=cur_device, dtype=torch.long).view(-1, 1).expand(-1, h * w).flatten()
                        h_index = torch.arange(h, device=cur_device, dtype=torch.long).view(1, -1, 1).expand(t, -1, w).flatten()
                        w_index = torch.arange(w, device=cur_device, dtype=torch.long).view(1, 1, -1).expand(t, h, -1).flatten()
                        cur_position_ids.append(torch.stack([t_index, h_index, w_index]) + max_pos_id)
                        
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_im_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == DEPTH_TOKEN_INDEX:
                        # Depth 与 image 完全相同的编码路径，mRoPE 2D 位置同样从 image_grid_thw 读取
                        cur_depth_feature = depth_features[batch_idx][cur_depth_idx]
                        cur_new_input_embeds.append(cur_depth_feature)
                        cur_new_labels.append(torch.full(
                            (cur_depth_feature.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_depth_feature.shape[0],), MODALITY_DEPTH,
                            dtype=torch.long, device=cur_device,
                        ))
                        # 复用 depth_images 的 grid_thw 计算 mRoPE 2D 位置
                        d_img = depth_images[batch_idx][cur_depth_idx]
                        if hasattr(d_img, 'image_grid_thw'):
                            if self.training:
                                t, h, w = d_img['image_grid_thw'][0]
                            else:
                                g = d_img['image_grid_thw']
                                if g.dim() == 3:
                                    t, h, w = g[0, 0, 0].item(), g[0, 0, 1].item(), g[0, 0, 2].item()
                                elif g.dim() == 2:
                                    t, h, w = g[0, 0].item(), g[0, 1].item(), g[0, 2].item()
                                else:
                                    t, h, w = g[0]
                        else:
                            if self.training:
                                t, h, w = d_img['video_grid_thw'][0]
                            else:
                                g = d_img['video_grid_thw']
                                if g.dim() == 3:
                                    t, h, w = g[0, 0, 0].item(), g[0, 0, 1].item(), g[0, 0, 2].item()
                                elif g.dim() == 2:
                                    t, h, w = g[0, 0].item(), g[0, 1].item(), g[0, 2].item()
                                else:
                                    t, h, w = g[0]
                        h = h // 2
                        w = w // 2
                        t_index = torch.arange(t, device=cur_device, dtype=torch.long).view(-1, 1).expand(-1, h * w).flatten()
                        h_index = torch.arange(h, device=cur_device, dtype=torch.long).view(1, -1, 1).expand(t, -1, w).flatten()
                        w_index = torch.arange(w, device=cur_device, dtype=torch.long).view(1, 1, -1).expand(t, h, -1).flatten()
                        cur_position_ids.append(torch.stack([t_index, h_index, w_index]) + max_pos_id)
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_depth_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == SPEECH_TOKEN_INDEX:
                        # Select the speech belonging to the current batch item.
                        # cur_sp_idx counts placeholders within that item; indexing
                        # the batch with it made every item use speech_features[0].
                        cur_speech_features = speech_features[batch_idx]
                        cur_new_input_embeds.append(cur_speech_features)
                        cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_speech_features.shape[0],), MODALITY_SPEECH,
                            dtype=torch.long, device=cur_device,
                        ))
                        cur_position_ids.append(torch.arange(0, cur_speech_features.shape[0], dtype=position_ids.dtype, device=position_ids.device).repeat(3, 1) + max_pos_id)
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_sp_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == POINTCLOUD_TOKEN_INDEX:
                        # 点云走 1D 位置序列（与 speech 同模式）：没有真实 2D 空间
                        cur_pc_features = pc_features[batch_idx][cur_pc_idx]   # (pc_tok, hidden)
                        cur_new_input_embeds.append(cur_pc_features)
                        cur_new_labels.append(torch.full(
                            (cur_pc_features.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_pc_features.shape[0],), MODALITY_POINTCLOUD,
                            dtype=torch.long, device=cur_device,
                        ))
                        cur_position_ids.append(
                            torch.arange(0, cur_pc_features.shape[0],
                                         dtype=position_ids.dtype,
                                         device=position_ids.device).repeat(3, 1) + max_pos_id
                        )
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_pc_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == IMU_TOKEN_INDEX:
                        # IMU 走 1D 位置序列（与 speech / pointcloud 同模式）
                        cur_imu_features = imu_features[batch_idx][cur_imu_idx]  # (imu_tok, hidden)
                        cur_new_input_embeds.append(cur_imu_features)
                        cur_new_labels.append(torch.full(
                            (cur_imu_features.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_imu_features.shape[0],), MODALITY_IMU,
                            dtype=torch.long, device=cur_device,
                        ))
                        cur_position_ids.append(
                            torch.arange(0, cur_imu_features.shape[0],
                                         dtype=position_ids.dtype,
                                         device=position_ids.device).repeat(3, 1) + max_pos_id
                        )
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_imu_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == RADAR_TOKEN_INDEX:
                        # Radar 走 1D 位置序列（与 speech / pointcloud / imu 同模式）
                        cur_radar_features = radar_features[batch_idx][cur_radar_idx]  # (radar_tok, hidden)
                        cur_new_input_embeds.append(cur_radar_features)
                        cur_new_labels.append(torch.full(
                            (cur_radar_features.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_radar_features.shape[0],), MODALITY_RADAR,
                            dtype=torch.long, device=cur_device,
                        ))
                        cur_position_ids.append(
                            torch.arange(0, cur_radar_features.shape[0],
                                         dtype=position_ids.dtype,
                                         device=position_ids.device).repeat(3, 1) + max_pos_id
                        )
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_radar_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == SMELL_TOKEN_INDEX:
                        # Smell 走 1D 位置序列（与 speech / pointcloud / imu / radar 同模式）
                        cur_smell_features = smell_features[batch_idx][cur_smell_idx]  # (smell_tok, hidden)
                        cur_new_input_embeds.append(cur_smell_features)
                        cur_new_labels.append(torch.full(
                            (cur_smell_features.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_smell_features.shape[0],), MODALITY_SMELL,
                            dtype=torch.long, device=cur_device,
                        ))
                        cur_position_ids.append(
                            torch.arange(0, cur_smell_features.shape[0],
                                         dtype=position_ids.dtype,
                                         device=position_ids.device).repeat(3, 1) + max_pos_id
                        )
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_smell_idx += 1


                    elif cur_input_ids[mm_token_indices[i+1]] == MOTION_TOKEN_INDEX:
                        # Motion 走 1D 位置序列（与 speech / pointcloud / imu / radar / smell 同模式）
                        cur_motion_features = motion_features[batch_idx][cur_motion_idx]  # (motion_tok, hidden)
                        cur_new_input_embeds.append(cur_motion_features)
                        cur_new_labels.append(torch.full(
                            (cur_motion_features.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_motion_features.shape[0],), MODALITY_MOTION,
                            dtype=torch.long, device=cur_device,
                        ))
                        cur_position_ids.append(
                            torch.arange(0, cur_motion_features.shape[0],
                                         dtype=position_ids.dtype,
                                         device=position_ids.device).repeat(3, 1) + max_pos_id
                        )
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_motion_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == TACTILE_TOKEN_INDEX:
                        # Tactile 与 depth 完全相同的编码路径，mRoPE 2D 位置从 image_grid_thw 读取
                        if tactile_features is None or not tactile_features[batch_idx]:
                            raise RuntimeError(
                                "Prompt contains <tactile> tokens but tactile features are missing. "
                                "Ensure mm_tactile_projector_type is set and modality_projectors['tactile'] "
                                "was initialized before loading non_lora_trainables.bin (see unisense/model/builder.py)."
                            )
                        cur_tactile_feature = tactile_features[batch_idx][cur_tactile_idx]
                        cur_new_input_embeds.append(cur_tactile_feature)
                        cur_new_labels.append(torch.full(
                            (cur_tactile_feature.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_tactile_feature.shape[0],), MODALITY_TACTILE,
                            dtype=torch.long, device=cur_device,
                        ))
                        # 复用 tactile_images 的 grid_thw 计算 mRoPE 2D 位置
                        t_img = tactile_images[batch_idx][cur_tactile_idx]
                        if hasattr(t_img, 'image_grid_thw'):
                            if self.training:
                                t, h, w = t_img['image_grid_thw'][0]
                            else:
                                g = t_img['image_grid_thw']
                                if g.dim() == 3:
                                    t, h, w = g[0, 0, 0].item(), g[0, 0, 1].item(), g[0, 0, 2].item()
                                elif g.dim() == 2:
                                    t, h, w = g[0, 0].item(), g[0, 1].item(), g[0, 2].item()
                                else:
                                    t, h, w = g[0]
                        else:
                            if self.training:
                                t, h, w = t_img['video_grid_thw'][0]
                            else:
                                g = t_img['video_grid_thw']
                                if g.dim() == 3:
                                    t, h, w = g[0, 0, 0].item(), g[0, 0, 1].item(), g[0, 0, 2].item()
                                elif g.dim() == 2:
                                    t, h, w = g[0, 0].item(), g[0, 1].item(), g[0, 2].item()
                                else:
                                    t, h, w = g[0]
                        h = h // 2
                        w = w // 2
                        t_index = torch.arange(t, device=cur_device, dtype=torch.long).view(-1, 1).expand(-1, h * w).flatten()
                        h_index = torch.arange(h, device=cur_device, dtype=torch.long).view(1, -1, 1).expand(t, -1, w).flatten()
                        w_index = torch.arange(w, device=cur_device, dtype=torch.long).view(1, 1, -1).expand(t, h, -1).flatten()
                        cur_position_ids.append(torch.stack([t_index, h_index, w_index]) + max_pos_id)
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_tactile_idx += 1

                    elif cur_input_ids[mm_token_indices[i+1]] == THERMAL_TOKEN_INDEX:
                        # Thermal 与 depth/tactile 完全相同的编码路径，mRoPE 2D 位置从 image_grid_thw 读取
                        if thermal_features is None or not thermal_features[batch_idx]:
                            raise RuntimeError(
                                "Prompt contains <thermal> tokens but thermal features are missing. "
                                "Ensure mm_thermal_projector_type is set and modality_projectors['thermal'] "
                                "was initialized before loading non_lora_trainables.bin."
                            )
                        cur_thermal_feature = thermal_features[batch_idx][cur_thermal_idx]
                        cur_new_input_embeds.append(cur_thermal_feature)
                        cur_new_labels.append(torch.full(
                            (cur_thermal_feature.shape[0],), IGNORE_INDEX,
                            device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_modality_ids.append(torch.full(
                            (cur_thermal_feature.shape[0],), MODALITY_THERMAL,
                            dtype=torch.long, device=cur_device,
                        ))
                        # 复用 thermal_images 的 grid_thw 计算 mRoPE 2D 位置（与 tactile 完全相同）
                        th_img = thermal_images[batch_idx][cur_thermal_idx]
                        if hasattr(th_img, 'image_grid_thw'):
                            if self.training:
                                t, h, w = th_img['image_grid_thw'][0]
                            else:
                                g = th_img['image_grid_thw']
                                if g.dim() == 3:
                                    t, h, w = g[0, 0, 0].item(), g[0, 0, 1].item(), g[0, 0, 2].item()
                                elif g.dim() == 2:
                                    t, h, w = g[0, 0].item(), g[0, 1].item(), g[0, 2].item()
                                else:
                                    t, h, w = g[0]
                        else:
                            if self.training:
                                t, h, w = th_img['video_grid_thw'][0]
                            else:
                                g = th_img['video_grid_thw']
                                if g.dim() == 3:
                                    t, h, w = g[0, 0, 0].item(), g[0, 0, 1].item(), g[0, 0, 2].item()
                                elif g.dim() == 2:
                                    t, h, w = g[0, 0].item(), g[0, 1].item(), g[0, 2].item()
                                else:
                                    t, h, w = g[0]
                        h = h // 2
                        w = w // 2
                        t_index = torch.arange(t, device=cur_device, dtype=torch.long).view(-1, 1).expand(-1, h * w).flatten()
                        h_index = torch.arange(h, device=cur_device, dtype=torch.long).view(1, -1, 1).expand(t, -1, w).flatten()
                        w_index = torch.arange(w, device=cur_device, dtype=torch.long).view(1, 1, -1).expand(t, h, -1).flatten()
                        cur_position_ids.append(torch.stack([t_index, h_index, w_index]) + max_pos_id)
                        max_pos_id = cur_position_ids[-1].max() + 1
                        cur_thermal_idx += 1

            if IMAGE_TOKEN_INDEX not in cur_input_ids: cur_im_idx += 1
            if DEPTH_TOKEN_INDEX not in cur_input_ids: cur_depth_idx += 1
            if SPEECH_TOKEN_INDEX not in cur_input_ids: cur_sp_idx += 1
            if POINTCLOUD_TOKEN_INDEX not in cur_input_ids: cur_pc_idx += 1
            if IMU_TOKEN_INDEX not in cur_input_ids: cur_imu_idx += 1
            if TACTILE_TOKEN_INDEX not in cur_input_ids: cur_tactile_idx += 1
            if THERMAL_TOKEN_INDEX not in cur_input_ids: cur_thermal_idx += 1
            if RADAR_TOKEN_INDEX not in cur_input_ids: cur_radar_idx += 1
            if SMELL_TOKEN_INDEX not in cur_input_ids: cur_smell_idx += 1
            if MOTION_TOKEN_INDEX not in cur_input_ids: cur_motion_idx += 1

            cur_new_input_embeds = [x.to(device=cur_input_embeds.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_modality_ids = torch.cat(cur_new_modality_ids)
            cur_position_ids = torch.cat(cur_position_ids, dim=-1)

            rope_delta = cur_position_ids.max() + 1 - len(cur_new_input_embeds)
            rope_deltas.append(rope_delta)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_modality_ids.append(cur_new_modality_ids)
            new_position_ids.append(cur_position_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if images is not None and any(image and 'pixel_values_videos' in image[0] for image in images):
            tokenizer_model_max_length = tokenizer_model_max_length * 3

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_position_ids = [x[..., :tokenizer_model_max_length] for x in new_position_ids]
            new_modality_ids = [x[:tokenizer_model_max_length] for x in new_modality_ids]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((3, batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_modality_ids_padded = torch.full(
            (batch_size, max_len), MODALITY_TEXT, dtype=torch.long,
            device=new_modality_ids[0].device if new_modality_ids else attention_mask.device,
        )

        for i, (cur_new_embed, cur_new_labels, cur_position_ids, cur_mod_ids) in enumerate(
            zip(new_input_embeds, new_labels, new_position_ids, new_modality_ids)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[:, i, -cur_len:] = cur_position_ids
                    new_modality_ids_padded[i, -cur_len:] = cur_mod_ids
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[:, i, :cur_len] = cur_position_ids
                    new_modality_ids_padded[i, :cur_len] = cur_mod_ids

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # Store modality_ids so UnisenseQwen35ForCausalLM can set thread-local state
        # before the LLM forward (used by ModalityAwareMoEFFN layers).
        self._pending_modality_ids = new_modality_ids_padded

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return (None, position_ids, attention_mask, past_key_values,
                new_input_embeds, new_labels, loss_align,
                loss_depth_caption, loss_pointcloud_caption, loss_imu_caption,
                loss_tactile_caption, loss_thermal_caption, loss_radar_caption,
                loss_smell_caption, loss_motion_caption, rope_deltas)


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights.get(
                    'model.language_model.embed_tokens.weight',
                    mm_projector_weights.get('model.embed_tokens.weight')
                )
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
                
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
