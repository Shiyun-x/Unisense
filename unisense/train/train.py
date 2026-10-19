# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import torch
_original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)

torch.load = patched_load

import os
import copy
import math
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import re
import json

import torch
import numpy as np
import torchaudio
from torchaudio.transforms import Resample

import transformers
import whisper
import tokenizers

from qwen_vl_utils import process_vision_info

from unisense.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                             DEPTH_TOKEN_INDEX, DEFAULT_DEPTH_TOKEN,
                             DEFAULT_SPEECH_TOKEN, DEFAULT_POINTCLOUD_TOKEN,
                             POINTCLOUD_TOKEN_INDEX, TACTILE_TOKEN_INDEX, DEFAULT_TACTILE_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from torch.utils.data import Dataset
from unisense.train.unisense_trainer import UnisenseTrainer

from unisense import conversation as conversation_lib
from unisense.model import *
from unisense.model.moe_ffn import MODALITY_ID_MAP
from unisense.mm_utils import tokenizer_image_token, tokenizer_image_speech_token, tokenizer_speech_token, process_highres_image, process_anyres_image, process_highres_image_crop_split

from PIL import Image
from unisense.train.preprocess import *
from unisense.train.data_settings import CONVERSATION_DATA

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

import glob as _glob


def apply_radar_lr_flip(radar_data):
    """左右镜像：MMFi filtered_mmwave 中 y 为左右轴，故 y → -y。

    Doppler 在该数据中近似为径向/非负量，不做符号翻转。返回新 dict（不改入参）。
    """
    points = radar_data['points'].clone()
    points[..., 1] *= -1
    return {
        'points': points,
        'point_mask': radar_data['point_mask'],
    }


def load_radar_segment(radar_dir, start, end, max_points=128, max_frames=32):
    """读取一段 mmWave 雷达帧 → (T, max_points, 5) + point_mask (T, max_points)。

    Args:
        radar_dir:  该样本的帧目录，如 ".../filtered_mmwave/E01/S01/A01"
        start, end: 1-based 闭区间，对应 MMFi 原始帧编号 frame{idx:03d}.bin
                    （HoloLLM 的 start_index/end_index）。filtered_mmwave 已去掉
                    空帧，范围内不存在的帧自动跳过，不能用 sorted(glob) 下标切片。
        max_points: 每帧 pad/截断到的点数 N_max
        max_frames: 段内帧数上限；超过则均匀下采样（tokenizer 还会进一步插值到 seq_len）

    Returns:
        dict {'points': FloatTensor(T, max_points, 5),
              'point_mask': BoolTensor(T, max_points)}
        每点 5 维 = [x, y, z, Doppler, intensity]。
    """
    seg = []
    for idx in range(start, end + 1):
        path = os.path.join(radar_dir, f'frame{idx:03d}.bin')
        if os.path.isfile(path):
            seg.append(path)
    if len(seg) == 0:
        bins = sorted(_glob.glob(os.path.join(radar_dir, 'frame*.bin')))
        seg = bins[:1]                                # 兜底：避免空段

    if max_frames and len(seg) > max_frames:
        idx = np.linspace(0, len(seg) - 1, max_frames).round().astype(int)
        seg = [seg[k] for k in idx]

    frames = []
    for b in seg:
        with open(b, 'rb') as f:
            raw = f.read()
        if len(raw) == 0:
            continue
        pts = np.frombuffer(raw, dtype=np.float64).reshape(-1, 5).astype(np.float32)
        if pts.shape[0] > 0:
            frames.append(pts)

    if len(frames) == 0:
        # 退化样本：1 个空帧（mask 全 False），RadarTokenizer 会输出 0 token
        return {
            'points': torch.zeros(1, max_points, 5, dtype=torch.float32),
            'point_mask': torch.zeros(1, max_points, dtype=torch.bool),
        }

    T = len(frames)
    points = torch.zeros(T, max_points, 5, dtype=torch.float32)
    point_mask = torch.zeros(T, max_points, dtype=torch.bool)
    for t, pts in enumerate(frames):
        n = pts.shape[0]
        if n > max_points:                            # 点太多则均匀下采样
            sel = np.linspace(0, n - 1, max_points).round().astype(int)
            pts = pts[sel]
            n = max_points
        points[t, :n] = torch.from_numpy(pts)
        point_mask[t, :n] = True
    return {'points': points, 'point_mask': point_mask}


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 datasets: str):
        super(LazySupervisedDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        # list_data_dict = [data for data in list_data_dict if "images" in data or "image" in data]
        list_data_dict = []
        list_image_path = []
        list_speech_path = []
        list_depth_path = []
        list_pointcloud_path = []
        list_imu_path = []
        list_tactile_path = []
        list_thermal_path = []
        list_radar_path = []
        list_smell_path = []
        list_motion_path = []
        list_target_modality = []
        for name in datasets.split("+"):
            dataset = CONVERSATION_DATA[name]

            data_path = dataset['data_path']
            if data_path.endswith('.json'):
                data = json.load(open(data_path, "r"))
            elif data_path.endswith('.jsonl'):
                data = [json.loads(q) for q in open(os.path.expanduser(data_path), "r")]
                # 数据集可能是 image-only / pointcloud-only / imu-only 等不同模态
                # 旧逻辑会过滤掉没有 image 字段的样本，对纯点云 / 纯 IMU 数据集需要放过
                _has_image_filter = any('image' in d and isinstance(d['image'], str) for d in data)
                _has_pc_filter = any('pointcloud' in d for d in data)
                _has_imu_filter = any('imu' in d for d in data)
                _has_tactile_filter = any('tactile' in d for d in data)
                _has_thermal_filter = any('thermal' in d for d in data)
                _has_radar_filter = any('radar' in d for d in data)
                _has_smell_filter = any('smell' in d for d in data)
                _has_motion_filter = any('motion' in d for d in data)
                if _has_image_filter and not _has_pc_filter and not _has_imu_filter and not _has_tactile_filter and not _has_thermal_filter and not _has_radar_filter and not _has_smell_filter and not _has_motion_filter:
                    data = [d for d in data if 'image' in d and isinstance(d['image'], str)]
            # filter text-only data
            if 'pretrain' not in name.lower():
                # 必须至少有一种模态字段
                data = [d for d in data if ('image' in d) or ('pointcloud' in d) or ('imu' in d) or ('tactile' in d) or ('thermal' in d) or ('radar' in d) or ('smell' in d) or ('motion' in d)]

            data_freq = dataset['frequency']
            sampled_data = data * data_freq
            print(f"Data from {data_path} are sampled from {len(data)} to {len(sampled_data)} conversations (x{data_freq}).")
            list_data_dict.extend(sampled_data)
            for sample in sampled_data:
                target_modalities = [
                    modality for modality in (
                        'depth', 'pointcloud', 'imu', 'radar', 'thermal',
                        'tactile', 'motion', 'smell')
                    if modality in sample
                ]
                if len(target_modalities) > 1:
                    raise ValueError(
                        f"Dataset {name!r} contains a sample with multiple added "
                        f"modalities {target_modalities}; homogeneous batching "
                        "requires one target modality per sample."
                    )
                list_target_modality.append(
                    target_modalities[0] if target_modalities else 'other')

            image_path = dataset.get('image_folder', None)
            list_image_path.extend([image_path] * len(sampled_data))

            speech_folder = dataset.get('speech_folder', None)
            list_speech_path.extend([speech_folder] * len(sampled_data))

            if 'depth_folder' in dataset:
                depth_folder = dataset['depth_folder']
                list_depth_path.extend([depth_folder] * len(sampled_data))
            else:
                list_depth_path.extend([None] * len(sampled_data))

            pointcloud_folder = dataset.get('pointcloud_folder', None)
            list_pointcloud_path.extend([pointcloud_folder] * len(sampled_data))

            imu_folder = dataset.get('imu_folder', None)
            list_imu_path.extend([imu_folder] * len(sampled_data))

            tactile_folder = dataset.get('tactile_folder', None)
            list_tactile_path.extend([tactile_folder] * len(sampled_data))

            thermal_folder = dataset.get('thermal_folder', None)
            list_thermal_path.extend([thermal_folder] * len(sampled_data))

            radar_folder = dataset.get('radar_folder', None)
            list_radar_path.extend([radar_folder] * len(sampled_data))

            smell_folder = dataset.get('smell_folder', None)
            list_smell_path.extend([smell_folder] * len(sampled_data))


            motion_folder = dataset.get('motion_folder', None)
            list_motion_path.extend([motion_folder] * len(sampled_data))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.list_image_path = list_image_path
        self.list_speech_path = list_speech_path
        self.list_depth_path = list_depth_path
        self.list_pointcloud_path = list_pointcloud_path
        self.list_imu_path = list_imu_path
        self.list_tactile_path = list_tactile_path
        self.list_thermal_path = list_thermal_path
        self.list_radar_path = list_radar_path
        self.list_smell_path = list_smell_path
        self.list_motion_path = list_motion_path
        self.list_target_modality = list_target_modality
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'image' in sample and 'speech' in sample:
                cur_len = cur_len + 0.5
            elif 'image' in sample:
                cur_len = cur_len
            else:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def target_modalities(self):
        """Stable primary-modality label used by homogeneous batch sampling."""
        return self.list_target_modality
    
    def _preprocess_image_qwen2vl(self, image, image_resolution):

        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        while attempt < max_attempt:
            try:
                # sample an item
                data_dict = self._sample_item(i)
                return data_dict
            except Exception as _e:
                attempt += 1
                if attempt == 1:  # 只打第一次完整 traceback
                    import traceback
                    print(f"Error in loading {i} (attempt {attempt}): {_e}")
                    traceback.print_exc()
                else:
                    print(f"Error in loading {i}, retrying...")
                i = random.randint(0, len(self.list_data_dict) - 1)
        raise RuntimeError(f"Failed to load a valid sample after {max_attempt} attempts.")

    def _sample_item(self, i) -> Dict[str, torch.Tensor]:
        # import pdb; pdb.set_trace()
        image = None
        sources = self.list_data_dict[i]
        suffix = None
        # import pdb; pdb.set_trace()
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        if 'speech_asr' in sources[0]:
            speech_asr = self.list_data_dict[i]['speech_asr']
            speech_asr = self.tokenizer(speech_asr, return_tensors="pt").input_ids
        
        if 'image' in sources[0]:
            image_files = self.list_data_dict[i]['image']
            image_folder = self.list_image_path[i]
            processor = self.data_args.image_processor
            
            image_total = []
            if not isinstance(image_files, list):
                image_files = [image_files]
            image_files = [os.path.join(image_folder, image_file) for image_file in image_files]
            
            for image_i, image_file in enumerate(image_files):

                # image_path = os.path.join(image_folder, image_file)
                image_path = image_file

                if not os.path.exists(image_path):
                    image_path = image_path.replace(".jpg", ".gif")

                if not os.path.exists(image_path):
                    image_path = image_path.replace(".jpg", ".png")
                    image_path = image_path.replace(".gif", ".png")

                image = Image.open(image_path).convert('RGB')
                
                if 'cavqa' in image_path.lower():
                    # cavqa数据集下采样2倍
                    image = image.resize((image.width // 2, image.height // 2), resample=Image.NEAREST)
                
                if 'sa1b' in image_path.lower() and 'spatialqa' in image_path.lower():
                    # spatialqa数据集下采样3倍
                    image = image.resize((image.width // 3, image.height // 3), resample=Image.NEAREST)
             
                # if image_i % 2 == 0:
                #     image = Image.open(image_path).convert('RGB')
                # else: # depth following rgb
                #     # import pdb; pdb.set_trace()
                #     depth_map = Image.open(image_path)

                #     # 归一化深度值到0-1范围
                #     depth_min = np.min(depth_map)
                #     depth_max = np.max(depth_map)
                    
                #     # 避免除以0（若深度图所有值相同）
                #     if depth_max - depth_min < 1e-6:
                #         normalized = np.zeros_like(depth_map, dtype=np.float32)
                #     else:
                #         normalized = (depth_map - depth_min) / (depth_max - depth_min)
                    
                #     # 转换为0-255的灰度值，并转为uint8类型
                #     gray_image = (normalized * 255).astype(np.uint8)

                #     # 沿最后一个维度堆叠3次
                #     if gray_image.shape[-1] == 1:
                #         gray_image = np.stack([gray_image, gray_image, gray_image], axis=-1)  

                #     # 转换回PIL Image
                #     image = Image.fromarray(gray_image)
                    
                if self.data_args.image_aspect_ratio in ("qwen2vl", "qwen25vl", "qwen35vl"):
                    image = self._preprocess_image_qwen2vl(image, image_resolution=self.data_args.image_resolution)
                    image = self.data_args.image_processor(images=image, return_tensors="pt")
                elif self.data_args.image_aspect_ratio == "highres":
                    image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
                elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                    image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints, self.data_args.overlap_ratio)
                elif self.data_args.image_aspect_ratio == "crop_split":
                    image = process_highres_image_crop_split(image, self.data_args)
                elif self.data_args.image_aspect_ratio == "plain_llava":
                    def make_image_pieces(image, up_scale):
                        if hasattr(self.data_args.image_processor, "crop_size"):
                            newsize = (self.data_args.image_processor.crop_size['height'] * up_scale, self.data_args.image_processor.crop_size['width'] * up_scale)
                        elif hasattr(self.data_args.image_processor, "size"):
                            newsize = (self.data_args.image_processor.size['height'] * up_scale, self.data_args.image_processor.size['width'] * up_scale)
                        image_up = image.resize(newsize)

                        M = image_up.size[0] // up_scale
                        N = image_up.size[1] // up_scale
                        image_up_array = np.array(image_up)
                        tiles = [Image.fromarray(image_up_array[x:x + M, y:y + N])
                                for x in range(0, image_up.size[0], M)
                                for y in range(0, image_up.size[1], N)]

                        return tiles
                    image = make_image_pieces(image, 1)
                    image = [processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image]
                image_total.append(image)

        # ── Depth 图独立加载（走同一套 image_processor，但单独存储）──────────────
        depth_total = []
        if 'depth' in sources[0]:
            depth_files = self.list_data_dict[i]['depth']
            depth_folder = self.list_depth_path[i]
            if not isinstance(depth_files, list):
                depth_files = [depth_files]
            depth_files = [os.path.join(depth_folder, df) for df in depth_files]
            for depth_file in depth_files:
                if not os.path.exists(depth_file):
                    depth_file = depth_file.replace(".jpg", ".png")
                depth_img = Image.open(depth_file).convert('RGB')
                if self.data_args.image_aspect_ratio in ("qwen2vl", "qwen25vl", "qwen35vl"):
                    depth_img = self._preprocess_image_qwen2vl(depth_img, image_resolution=self.data_args.image_resolution)
                    depth_img = self.data_args.image_processor(images=depth_img, return_tensors="pt")
                elif self.data_args.image_aspect_ratio == "highres":
                    depth_img = process_highres_image(depth_img, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
                elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                    depth_img = process_anyres_image(depth_img, self.data_args.image_processor, self.data_args.image_grid_pinpoints, self.data_args.overlap_ratio)
                depth_total.append(depth_img)

        # ── Tactile 图独立加载（与 depth 完全对称，走同一套 image_processor）──────
        tactile_total = []
        if 'tactile' in sources[0]:
            tactile_files = self.list_data_dict[i]['tactile']
            tactile_folder = self.list_tactile_path[i]
            if not isinstance(tactile_files, list):
                tactile_files = [tactile_files]
            tactile_files = [os.path.join(tactile_folder, tf) for tf in tactile_files]
            for tactile_file in tactile_files:
                if not os.path.exists(tactile_file):
                    tactile_file = tactile_file.replace(".jpg", ".png")
                tactile_img = Image.open(tactile_file).convert('RGB')
                if self.data_args.image_aspect_ratio in ("qwen2vl", "qwen25vl", "qwen35vl"):
                    tactile_img = self._preprocess_image_qwen2vl(tactile_img, image_resolution=self.data_args.image_resolution)
                    tactile_img = self.data_args.image_processor(images=tactile_img, return_tensors="pt")
                elif self.data_args.image_aspect_ratio == "highres":
                    tactile_img = process_highres_image(tactile_img, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
                elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                    tactile_img = process_anyres_image(tactile_img, self.data_args.image_processor, self.data_args.image_grid_pinpoints, self.data_args.overlap_ratio)
                tactile_total.append(tactile_img)

        # ── Thermal 图独立加载（与 depth/tactile 完全对称，走同一套 image_processor）──────
        thermal_total = []
        if 'thermal' in sources[0]:
            thermal_files = self.list_data_dict[i]['thermal']
            thermal_folder = self.list_thermal_path[i]
            if not isinstance(thermal_files, list):
                thermal_files = [thermal_files]
            thermal_files = [os.path.join(thermal_folder, tf) for tf in thermal_files]
            for thermal_file in thermal_files:
                if not os.path.exists(thermal_file):
                    thermal_file = thermal_file.replace(".jpg", ".png")
                thermal_img = Image.open(thermal_file).convert('RGB')
                if self.data_args.image_aspect_ratio in ("qwen2vl", "qwen25vl", "qwen35vl"):
                    thermal_img = self._preprocess_image_qwen2vl(thermal_img, image_resolution=self.data_args.image_resolution)
                    thermal_img = self.data_args.image_processor(images=thermal_img, return_tensors="pt")
                elif self.data_args.image_aspect_ratio == "highres":
                    thermal_img = process_highres_image(thermal_img, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
                elif self.data_args.image_aspect_ratio == "anyres" or "anyres_max" in self.data_args.image_aspect_ratio:
                    thermal_img = process_anyres_image(thermal_img, self.data_args.image_processor, self.data_args.image_grid_pinpoints, self.data_args.overlap_ratio)
                thermal_total.append(thermal_img)

        elif 'video' in sources[0]:
            video_files = self.list_data_dict[i]['video']
            video_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            if not isinstance(video_files, list):
                video_files = [video_files]
            
            for video_file in video_files:
                video_path = os.path.join(video_folder, video_file)
                if self.data_args.image_aspect_ratio in ("qwen2vl", "qwen25vl", "qwen35vl"):
                    total_pixels = 8192 * 28 * 28
                    suffix = video_file.split('.')[-1]
                    if suffix == 'mp4' or suffix == 'mkv':
                        messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "total_pixels": total_pixels, "min_pixels": 28 * 28}]}]
                        image_inputs, video_inputs = process_vision_info(messages)
                        image = processor(images=None, videos=video_inputs[0], return_tensors="pt")
                    elif suffix == 'jpg' or suffix == 'png' or suffix == 'jpeg':
                        image = Image.open(video_path).convert('RGB')
                        
                        image = np.array(image)
                        non_white_rows = np.any(image.sum(axis=-1) < 720, axis=1)
                        non_white_cols = np.any(image.sum(axis=-1) < 720, axis=0)
                        cat_image = image[non_white_rows][:, non_white_cols]

                        img_height, img_width, _ = cat_image.shape
                        piece_width, piece_height = img_width // 4, img_height // 4
                        cat_image = Image.fromarray(cat_image)

                        images = []
                        for row in range(4):
                            for col in range(4):
                                left = col * piece_width
                                upper = row * piece_height
                                right = left + piece_width
                                lower = upper + piece_height
                                cropped_img = cat_image.crop((left, upper, right, lower))
                                images.append(cropped_img)
                        images = [processor.preprocess(img, return_tensors='pt') for img in images]
                        
                        pixel_values_videos = torch.cat([x['pixel_values'] for x in images])
                        video_grid_thw = images[0]['image_grid_thw']
                        video_grid_thw[0,0] = len(images)
                        image = dict(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw)
                    else:
                        video_root = video_path
                        video_frames = os.listdir(video_path)
                        video_path = [os.path.join(video_root, x) for x in video_frames]
                        max_pixels = int(total_pixels / len(video_frames) * 2)
                        messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "max_pixels": max_pixels}]}]  
                        image_inputs, video_inputs = process_vision_info(messages)
                        image = processor(images=None, videos=video_inputs[0], return_tensors="pt")
                else:
                    raise NotImplementedError
        
        # preprocess speech generation part:
        if 'tgt_unit' in sources[0]:
            tgt_units = self.list_data_dict[i]['tgt_unit']
            numbers = re.findall(r'<(\d+)>', tgt_units)
            numbers = [int(num) for num in numbers]
        # preprocess speech file:   
        if 'speech' in sources[0]:
            speech_files = self.list_data_dict[i]['speech']
            speech_folder = self.list_speech_path[i]
            processor = self.data_args.speech_processor

            if not isinstance(speech_files, list):
                speech_files = [speech_files]

            speech_files = [os.path.join(speech_folder, speech_file) for speech_file in speech_files]
            
            speech_total = []
            for speech_file in speech_files:
                wav, sample_rate = torchaudio.load(speech_file)
                target_sample_rate = 16000
                resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                wav = resample_transform(wav)
                if wav.ndim != 1:
                    wav = wav[0]
                    
                speech = processor(raw_speech=wav, 
                                sampling_rate=target_sample_rate, 
                                return_tensors="pt", 
                                return_attention_mask=True)
                speech_total.append(speech["input_features"].squeeze())  # (mels, lengths), e.g. (128, 3000)

            if len(speech_total) > 1:
                speech = torch.stack(speech_total, dim=0)
            else: # always in this branch in current version
                speech = speech_total[0]  
        
        if 'long_speech' in sources[0]:
            speech_files = self.list_data_dict[i]['long_speech']
            speech_folder = self.data_args.speech_folder
            processor = self.data_args.speech_processor

            if not isinstance(speech_files, list):
                speech_files = [speech_files]

            speech_files = [os.path.join(speech_folder, speech_file) for speech_file in speech_files]
            
            speech_total = []
            for speech_file in speech_files:
                
                speech_total = []
                wav, sample_rate = torchaudio.load(speech_file)
                target_sample_rate = 16000
                resample_transform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                wav = resample_transform(wav)
                if wav.ndim != 1:
                    wav = wav[0]
                if self.data_args.speech_overlap:
                    temp_wav = wav.unfold(0, target_sample_rate * 30, target_sample_rate * self.data_args.speech_overlap_time)
                    temp_wav = [wav_clip for wav_clip in temp_wav]
                    clip_num = len(temp_wav)
                    temp_wav.append(wav[target_sample_rate * self.data_args.speech_overlap_time * clip_num:])
                    for wav_clip in temp_wav:
                        speech_tensor = processor(raw_speech=wav_clip, 
                                            sampling_rate=target_sample_rate, 
                                            return_tensors="pt", 
                                            return_attention_mask=True)["input_features"].squeeze() # (128,3000)
                        speech_total.append(speech_tensor)
                else:
                    whipser_len = target_sample_rate * 30
                    speech_num = wav.shape[0] // whipser_len + 1
                    for speech_num_idx in range(speech_num):
                        temp_wav = wav[speech_num_idx*whipser_len:(speech_num_idx+1)*whipser_len]
                        speech_tensor = processor(raw_speech=temp_wav, 
                                            sampling_rate=target_sample_rate, 
                                            return_tensors="pt", 
                                            return_attention_mask=True)["input_features"].squeeze() # (128,3000)
                        speech_total.append(speech_tensor)
                
                
            if len(speech_total) > 1:
                speech = torch.stack(speech_total, dim=0)
            else: # always in this branch in current version
                speech = speech_total[0]   
            # print('speech input shape is :', speech.shape)
        # ── 点云数据加载（预处理 .pth: sampled_features + group_idx） ──
        pointcloud_data = None
        if 'pointcloud' in sources[0]:
            pc_files = self.list_data_dict[i]['pointcloud']
            pc_folder = self.list_pointcloud_path[i]
            if not isinstance(pc_files, list):
                pc_files = [pc_files]
            # 当前每 batch 支持 1 个点云
            pc_path = os.path.join(pc_folder, pc_files[0])
            pc_raw = torch.load(pc_path, weights_only=False, map_location='cpu')
            pointcloud_data = {
                'sampled_features': torch.from_numpy(pc_raw['sampled_features']).float()
                    if isinstance(pc_raw['sampled_features'], np.ndarray) else pc_raw['sampled_features'].float(),
                'group_idx': torch.from_numpy(pc_raw['group_idx']).long()
                    if isinstance(pc_raw['group_idx'], np.ndarray) else pc_raw['group_idx'].long(),
            }

        # ── IMU 数据加载（.npy: (T, 6) 时间序列，加速度 xyz + 陀螺仪 xyz） ──
        imu_data = None
        if 'imu' in sources[0]:
            imu_files = self.list_data_dict[i]['imu']
            imu_folder = self.list_imu_path[i]
            if not isinstance(imu_files, list):
                imu_files = [imu_files]
            # 当前每 batch 支持 1 个 IMU
            imu_path = os.path.join(imu_folder, imu_files[0])
            imu_raw = np.load(imu_path)                        # (T, 6) float64
            imu_data = torch.from_numpy(imu_raw).float()       # (T, 6)

        # ── Radar 数据加载（多帧 mmWave .bin，每帧 (n_i, 5) [x,y,z,Doppler,intensity]） ──
        # jsonl 提供 radar 目录（如 "E01/S01/A01"）+ start/end（1-based 闭区间，
        # MMFi 原始帧编号 frame{idx:03d}.bin；空帧已过滤则范围内自动跳过）。
        radar_data = None
        if 'radar' in sources[0]:
            radar_rel = self.list_data_dict[i]['radar']
            radar_folder = self.list_radar_path[i]
            radar_dir = os.path.join(radar_folder, radar_rel)
            start = int(self.list_data_dict[i].get('start', 1))
            end = int(self.list_data_dict[i].get('end', start))
            radar_data = load_radar_segment(
                radar_dir, start, end,
                max_points=getattr(self.data_args, 'radar_max_points', 128),
                max_frames=getattr(self.data_args, 'radar_max_frames', 32),
            )
            # jsonl ``lr_flip: true``：点云 y 取反 + 文本已在生成脚本里 left↔right
            if self.list_data_dict[i].get('lr_flip'):
                radar_data = apply_radar_lr_flip(radar_data)

        # ── Smell 数据加载（.npy: (T, 6) 定长多通道气体传感器时间序列） ──
        # 与 IMU 完全一致的结构（固定 shape，无需 padding/mask）。
        smell_data = None
        if 'smell' in sources[0]:
            smell_files = self.list_data_dict[i]['smell']
            smell_folder = self.list_smell_path[i]
            if not isinstance(smell_files, list):
                smell_files = [smell_files]
            # 当前每 batch 支持 1 个 smell
            smell_path = os.path.join(smell_folder, smell_files[0])
            smell_raw = np.load(smell_path)                        # (T, 6) float32
            smell_data = torch.from_numpy(smell_raw).float()       # (T, 6)

        # ── Motion 骨骼轨迹数据加载（.npy: (320, 24, 3) float32，定长时序，无需归一化）──
        motion_data = None
        if 'motion' in sources[0]:
            motion_files = self.list_data_dict[i]['motion']
            motion_folder = self.list_motion_path[i]
            if not isinstance(motion_files, list):
                motion_files = [motion_files]
            # 当前每 batch 支持 1 个 motion
            motion_path = os.path.join(motion_folder, motion_files[0])
            motion_raw = np.load(motion_path)                      # (320, 24, 3) float32
            motion_data = torch.from_numpy(motion_raw).float()     # (320, 24, 3)

        # import pdb; pdb.set_trace()
        if ('image' in sources[0] or 'speech' in sources[0] or 'long_speech' in sources[0]
                or 'pointcloud' in sources[0] or 'imu' in sources[0] or 'depth' in sources[0]
                or 'tactile' in sources[0] or 'thermal' in sources[0] or 'radar' in sources[0]
                or 'smell' in sources[0] or 'motion' in sources[0]):
            # <depth>/<pointcloud>/<imu> 各有独立 token index，preprocess_multimodal 不处理它们
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ('image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i])
        has_speech = ('speech' in self.list_data_dict[i]) or ('long_speech' in self.list_data_dict[i])
        has_pointcloud = ('pointcloud' in self.list_data_dict[i])
        has_imu = ('imu' in self.list_data_dict[i])
        has_depth = ('depth' in self.list_data_dict[i])
        has_tactile = ('tactile' in self.list_data_dict[i])
        has_thermal = ('thermal' in self.list_data_dict[i])
        has_radar = ('radar' in self.list_data_dict[i])
        has_smell = ('smell' in self.list_data_dict[i])
        has_motion = ('motion' in self.list_data_dict[i])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image,
            has_speech=has_speech,
            has_pointcloud=has_pointcloud,
            has_imu=has_imu,
            has_depth=has_depth,
            has_tactile=has_tactile,
            has_thermal=has_thermal)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        if has_image:
            data_dict['image'] = image_total
            if not isinstance(data_dict['image'], list):
                data_dict['image'] = [data_dict['image']]
        elif self.data_args.is_multimodal and not has_pointcloud and not has_imu and not has_tactile and not has_thermal and not has_radar and not has_smell and not has_motion:
            # 纯点云 / 纯 IMU / 纯 radar / 纯 smell / 纯 motion 数据：不塞 dummy image（避免白跑 ViT；它们走自己的 tokenizer+ViT 路径）
            if self.data_args.image_aspect_ratio in ("qwen2vl", "qwen25vl", "qwen35vl"):
                tmp_image = 'examples/placeholder.jpg'
                tmp_image = Image.open(tmp_image).convert('RGB')
                data_dict['image'] = self.data_args.image_processor(images=tmp_image, return_tensors="pt")
            elif hasattr(self.data_args.image_processor, "crop_size"):
                data_dict['image'] = torch.stack([torch.zeros(3, self.data_args.image_processor.crop_size['height'], self.data_args.image_processor.crop_size['width'])] * 2)
            elif hasattr(self.data_args.image_processor, "size"):
                data_dict['image'] = torch.stack([torch.zeros(3, self.data_args.image_processor.size['height'], self.data_args.image_processor.size['width'])] * 2)

                
        # append speech
        if 'speech' in self.list_data_dict[i] or 'long_speech' in self.list_data_dict[i]:
            data_dict['speech'] = speech
        elif self.data_args.is_speech_multimodal:
            data_dict['speech'] = torch.zeros(128, 3000)
        
        if 'speech_asr' in self.list_data_dict[i]:
            data_dict['speech_asr'] = speech_asr.squeeze()
        elif self.data_args.is_speech_multimodal:
            data_dict['speech_asr'] = torch.zeros(100).to(device=data_dict["input_ids"].device, dtype=data_dict["input_ids"].dtype)
        
        if 'tgt_unit' in self.list_data_dict[i]:
            data_dict['tgt_unit'] = torch.tensor(numbers)
        
        # depth images（独立于 RGB image 存储）
        if depth_total:
            data_dict['depth_image'] = depth_total

        # tactile images（独立存储，与 depth 完全对称）
        if tactile_total:
            data_dict['tactile_image'] = tactile_total

        # thermal images（独立存储，与 tactile/depth 完全对称）
        if thermal_total:
            data_dict['thermal_image'] = thermal_total

        # point cloud
        if pointcloud_data is not None:
            data_dict['pointcloud'] = pointcloud_data

        # imu
        if imu_data is not None:
            data_dict['imu'] = imu_data

        # radar
        if radar_data is not None:
            data_dict['radar'] = radar_data

        # smell
        if smell_data is not None:
            data_dict['smell'] = smell_data


        # motion
        if motion_data is not None:
            data_dict['motion'] = motion_data

        # Stable per-modality identity for instance-aware contrastive losses.
        # The raw modality field identifies the underlying signal/image; the
        # configured root prevents unrelated datasets with the same relative
        # filename from colliding. Radar segments also need their frame range
        # and mirror transform because one directory contains many sequences.
        modality_roots = {
            'depth': self.list_depth_path[i],
            'pointcloud': self.list_pointcloud_path[i],
            'imu': self.list_imu_path[i],
            'tactile': self.list_tactile_path[i],
            'thermal': self.list_thermal_path[i],
            'radar': self.list_radar_path[i],
            'smell': self.list_smell_path[i],
            'motion': self.list_motion_path[i],
        }
        modality_instance_ids = {}
        raw_sample = self.list_data_dict[i]
        for modality, root in modality_roots.items():
            if modality not in raw_sample:
                continue
            identity = {
                'modality': modality,
                'root': root,
                'value': raw_sample[modality],
            }
            if modality == 'radar':
                identity.update({
                    'start': int(raw_sample.get('start', 1)),
                    'end': int(raw_sample.get(
                        'end', raw_sample.get('start', 1))),
                    'lr_flip': bool(raw_sample.get('lr_flip', False)),
                })
            modality_instance_ids[modality] = json.dumps(
                identity, ensure_ascii=False, sort_keys=True,
                separators=(',', ':'))
        data_dict['modality_instance_ids'] = modality_instance_ids

        if 'caption' in self.list_data_dict[i]:
            if getattr(self.data_args, 'object_depth', False) and 'objects' in self.list_data_dict[i]:
                # object_depth mode: prepend objects description + separator before caption.
                # Objects tokens go through self-attention only (no cross-attention to image).
                # Caption tokens do both self-attention (seeing objects context) and
                # cross-attention (attending to RGB+depth image features).
                objects_raw = self.list_data_dict[i]['objects']
                caption_raw = self.list_data_dict[i]['caption']
                objects_str = objects_raw[0] if isinstance(objects_raw, list) else objects_raw
                caption_str = caption_raw[0] if isinstance(caption_raw, list) else caption_raw

                objects_ids = self.tokenizer(objects_str, add_special_tokens=False).input_ids
                caption_ids_list = self.tokenizer(caption_str, add_special_tokens=False).input_ids
                sep_id = self.tokenizer.eos_token_id  # use EOS as objects/caption separator
                full_ids = objects_ids + [sep_id] + caption_ids_list
                # objects_len: number of leading tokens that are *not* part of the caption
                # (all object tokens + the separator token)
                data_dict['caption'] = torch.tensor([full_ids], dtype=torch.long)
                data_dict['objects_len'] = len(objects_ids) + 1
            else:
                caption_ids = self.tokenizer(self.list_data_dict[i]['caption']).input_ids
                data_dict['caption'] = torch.tensor([caption_ids], dtype=torch.long)
                data_dict['objects_len'] = 0
        else:
            data_dict['caption'] = None
            data_dict['objects_len'] = 0

        data_dict['target_modality'] = self.list_target_modality[i]
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    enforce_target_modality_homogeneous: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        homogeneous_modality = None
        if self.enforce_target_modality_homogeneous:
            modalities = {instance.get('target_modality') for instance in instances}
            if len(modalities) != 1:
                raise RuntimeError(
                    f"mixed target modalities in one micro-batch: {sorted(modalities)}")
            homogeneous_modality = next(iter(modalities))
            if homogeneous_modality not in MODALITY_ID_MAP:
                raise RuntimeError(
                    f"unsupported homogeneous target modality: {homogeneous_modality!r}")
        input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if homogeneous_modality is not None:
            batch['target_modality_id'] = torch.tensor(
                MODALITY_ID_MAP[homogeneous_modality], dtype=torch.long)

        # Mixed-modality batches (e.g. image-caption + pointcloud-only) are common
        # when per_device_train_batch_size > 1. Do not key off instances[0] only.
        if any('image' in instance for instance in instances):
            batch['images'] = [instance.get('image', []) for instance in instances]

        if any('depth_image' in instance for instance in instances):
            batch['depth_images'] = [instance.get('depth_image', []) for instance in instances]

        if any('tactile_image' in instance for instance in instances):
            batch['tactile_images'] = [instance.get('tactile_image', []) for instance in instances]

        if any('thermal_image' in instance for instance in instances):
            batch['thermal_images'] = [instance.get('thermal_image', []) for instance in instances]

        if 'speech' in instances[0]:
            speeches = [instance['speech'] for instance in instances]
            batch['speeches'] = speeches

        if any('pointcloud' in instance for instance in instances):
            batch['pointclouds'] = [instance.get('pointcloud', []) for instance in instances]

        if any('imu' in instance for instance in instances):
            batch['imus'] = [instance.get('imu', []) for instance in instances]

        if any('radar' in instance for instance in instances):
            batch['radars'] = [instance.get('radar', []) for instance in instances]

        if any('smell' in instance for instance in instances):
            batch['smell_signals'] = [instance.get('smell', []) for instance in instances]


        if any('motion' in instance for instance in instances):
            batch['motions'] = [instance.get('motion', []) for instance in instances]

        if 'speech_asr' in instances[0]:
            speeches_asr = [instance['speech_asr'] for instance in instances]
            batch['speeches_asr'] = speeches_asr
        
        if 'tgt_unit' in instances[0]:
            target_units = [instance['tgt_unit'] for instance in instances]
            if all(x is not None and x.shape == target_units[0].shape for x in target_units):
                batch['tgt_units'] = torch.stack(target_units)

            else:
                target_units = torch.nn.utils.rnn.pad_sequence(target_units,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX)
                batch['tgt_units'] = target_units
        
        # # import pdb; pdb.set_trace()
        # if 'caption' in instances[0]:
        #     caption = [instance['caption'] for instance in instances]
        #     batch['caption'] = caption
        caption = [instance['caption'] for instance in instances]
        batch['caption'] = caption
        objects_len = [instance.get('objects_len', 0) for instance in instances]
        batch['objects_len'] = objects_len
        batch['modality_instance_ids'] = [
            instance.get('modality_instance_ids', {}) for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                enforce_target_modality_homogeneous=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                datasets=data_args.datasets)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        enforce_target_modality_homogeneous=(
            enforce_target_modality_homogeneous),
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def train(attn_implementation="flash_attention_2"):
    global local_rank

    # 在 5090(Blackwell/sm_120) 上，cuDNN 多条路径（SDPA flash 后端、nn.MultiheadAttention
    # 内部 C++ 快路径、Conv2d backward）均会抛 CUDNN_STATUS_INTERNAL_ERROR。
    # LLM 走 flash_attn 扩展 + fla Triton kernel，完全不经过 cuDNN，因此对 Blackwell
    # 直接全关 cuDNN 不影响训练速度和正确性；H100(sm_90) 等旧架构不触发此分支。
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            torch.backends.cudnn.enabled = False       # 全关 cuDNN（LLM 不走 cuDNN，安全）
            torch.backends.cuda.enable_cudnn_sdp(False)  # 冗余但无害，保留作文档
    except Exception:
        pass

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # Reuse Stage-2 smell normalization when starting expert tuning, including
    # older checkpoints that saved these values only in config.json.
    if (training_args.load_non_lora_weights_from
            and model_args.mm_smell_projector_type is not None):
        checkpoint_config_path = os.path.join(
            training_args.load_non_lora_weights_from, 'config.json')
        if os.path.isfile(checkpoint_config_path):
            with open(checkpoint_config_path) as config_file:
                checkpoint_config = json.load(config_file)
            for name in (
                'smell_standardize', 'smell_channel_mean', 'smell_channel_std'
            ):
                if name in checkpoint_config:
                    setattr(model_args, name, checkpoint_config[name])
    # Unisense creates modality projectors, caption decoders, and MoE modules before
    # constructing Trainer. Trainer seeds the RNG in its own __init__, which is
    # too late for those randomly initialized modules. Seed here as well so that
    # fresh runs with the same --seed start from identical added-module weights.
    if training_args.full_determinism:
        transformers.enable_full_determinism(training_args.seed)
    else:
        transformers.set_seed(training_args.seed)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    # Must be initialized before from_pretrained so ZeRO-3's zero.Init() wraps model creation.
    # Resolve "auto" batch-size fields upfront so zero.Init() (called inside from_pretrained)
    # receives concrete integers instead of the string "auto".
    if training_args.deepspeed:
        import json as _json
        from transformers.integrations import HfDeepSpeedConfig
        with open(training_args.deepspeed) as _f:
            _ds_cfg = _json.load(_f)
        _world_size = int(os.environ.get("WORLD_SIZE", 1))
        _micro_bs = training_args.per_device_train_batch_size
        _gas = training_args.gradient_accumulation_steps
        _ds_cfg["train_micro_batch_size_per_gpu"] = _micro_bs
        _ds_cfg["gradient_accumulation_steps"] = _gas
        _ds_cfg["train_batch_size"] = _micro_bs * _gas * _world_size
        hf_deepspeed_config = HfDeepSpeedConfig(_ds_cfg)

    if model_args.vision_tower is not None or \
        model_args.speech_tower is not None:
        model = UnisenseQwen35ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if is_main_process():
        # ── Attention implementation check ──────────────────────────────
        print(f"[attn] _attn_implementation = {getattr(model.config, '_attn_implementation', 'N/A')}")
        try:
            from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
            print(f"[attn] flash_linear_attention (fla) available = {is_flash_linear_attention_available()}")
            print(f"[attn] causal_conv1d available               = {is_causal_conv1d_available()}")
            llm = getattr(model.model, 'language_model', None)
            if llm is not None:
                layer0 = llm.layers[0]
                if hasattr(layer0, 'linear_attn'):
                    fn = layer0.linear_attn.chunk_gated_delta_rule
                    print(f"[attn] GatedDeltaNet kernel = {fn.__module__}.{fn.__name__}")
                if hasattr(layer0, 'self_attn'):
                    print(f"[attn] full-attn layer0 impl = {layer0.self_attn.config._attn_implementation}")
        except Exception as e:
            print(f"[attn] check failed: {e}")
        # ────────────────────────────────────────────────────────────────

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.moe_enable:
        from unisense.model.moe_ffn import apply_moe_ffn_surgery
        _shared_dense = bool(model_args.moe_shared_dense)
        _shared_pool = bool(model_args.moe_shared_pool)
        if _shared_dense and _shared_pool:
            raise ValueError("moe_shared_dense and moe_shared_pool are mutually exclusive")
        if _shared_dense:
            rank0_print("Applying SharedDense FFN surgery (MSC ablation)...")
            _bal_type = "none"
            _bal_weight = 0.0
            _div_weight = 0.0
        elif _shared_pool:
            rank0_print("Applying shared expert-pool MoE FFN surgery...")
            _bal_type = model_args.moe_balance_loss_type
            _bal_weight = model_args.moe_balance_loss_weight
            _div_weight = model_args.moe_diversity_loss_weight
        else:
            rank0_print("Applying MoE FFN surgery...")
            _bal_type = model_args.moe_balance_loss_type
            _bal_weight = model_args.moe_balance_loss_weight
            _div_weight = model_args.moe_diversity_loss_weight
        _moe_group_indices = [int(x) for x in model_args.moe_group_indices.split(',')]
        apply_moe_ffn_surgery(
            model.get_model(),
            modal_list=model_args.moe_modal_list.split(','),
            num_experts_per_modal=model_args.moe_num_experts,
            r=model_args.moe_lora_r,
            lora_alpha=model_args.moe_lora_alpha,
            proj_mode=model_args.moe_proj_mode,
            full_attn_only=model_args.moe_full_attn_only,
            balance_loss_type=_bal_type,
            group_indices=_moe_group_indices,
            shared_dense=_shared_dense,
            shared_pool=_shared_pool,
        )
        # Save MoE config so builder.py can reconstruct the architecture at eval time
        model.config.moe_enable = True
        model.config.moe_shared_dense = _shared_dense
        model.config.moe_shared_pool = _shared_pool
        model.config.moe_modal_list = model_args.moe_modal_list
        model.config.moe_num_experts = model_args.moe_num_experts
        model.config.moe_lora_r = model_args.moe_lora_r
        model.config.moe_lora_alpha = model_args.moe_lora_alpha
        model.config.moe_proj_mode = model_args.moe_proj_mode
        model.config.moe_group_indices = model_args.moe_group_indices
        model.config.moe_full_attn_only = model_args.moe_full_attn_only
        model.config.moe_balance_loss_type = _bal_type
        model.config.moe_balance_loss_weight = _bal_weight
        model.config.moe_diversity_loss_weight = _div_weight
        # Runtime MoE-aux schedule state. Trainer fills these after max_steps
        # has been resolved, converting optimizer-step boundaries and the
        # user-facing diversity ratio to per-rank forwards.
        model.config.moe_diversity_warmup_forwards = 0
        model.config.moe_aux_start_forward = 0
        model.config.moe_diagnostic_interval = model_args.moe_diagnostic_interval
        model.config.moe_diagnostic_gradient_accumulation_steps = training_args.gradient_accumulation_steps

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.target_modules.split(','),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        # fix bugs after special token with use_fast=True
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if (
        "qwen2vl" == model_args.version
        or "qwen2vl_original" == model_args.version
        or "qwen25vl" == model_args.version
        or "qwen35vl" == model_args.version
        or "qwen35_all_round_no_think" == model_args.version
        or "qwen35_all_round" == model_args.version
        or "qwen35_last_think_all_round" == model_args.version
    ):
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen2vl"]
    elif "qwen_2" == model_args.version:
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    elif "llama_3" == model_args.version:
        # set unknown token and pad token to the first reserved special token
        if tokenizer.unk_token is None:
            tokenizer.unk_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    elif "llama_3_1" == model_args.version:
        # set unknown token and pad token to the first reserved special token
        if tokenizer.unk_token is None:
            tokenizer.unk_token = "<|reserved_special_token_2|>"
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3_1"]
    else:
        #  for Qwen-2 plain training
        if tokenizer.unk_token is None:
            tokenizer.pad_token = "<|endoftext|>"
        else:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # import pdb; pdb.set_trace()
    if model_args.vision_tower is not None:
        _emb_weight = model.get_input_embeddings().weight
        if hasattr(_emb_weight, 'ds_id'):
            from deepspeed import zero
            with zero.GatheredParameters([_emb_weight]):
                model_args.text_embedding = _emb_weight.data.detach().cpu().clone()
        else:
            model_args.text_embedding = _emb_weight
        model_args.use_flash_attn = (attn_implementation == "flash_attention_2")
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
        
        data_args.video_processor = copy.deepcopy(vision_tower.image_processor)
        data_args.is_multimodal = True

        model.config.image_grid = data_args.image_grid
        model.config.image_global = data_args.image_global
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.speech_overlap = data_args.speech_overlap
        model.config.speech_overlap_time = data_args.speech_overlap_time
        model.config.overlap_ratio = data_args.overlap_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        
        model.config.down_factor_1 = model_args.down_factor_1
        model.config.down_factor_2 = model_args.down_factor_2
        model.config.train_modality = model_args.train_modality
        model.config.check_data_modality = model_args.check_data_modality
        
        model.config.asr_align = model_args.asr_align
        

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_vision_mlp_adapter = training_args.freeze_mm_vision_mlp_adapter
        if training_args.freeze_mm_vision_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        if model_args.optimize_vision_tower:
            vision_tower.requires_grad_(True)
            # print('Optimize last 1/2 layers in vision tower')
            # total_num = len(vision_tower.vision_tower.vision_model.encoder.layers)
            # for _idx in range(total_num//2, total_num):
            #     vision_tower.vision_tower.vision_model.encoder.layers[_idx].requires_grad_(True)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Start: add speech module
    if model_args.generate:
        model.initialize_speech_generator(
            model_args=model_args
        )
        
        speech_generator = model.get_speech_generator()
        speech_generator.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        
        model.config.generate = model_args.generate
        model.config.ctc_decoder_config = model_args.ctc_decoder_config
        model.config.ctc_upsample_factor = model_args.ctc_upsample_factor
        model.config.ctc_loss_weight = model_args.ctc_loss_weight
        model.config.unit_vocab_size = model_args.unit_vocab_size
        
        model.config.tune_speech_generator_only = model_args.tune_speech_generator_only
        if model_args.tune_speech_generator_only:
            model.requires_grad_(False)
            for p in model.speech_generator.parameters():
                p.requires_grad = True
        
    if model_args.speech_tower is not None:
        speech_tower_name = model_args.speech_tower

        # load_non_lora_weights_from will load speech projector from non_lora_trainables.bin,
        # so pretrain_mm_speech_mlp_adapter is redundant and should not be loaded.
        if training_args.load_non_lora_weights_from:
            model_args.pretrain_mm_speech_mlp_adapter = None

        model.get_model().initialize_speech_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        speech_tower = model.get_speech_tower()
        speech_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.speech_processor = copy.deepcopy(speech_tower.speech_processor)
        # if "whisper" in speech_tower_name.lower():
        #     from transformers import WhisperFeatureExtractor
        #     data_args.speech_processor = WhisperFeatureExtractor.from_pretrained(model_args.speech_tower)
        # else:
        #     raise NotImplementedError
            
        data_args.speech_processor_type = speech_tower_name.lower()
        data_args.is_speech_multimodal = True
        
        model.config.train_modality = model_args.train_modality
        model.config.check_data_modality = model_args.check_data_modality
        model.config.asr_align = model_args.asr_align
        model.config.align_temperature = model_args.align_temperature
        model.config.weight_lambda = model_args.weight_lambda
        model.config.depth_caption_weight = (
            model_args.modality_alignment_weight
            if model_args.modality_alignment_weight is not None
            else model_args.depth_caption_weight
        )
        model.config.depth_decorr_weight = model_args.depth_decorr_weight
        model.config.depth_decorr_margin = model_args.depth_decorr_margin
        model.config.object_depth = model_args.object_depth
        model.config.tactile_caption_weight = (
            model_args.modality_alignment_weight
            if model_args.modality_alignment_weight is not None
            else model_args.tactile_caption_weight
        )
        data_args.object_depth = model_args.object_depth
        model.config.coca_image_input = model_args.coca_image_input
        model.config.speech_encoder_ds_rate = model_args.speech_encoder_ds_rate
        model.config.speech_encoder_hidden_size = model_args.speech_encoder_hidden_size
        model.config.speech_norm = model_args.speech_norm
        model.config.speech_learn = model_args.speech_learn
        model.config.align_norm = model_args.align_norm
        model.config.align_type = model_args.align_type
        
        

        model.config.tune_speech_projector_only = training_args.tune_speech_projector_only = model_args.tune_speech_projector_only
        if model_args.tune_speech_projector_only:
            model.requires_grad_(False)  # BEAWARE: Need to remove in case vision tower become false
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = True
        elif training_args.freeze_mm_speech_mlp_adapter:
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = False
        else:
            # Default: speech projector is trainable (e.g. Stage 1.5 / Stage 2 with LoRA)
            for p in model.get_model().mm_speech_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_speech_mlp_adapter = training_args.freeze_mm_speech_mlp_adapter
                
        if training_args.bits in [4, 8]:
            model.get_model().mm_speech_projector.to(dtype=compute_dtype, device=training_args.device)
    
    if training_args.load_non_lora_weights_from:
        compute_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
        non_lora_bin = os.path.join(training_args.load_non_lora_weights_from, 'non_lora_trainables.bin')
        mm_projector_weights = torch.load(non_lora_bin, map_location='cpu')
        # Stage 1 (no LoRA) saves keys as "model.xxx" (1 prefix).
        # Stage 1.5 (LoRA, via model.base_model.named_parameters) saves as "model.model.xxx" (2 prefixes).
        # Strip to bare module name for loading into model.get_model() (UnisenseQwen3_5_VLModel).
        def _strip_non_lora_prefix(k):
            if k.startswith("model.model."):
                return k[12:]   # LoRA checkpoint: "model.model.xxx" -> "xxx"
            elif k.startswith("model."):
                return k[6:]    # Non-LoRA checkpoint: "model.xxx" -> "xxx"
            return k
        mm_projector_weights = {_strip_non_lora_prefix(k): v.to(compute_dtype) for k, v in mm_projector_weights.items()}
        if training_args.load_non_lora_ignore_moe:
            # Stage-2 alignment runs keep MoE parameters requires_grad=True so
            # they remain in the optimizer/ZeRO graph, and freeze them with
            # gradient hooks.  The generic non-LoRA saver therefore includes
            # their untouched initialization.  An explicit opt-in filter lets
            # Stage 3 load only projector/tokenizer/alignment state and create
            # its requested MoE topology from a clean seeded initialization.
            moe_state_markers = (
                ".moe_routers.",
                ".moe_experts.",
                ".moe_shared_expert.",
                ".moe_shared_router.",
                ".moe_shared_experts.",
                "_moe_forward_count",
            )
            moe_keys = [
                key for key in mm_projector_weights
                if any(marker in key for marker in moe_state_markers)
            ]
            for key in moe_keys:
                del mm_projector_weights[key]
            print(
                "Ignored {} saved MoE keys from {}; the current run's MoE "
                "remains freshly initialized.".format(len(moe_keys), non_lora_bin)
            )
        print("Loading non-LoRA weights (projectors):", mm_projector_weights.keys())
        status = model.get_model().load_state_dict(mm_projector_weights, strict=False)
        print('load non_lora_trainables, unexpected_keys:{}'.format(status.unexpected_keys))
        model.to(compute_dtype)

    if training_args.lora_enable and training_args.load_lora_weights_from:
        # NOTE: peft>=0.13 requires explicit adapter_name. The LoRA adapter created by
        # get_peft_model defaults to "default", so we use the same name here.
        model.load_adapter(training_args.load_lora_weights_from, adapter_name="default")
        print(f"Loading LoRA weights from {training_args.load_lora_weights_from}")

    # Projection heads stay in fp32 (especially their LayerNorms); alignment
    # forwards cast inputs to the head dtype. Both independently initialized C2
    # heads remain trainable; only the Qwen backbone supplying text targets is
    # frozen.
    _alignment_modalities = (
        'depth', 'pointcloud', 'imu', 'tactile', 'thermal',
        'radar', 'smell', 'motion')
    _dm = model.get_model()
    for _modality in _alignment_modalities:
        _is_c2 = (getattr(
            model.config, f'{_modality}_alignment_type', None)
            == 'coca_staged_qwen')
        for _branch in ('student', 'text'):
            _head = getattr(
                _dm, f'{_modality}_alignment_{_branch}_head', None)
            if _head is not None:
                _head.to(device=training_args.device, dtype=torch.float32)
                if _is_c2:
                    # freeze_backbone is applied before modality initialization
                    # in the usual path, but make the C2 contract explicit and
                    # robust to modules restored by other initialization paths.
                    _head.requires_grad_(True)
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    # ── Pointcloud tokenizer + projector + CoCa：dtype 移动 + freeze 控制 ──
    # 必须 AFTER load_non_lora_weights_from 但放在 depth freeze 之前。
    if getattr(model_args, 'mm_pointcloud_projector_type', None) is not None:
        _dm = model.get_model()
        _pc_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if hasattr(_dm, 'pointcloud_tokenizer'):
            _dm.pointcloud_tokenizer.to(dtype=_pc_dtype, device=training_args.device)
        if hasattr(_dm, 'modality_projectors') and 'pointcloud' in _dm.modality_projectors:
            _dm.modality_projectors['pointcloud'].to(dtype=_pc_dtype, device=training_args.device)
        if hasattr(_dm, 'pointcloud_caption_module'):
            _dm.pointcloud_caption_module.to(dtype=_pc_dtype, device=training_args.device)
        if hasattr(_dm, 'pointcloud_alignment_metric_head'):
            _dm.pointcloud_alignment_metric_head.to(device=training_args.device)
        if hasattr(_dm, 'pointcloud_alignment_student_head'):
            _dm.pointcloud_alignment_student_head.to(device=training_args.device)
        if hasattr(_dm, 'pointcloud_alignment_text_head'):
            _dm.pointcloud_alignment_text_head.to(device=training_args.device)

        if training_args.freeze_mm_pointcloud_mlp_adapter:
            assert training_args.unfreeze_pointcloud_at_ratio == 0.0, \
                "freeze_mm_pointcloud_mlp_adapter and unfreeze_pointcloud_at_ratio are mutually exclusive."
            if hasattr(_dm, 'pointcloud_tokenizer'):
                _dm.pointcloud_tokenizer.requires_grad_(False)
            if hasattr(_dm, 'modality_projectors') and 'pointcloud' in _dm.modality_projectors:
                _dm.modality_projectors['pointcloud'].requires_grad_(False)
            if hasattr(_dm, 'pointcloud_caption_module'):
                _dm.pointcloud_caption_module.requires_grad_(False)
            if hasattr(_dm, 'pointcloud_alignment_metric_head'):
                _dm.pointcloud_alignment_metric_head.requires_grad_(False)
            if hasattr(_dm, 'pointcloud_alignment_student_head'):
                _dm.pointcloud_alignment_student_head.requires_grad_(False)
            if hasattr(_dm, 'pointcloud_alignment_text_head'):
                _dm.pointcloud_alignment_text_head.requires_grad_(False)
            rank0_print("Frozen pointcloud tokenizer + projector + alignment modules "
                        "(freeze_mm_pointcloud_mlp_adapter=True).")

    # ── IMU tokenizer + projector + CoCa：dtype 移动 + freeze 控制 ──
    # 必须 AFTER load_non_lora_weights_from 但放在 depth freeze 之前。
    if getattr(model_args, 'mm_imu_projector_type', None) is not None:
        _dm = model.get_model()
        _imu_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if hasattr(_dm, 'imu_tokenizer'):
            _dm.imu_tokenizer.to(dtype=_imu_dtype, device=training_args.device)
        if hasattr(_dm, 'modality_projectors') and 'imu' in _dm.modality_projectors:
            _dm.modality_projectors['imu'].to(dtype=_imu_dtype, device=training_args.device)
        if hasattr(_dm, 'imu_caption_module'):
            _dm.imu_caption_module.to(dtype=_imu_dtype, device=training_args.device)

        if training_args.freeze_mm_imu_mlp_adapter:
            assert training_args.unfreeze_imu_at_ratio == 0.0, \
                "freeze_mm_imu_mlp_adapter and unfreeze_imu_at_ratio are mutually exclusive."
            if hasattr(_dm, 'imu_tokenizer'):
                _dm.imu_tokenizer.requires_grad_(False)
            if hasattr(_dm, 'modality_projectors') and 'imu' in _dm.modality_projectors:
                _dm.modality_projectors['imu'].requires_grad_(False)
            if hasattr(_dm, 'imu_caption_module'):
                _dm.imu_caption_module.requires_grad_(False)
            rank0_print("Frozen imu tokenizer + projector + CoCa "
                        "(freeze_mm_imu_mlp_adapter=True).")

    # ── Radar tokenizer + projector + CoCa：dtype 移动 + freeze 控制 ──
    # 必须 AFTER load_non_lora_weights_from 但放在 depth freeze 之前。
    if getattr(model_args, 'mm_radar_projector_type', None) is not None:
        _dm = model.get_model()
        _radar_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if hasattr(_dm, 'radar_tokenizer'):
            _dm.radar_tokenizer.to(dtype=_radar_dtype, device=training_args.device)
        if hasattr(_dm, 'modality_projectors') and 'radar' in _dm.modality_projectors:
            _dm.modality_projectors['radar'].to(dtype=_radar_dtype, device=training_args.device)
        if hasattr(_dm, 'radar_caption_module'):
            _dm.radar_caption_module.to(dtype=_radar_dtype, device=training_args.device)

        if training_args.freeze_mm_radar_mlp_adapter:
            assert training_args.unfreeze_radar_at_ratio == 0.0, \
                "freeze_mm_radar_mlp_adapter and unfreeze_radar_at_ratio are mutually exclusive."
            if hasattr(_dm, 'radar_tokenizer'):
                _dm.radar_tokenizer.requires_grad_(False)
            if hasattr(_dm, 'modality_projectors') and 'radar' in _dm.modality_projectors:
                _dm.modality_projectors['radar'].requires_grad_(False)
            if hasattr(_dm, 'radar_caption_module'):
                _dm.radar_caption_module.requires_grad_(False)
            rank0_print("Frozen radar tokenizer + projector + CoCa "
                        "(freeze_mm_radar_mlp_adapter=True).")

    # ── Smell tokenizer + projector + CoCa：dtype 移动 + freeze 控制 ──
    # 必须 AFTER load_non_lora_weights_from 但放在 depth freeze 之前。
    if getattr(model_args, 'mm_smell_projector_type', None) is not None:
        _dm = model.get_model()
        _smell_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if hasattr(_dm, 'smell_tokenizer'):
            _dm.smell_tokenizer.to(dtype=_smell_dtype, device=training_args.device)
        if hasattr(_dm, 'modality_projectors') and 'smell' in _dm.modality_projectors:
            _dm.modality_projectors['smell'].to(dtype=_smell_dtype, device=training_args.device)
        if hasattr(_dm, 'smell_caption_module'):
            _dm.smell_caption_module.to(dtype=_smell_dtype, device=training_args.device)

        if training_args.freeze_mm_smell_mlp_adapter:
            assert training_args.unfreeze_smell_at_ratio == 0.0, \
                "freeze_mm_smell_mlp_adapter and unfreeze_smell_at_ratio are mutually exclusive."
            if hasattr(_dm, 'smell_tokenizer'):
                _dm.smell_tokenizer.requires_grad_(False)
            if hasattr(_dm, 'modality_projectors') and 'smell' in _dm.modality_projectors:
                _dm.modality_projectors['smell'].requires_grad_(False)
            if hasattr(_dm, 'smell_caption_module'):
                _dm.smell_caption_module.requires_grad_(False)
            rank0_print("Frozen smell tokenizer + projector + CoCa "
                        "(freeze_mm_smell_mlp_adapter=True).")

    # ── Motion tokenizer + projector + CoCa：dtype 移动 + freeze 控制 ──
    # 必须 AFTER load_non_lora_weights_from 但放在 depth freeze 之前。
    if getattr(model_args, 'mm_motion_projector_type', None) is not None:
        _dm = model.get_model()
        _motion_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if hasattr(_dm, 'motion_tokenizer'):
            _dm.motion_tokenizer.to(dtype=_motion_dtype, device=training_args.device)
        if hasattr(_dm, 'modality_projectors') and 'motion' in _dm.modality_projectors:
            _dm.modality_projectors['motion'].to(dtype=_motion_dtype, device=training_args.device)
        if hasattr(_dm, 'motion_caption_module'):
            _dm.motion_caption_module.to(dtype=_motion_dtype, device=training_args.device)

        if training_args.freeze_mm_motion_mlp_adapter:
            assert training_args.unfreeze_motion_at_ratio == 0.0, \
                "freeze_mm_motion_mlp_adapter and unfreeze_motion_at_ratio are mutually exclusive."
            if hasattr(_dm, 'motion_tokenizer'):
                _dm.motion_tokenizer.requires_grad_(False)
            if hasattr(_dm, 'modality_projectors') and 'motion' in _dm.modality_projectors:
                _dm.modality_projectors['motion'].requires_grad_(False)
            if hasattr(_dm, 'motion_caption_module'):
                _dm.motion_caption_module.requires_grad_(False)
            rank0_print("Frozen motion tokenizer + projector + CoCa "
                        "(freeze_mm_motion_mlp_adapter=True).")

    # ── Tactile projector + CoCa：dtype 移动 + freeze 控制 ──────────────────
    # 必须 AFTER load_non_lora_weights_from 但放在 depth freeze 之前。
    if getattr(model_args, 'mm_tactile_projector_type', None) is not None:
        _dm = model.get_model()
        _tac_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if hasattr(_dm, 'modality_projectors') and 'tactile' in _dm.modality_projectors:
            _dm.modality_projectors['tactile'].to(dtype=_tac_dtype, device=training_args.device)
        if hasattr(_dm, 'tactile_caption_module'):
            _dm.tactile_caption_module.to(dtype=_tac_dtype, device=training_args.device)

        if training_args.freeze_mm_tactile_mlp_adapter:
            assert training_args.unfreeze_tactile_at_ratio == 0.0, \
                "freeze_mm_tactile_mlp_adapter and unfreeze_tactile_at_ratio are mutually exclusive."
            if hasattr(_dm, 'modality_projectors') and 'tactile' in _dm.modality_projectors:
                _dm.modality_projectors['tactile'].requires_grad_(False)
            if hasattr(_dm, 'tactile_caption_module'):
                _dm.tactile_caption_module.requires_grad_(False)
            rank0_print("Frozen tactile projector + CoCa (freeze_mm_tactile_mlp_adapter=True).")

    _unfreeze_tactile_callback = None
    if training_args.unfreeze_tactile_at_ratio > 0.0:
        assert not training_args.freeze_mm_tactile_mlp_adapter, \
            "unfreeze_tactile_at_ratio requires freeze_mm_tactile_mlp_adapter=False."
        _dm = model.get_model()
        _tac_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('modality_projectors.tactile' in _n
                 or 'tactile_caption_module' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _tac_hook_handles.append(_h)
        _tac_ratio = training_args.unfreeze_tactile_at_ratio

        class _UnfreezeTactileCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _tac_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _tac_ratio):
                        cnt = len(_tac_hook_handles)
                        for _h in _tac_hook_handles:
                            _h.remove()
                        _tac_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeTactile] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_tac_ratio:.0%}). Tactile proj+CoCa now trainable."
                        )

        _unfreeze_tactile_callback = _UnfreezeTactileCallback()
        rank0_print(
            f"Phased training: tactile proj+CoCa gradient-zeroed for first "
            f"{_tac_ratio:.0%} of steps ({len(_tac_hook_handles)} hooks registered)."
        )

    # ── Depth projector + alignment modules: dtype/device + freeze control ──
    # Must come AFTER load_non_lora_weights_from so weights are loaded first.
    if getattr(model_args, 'mm_depth_projector_type', None) is not None:
        _dm = model.get_model()
        _depth_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        if (hasattr(_dm, 'modality_projectors')
                and 'depth' in _dm.modality_projectors):
            _dm.modality_projectors['depth'].to(
                dtype=_depth_dtype, device=training_args.device)
        if hasattr(_dm, 'depth_caption_module'):
            _dm.depth_caption_module.to(
                dtype=_depth_dtype, device=training_args.device)
        # Keep LayerNorm in its construction dtype, as for pointcloud. The
        # H1 forward casts its inputs to the metric-head parameter dtype.
        if hasattr(_dm, 'depth_alignment_metric_head'):
            _dm.depth_alignment_metric_head.to(device=training_args.device)
        if hasattr(_dm, 'depth_alignment_student_head'):
            _dm.depth_alignment_student_head.to(device=training_args.device)
        if hasattr(_dm, 'depth_alignment_text_head'):
            _dm.depth_alignment_text_head.to(device=training_args.device)

    if training_args.freeze_mm_depth_mlp_adapter:
        assert training_args.unfreeze_depth_at_ratio == 0.0, \
            "freeze_mm_depth_mlp_adapter and unfreeze_depth_at_ratio are mutually exclusive."
        _dm = model.get_model()
        # 注意：当同时启用 depth 和 pointcloud 时，整个 modality_projectors 被冻结会
        # 误伤 pointcloud。这里仅冻结 'depth' 子模块。
        if (hasattr(_dm, 'modality_projectors')
                and 'depth' in _dm.modality_projectors):
            _dm.modality_projectors['depth'].requires_grad_(False)
        if hasattr(_dm, 'depth_caption_module'):
            _dm.depth_caption_module.requires_grad_(False)
        if hasattr(_dm, 'depth_alignment_metric_head'):
            _dm.depth_alignment_metric_head.requires_grad_(False)
        if hasattr(_dm, 'depth_alignment_student_head'):
            _dm.depth_alignment_student_head.requires_grad_(False)
        if hasattr(_dm, 'depth_alignment_text_head'):
            _dm.depth_alignment_text_head.requires_grad_(False)
        rank0_print("Frozen depth projector + alignment modules "
                    "(freeze_mm_depth_mlp_adapter=True). "
                    "Only MoE/other trainable params will be saved to non_lora_trainables.bin.")

    _unfreeze_depth_callback = None
    if training_args.unfreeze_depth_at_ratio > 0.0:
        assert not training_args.freeze_mm_depth_mlp_adapter, \
            "unfreeze_depth_at_ratio requires freeze_mm_depth_mlp_adapter=False."
        _dm = model.get_model()
        _depth_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('modality_projectors.depth' in _n
                 or 'depth_caption_module' in _n
                 or 'depth_alignment_metric_head' in _n
                 or 'depth_alignment_student_head' in _n
                 or 'depth_alignment_text_head' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _depth_hook_handles.append(_h)
        _ratio = training_args.unfreeze_depth_at_ratio

        class _UnfreezeDepthCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _depth_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _ratio):
                        cnt = len(_depth_hook_handles)
                        for _h in _depth_hook_handles:
                            _h.remove()
                        _depth_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeDepth] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_ratio:.0%}). Depth alignment modules now trainable."
                        )

        _unfreeze_depth_callback = _UnfreezeDepthCallback()
        rank0_print(
            f"Phased training: depth alignment modules gradient-zeroed for first {_ratio:.0%} of steps "
            f"({len(_depth_hook_handles)} hooks registered)."
        )

    # ── Pointcloud tokenizer + projector + CoCa phased-unfreeze ──────────
    _unfreeze_pointcloud_callback = None
    if training_args.unfreeze_pointcloud_at_ratio > 0.0:
        assert not training_args.freeze_mm_pointcloud_mlp_adapter, \
            "unfreeze_pointcloud_at_ratio requires freeze_mm_pointcloud_mlp_adapter=False."
        _dm = model.get_model()
        _pc_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('pointcloud_tokenizer' in _n
                 or 'modality_projectors.pointcloud' in _n
                 or 'pointcloud_caption_module' in _n
                 or 'pointcloud_alignment_metric_head' in _n
                 or 'pointcloud_alignment_student_head' in _n
                 or 'pointcloud_alignment_text_head' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _pc_hook_handles.append(_h)
        _pc_ratio = training_args.unfreeze_pointcloud_at_ratio

        class _UnfreezePointcloudCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _pc_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _pc_ratio):
                        cnt = len(_pc_hook_handles)
                        for _h in _pc_hook_handles:
                            _h.remove()
                        _pc_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezePointcloud] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_pc_ratio:.0%}). Pointcloud tokenizer+proj+CoCa now trainable."
                        )

        _unfreeze_pointcloud_callback = _UnfreezePointcloudCallback()
        rank0_print(
            f"Phased training: pointcloud tokenizer+proj+CoCa gradient-zeroed for first "
            f"{_pc_ratio:.0%} of steps ({len(_pc_hook_handles)} hooks registered)."
        )

    # ── IMU tokenizer + projector + CoCa phased-unfreeze ─────────────────
    _unfreeze_imu_callback = None
    if training_args.unfreeze_imu_at_ratio > 0.0:
        assert not training_args.freeze_mm_imu_mlp_adapter, \
            "unfreeze_imu_at_ratio requires freeze_mm_imu_mlp_adapter=False."
        _dm = model.get_model()
        _imu_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('imu_tokenizer' in _n
                 or 'modality_projectors.imu' in _n
                 or 'imu_caption_module' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _imu_hook_handles.append(_h)
        _imu_ratio = training_args.unfreeze_imu_at_ratio

        class _UnfreezeImuCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _imu_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _imu_ratio):
                        cnt = len(_imu_hook_handles)
                        for _h in _imu_hook_handles:
                            _h.remove()
                        _imu_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeImu] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_imu_ratio:.0%}). IMU tokenizer+proj+CoCa now trainable."
                        )

        _unfreeze_imu_callback = _UnfreezeImuCallback()
        rank0_print(
            f"Phased training: imu tokenizer+proj+CoCa gradient-zeroed for first "
            f"{_imu_ratio:.0%} of steps ({len(_imu_hook_handles)} hooks registered)."
        )

    # ── Thermal projector + CoCa phased-unfreeze ──────────────────────────
    _unfreeze_thermal_callback = None
    if training_args.unfreeze_thermal_at_ratio > 0.0:
        assert not training_args.freeze_mm_thermal_mlp_adapter, \
            "unfreeze_thermal_at_ratio requires freeze_mm_thermal_mlp_adapter=False."
        _dm = model.get_model()
        _thermal_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('modality_projectors.thermal' in _n
                 or 'thermal_caption_module' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _thermal_hook_handles.append(_h)
        _thermal_ratio = training_args.unfreeze_thermal_at_ratio

        class _UnfreezeThermalCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _thermal_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _thermal_ratio):
                        cnt = len(_thermal_hook_handles)
                        for _h in _thermal_hook_handles:
                            _h.remove()
                        _thermal_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeThermal] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_thermal_ratio:.0%}). Thermal proj+CoCa now trainable."
                        )

        _unfreeze_thermal_callback = _UnfreezeThermalCallback()
        rank0_print(
            f"Phased training: thermal proj+CoCa gradient-zeroed for first "
            f"{_thermal_ratio:.0%} of steps ({len(_thermal_hook_handles)} hooks registered)."
        )
    # ── Radar tokenizer + projector + CoCa phased-unfreeze ───────────────
    _unfreeze_radar_callback = None
    if training_args.unfreeze_radar_at_ratio > 0.0:
        assert not training_args.freeze_mm_radar_mlp_adapter, \
            "unfreeze_radar_at_ratio requires freeze_mm_radar_mlp_adapter=False."
        _dm = model.get_model()
        _radar_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('radar_tokenizer' in _n
                 or 'modality_projectors.radar' in _n
                 or 'radar_caption_module' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _radar_hook_handles.append(_h)
        _radar_ratio = training_args.unfreeze_radar_at_ratio

        class _UnfreezeRadarCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _radar_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _radar_ratio):
                        cnt = len(_radar_hook_handles)
                        for _h in _radar_hook_handles:
                            _h.remove()
                        _radar_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeRadar] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_radar_ratio:.0%}). Radar tokenizer+proj+CoCa now trainable."
                        )

        _unfreeze_radar_callback = _UnfreezeRadarCallback()
        rank0_print(
            f"Phased training: radar tokenizer+proj+CoCa gradient-zeroed for first "
            f"{_radar_ratio:.0%} of steps ({len(_radar_hook_handles)} hooks registered)."
        )
    # ── Smell tokenizer + projector + CoCa phased-unfreeze ───────────────
    _unfreeze_smell_callback = None
    if training_args.unfreeze_smell_at_ratio > 0.0:
        assert not training_args.freeze_mm_smell_mlp_adapter, \
            "unfreeze_smell_at_ratio requires freeze_mm_smell_mlp_adapter=False."
        _dm = model.get_model()
        _smell_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('smell_tokenizer' in _n
                 or 'modality_projectors.smell' in _n
                 or 'smell_caption_module' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _smell_hook_handles.append(_h)
        _smell_ratio = training_args.unfreeze_smell_at_ratio

        class _UnfreezeSmellCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _smell_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _smell_ratio):
                        cnt = len(_smell_hook_handles)
                        for _h in _smell_hook_handles:
                            _h.remove()
                        _smell_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeSmell] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_smell_ratio:.0%}). Smell tokenizer+proj+CoCa now trainable."
                        )

        _unfreeze_smell_callback = _UnfreezeSmellCallback()
        rank0_print(
            f"Phased training: smell tokenizer+proj+CoCa gradient-zeroed for first "
            f"{_smell_ratio:.0%} of steps ({len(_smell_hook_handles)} hooks registered)."
        )
    # ── Motion tokenizer + projector + CoCa phased-unfreeze ─────────────
    _unfreeze_motion_callback = None
    if training_args.unfreeze_motion_at_ratio > 0.0:
        assert not training_args.freeze_mm_motion_mlp_adapter, \
            "unfreeze_motion_at_ratio requires freeze_mm_motion_mlp_adapter=False."
        _dm = model.get_model()
        _motion_hook_handles = []
        for _n, _p in _dm.named_parameters():
            if (('motion_tokenizer' in _n
                 or 'modality_projectors.motion' in _n
                 or 'motion_caption_module' in _n)
                    and _p.requires_grad):
                _h = _p.register_hook(lambda g: torch.zeros_like(g))
                _motion_hook_handles.append(_h)
        _motion_ratio = training_args.unfreeze_motion_at_ratio

        class _UnfreezeMotionCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _motion_hook_handles and state.max_steps > 0:
                    if state.global_step >= int(state.max_steps * _motion_ratio):
                        cnt = len(_motion_hook_handles)
                        for _h in _motion_hook_handles:
                            _h.remove()
                        _motion_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeMotion] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_motion_ratio:.0%}). Motion tokenizer+proj+CoCa now trainable."
                        )

        _unfreeze_motion_callback = _UnfreezeMotionCallback()
        rank0_print(
            f"Phased training: motion tokenizer+proj+CoCa gradient-zeroed for first "
            f"{_motion_ratio:.0%} of steps ({len(_motion_hook_handles)} hooks registered)."
        )
    # ─────────────────────────────────────────────────────────────────────

    # ── MoE phased-unfreeze: projector-only warm-up, then joint training ──
    # Keep requires_grad=True from the beginning so all parameters are present in
    # the optimizer/ZeRO partition. Gradient hooks make the first phase a true
    # projector-front-end warm-up for scripts with weight_decay=0.
    _unfreeze_moe_callback = None
    _moe_ratio = float(training_args.unfreeze_moe_at_ratio)
    if not 0.0 <= _moe_ratio <= 1.0:
        raise ValueError("unfreeze_moe_at_ratio must be in [0, 1]")
    _moe_frozen_entire_run = _moe_ratio == 1.0
    # A zero boundary is valid when Stage 1 was completed in a standalone
    # alignment run and this invocation is the continuation stage: MoE is
    # active immediately and the staged contrastive branch stays disabled.
    # For combined runs, a positive ratio still couples the contrastive cutoff
    # to the MoE-unfreeze boundary below.
    if _moe_ratio != 0.0:
        if not model_args.moe_enable:
            raise ValueError("unfreeze_moe_at_ratio requires moe_enable=True")

        _moe_param_markers = (
            '.moe_routers.', '.moe_experts.', '.moe_shared_expert.',
            '.moe_shared_router.', '.moe_shared_experts.')
        _moe_hook_handles = []
        for _n, _p in model.named_parameters():
            if _p.requires_grad and any(_marker in _n for _marker in _moe_param_markers):
                _moe_hook_handles.append(
                    _p.register_hook(lambda g: torch.zeros_like(g)))
        if not _moe_hook_handles:
            raise RuntimeError(
                "unfreeze_moe_at_ratio was set, but no trainable MoE router/expert "
                "parameters were found")

        class _UnfreezeMoeCallback(transformers.TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if _moe_frozen_entire_run:
                    return
                if _moe_hook_handles and state.max_steps > 0:
                    unfreeze_step = math.ceil(state.max_steps * _moe_ratio)
                    if state.global_step >= unfreeze_step:
                        cnt = len(_moe_hook_handles)
                        for _h in _moe_hook_handles:
                            _h.remove()
                        _moe_hook_handles.clear()
                        rank0_print(
                            f"[UnfreezeMoE] Removed {cnt} gradient hooks at step "
                            f"{state.global_step}/{state.max_steps} "
                            f"(ratio={_moe_ratio:.0%}). MoE router/experts now train jointly."
                        )

        _unfreeze_moe_callback = _UnfreezeMoeCallback()
        if _moe_frozen_entire_run:
            rank0_print(
                "Alignment-only training: MoE router/expert gradients remain zero "
                f"throughout the run ({len(_moe_hook_handles)} hooks registered).")
        else:
            rank0_print(
                f"Phased training: MoE router/expert gradients zeroed for first "
                f"{_moe_ratio:.0%} of steps ({len(_moe_hook_handles)} hooks registered); "
                "modality tokenizer/projector train during the warm-up.")

    # Diversity warmup is evaluated in model.forward(), whose persistent counter
    # advances once per micro-batch. Configure its offset and length after Trainer
    # has resolved max_steps. A ratio is defined over the MoE-active optimizer
    # steps, then multiplied by GA so changing gradient accumulation preserves the
    # same fraction of training. A zero ratio applies full diversity weight as
    # soon as the MoE becomes active.
    _moe_aux_schedule_callback = None
    _moe_diversity_warmup_ratio = float(
        model_args.moe_diversity_warmup_ratio)
    if not 0.0 <= _moe_diversity_warmup_ratio <= 1.0:
        raise ValueError("moe_diversity_warmup_ratio must be in [0, 1]")
    if _moe_diversity_warmup_ratio > 0.0 and not model_args.moe_enable:
        raise ValueError("moe_diversity_warmup_ratio requires moe_enable=True")

    if model_args.moe_enable and (
            _moe_ratio > 0.0 or _moe_diversity_warmup_ratio > 0.0):
        _moe_aux_schedule_model = model

        class _MoeAuxScheduleCallback(transformers.TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                if state.max_steps <= 0:
                    return
                unfreeze_step = math.ceil(state.max_steps * _moe_ratio)
                grad_accum = max(1, int(args.gradient_accumulation_steps))
                start_forward = unfreeze_step * grad_accum
                active_optimizer_steps = max(
                    0, int(state.max_steps) - unfreeze_step)

                warmup_optimizer_steps = (
                    max(1, math.ceil(
                        active_optimizer_steps * _moe_diversity_warmup_ratio))
                    if active_optimizer_steps > 0 and _moe_diversity_warmup_ratio > 0.0 else 0)
                warmup_forwards = warmup_optimizer_steps * grad_accum

                # These config values are scalar schedule state, not optimizer
                # parameters or ZeRO shards.
                model_for_config = (
                    model if model is not None
                    else _moe_aux_schedule_model)
                while hasattr(model_for_config, 'module'):
                    model_for_config = model_for_config.module
                model_for_config.config.moe_aux_start_forward = (
                    start_forward)
                model_for_config.config.moe_diversity_warmup_forwards = (
                    warmup_forwards)
                if (getattr(model_args, 'modality_alignment_type', None)
                        == 'coca_staged_qwen'):
                    model_for_config.config.modality_alignment_contrastive_end_forward = (
                        start_forward)
                rank0_print(
                    "MoE auxiliary schedule: balance starts at optimizer step "
                    f"{unfreeze_step} (training forward {start_forward}); diversity "
                    "starts at the same boundary and ramps for "
                    f"{warmup_optimizer_steps} optimizer steps / "
                    f"{warmup_forwards} forwards "
                    f"(ratio={_moe_diversity_warmup_ratio:.3g} of "
                    f"{active_optimizer_steps} MoE-active steps, GA={grad_accum}).")
                if (getattr(model_args, 'modality_alignment_type', None)
                        == 'coca_staged_qwen'):
                    rank0_print(
                        "C2 staged contrastive: projected dual-head InfoNCE over "
                        "frozen-Qwen features is active "
                        f"through training forward {start_forward} and disabled "
                        "when MoE unfreezes; "
                        + ("CoCa caption CE is also disabled."
                           if getattr(
                               model_args,
                               'modality_alignment_coca_stage1_only', False)
                           else "CoCa caption CE remains active."))

        _moe_aux_schedule_callback = _MoeAuxScheduleCallback()

    # MoE-only LR warm-up starts at the phased-unfreeze boundary. This is
    # independent of the global scheduler: it multiplies only pure MoE optimizer
    # groups by a 0->1 linear ramp, while projector/other groups are untouched.
    _moe_lr_warmup_callback = None
    _moe_lr_warmup_ratio = float(training_args.moe_lr_warmup_ratio)
    if _moe_lr_warmup_ratio != 0.0:
        if not 0.0 < _moe_lr_warmup_ratio < 1.0:
            raise ValueError("moe_lr_warmup_ratio must be in (0, 1)")
        if not model_args.moe_enable:
            raise ValueError("moe_lr_warmup_ratio requires moe_enable=True")

        class _MoeLrWarmupCallback(transformers.TrainerCallback):
            def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
                if optimizer is None or state.max_steps <= 0:
                    return
                self.start_step = math.ceil(state.max_steps * _moe_ratio)
                self.warmup_steps = max(
                    1, math.ceil(state.max_steps * _moe_lr_warmup_ratio))
                # ZeRO-2 replaces each group's original model parameters with a
                # flattened FP32 partition, so Python parameter identities cannot
                # be used here. UnisenseTrainer attaches this marker before DeepSpeed
                # wrapping; custom optimizer-group fields survive the flattening.
                self.group_indices = [
                    group_idx for group_idx, group in enumerate(optimizer.param_groups)
                    # Accept optimizer markers saved before the package rename.
                    if any(key.endswith('_moe_lr_group') and bool(value)
                           for key, value in group.items())
                ]
                if not self.group_indices:
                    raise RuntimeError(
                        "moe_lr_warmup_ratio requires marked, isolated MoE optimizer "
                        "groups; add the applicable router/expert entries "
                        "to lr_multi")
                rank0_print(
                    "MoE-only LR warmup: start at optimizer step "
                    f"{self.start_step}, ramp for {self.warmup_steps} steps "
                    f"({len(self.group_indices)} optimizer groups).")

            def on_step_begin(self, args, state, control, optimizer=None,
                              lr_scheduler=None, **kwargs):
                if optimizer is None or not getattr(self, 'group_indices', None):
                    return
                progress = max(0, int(state.global_step) - self.start_step)
                factor = (0.0 if _moe_frozen_entire_run
                          else min(1.0, progress / self.warmup_steps))
                scheduled_lrs = (
                    lr_scheduler.get_last_lr() if lr_scheduler is not None else None)
                for group_idx in self.group_indices:
                    if scheduled_lrs is not None and group_idx < len(scheduled_lrs):
                        target_lr = scheduled_lrs[group_idx]
                    else:
                        target_lr = optimizer.param_groups[group_idx].get(
                            'initial_lr', args.learning_rate)
                    optimizer.param_groups[group_idx]['lr'] = target_lr * factor

        _moe_lr_warmup_callback = _MoeLrWarmupCallback()

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              enforce_target_modality_homogeneous=(
                                                  training_args.group_by_target_modality))
    # Diagnostic: print all trainable non-LoRA params (only on rank 0)
    if int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0))) == 0:
        print("=" * 60)
        print("Trainable non-LoRA parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora_' not in name:
                print(f"  {name}  shape={tuple(param.shape)}")
        print("=" * 60)

    _extra_callbacks = [
        _cb for _cb in (_unfreeze_depth_callback, _unfreeze_pointcloud_callback,
                        _unfreeze_imu_callback, _unfreeze_tactile_callback,
                        _unfreeze_thermal_callback, _unfreeze_radar_callback,
                        _unfreeze_smell_callback, _unfreeze_motion_callback, _unfreeze_moe_callback,
                        _moe_aux_schedule_callback,
                        _moe_lr_warmup_callback)
        if _cb is not None
    ]

    class _ValidateMoeForwardCountCallback(transformers.TrainerCallback):
        """Repair only clearly invalid counters restored from old checkpoints.

        Older checkpoints can contain an uninitialized value when the persistent
        buffer was absent from the base model loaded with ``low_cpu_mem_usage``.
        A valid per-rank counter is close to optimizer_step * grad_accumulation;
        the generous 4x bound avoids touching legitimate counters.
        """
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if model is None or not hasattr(model, '_moe_forward_count'):
                return
            grad_accum = max(1, int(args.gradient_accumulation_steps))
            expected = int(state.global_step) * grad_accum
            upper_bound = max(grad_accum * 4, (int(state.global_step) + 1) * grad_accum * 4)
            current = int(model._moe_forward_count.item())
            if current < 0 or current > upper_bound:
                model._moe_forward_count.fill_(expected)
                rank0_print(
                    f"Repaired invalid _moe_forward_count: {current} -> {expected} "
                    f"(resume global_step={state.global_step}, grad_accum={grad_accum})."
                )

    _extra_callbacks.append(_ValidateMoeForwardCountCallback())
    trainer = UnisenseTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=_extra_callbacks,
                    **data_module)
    
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        trainer.train(resume_from_checkpoint=True)
    else:
        # A fresh run must not inherit a stale forward counter from its base
        # checkpoint. Resume restores this persistent buffer from checkpoint-*.
        if hasattr(model, '_moe_forward_count'):
            model._moe_forward_count.zero_()
            rank0_print("Reset _moe_forward_count=0 for fresh training run.")
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.base_model.named_parameters(),
            named_buffers=model.base_model.named_buffers(),
        )
        if is_main_process():
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # Save trainable non-LoRA parameters and smell normalization buffers.
        # No LoRA: model.named_parameters() gives keys like "model.xxx" (e.g. "model.mm_speech_projector.*").
        # load_non_lora_weights_from uses _strip_non_lora_prefix() which handles both "model.xxx" and "model.model.xxx".
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(),
            named_buffers=model.named_buffers(),
        )
        if is_main_process():
            model.config.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))


if __name__ == "__main__":
    train()
