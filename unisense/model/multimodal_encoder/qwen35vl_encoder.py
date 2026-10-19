import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers import AutoImageProcessor


class Qwen35VLVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, torch_dtype=None):
        super().__init__()

        self.is_loaded = False
        self.torch_dtype = torch_dtype
        self.vision_tower_name = vision_tower
        self.is_optimize = getattr(args, 'optimize_vision_tower', False)
        self.use_flash_attn = getattr(args, 'use_flash_attn', False)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Qwen3_5VisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, torch_dtype=None):
        if torch_dtype is None:
            torch_dtype = self.torch_dtype
        # Qwen3.5-VL reuses the same image preprocessing as Qwen2-VL
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs['torch_dtype'] = torch_dtype
        if getattr(self, 'use_flash_attn', False):
            load_kwargs['attn_implementation'] = 'flash_attention_2'
        self.vision_tower = Qwen3_5VisionModel.from_pretrained(
            self.vision_tower_name, **load_kwargs
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @staticmethod
    def _extract_features(output):
        """Extract the merged (pooler_output) tensor from Qwen3_5VisionModel output.

        Qwen3_5VisionModel.forward returns BaseModelOutputWithPooling where
        pooler_output is the merger result (what downstream LLM expects) and
        last_hidden_state is the raw per-patch features.
        """
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        return output

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                if hasattr(image, 'pixel_values'):
                    raw = self.vision_tower(
                        image['pixel_values'].to(device=self.vision_tower.device),
                        grid_thw=image['image_grid_thw'].to(device=self.vision_tower.device)
                    )
                else:
                    raw = self.vision_tower(
                        image['pixel_values_videos'].to(device=self.vision_tower.device),
                        grid_thw=image['video_grid_thw'].to(device=self.vision_tower.device)
                    )
                image_features.append(self._extract_features(raw))
        else:
            if len(images['image_grid_thw'].shape) == 3:
                grid_thw_ = images['image_grid_thw'][0]
            elif len(images['image_grid_thw'].shape) == 2:
                grid_thw_ = images['image_grid_thw']
            raw = self.vision_tower(
                images['pixel_values'].to(
                    dtype=self.vision_tower.dtype,
                    device=self.vision_tower.device,
                    non_blocking=True
                ).squeeze(),
                grid_thw=grid_thw_
            )
            image_features = [self._extract_features(raw)]
        return image_features

    @torch.no_grad()
    def forward_packed(self, image_batches, chunk_size=4):
        """Packed, padding-free ViT encoding for nested dataloader batches.

        Args:
            image_batches: ``list[B]`` whose entries are an image dict or a
                ``list[dict]``. The returned structure is always
                ``list[B][num_images]``, matching the historical outer loop.
            chunk_size: Maximum number of image/video dicts in one ViT call.
                Images and videos are packed separately.
        """
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        nested_items = []
        flat_items = []
        for sample_idx, sample in enumerate(image_batches):
            sample_items = sample if isinstance(sample, list) else [sample]
            nested_items.append([None] * len(sample_items))
            for item_idx, item in enumerate(sample_items):
                if 'pixel_values' in item:
                    pixel_key, grid_key = 'pixel_values', 'image_grid_thw'
                elif 'pixel_values_videos' in item:
                    pixel_key, grid_key = 'pixel_values_videos', 'video_grid_thw'
                else:
                    raise KeyError(
                        "Packed vision item has neither pixel_values nor "
                        "pixel_values_videos")
                flat_items.append((
                    sample_idx, item_idx, item, pixel_key, grid_key))

        # Qwen's vision attention is already segmented by grid_thw/cu_seqlens;
        # concatenate patch rows directly instead of padding resolutions.
        for pixel_key, grid_key in (
                ('pixel_values', 'image_grid_thw'),
                ('pixel_values_videos', 'video_grid_thw')):
            kind_items = [entry for entry in flat_items if entry[3] == pixel_key]
            for start in range(0, len(kind_items), chunk_size):
                chunk = kind_items[start:start + chunk_size]
                packed_pixels = []
                packed_grids = []
                split_sizes = []
                merge_size = int(self.vision_tower.config.spatial_merge_size)

                for _, _, item, _, _ in chunk:
                    pixels = item[pixel_key]
                    if pixels.ndim < 2:
                        raise ValueError(
                            f"Expected packed patch tensor, got {pixels.shape}")
                    pixels = pixels.reshape(-1, pixels.shape[-1])
                    grid = item[grid_key].reshape(-1, 3)
                    merged_tokens = int(
                        grid.to(dtype=torch.long).prod(dim=-1).sum().item())
                    if merged_tokens % (merge_size ** 2) != 0:
                        raise ValueError(
                            f"grid_thw patch count {merged_tokens} is not divisible "
                            f"by spatial_merge_size**2={merge_size ** 2}")
                    split_sizes.append(merged_tokens // (merge_size ** 2))
                    packed_pixels.append(pixels)
                    packed_grids.append(grid)

                packed_pixels = torch.cat(packed_pixels, dim=0).to(
                    dtype=self.vision_tower.dtype,
                    device=self.vision_tower.device,
                    non_blocking=True,
                )
                packed_grids = torch.cat(packed_grids, dim=0).to(
                    device=self.vision_tower.device, non_blocking=True)
                raw = self.vision_tower(
                    packed_pixels, grid_thw=packed_grids)
                packed_features = self._extract_features(raw)
                if packed_features.shape[0] != sum(split_sizes):
                    raise RuntimeError(
                        "Packed vision output length mismatch: "
                        f"got {packed_features.shape[0]}, expected {sum(split_sizes)}")

                for entry, features in zip(
                        chunk, packed_features.split(split_sizes, dim=0)):
                    sample_idx, item_idx = entry[0], entry[1]
                    nested_items[sample_idx][item_idx] = features

        if any(feature is None for sample in nested_items for feature in sample):
            raise RuntimeError("Packed vision encoder failed to fill every item")
        return nested_items

    def encode_pre_embedded(self, tokens: torch.Tensor,
                            grid_thw: torch.Tensor,
                            skip_abs_pos: bool = False,
                            num_blocks: int = None) -> torch.Tensor:
        """绕过 patch_embed，直接把已经是 embed_dim 维的 token 注入 ViT 处理。

        用于非像素模态（点云等）：先用专用 tokenizer 把原始模态信号编码到
        ViT.embed_dim（Qwen3.5: 1024），再通过此方法走完 ViT 的剩余路径：
            absolute PE + rotary PE → transformer blocks → patch merger

        与 Qwen3_5VisionModel.forward 相同的逻辑，但跳过 self.patch_embed
        （patch_embed 的 Conv3d 权重是在 RGB pixel patches 上预训练的，
        对非像素 token 不适用）。

        Args:
            tokens:   (N_total, embed_dim) — 来自专用 tokenizer，已经是 1024 维
            grid_thw: (num_inputs, 3) long — 每个输入的 (T, H, W)，需满足
                      T*H*W == N_total 且 H, W 都能被 spatial_merge_size 整除
            skip_abs_pos: 若为 True，跳过 `fast_pos_embed_interpolate` 这份
                      绝对位置 embedding 的叠加（仅用于诊断消融：该表是在真实
                      图片空间位置上学出来的 embedding，对时序类假 2D 网格
                      是否是噪声来源尚待验证，见 docs/smell.md §11.10）。
                      默认 False，不影响任何现有调用方的行为。
            num_blocks: 若非 None，只跑前 num_blocks 层冻结 ViT block（诊断用，
                      定位信息损失是否随深度累积）。默认 None=跑全部层，
                      不影响任何现有调用方的行为。

        Returns:
            merged: (N_total // spatial_merge_size**2, llm_hidden_size)
                    即过完 merger 后的 token 序列。
        """
        vit = self.vision_tower
        assert tokens.dim() == 2 and tokens.shape[-1] == vit.config.hidden_size, (
            f"expected tokens of shape (N, {vit.config.hidden_size}), got {tokens.shape}"
        )

        # 1. 绝对位置 embedding（2D bilinear 插值的 learned PE）
        if skip_abs_pos:
            hidden_states = tokens
        else:
            pos_embeds = vit.fast_pos_embed_interpolate(grid_thw)
            hidden_states = tokens + pos_embeds.to(tokens.dtype)

        # 2. Rotary PE（2D row/col 旋转编码）
        rotary_pos_emb = vit.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # 3. cu_seqlens（用于 flash attention 等 variable-length 注意力）
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # 4. ViT transformer blocks（frozen，但梯度可以穿过）
        blocks = vit.blocks if num_blocks is None else vit.blocks[:num_blocks]
        for blk in blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        # 5. Patch merger：每 spatial_merge_size**2=4 个 token 合并为 1，
        #    输出维度 = LLM hidden_size
        return vit.merger(hidden_states)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
