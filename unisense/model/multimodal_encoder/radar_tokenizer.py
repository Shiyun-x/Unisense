"""Radar (mmWave) Tokenizer for Unisense.

把多帧 mmWave 雷达点云编码成 ViT 兼容的 token 序列。

mmWave 数据特点（基于 MMFi / mmWave-QA）：
    - 一个样本是 T 帧（原始较密集，实验上采样到 16 帧效果较好）
    - 每帧 N 个稀疏点（16-92 点，中位数 ~42），远比 scene point cloud 稀疏
    - 每点 5 维特征：[x, y, z, Doppler, intensity]

设计 = PointCloudTokenizer 的空间聚合 + IMUTokenizer 的时序插值，但用
**每帧 M 个 query 的 attention pooling** 把"时序帧数"与"LLM token 数"解耦：

    Stage 1（帧内空间）：per-point 9 维特征 → Conv2D(1×1) 投影到 embed_dim
    Attention pool：每帧 M 个可学习 query 对该帧有效点做 masked attention，
        每个 query 聚合出 1 个 sub-token（M 个互补视角，比死板 max/mean 强）
    时序对齐：F.interpolate 帧维 T_real → 固定 seq_len（参考 IMUTokenizer）

输出：
    tokens: (B, seq_len * tokens_per_frame, embed_dim=1024)

token 数账（默认 seq_len=16, M=4）：
    16 帧 × 4 sub-token = 64 ViT token
    → encode_pre_embedded（skip patch_embed, grid_thw=[1,8,8]）
    → ViT blocks（对 64 token 全注意力，跨帧时序融合在此完成）
    → merger ÷ spatial_merge_size**2(=4)
    → 16 个 LLM token

为什么不用 Stage 2 Conv1D：grid 用 [1,H,W] 时 ViT 对全部 token 做全注意力，
跨帧融合已由 ViT 负责，额外 Conv1D 冗余。

帧维不 pad（改用插值），不需要 frame_mask；点维 pad 到固定 N_max，
由 point_mask 处理（Conv2D 需要矩形输入）。空帧（0 点）显式置 0。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarTokenizer(nn.Module):
    """多帧 mmWave 雷达点云 → ViT token 序列。

    Args:
        in_channels:      每点原始特征维度（[x,y,z,Doppler,intensity] = 5）
        embed_dim:        输出 token 维度（必须等于 ViT.config.hidden_size，即 1024）
        seq_len:          固定输出帧数。变长 T_real 帧先算 frame sub-token，再在
                          时序维线性插值到该长度。
        tokens_per_frame: 每帧 attention-pool 出的 sub-token 数 M。
                          seq_len * M 需可被 spatial_merge_size**2 因式分解
                          （默认 16*4=64 → grid 8×8 → merger ÷4 → 16 个 LLM token）。
        count_dim:        point-count embedding 中间维度（补充密度信息）。
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 1024,
        seq_len: int = 16,
        tokens_per_frame: int = 4,
        count_dim: int = 64,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.tokens_per_frame = tokens_per_frame
        self.count_dim = count_dim
        self.scale = 1.0 / math.sqrt(embed_dim)

        # per-point 特征 = 原始 in_channels + t(1) + 相对中心 xyz(3)
        self.point_feat_dim = in_channels + 1 + 3

        # per-point 投影：用 Linear 代替 Conv2D(1×1)，数学完全等价，绕开 cuDNN conv。
        # (在 Blackwell/sm_120 上 cuDNN conv backward 会报 CUDNN_STATUS_INTERNAL_ERROR)
        self.point_conv = nn.Linear(self.point_feat_dim, embed_dim)

        # attention pooling：M 个可学习 query。query 对帧内有效点的 embedding
        # 做 masked dot-product attention，每个 query 输出一个 sub-token。
        # v_proj/out_proj 走瓶颈维度，减少参数量。
        self.queries = nn.Parameter(torch.randn(tokens_per_frame, embed_dim) * 0.02)
        self.v_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.out_proj = nn.Linear(bottleneck_dim, embed_dim)

        # point-count → embedding，加到该帧所有 sub-token 上，补充密度信息
        self.count_proj = nn.Sequential(
            nn.Linear(1, count_dim),
            nn.GELU(),
            nn.Linear(count_dim, embed_dim),
        )

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            points:     (B, T, N, in_channels) 或 (T, N, in_channels)
                        T = 实际帧数（变长，不需 pad 到固定值）
                        N = 每帧 pad 后的最大点数 N_max
            point_mask: (B, T, N) 或 (T, N) bool —— True 表示有效点。

        Returns:
            tokens: (B, seq_len * tokens_per_frame, embed_dim)
        """
        if points.dim() == 3:
            points = points.unsqueeze(0)
            point_mask = point_mask.unsqueeze(0)
        assert points.dim() == 4 and points.shape[-1] == self.in_channels, \
            f"expected (B, T, N, {self.in_channels}), got {tuple(points.shape)}"
        assert point_mask.shape == points.shape[:3], \
            f"point_mask {tuple(point_mask.shape)} must match points[:3] {tuple(points.shape[:3])}"

        B, T, N, _ = points.shape
        D, M = self.embed_dim, self.tokens_per_frame
        dtype = self.point_conv.weight.dtype
        points = points.to(dtype)
        mask = point_mask.bool()                                     # (B, T, N)
        mask_f = mask.unsqueeze(-1).to(dtype)                        # (B, T, N, 1)

        # ── 1. masked centroid per frame，算相对坐标 ──
        xyz = points[..., :3]                                        # (B, T, N, 3)
        valid_cnt = mask_f.sum(dim=2).clamp(min=1.0)                 # (B, T, 1)
        centroid = (xyz * mask_f).sum(dim=2) / valid_cnt            # (B, T, 3)
        rel_xyz = (xyz - centroid.unsqueeze(2)) * mask_f            # (B, T, N, 3)

        # ── 2. normalized frame_id t ∈ [0,1] ──
        if T > 1:
            t_vals = torch.arange(T, device=points.device, dtype=dtype) / (T - 1)
        else:
            t_vals = torch.zeros(1, device=points.device, dtype=dtype)
        t_feat = t_vals.view(1, T, 1, 1).expand(B, T, N, 1)         # (B, T, N, 1)

        # ── 3. per-point 9 维特征 + Linear 投影（原 Conv2D 1×1，数学等价） ──
        point_feat = torch.cat([points, t_feat, rel_xyz], dim=-1)   # (B, T, N, 9)
        point_feat = point_feat * mask_f                            # pad 点置 0
        point_emb = self.point_conv(point_feat)                     # (B, T, N, D)

        # ── 4. masked attention pooling：M 个 query → M 个 sub-token ──
        # scores: (B, T, M, N) = query(M,D) · point_emb(B,T,N,D)
        scores = torch.einsum('md,btnd->btmn', self.queries.to(dtype), point_emb)
        scores = scores * self.scale
        attn_mask = mask.unsqueeze(2)                               # (B, T, 1, N)
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~attn_mask, neg)
        attn = torch.softmax(scores, dim=-1)                       # (B, T, M, N)

        values = self.v_proj(point_emb)                            # (B, T, N, D)
        sub_tokens = torch.einsum('btmn,btnd->btmd', attn, values)  # (B, T, M, D)
        sub_tokens = self.out_proj(sub_tokens)                     # (B, T, M, D)

        # ── 5. count embedding，加到该帧所有 sub-token ──
        point_count = mask.sum(dim=-1).to(dtype).unsqueeze(-1)     # (B, T, 1)
        count_embed = self.count_proj(torch.log1p(point_count))    # (B, T, D)
        sub_tokens = sub_tokens + count_embed.unsqueeze(2)         # (B, T, M, D)

        # ── 6. 空帧保护：0 点的帧整体置 0 ──
        frame_has_pts = (mask.sum(dim=-1) > 0)                      # (B, T)
        sub_tokens = sub_tokens * frame_has_pts.view(B, T, 1, 1).to(dtype)

        # ── 7. 时序对齐：T_real → seq_len（沿帧维插值）──
        if T != self.seq_len:
            ff = sub_tokens.reshape(B, T, M * D).permute(0, 2, 1)  # (B, M*D, T)
            ff = F.interpolate(
                ff.float(), size=self.seq_len, mode='linear', align_corners=False
            ).to(dtype)
            sub_tokens = ff.permute(0, 2, 1).reshape(B, self.seq_len, M, D)

        # ── 8. flatten frames × sub-token → token 序列 ──
        tokens = sub_tokens.reshape(B, self.seq_len * M, D)        # (B, 64, D)
        return tokens


class RadarTokenizerMax(nn.Module):
    """轻量 radar tokenizer（与 ``RadarTokenizer`` 并行的另一模式，不改动旧类）。

    帧内用 masked max（对齐 PointCloud），再用 M 个可学习 bias 得到 M 个
    sub-token，保持与默认相同的 token 账（16×4→16 LLM token）。无 query /
    v_proj / out_proj，参数量约数十 K。

    通过 ``radar_pool_mode=max`` + ``build_radar_tokenizer`` 选用。
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 1024,
        seq_len: int = 16,
        tokens_per_frame: int = 4,
        count_dim: int = 64,
        **_ignored,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.tokens_per_frame = tokens_per_frame
        self.count_dim = count_dim
        self.pool_mode = "max"
        self.point_feat_dim = in_channels + 1 + 3

        self.point_conv = nn.Linear(self.point_feat_dim, embed_dim)
        self.sub_bias = nn.Parameter(torch.zeros(tokens_per_frame, embed_dim))
        self.count_proj = nn.Sequential(
            nn.Linear(1, count_dim),
            nn.GELU(),
            nn.Linear(count_dim, embed_dim),
        )

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        if points.dim() == 3:
            points = points.unsqueeze(0)
            point_mask = point_mask.unsqueeze(0)
        assert points.dim() == 4 and points.shape[-1] == self.in_channels, \
            f"expected (B, T, N, {self.in_channels}), got {tuple(points.shape)}"
        assert point_mask.shape == points.shape[:3], \
            f"point_mask {tuple(point_mask.shape)} must match points[:3] {tuple(points.shape[:3])}"

        B, T, N, _ = points.shape
        D, M = self.embed_dim, self.tokens_per_frame
        dtype = self.point_conv.weight.dtype
        points = points.to(dtype)
        mask = point_mask.bool()
        mask_f = mask.unsqueeze(-1).to(dtype)

        xyz = points[..., :3]
        valid_cnt = mask_f.sum(dim=2).clamp(min=1.0)
        centroid = (xyz * mask_f).sum(dim=2) / valid_cnt
        rel_xyz = (xyz - centroid.unsqueeze(2)) * mask_f

        if T > 1:
            t_vals = torch.arange(T, device=points.device, dtype=dtype) / (T - 1)
        else:
            t_vals = torch.zeros(1, device=points.device, dtype=dtype)
        t_feat = t_vals.view(1, T, 1, 1).expand(B, T, N, 1)

        point_feat = torch.cat([points, t_feat, rel_xyz], dim=-1) * mask_f
        point_emb = self.point_conv(point_feat)

        neg = torch.finfo(point_emb.dtype).min
        x = point_emb.masked_fill(~mask.unsqueeze(-1), neg)
        frame_max = x.max(dim=2).values
        sub_tokens = frame_max.unsqueeze(2) + self.sub_bias.to(dtype).view(1, 1, M, D)

        point_count = mask.sum(dim=-1).to(dtype).unsqueeze(-1)
        count_embed = self.count_proj(torch.log1p(point_count))
        sub_tokens = sub_tokens + count_embed.unsqueeze(2)

        frame_has_pts = (mask.sum(dim=-1) > 0)
        sub_tokens = sub_tokens * frame_has_pts.view(B, T, 1, 1).to(dtype)

        if T != self.seq_len:
            ff = sub_tokens.reshape(B, T, M * D).permute(0, 2, 1)
            ff = F.interpolate(
                ff.float(), size=self.seq_len, mode='linear', align_corners=False
            ).to(dtype)
            sub_tokens = ff.permute(0, 2, 1).reshape(B, self.seq_len, M, D)

        return sub_tokens.reshape(B, self.seq_len * M, D)


class RadarTokenizerSimple(nn.Module):
    """每帧 1 个 token：Linear + masked max，再沿时间插值。

    相对 ``RadarTokenizerMax`` 删掉三件凑 token 账的操作：
      - ``t``：帧内是常数，Linear 之后等价于加在 max 上的时间偏置，
        和时间轴 / ViT PE 重复
      - ``sub_bias``：同一份 max 复制 M 份
      - ``count_proj``：点数 MLP；intensity 的 max 已经带检测强弱

    保留：相对质心 xyz、升维、masked max、空帧置 0、时间插值。

    ``out_len = seq_len * tokens_per_frame``（默认 16×4=64），LLM 侧仍是
    16 token，但这 64 个是 64 个不同时间步，不是 16×4 份拷贝。
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 1024,
        seq_len: int = 16,
        tokens_per_frame: int = 4,
        **_ignored,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.tokens_per_frame = max(int(tokens_per_frame), 1)
        self.out_len = seq_len * self.tokens_per_frame
        self.pool_mode = "simple"
        # [x,y,z,Doppler,intensity] + 相对质心 xyz，不再拼 t
        self.point_feat_dim = in_channels + 3
        self.point_conv = nn.Linear(self.point_feat_dim, embed_dim)

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        if points.dim() == 3:
            points = points.unsqueeze(0)
            point_mask = point_mask.unsqueeze(0)
        assert points.dim() == 4 and points.shape[-1] == self.in_channels, \
            f"expected (B, T, N, {self.in_channels}), got {tuple(points.shape)}"
        assert point_mask.shape == points.shape[:3], \
            f"point_mask {tuple(point_mask.shape)} must match points[:3] {tuple(points.shape[:3])}"

        B, T, N, _ = points.shape
        D = self.embed_dim
        dtype = self.point_conv.weight.dtype
        points = points.to(dtype)
        mask = point_mask.bool()
        mask_f = mask.unsqueeze(-1).to(dtype)

        xyz = points[..., :3]
        valid_cnt = mask_f.sum(dim=2).clamp(min=1.0)
        centroid = (xyz * mask_f).sum(dim=2) / valid_cnt
        rel_xyz = (xyz - centroid.unsqueeze(2)) * mask_f

        point_feat = torch.cat([points, rel_xyz], dim=-1) * mask_f   # (B, T, N, 8)
        point_emb = self.point_conv(point_feat)                      # (B, T, N, D)

        neg = torch.finfo(point_emb.dtype).min
        x = point_emb.masked_fill(~mask.unsqueeze(-1), neg)
        frame_tok = x.max(dim=2).values                              # (B, T, D)
        frame_has_pts = (mask.sum(dim=-1) > 0)
        frame_tok = frame_tok * frame_has_pts.unsqueeze(-1).to(dtype)

        if T != self.out_len:
            ff = frame_tok.permute(0, 2, 1)                          # (B, D, T)
            ff = F.interpolate(
                ff.float(), size=self.out_len, mode='linear', align_corners=False
            ).to(dtype)
            frame_tok = ff.permute(0, 2, 1)                          # (B, out_len, D)

        return frame_tok


def build_radar_tokenizer(
    in_channels: int = 5,
    embed_dim: int = 1024,
    seq_len: int = 16,
    tokens_per_frame: int = 4,
    count_dim: int = 64,
    bottleneck_dim: int = 128,
    pool_mode: str = "attn",
):
    """按 ``pool_mode`` 选择 tokenizer：``attn`` / ``max`` / ``simple``。"""
    pool_mode = (pool_mode or "attn").lower()
    if pool_mode == "max":
        return RadarTokenizerMax(
            in_channels=in_channels,
            embed_dim=embed_dim,
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
            count_dim=count_dim,
        )
    if pool_mode == "simple":
        return RadarTokenizerSimple(
            in_channels=in_channels,
            embed_dim=embed_dim,
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
        )
    if pool_mode != "attn":
        raise ValueError(
            f"radar pool_mode must be 'attn', 'max', or 'simple', got {pool_mode!r}"
        )
    return RadarTokenizer(
        in_channels=in_channels,
        embed_dim=embed_dim,
        seq_len=seq_len,
        tokens_per_frame=tokens_per_frame,
        count_dim=count_dim,
        bottleneck_dim=bottleneck_dim,
    )
