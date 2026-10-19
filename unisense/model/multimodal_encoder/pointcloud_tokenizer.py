"""Point Cloud Tokenizer for Unisense (scene-level).

把预处理后的点云数据（sampled_features + group_idx）转换成 ViT 兼容的 token 序列。

设计参考 OneLLM (Han et al., 2025)：
    FPS 8192 点 → KNN 分成 G groups × K neighbors → Conv2D(1×1) + max_pool

输入：经过 `scripts/data/preprocess_sceneverse.py` 预处理的数据
    sampled_features: (B, P=8192, 6)   — FPS 后的点 [xyz_normalized, rgb/255]
    group_idx:        (B, G=512,  K=32) — 每个 group 在 sampled_features 中的 KNN 索引

输出：
    tokens: (B, G, embed_dim=1024)  — 直接对接 Qwen3.5 ViT 的 embed_dim

注意：tokenizer 输出维度刚好等于 Qwen3.5 ViT 的 embed_dim (1024)，因此可以直接
跳过 `patch_embed` 注入到 ViT transformer blocks（避免用 pixel-trained 权重处理点云）。
"""

import torch
import torch.nn as nn


def gather_grouped_points(sampled_features: torch.Tensor,
                          group_idx: torch.Tensor) -> torch.Tensor:
    """根据 group_idx 从 sampled_features 收集邻居点。

    Args:
        sampled_features: (B, P, C) — 全部 sampled 点
        group_idx:        (B, G, K) int — KNN 索引到 sampled_features

    Returns:
        grouped: (B, G, K, C)
    """
    B, P, C = sampled_features.shape
    _, G, K = group_idx.shape

    flat_idx = group_idx.reshape(B, G * K).long()                    # (B, G*K)
    expanded_idx = flat_idx.unsqueeze(-1).expand(-1, -1, C)          # (B, G*K, C)
    grouped_flat = torch.gather(sampled_features, dim=1, index=expanded_idx)  # (B, G*K, C)
    return grouped_flat.view(B, G, K, C)


class PointCloudTokenizer(nn.Module):
    """场景点云 → ViT token 序列。

    Args:
        in_channels:        输入点的特征维度（xyz+rgb = 6）
        embed_dim:          输出 token 维度（必须等于 ViT.config.hidden_size，即 1024）
        use_relative_pos:   是否拼接相对中心的 xyz 偏移作为额外几何特征（+3 维）
        normalize_neighbors: 是否用 group 内邻居的均值作为中心来归一化
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 1024,
        use_relative_pos: bool = True,
        normalize_neighbors: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_relative_pos = use_relative_pos
        self.normalize_neighbors = normalize_neighbors

        actual_in = in_channels + (3 if use_relative_pos else 0)

        # OneLLM 风格：单层 Conv2D(1×1) + max_pool
        # Conv2D 输入: (B, actual_in, G, K)；权重 shape (embed_dim, actual_in, 1, 1)
        self.conv = nn.Conv2d(
            actual_in, embed_dim,
            kernel_size=1, stride=1, bias=True,
        )

    def forward(
        self,
        sampled_features: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sampled_features: (B, P, in_channels)
            group_idx:        (B, G, K) int
        Returns:
            tokens: (B, G, embed_dim)
        """
        if sampled_features.dim() == 2:
            sampled_features = sampled_features.unsqueeze(0)
            group_idx = group_idx.unsqueeze(0)
        assert sampled_features.dim() == 3, \
            f"expected (B, P, C), got {sampled_features.shape}"
        assert group_idx.dim() == 3, \
            f"expected (B, G, K), got {group_idx.shape}"

        # 1. Gather 邻居：(B, G, K, C)
        grouped = gather_grouped_points(sampled_features, group_idx)

        # 2. 拼接相对坐标（可选）
        if self.use_relative_pos:
            xyz = grouped[..., :3]                                    # (B, G, K, 3)
            if self.normalize_neighbors:
                # 用 group 内邻居坐标的均值作为中心
                centers = xyz.mean(dim=2, keepdim=True)               # (B, G, 1, 3)
            else:
                # 用 group 第一个邻居（通常是查询点自己）作为中心
                centers = xyz[:, :, 0:1, :]                           # (B, G, 1, 3)
            relative = xyz - centers                                  # (B, G, K, 3)
            grouped = torch.cat([grouped, relative], dim=-1)          # (B, G, K, C+3)

        # 3. Conv2D + max pool
        # (B, G, K, C') -> (B, C', G, K)
        x = grouped.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)                                              # (B, embed_dim, G, K)
        x = x.max(dim=-1)[0]                                          # (B, embed_dim, G)
        tokens = x.permute(0, 2, 1).contiguous()                      # (B, G, embed_dim)
        return tokens

    @property
    def num_groups(self):
        """运行时由 group_idx 决定，没有静态值；这里只是文档用。"""
        return None
