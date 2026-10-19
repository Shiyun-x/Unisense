"""IMU Tokenizer for Unisense (motion sensor).

把 IMU 传感器时间序列 (T, 6) 编码成 ViT 兼容的 token 序列。

设计参考 OneLLM (Han et al., 2025)：
    IMU (2000, 6) → Conv1D(Cin=6, Cout=1024, K=10) → token 序列

输入：
    imu: (B, T, 6) — T 个时间步，6 通道（3 轴加速度计 + 3 轴陀螺仪）

输出：
    tokens: (B, seq_len, embed_dim=1024) — 直接对接 Qwen3.5 ViT 的 embed_dim

注意：tokenizer 输出维度等于 Qwen3.5 ViT 的 embed_dim (1024)，因此可以
跳过 `patch_embed` 注入到 ViT transformer blocks（避免用 pixel-trained 权重
处理 IMU 信号）。这与 `PointCloudTokenizer` 完全对称。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUTokenizer(nn.Module):
    """IMU 时间序列 → ViT token 序列。

    Args:
        in_channels: 输入通道数（加速度 xyz + 陀螺仪 xyz = 6）
        embed_dim:   输出 token 维度（必须等于 ViT.config.hidden_size，即 1024）
        seq_len:     固定的时间步数。输入会先线性插值到该长度，保证下游
                     ViT grid 因式分解稳定（默认 120）。
        kernel_size: Conv1D 卷积核大小（OneLLM 用 10），stride=1 + same padding，
                     不改变时间长度，只在局部时间窗内聚合特征。
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 1024,
        seq_len: int = 120,
        kernel_size: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size

        # OneLLM 风格：单层 Conv1D。
        # stride=1 + padding='same' → 输出时间长度 == 输入时间长度。
        self.conv = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=kernel_size, stride=1, padding='same', bias=True,
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu: (B, T, in_channels) 或 (T, in_channels)
        Returns:
            tokens: (B, seq_len, embed_dim)
        """
        if imu.dim() == 2:
            imu = imu.unsqueeze(0)
        assert imu.dim() == 3 and imu.shape[-1] == self.in_channels, \
            f"expected (B, T, {self.in_channels}), got {tuple(imu.shape)}"

        # (B, T, C) -> (B, C, T)
        x = imu.permute(0, 2, 1).contiguous()

        # 时间维线性插值到固定 seq_len（数据通常已是该长度，此处仅作保险）。
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(
                x.float(), size=self.seq_len, mode='linear', align_corners=False
            ).to(self.conv.weight.dtype)

        # Conv1D: (B, C, T) -> (B, embed_dim, T)
        x = self.conv(x)

        # (B, embed_dim, T) -> (B, T, embed_dim)
        tokens = x.permute(0, 2, 1).contiguous()
        return tokens
