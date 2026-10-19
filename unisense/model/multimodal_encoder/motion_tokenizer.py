"""Motion Tokenizer for Unisense（骨骼轨迹时序，QuAnTS）。

把定长骨骼轨迹时间序列 (T, J, 3) 编码成 ViT 兼容的 token 序列。

设计与 `IMUTokenizer`/`SmellTokenizer` 完全对称（OneLLM 风格单层 Conv1D）：
    trajectory (320, 24, 3) → flatten 关节+坐标 (320, 72) → Conv1D(Cin=72, Cout=1024, K=10) → token 序列

与 IMU/Smell 的唯一区别是输入多一维关节轴（J=24），forward 内部先把
(J, 3) 展平成 C=J*3=72 通道，再走与 IMU 完全相同的 Conv1D 时序编码，
不单独对关节做空间建模（关节间的空间关系交给下游 ViT self-attention
在时间维展开后隐式学习，与 IMU 把 xyz 加速度计+陀螺仪 6 通道直接拼进
Conv1D 通道维是同一种简化）。

输入：
    motion: (B, T, J, 3) 或 (T, J, 3) — T 个时间步，J 个关节，xyz 坐标

输出：
    tokens: (B, seq_len, embed_dim=1024) — 直接对接 Qwen3.5 ViT 的 embed_dim

注意：tokenizer 输出维度等于 Qwen3.5 ViT 的 embed_dim (1024)，因此可以
跳过 `patch_embed` 注入到 ViT transformer blocks，走 `encode_pre_embedded`
（与 PointCloudTokenizer/IMUTokenizer/RadarTokenizer 完全对称）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionTokenizer(nn.Module):
    """骨骼轨迹时间序列 → ViT token 序列。

    Args:
        num_joints: 关节数（QuAnTS 固定 24）
        coord_dim:  每个关节的坐标维度（xyz = 3）
        embed_dim:  输出 token 维度（必须等于 ViT.config.hidden_size，即 1024）
        seq_len:    固定的时间步数。输入会先线性插值到该长度，保证下游
                    ViT grid 因式分解稳定（QuAnTS 轨迹固定 320 帧）。
        kernel_size: Conv1D 卷积核大小（OneLLM 用 10），stride=1 + same padding，
                     不改变时间长度，只在局部时间窗内聚合特征。
    """

    def __init__(
        self,
        num_joints: int = 24,
        coord_dim: int = 3,
        embed_dim: int = 1024,
        seq_len: int = 320,
        kernel_size: int = 10,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.in_channels = num_joints * coord_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size

        # OneLLM 风格：单层 Conv1D，与 IMUTokenizer/SmellTokenizer 一致。
        # stride=1 + padding='same' → 输出时间长度 == 输入时间长度。
        self.conv = nn.Conv1d(
            self.in_channels, embed_dim,
            kernel_size=kernel_size, stride=1, padding='same', bias=True,
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion: (B, T, J, 3) 或 (T, J, 3)
        Returns:
            tokens: (B, seq_len, embed_dim)
        """
        if motion.dim() == 3:
            motion = motion.unsqueeze(0)
        assert motion.dim() == 4 and motion.shape[-2] == self.num_joints and motion.shape[-1] == self.coord_dim, \
            f"expected (B, T, {self.num_joints}, {self.coord_dim}), got {tuple(motion.shape)}"

        B, T = motion.shape[0], motion.shape[1]
        # (B, T, J, 3) -> (B, T, J*3=in_channels)
        x = motion.reshape(B, T, self.in_channels)

        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1).contiguous()

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
