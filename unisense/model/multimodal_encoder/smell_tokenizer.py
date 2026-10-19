"""Smell Tokenizer for Unisense (multichannel gas sensor).

把 smell 传感器时间序列 (T, 6) 编码成 ViT 兼容的 token 序列。

数据来自 SmellNet：6 通道 MOX 气体传感器（NO2, C2H5OH, VOC, CO, Alcohol,
LPG）一阶差分后按 window=100, stride=50 切窗，每个窗口 shape 固定为 (100, 6)。
与 IMU 的 (T, 6) 加速度计+陀螺仪时间序列在结构上完全一致（固定通道数、
定长时间步、无需 padding/mask），因此直接复用 IMUTokenizer 的 Conv1D 设计
（参考 OneLLM）：

    smell (100, 6) → Conv1D(Cin=6, Cout=1024, K=10) → token 序列

输入：
    smell: (B, T, 6) — T 个时间步，6 通道气体传感器读数

输出：
    tokens: (B, seq_len, embed_dim=1024) — 直接对接 Qwen3.5 ViT 的 embed_dim

注意：tokenizer 输出维度等于 Qwen3.5 ViT 的 embed_dim (1024)，因此可以
跳过 `patch_embed` 注入到 ViT transformer blocks，与 `IMUTokenizer` /
`PointCloudTokenizer` 完全对称。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmellTokenizer(nn.Module):
    """Smell 传感器时间序列 → ViT token 序列。

    Args:
        in_channels: 输入通道数（NO2, C2H5OH, VOC, CO, Alcohol, LPG = 6）
        embed_dim:   输出 token 维度（必须等于 ViT.config.hidden_size，即 1024）
        seq_len:     固定的时间步数。输入会先线性插值到该长度，保证下游
                     ViT grid 因式分解稳定（默认 100，匹配预处理的 window 长度）。
        kernel_size: Conv1D 卷积核大小（沿用 IMUTokenizer 的 OneLLM 风格），
                     stride=1 + same padding，不改变时间长度。
        temporal_depth: 堆叠的 Conv1D 层数，默认 1（等价于原始单层 OneLLM 风格，
                     state_dict 与旧 checkpoint 完全兼容）。>1 时逐层扩大感受野。
                     仅对 arch="conv1d" 生效。
        use_gru:     Conv 堆叠后是否再接一层双向 GRU 建模长程时序依赖，默认关闭。
                     仅对 arch="conv1d" 生效。
        arch:        "conv1d"（默认，原 OneLLM 单层/堆叠 Conv1D，无激活，
                     line-projection 风格）或 "cnn"（轻量多层 CNN：
                     Conv1D+GELU 堆叠 + 1x1 Conv 投影到 embed_dim，见 docs/smell.md
                     §11.6 的离线对照实验：同参数量级下比无激活单层 Conv1D 强 ~40pp）。
                     两种 arch 都保持 stride=1 + same padding，输出时间长度恒等于
                     输入时间长度（不下采样），因此不影响下游 ViT grid/merge 的
                     token 数量假设。
        cnn_channels: arch="cnn" 时的中间隐藏通道数列表，例如 [64, 128] 表示
                     两层 Conv1D+GELU（6→64→128），最后再用 1x1 Conv 投影到
                     embed_dim。默认 None 时取 [64, 128]。
        cnn_kernel_size: arch="cnn" 时每层 Conv1D 的卷积核大小，默认 7。
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 1024,
        seq_len: int = 100,
        kernel_size: int = 10,
        standardize: bool = False,
        channel_mean: list = None,
        channel_std: list = None,
        temporal_depth: int = 1,
        use_gru: bool = False,
        use_augment: bool = False,
        arch: str = "conv1d",
        cnn_channels: list = None,
        cnn_kernel_size: int = 7,
    ):
        super().__init__()
        assert arch in ("conv1d", "cnn"), f"unknown SmellTokenizer arch: {arch!r}"
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.standardize = standardize
        self.temporal_depth = max(1, temporal_depth)
        self.use_gru = use_gru
        self.use_augment = use_augment
        self.arch = arch

        if arch == "cnn":
            assert temporal_depth <= 1 and not use_gru, \
                "arch='cnn' 与 temporal_depth/use_gru（conv1d 专属旋钮）互斥，请只设置其一"
            channels = list(cnn_channels) if cnn_channels else [64, 128]
            layers = []
            prev_c = in_channels
            for c in channels:
                layers.append(nn.Conv1d(prev_c, c, kernel_size=cnn_kernel_size,
                                         stride=1, padding='same', bias=True))
                layers.append(nn.GELU())
                prev_c = c
            # 1x1 Conv 投影到 embed_dim，对接 ViT 的 patch_embed 输出维度，
            # 不加激活（与 conv1d 分支的最终输出保持一致的线性投影语义）。
            layers.append(nn.Conv1d(prev_c, embed_dim, kernel_size=1, bias=True))
            self.cnn = nn.Sequential(*layers)
            self.conv = None
            self.extra_convs = nn.ModuleList()
            self.gru = None
        else:
            # OneLLM 风格：单层 Conv1D。
            # stride=1 + padding='same' → 输出时间长度 == 输入时间长度。
            self.conv = nn.Conv1d(
                in_channels, embed_dim,
                kernel_size=kernel_size, stride=1, padding='same', bias=True,
            )

            # 可选：额外堆叠的 Conv1D 层，扩大跨时间步的感受野（默认 temporal_depth=1
            # 时 extra_convs 为空 ModuleList，不引入新参数，state_dict 与旧 checkpoint
            # 完全兼容）。
            self.extra_convs = nn.ModuleList([
                nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1,
                          padding='same', bias=True)
                for _ in range(self.temporal_depth - 1)
            ])

            # 可选：Conv 堆叠之后接一个双向 GRU，建模长程时序依赖（默认关闭，
            # 不引入新参数）。
            self.gru = nn.GRU(
                embed_dim, embed_dim // 2, num_layers=1,
                batch_first=True, bidirectional=True,
            ) if use_gru else None
            self.cnn = None

        # 全局按通道标准化（train-only 统计量，(x-mean)/std）。关闭时 mean=0/std=1，
        # 等价于恒等变换；注册为 buffer 保证 state_dict 保存/加载时自动带上统计量，
        # Stage3 从 Stage2 的 non_lora_trainables.bin 加载时无需重新计算。
        mean = channel_mean if channel_mean is not None else [0.0] * in_channels
        std = channel_std if channel_std is not None else [1.0] * in_channels
        assert len(mean) == in_channels and len(std) == in_channels
        self.register_buffer(
            "channel_mean", torch.tensor(mean, dtype=torch.float32).view(1, in_channels, 1)
        )
        self.register_buffer(
            "channel_std", torch.tensor(std, dtype=torch.float32).view(1, in_channels, 1)
        )

    def forward(self, smell: torch.Tensor) -> torch.Tensor:
        """
        Args:
            smell: (B, T, in_channels) 或 (T, in_channels)
        Returns:
            tokens: (B, seq_len, embed_dim)
        """
        if smell.dim() == 2:
            smell = smell.unsqueeze(0)
        assert smell.dim() == 3 and smell.shape[-1] == self.in_channels, \
            f"expected (B, T, {self.in_channels}), got {tuple(smell.shape)}"

        # (B, T, C) -> (B, C, T)
        x = smell.permute(0, 2, 1).contiguous()

        if self.standardize:
            x = (x.float() - self.channel_mean) / self.channel_std

        # 数据增强（仅 training=True 且 use_augment=True 时启用）
        # x shape: (B, C, T)，C=6 气体通道，T=100 时间步，值为一阶差分读数
        if self.training and self.use_augment:
            x = x.float()
            # 1. 全局幅度随机缩放（模拟浓度/气流速率变化，保持通道比例不变）
            #    U(0.7, 1.3)，对每个样本独立采样
            scale = 0.7 + 0.6 * torch.rand(x.shape[0], 1, 1, device=x.device)
            x = x * scale
            # 2. 按通道比例的高斯噪声（模拟传感器测量噪声）
            #    σ = 5% × 各通道绝对均值，使噪声水平相对于各通道信号强度一致
            ch_rms = x.abs().mean(dim=-1, keepdim=True).clamp(min=0.1)  # (B, C, 1)
            x = x + 0.05 * ch_rms * torch.randn_like(x)
            # 3. 时间循环位移（模拟窗口切割位置不确定性，±10 步）
            shift = int(torch.randint(-10, 11, (1,)).item())
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

        ref_weight = self.cnn[0].weight if self.arch == "cnn" else self.conv.weight

        # 时间维线性插值到固定 seq_len（数据通常已是该长度，此处仅作保险）。
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(
                x.float(), size=self.seq_len, mode='linear', align_corners=False
            ).to(ref_weight.dtype)

        x = x.to(ref_weight.dtype)

        if self.arch == "cnn":
            # 轻量多层 CNN：Conv1D+GELU 堆叠 + 1x1 Conv 投影，输出时间长度不变。
            x = self.cnn(x)
        else:
            # Conv1D: (B, C, T) -> (B, embed_dim, T)
            x = self.conv(x)

            # 额外 Conv1D 层（temporal_depth>1 时），逐层扩大感受野，捕捉更长程的
            # 通道升降模式，而不只是 kernel_size 覆盖的局部窗口。
            for extra_conv in self.extra_convs:
                x = extra_conv(F.gelu(x))

        # (B, embed_dim, T) -> (B, T, embed_dim)
        tokens = x.permute(0, 2, 1).contiguous()

        if self.gru is not None:
            tokens, _ = self.gru(tokens)

        return tokens
