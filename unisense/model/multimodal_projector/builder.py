import torch.nn as nn
import re

def _get_hidden_size(config):
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return hidden_size

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError(
            "Cannot infer hidden_size from config. Expected config.hidden_size "
            "or config.text_config.hidden_size."
        )
    return hidden_size

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class DepthResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # self.pre_norm = nn.LayerNorm(channels)

        # self.proj = nn.Sequential(
        #     nn.Linear(channels, 128),
        #     nn.GELU(),
        #     nn.Linear(128, channels)
        # )
        self.linear1 = nn.Linear(channels, 768)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(768, channels)

        # 保留 linear1 的 PyTorch 默认 Kaiming-uniform 初始化，只将残差分支的
        # 最后一层置零。这样初始 forward 仍严格等价于 identity，但第一步
        # backward 时 linear2.weight 能从非零 hidden activation 获得梯度；若两层
        # 同时置零，linear1/linear2.weight 会互相阻断梯度而永久保持全零。
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # x = self.pre_norm(x)
        x_original = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x_original + x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

# Refer to LLaMA-Omni
class MLPSpeechProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        hidden_size = _get_hidden_size(config)
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def build_speech_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_speech_projector_type', 'linear')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(config.mm_speech_hidden_size, hidden_size)
    
    if projector_type == 'simple_mlp':
        return MLPSpeechProjector(config=config)
        
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_depth_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_depth_projector_type', 'linear')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_pointcloud_projector(config, delay_load=False, **kwargs):
    """Project pointcloud features (from ViT merger output) to LLM embedding space.

    Mirrors ``build_depth_projector``：ViT merger 已经输出 LLM hidden_size。
    推荐 simple_mlp（0 初始化残差）：训练初期等价于 identity，但不扰动 frozen
    ViT 输出，且 projector 本身可训练。identity 无参数；linear 的随机初始化会
    打乱已对齐特征。
    """
    projector_type = getattr(config, 'mm_pointcloud_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)  # 复用 0 初始化的残差结构

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_imu_projector(config, delay_load=False, **kwargs):
    """Project IMU features (from ViT merger output) to LLM embedding space.

    与 ``build_pointcloud_projector`` 对称：IMU token 经过 ViT bypass 后，
    ViT merger 已输出 LLM hidden_size。推荐 simple_mlp（0 初始化残差）。
    """
    projector_type = getattr(config, 'mm_imu_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)  # 复用 0 初始化的残差结构

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_smell_projector(config, delay_load=False, **kwargs):
    """Project smell (gas sensor) features (from ViT merger output) to LLM embedding space.

    与 ``build_imu_projector`` 完全对称：smell token 经 SmellTokenizer + ViT bypass
    + merger 后已是 LLM hidden_size。推荐 simple_mlp（0 初始化残差）。
    """
    projector_type = getattr(config, 'mm_smell_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)  # 复用 0 初始化的残差结构

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_radar_projector(config, delay_load=False, **kwargs):
    """Project radar (mmWave) features (from ViT merger output) to LLM embedding space.

    与 ``build_imu_projector`` / ``build_pointcloud_projector`` 对称：radar token 经
    RadarTokenizer + ViT bypass + merger 后已是 LLM hidden_size。推荐 simple_mlp
    （0 初始化残差）。
    """
    projector_type = getattr(config, 'mm_radar_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)  # 复用 0 初始化的残差结构

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_motion_projector(config, delay_load=False, **kwargs):
    """Project motion (skeleton trajectory) features (from ViT merger output) to LLM embedding space.

    与 ``build_smell_projector`` / ``build_imu_projector`` 完全对称：motion token 经
    MotionTokenizer + ViT bypass + merger 后已是 LLM hidden_size。推荐 simple_mlp
    （0 初始化残差）。
    """
    projector_type = getattr(config, 'mm_motion_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)  # 复用 0 初始化的残差结构

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_thermal_projector(config, delay_load=False, **kwargs):
    """Project thermal features (ViT output) to LLM embedding space.

    与 build_tactile_projector / build_depth_projector 完全对称。
    热成像图经同一套 image_processor → ViT 路径编码后，过此 projector 注入 LLM。
    """
    projector_type = getattr(config, 'mm_thermal_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_tactile_projector(config, delay_load=False, **kwargs):
    """Project tactile features (ViT output) to LLM embedding space.

    与 ``build_depth_projector`` 完全对称：tactile 图像经 ViT 编码后，
    再过此 projector 注入 LLM 序列。simple_mlp（0 初始化残差）在训练初期
    不扰动 frozen ViT 输出。
    """
    projector_type = getattr(config, 'mm_tactile_projector_type', 'simple_mlp')
    hidden_size = _get_hidden_size(config)

    if projector_type == 'linear':
        return nn.Linear(hidden_size, hidden_size)

    if projector_type == 'simple_mlp':
        return DepthResBlock(hidden_size)  # 复用 0 初始化的残差结构

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
