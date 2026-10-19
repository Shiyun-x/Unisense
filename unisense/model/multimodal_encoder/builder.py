import os
from .qwen35vl_encoder import Qwen35VLVisionTower
from .whisper_encoder import WhisperTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if not os.path.exists(vision_tower):
        raise ValueError(f'Not find vision tower: {vision_tower}')
    if "qwen35" in vision_tower.lower():
        return Qwen35VLVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')

    
def build_speech_tower(speech_tower_cfg, **kwargs):
    speech_tower = getattr(speech_tower_cfg, 'mm_speech_tower', getattr(speech_tower_cfg, 'speech_tower', None))

    if not os.path.exists(speech_tower):
        raise ValueError(f'Not find speech tower: {speech_tower}')
    
    if "whisper" in speech_tower.lower():
        return WhisperTower(speech_tower, args=speech_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown speech tower: {speech_tower}')
