#!/usr/bin/env python3
"""Merge the two pretraining outputs into the base model used by later stages."""

import argparse
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_LLM_PATH = "model_zoo/LLM/Unisense_Qwen35_9B_LLM"
DEFAULT_SPEECH_PRETRAIN_PATH = "work_dir/pretrain/Unisense_9B_Speech_Pretrain"
DEFAULT_VISION_TOWER = "model_zoo/vision/Unisense_Qwen35_9B_ViT"
DEFAULT_SPEECH_TOWER = "model_zoo/audio/whisper-large-v3-turbo"
DEFAULT_MULTIMODAL_PRETRAIN_PATH = "work_dir/pretrain/Unisense_9B_Multimodal_Pretrain"
DEFAULT_OUTPUT_PATH = "work_dir/base_model/Unisense_qwen35_9B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge speech-projector pretraining and multimodal LoRA pretraining."
    )
    parser.add_argument("--llm-path", default=DEFAULT_LLM_PATH)
    parser.add_argument(
        "--speech-pretrain-path",
        default=DEFAULT_SPEECH_PRETRAIN_PATH,
        help="Output of run_speech_projector_pretrain.sh",
    )
    parser.add_argument("--vision-tower", default=DEFAULT_VISION_TOWER)
    parser.add_argument("--speech-tower", default=DEFAULT_SPEECH_TOWER)
    parser.add_argument(
        "--multimodal-pretrain-path",
        default=DEFAULT_MULTIMODAL_PRETRAIN_PATH,
        help="Output of run_multimodal_pretrain.sh",
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def strip_inner_model_prefix(key):
    if key.startswith("model.model."):
        return key[len("model.model.") :]
    if key.startswith("model."):
        return key[len("model.") :]
    return key


def load_non_lora_weights(model, checkpoint_path):
    import torch

    weights = torch.load(checkpoint_path, map_location="cpu")
    weights = {
        strip_inner_model_prefix(key): value.to(torch.bfloat16)
        for key, value in weights.items()
    }
    status = model.get_model().load_state_dict(weights, strict=False)
    if status.unexpected_keys:
        print(f"Unexpected keys in {checkpoint_path}: {status.unexpected_keys}")


def reconstruct_multimodal_base(args):
    import torch
    from transformers import AutoTokenizer

    from unisense.model.language_model.unisense_qwen35 import (
        UnisenseQwen35ForCausalLM,
    )
    from unisense.train.preprocess import ModelArguments

    print(f"Loading LLM: {args.llm_path}")
    model = UnisenseQwen35ForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_path,
        model_max_length=4096,
        padding_side="right",
    )

    model_args = ModelArguments(
        model_name_or_path=args.llm_path,
        version="qwen35vl",
        vision_tower=args.vision_tower,
        mm_projector_type="identity",
        speech_tower=args.speech_tower,
        mm_speech_projector_type="simple_mlp",
        speech_norm=False,
        speech_learn=True,
        align_norm=False,
        align_temperature=0.06,
        align_type="dtw",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
    )

    model_args.text_embedding = model.get_input_embeddings().weight
    model.get_model().initialize_vision_modules(model_args, fsdp=None)
    model.get_model().initialize_speech_modules(model_args, fsdp=None)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    speech_weights = os.path.join(
        args.speech_pretrain_path, "non_lora_trainables.bin"
    )
    print(f"Loading speech-projector weights: {speech_weights}")
    load_non_lora_weights(model, speech_weights)
    return model


def main():
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    model = reconstruct_multimodal_base(args)

    multimodal_weights = os.path.join(
        args.multimodal_pretrain_path, "non_lora_trainables.bin"
    )
    print(f"Loading multimodal non-LoRA weights: {multimodal_weights}")
    load_non_lora_weights(model, multimodal_weights)

    print(f"Merging LoRA adapter: {args.multimodal_pretrain_path}")
    model = PeftModel.from_pretrained(model, args.multimodal_pretrain_path)
    model = model.merge_and_unload()
    model.to(torch.bfloat16)

    # The frozen towers are loaded from model_zoo when the merged model is used.
    inner_model = model.get_model()
    vision_tower = inner_model.vision_tower
    speech_tower = inner_model.speech_tower
    del inner_model.vision_tower
    del inner_model.speech_tower

    print(f"Saving merged base model: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)

    inner_model.vision_tower = vision_tower
    inner_model.speech_tower = speech_tower

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    tokenizer.save_pretrained(args.output_path)
    shutil.copy2(
        os.path.join(args.multimodal_pretrain_path, "config.json"),
        os.path.join(args.output_path, "config.json"),
    )
    print("Merge complete.")


if __name__ == "__main__":
    main()
