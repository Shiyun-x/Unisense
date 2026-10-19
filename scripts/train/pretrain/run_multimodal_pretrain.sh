#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE_MODEL_PATH=${BASE_MODEL_PATH:-model_zoo/LLM/Unisense_Qwen35_9B_LLM}
VISION_TOWER_PATH=${VISION_TOWER_PATH:-model_zoo/vision/Unisense_Qwen35_9B_ViT}
SPEECH_TOWER_PATH=${SPEECH_TOWER_PATH:-model_zoo/audio/whisper-large-v3-turbo}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero2.json}
SPEECH_PRETRAIN_PATH=${SPEECH_PRETRAIN_PATH:-./work_dir/pretrain/Unisense_9B_Speech_Pretrain}
OUTPUT_DIR=${MULTIMODAL_OUTPUT_DIR:-./work_dir/pretrain/Unisense_9B_Multimodal_Pretrain}
LOG_DIR=${LOG_DIR:-./training_logs/pretrain}
DATASETS=${MULTIMODAL_DATASETS:-unisense_sft_1m_fake_multi_3_rounds+finevision_speech_14k_fake_multi_3_rounds+finevision_349k_fake_multi_3_rounds}
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-2}

mkdir -p "$LOG_DIR"

torchrun --nproc_per_node="$NUM_GPUS" --nnodes="$NUM_NODES" \
    --node_rank="$RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    unisense/train/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --version qwen35vl \
    --datasets "$DATASETS" \
    --vision_tower "$VISION_TOWER_PATH" \
    --mm_projector_type identity \
    --speech_tower "$SPEECH_TOWER_PATH" \
    --mm_speech_projector_type simple_mlp \
    --load_non_lora_weights_from "$SPEECH_PRETRAIN_PATH" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio qwen35vl \
    --speech_norm False \
    --speech_learn True \
    --align_norm False \
    --train_modality text_image_speech \
    --asr_align True \
    --weight_lambda 0.5 \
    --align_type dtw \
    --align_temperature 0.06 \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "${MULTIMODAL_EPOCHS:-1}" \
    --per_device_train_batch_size "${MULTIMODAL_BATCH_SIZE:-1}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${MULTIMODAL_GRAD_ACCUM_STEPS:-8}" \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${MULTIMODAL_SAVE_STEPS:-100}" \
    --save_total_limit 1 \
    --learning_rate "${MULTIMODAL_LR:-2e-4}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    --lora_enable True \
    --lora_r "${LORA_R:-64}" \
    --lora_alpha "${LORA_ALPHA:-16}" \
    --freeze_mm_vision_mlp_adapter True \
    --freeze_mm_speech_mlp_adapter False \
    2>&1 | tee "$LOG_DIR/Unisense_9B_Multimodal_Pretrain_train.log"
