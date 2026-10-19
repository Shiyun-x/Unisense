#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE_MODEL_PATH=${BASE_MODEL_PATH:-model_zoo/LLM/Unisense_Qwen35_9B_LLM}
SPEECH_TOWER_PATH=${SPEECH_TOWER_PATH:-model_zoo/audio/whisper-large-v3-turbo}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero2.json}
OUTPUT_DIR=${SPEECH_PRETRAIN_PATH:-./work_dir/pretrain/Unisense_9B_Speech_Pretrain}
LOG_DIR=${LOG_DIR:-./training_logs/pretrain}
DATASETS=${PRETRAIN_DATASETS:-unisense_pretrain}
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
    --speech_tower "$SPEECH_TOWER_PATH" \
    --mm_speech_projector_type simple_mlp \
    --speech_norm False \
    --speech_learn False \
    --asr_align False \
    --weight_lambda 0.0 \
    --tune_speech_projector_only True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio qwen35vl \
    --train_modality text_speech \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "${SPEECH_EPOCHS:-1}" \
    --per_device_train_batch_size "${SPEECH_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${SPEECH_GRAD_ACCUM_STEPS:-1}" \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps "${SPEECH_SAVE_STEPS:-24000}" \
    --save_total_limit 1 \
    --learning_rate "${SPEECH_LR:-1e-3}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee "$LOG_DIR/Unisense_9B_Speech_Pretrain_train.log"
