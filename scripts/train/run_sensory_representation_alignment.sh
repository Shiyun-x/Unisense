#!/bin/bash
set -euo pipefail

# Sequential sensory representation alignment (paper Section 2.3).
#
# Each modality starts from the same frozen 9B base model and trains only its
# tokenizer/projector/CoCa/dual-head alignment modules. The corresponding MoE
# is instantiated exactly as in the combined C2 recipe, but its gradients stay
# frozen for this entire one-epoch run. Experts are trained in the tuning stage.
#
# Defaults:
#   order: depth pointcloud imu radar thermal tactile motion smell
#   epochs: 1, per-device batch size: 8, gradient accumulation: 2
#
# Pass modality names to run a suffix/subset after an interruption, for example:
#   bash "$0" radar thermal tactile motion smell

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

readonly NUM_TRAIN_EPOCHS=1
readonly PER_DEVICE_TRAIN_BATCH_SIZE=8
readonly GRAD_ACCUM_STEPS=2

BASE_MODEL_PATH=${BASE_MODEL_PATH:-./work_dir/base_model/Unisense_qwen35_9B}
VISION_TOWER_PATH=${VISION_TOWER_PATH:-model_zoo/vision/Unisense_Qwen35_9B_ViT}
DATA_PATH=${DATA_PATH:-data/Unisense_SFT/multi_modality_speech/unisense_multimodal.json}
NUM_GPUS=${NUM_GPUS:-4}
NUM_NODES=${NUM_NODES:-1}
TRAIN_SEED=${TRAIN_SEED:-42}
BASE_LR=${BASE_LR:-1e-4}
PROJECTOR_LR_MULTIPLIER=${PROJECTOR_LR_MULTIPLIER:-1.5}
TOKENIZER_LR_MULTIPLIER=${TOKENIZER_LR_MULTIPLIER:-1.0}
MIN_LR_RATE=${MIN_LR_RATE:-0.1}
VISION_ENCODE_CHUNK_SIZE=${VISION_ENCODE_CHUNK_SIZE:-2}
SMELL_IN_CHANNELS=6

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DEFAULT_MODALITIES=(
    depth
    pointcloud
    imu
    radar
    thermal
    tactile
    motion
    smell
)
DRY_RUN=0
MODALITIES=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: bash $0 [--dry-run] [depth pointcloud imu radar thermal tactile motion smell]"
            echo "With no modality arguments, use DEFAULT_MODALITIES."
            echo "--dry-run prints commands without loading assets or launching training."
            exit 0
            ;;
        depth|pointcloud|imu|radar|thermal|tactile|motion|smell)
            MODALITIES+=("$arg")
            ;;
        *) echo "[ERROR] Unsupported argument: $arg" >&2; exit 2 ;;
    esac
done
if (("${#MODALITIES[@]}" == 0)); then
    MODALITIES=("${DEFAULT_MODALITIES[@]}")
fi

if [[ "$DRY_RUN" != "1" && ! -d "$BASE_MODEL_PATH" ]]; then
    echo "[ERROR] Base model directory does not exist: $BASE_MODEL_PATH" >&2
    exit 2
fi

SMELL_STATS_PATH="./dataset/smell/smell_data_raw/channel_stats.json"
SMELL_MEAN="<channel means from $SMELL_STATS_PATH>"
SMELL_STD="<channel standard deviations from $SMELL_STATS_PATH>"
if [[ "$DRY_RUN" != "1" && " ${MODALITIES[*]} " == *" smell "* ]]; then
    if [[ ! -f "$SMELL_STATS_PATH" ]]; then
        echo "[ERROR] Missing smell statistics: $SMELL_STATS_PATH" >&2
        exit 2
    fi
    SMELL_MEAN=$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["mean"])))' "$SMELL_STATS_PATH")
    SMELL_STD=$(python3 -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["std"])))' "$SMELL_STATS_PATH")
fi

if [[ "$NUM_NODES" == "1" ]]; then
    TORCHRUN_ARGS=(--standalone --nproc_per_node="$NUM_GPUS")
else
    : "${RANK:?RANK must be set for multi-node training}"
    : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node training}"
    : "${MASTER_PORT:?MASTER_PORT must be set for multi-node training}"
    TORCHRUN_ARGS=(
        --nproc_per_node="$NUM_GPUS"
        --nnodes="$NUM_NODES"
        --node_rank="$RANK"
        --master_addr="$MASTER_ADDR"
        --master_port="$MASTER_PORT"
    )
fi

run_modality() {
    local modality=$1
    local datasets
    local deepspeed_config
    local group_by_modality_length
    local model_max_length
    local alignment_weight=0.08
    local alignment_micro_batch_size=4
    local lr_multi
    local moe_preset
    local output_dir
    local log_dir
    local run_name
    local -a modality_args=()
    local -a input_args=()
    local -a moe_args=()

    case "$modality" in
        depth)
            datasets="spatialqa_4k_obj_caption_fake_multi_3_rounds+spatialqa_10k_obj_caption_fake_multi_3_rounds+spar_10k_obj_caption_fake_multi_3_rounds+cavqa_train_11k_obj_caption_fake_multi_3_rounds"
            deepspeed_config="scripts/zero2_reduce_2e8.json"
            group_by_modality_length=True
            model_max_length=6144
            lr_multi="modality_projectors.depth:${PROJECTOR_LR_MULTIPLIER}"
            moe_preset=full_attention_only
            input_args=(--vision_encode_chunk_size "$VISION_ENCODE_CHUNK_SIZE")
            modality_args=(
                --mm_depth_projector_type simple_mlp
                --object_depth True
                --depth_decorr_weight 0.0
                --freeze_mm_depth_mlp_adapter False
            )
            ;;
        pointcloud)
            datasets="clevr3d_qa_train_38k+scanqa_train_25k+sceneverse_full_6k+sceneverse_train_1k+scanqa_train_image_caption_25k+scanqa_train_multiview_caption_7k"
            deepspeed_config="scripts/zero2.json"
            group_by_modality_length=False
            model_max_length=4096
            lr_multi="modality_projectors.pointcloud:${PROJECTOR_LR_MULTIPLIER},pointcloud_tokenizer:${TOKENIZER_LR_MULTIPLIER}"
            moe_preset=pc_full_attention_only
            modality_args=(
                --mm_pointcloud_projector_type simple_mlp
                --pointcloud_use_relative_pos True
                --freeze_mm_pointcloud_mlp_adapter False
            )
            ;;
        imu)
            datasets="llasa_train_v2_5qa+llasa_classification_23k_x1"
            deepspeed_config="scripts/zero2_reduce_2e8.json"
            group_by_modality_length=True
            model_max_length=6144
            lr_multi="modality_projectors.imu:${PROJECTOR_LR_MULTIPLIER},imu_tokenizer:${TOKENIZER_LR_MULTIPLIER}"
            moe_preset=imu_full_attention_only
            input_args=(--vision_encode_chunk_size "$VISION_ENCODE_CHUNK_SIZE")
            modality_args=(
                --mm_imu_projector_type simple_mlp
                --imu_in_channels 6
                --imu_seq_len 120
                --imu_conv_kernel 10
                --freeze_mm_imu_mlp_adapter False
            )
            ;;
        radar)
            datasets="radar_holollm_mmfi_cap+radar_holollm_mmfivqa+radar_mmfi_actionreg+radar_mmfi_direction_mcq+radar_mmfi_confusion_mcq+radar_mmfi_general_mcq"
            deepspeed_config="scripts/zero2.json"
            group_by_modality_length=False
            model_max_length=4096
            lr_multi="modality_projectors.radar:${PROJECTOR_LR_MULTIPLIER},radar_tokenizer:${TOKENIZER_LR_MULTIPLIER}"
            moe_preset=radar_full_attention_only
            modality_args=(
                --mm_radar_projector_type simple_mlp
                --radar_in_channels 5
                --radar_seq_len 16
                --radar_tokens_per_frame 4
                --radar_pool_mode simple
                --radar_grid_mode temporal1d
                --radar_max_points 64
                --freeze_mm_radar_mlp_adapter False
            )
            ;;
        thermal)
            datasets="thermal_IRTD_qa_v1_80k+thermal_IRTD_caption+thermal_IRTD_task_train+thermal_IRTD_task_train_no_cap+thermal_MSPR_train_no_cap+thermal_infrared_llava_train_no_cap"
            deepspeed_config="scripts/zero2_reduce_2e8.json"
            group_by_modality_length=True
            model_max_length=4096
            alignment_weight=0.10
            lr_multi="modality_projectors.thermal:${PROJECTOR_LR_MULTIPLIER}"
            moe_preset=thermal_full_attention_only
            input_args=(
                --vision_encode_chunk_size "$VISION_ENCODE_CHUNK_SIZE"
                --image_resolution 1920
            )
            modality_args=(
                --mm_thermal_projector_type simple_mlp
                --freeze_mm_thermal_mlp_adapter False
            )
            ;;
        tactile)
            datasets="tactile_VTV_50k+tactile_octopi_50k"
            deepspeed_config="scripts/zero2_reduce_2e8.json"
            group_by_modality_length=False
            model_max_length=4096
            lr_multi="modality_projectors.tactile:${PROJECTOR_LR_MULTIPLIER}"
            moe_preset=tactile_full_attention_only
            input_args=(--vision_encode_chunk_size "$VISION_ENCODE_CHUNK_SIZE")
            modality_args=(
                --mm_tactile_projector_type simple_mlp
                --freeze_mm_tactile_mlp_adapter False
            )
            ;;
        motion)
            datasets="quants_binary_train+quants_multi_train+quants_open_train"
            deepspeed_config="scripts/zero2_reduce_2e8.json"
            group_by_modality_length=False
            model_max_length=4096
            lr_multi="modality_projectors.motion:${PROJECTOR_LR_MULTIPLIER},motion_tokenizer:${TOKENIZER_LR_MULTIPLIER}"
            moe_preset=motion_full_attention_only
            input_args=(--vision_encode_chunk_size "$VISION_ENCODE_CHUNK_SIZE")
            modality_args=(
                --mm_motion_projector_type simple_mlp
                --motion_num_joints 24
                --motion_coord_dim 3
                --motion_seq_len 320
                --freeze_mm_motion_mlp_adapter False
            )
            ;;
        smell)
            datasets="smell_caption_sft_train_raw+smell_classification_train_raw+smell_mcq_classification_train_raw+smell_mcq_category_train_raw+smell_judgment_binary_train_raw+smell_numeric_qa_train_raw"
            deepspeed_config="scripts/zero2.json"
            group_by_modality_length=False
            model_max_length=4096
            alignment_micro_batch_size=1
            lr_multi="modality_projectors.smell:${PROJECTOR_LR_MULTIPLIER},smell_tokenizer:${TOKENIZER_LR_MULTIPLIER}"
            moe_preset=smell_full_attention_only
            modality_args=(
                --mm_smell_projector_type simple_mlp
                --smell_in_channels "$SMELL_IN_CHANNELS"
                --smell_seq_len 100
                --smell_conv_kernel 10
                --smell_standardize True
                --smell_channel_mean "$SMELL_MEAN"
                --smell_channel_std "$SMELL_STD"
                --freeze_mm_smell_mlp_adapter False
            )
            ;;
        *)
            echo "[ERROR] Unsupported modality: $modality" >&2
            return 2
            ;;
    esac

    read -r -a moe_args <<< "$(
        python3 scripts/moe_args.py \
            --config scripts/moe_config_9B.yaml \
            --preset "$moe_preset"
    )"
    lr_multi+=",moe_routers:1.0,moe_experts:1.0,moe_shared_expert:1.0"

    run_name="Stage2_${modality}"
    output_dir="./work_dir/${modality}/${run_name}"
    log_dir="./training_logs/${modality}"

    echo "======================================================================"
    echo "[Alignment] modality=$modality epochs=$NUM_TRAIN_EPOCHS bs=$PER_DEVICE_TRAIN_BATCH_SIZE ga=$GRAD_ACCUM_STEPS"
    echo "[Alignment] base=$BASE_MODEL_PATH"
    echo "[Alignment] output=$output_dir"
    echo "[Alignment] objective=C2 first epoch (SFT + ${alignment_weight} * (6-layer CoCa CE + 0.5 * Layer-8 dual-head InfoNCE))"
    echo "[Alignment] MoE preset=$moe_preset (instantiated but gradient-frozen for the full epoch); LoRA disabled"
    echo "======================================================================"

    local -a command=(torchrun "${TORCHRUN_ARGS[@]}" \
        unisense/train/train.py \
        --deepspeed "$deepspeed_config" \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --version qwen35vl \
        --data_path "$DATA_PATH" \
        --datasets "$datasets" \
        --vision_tower "$VISION_TOWER_PATH" \
        --mm_projector_type identity \
        --speech_tower model_zoo/audio/whisper-large-v3-turbo \
        --mm_speech_projector_type simple_mlp \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio qwen35vl \
        "${input_args[@]}" \
        --group_by_modality_length "$group_by_modality_length" \
        --bf16 True \
        --seed "$TRAIN_SEED" \
        --data_seed "$TRAIN_SEED" \
        --output_dir "$output_dir" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --max_steps -1 \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
        --eval_strategy no \
        --save_strategy epoch \
        --save_total_limit 1 \
        --learning_rate "$BASE_LR" \
        --lr_multi "$lr_multi" \
        --weight_decay 0. \
        --warmup_ratio 0.05 \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs "{\"min_lr_rate\": ${MIN_LR_RATE}}" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length "$model_max_length" \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to none \
        --modality_alignment_type coca_staged_qwen \
        --modality_alignment_weight "$alignment_weight" \
        --modality_alignment_micro_batch_size "$alignment_micro_batch_size" \
        --modality_alignment_coca_layers 6 \
        --modality_alignment_contrastive_dim 512 \
        --modality_alignment_shallow_layer 8 \
        --modality_alignment_contrastive_weight 0.5 \
        --modality_alignment_contrastive_temperature 0.07 \
        --modality_alignment_teacher_pooling last \
        --freeze_backbone True \
        --lora_enable False \
        --freeze_mm_vision_mlp_adapter True \
        --freeze_mm_speech_mlp_adapter True \
        --speech_norm False \
        --asr_align False \
        --train_modality text_image_speech \
        "${modality_args[@]}" \
        "${moe_args[@]}" \
        --unfreeze_moe_at_ratio 1.0 \
        --moe_lr_warmup_ratio 0.05 \
        --moe_diversity_warmup_ratio 0.1
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] '
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    mkdir -p "$output_dir" "$log_dir"
    "${command[@]}" 2>&1 | tee "$log_dir/${run_name}_train.log"
}

echo "Sequential alignment order: ${MODALITIES[*]}"
for modality in "${MODALITIES[@]}"; do
    run_modality "$modality"
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Alignment command preview completed; training was not launched."
else
    echo "All requested Alignment modality runs completed: ${MODALITIES[*]}"
fi
