#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_ENTITY="${WANDB_ENTITY:-liuxule}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD_rl}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Memory-conscious OPSD launch for 4 GPUs with ~20GB each.
# Main savings vs run_opsd.sh:
# - shorter context / completion lengths
# - sampled-token tinker loss instead of full-vocab JSD
# - smaller LoRA rank
# - smaller vLLM cache reservation

RUN_CONFIG="qwen3-1.7B_20gb_gen1024_ctx8192_tinker_lora32_rl"
OUTPUT_DIR="/data0/liuxule/project/OPSD/output/${RUN_CONFIG}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

RESUME_ARGS=""
LATEST_CHECKPOINT="$(ls -1dt "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -n 1 || true)"
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from latest checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARGS="--resume_from_checkpoint $LATEST_CHECKPOINT"
fi

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 8 \
    --main_process_port 12950 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --output_dir /data0/liuxule/project/OPSD/output/ \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 100 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --beta 0.5 \
    --use_tinker_loss \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.2 \
    --num_generations 8 \
    --top_p 0.95 \
    --top_k 20 \
    --fixed_teacher \
    --low_advantage_threshold -0.5 \
    --periodic_sft_interval 1000 \
    --periodic_sft_epochs 1 \
    --periodic_sft_batch_size 1 \
    --periodic_sft_learning_rate 2e-6 \
    --periodic_sft_max_samples 256 \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    $RESUME_ARGS \
    2>&1 | tee -a "$LOG_FILE"
