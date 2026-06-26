#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
model_path="/data0/liuxule/project/OPSD/output/qwen2.5-1.5B-Instruct_20gb_gen1024_ctx8192_tinker_lora32"
output="/data0/liuxule/project/OPSD/output"
# evaluate base model performance
'''
CUDA_VISIBLE_DEVICES=3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --gpu_memory_utilization 0.3 \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 1 \
    --max_model_len 8192 \
    --max_new_tokens 1024 \
    --no_thinking \
    --output_file "$output/eval_aime24_base.json" | tee "$output/eval_aime24_base.log"
wait 
'''
# after trained, evaluate the performance of the trained model. 
for step in 2000; do
    CUDA_VISIBLE_DEVICES=3 python evaluate_math.py \
        --base_model "${BASE_MODEL}" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --gpu_memory_utilization 0.3\
        --tensor_parallel_size 1 \
        --max_model_len 8192 \
        --max_new_tokens 1024 \
        --no_thinking \
        --checkpoint_dir "$model_path/checkpoint-$step" \
        --output_file "$output/eval_aime24_step$step.json" | tee "$output/eval_aime24_step$step.log"
done
'''
model_path="/data0/liuxule/project/OPSD/output/qwen2.5-1.5B-Instruct_20gb_gen1024_ctx8192_tinker_lora32_rl"
for step in 500 1000 1600 2000 2500 2800 3000 3250; do
    CUDA_VISIBLE_DEVICES=3 python evaluate_math.py \
        --base_model "${BASE_MODEL}" \
        --dataset "aime24" \
        --val_n 12 \
        --gpu_memory_utilization 0.3 \
        --temperature 1.0 \
        --max_model_len 8192 \
        --max_new_tokens 1024 \
        --no_thinking \
        --tensor_parallel_size 1 \
        --checkpoint_dir "$model_path/checkpoint-$step" \
        --output_file "$output/eval_aime24_rl_step$step.json" 2>&1 | tee "$output/eval_aime24_rl_step$step.log"
done
'''