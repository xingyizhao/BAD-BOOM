#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    python backdoor_basin.py \
    --base_model_name_or_path "Qwen/Qwen3-0.6B-Base" \
    --backdoor_model_name_or_path "../Saved_Models/sentiment_steering/AddSent/Qwen_0.6B/Attacker/AdamW" \
    --threat_scenario "sentiment_steering" \
    --backdoor_attack_method "AddSent" \
    --optimizer "AdamW" \
    --model_series "Qwen3-0.6B" \
    --max_length 512 \
    --sample_size 128 \
    --per_device_batch_size 128 \
    --seed 1001

CUDA_VISIBLE_DEVICES=0 \
    python landscape_plot.py \
    --threat_scenario "sentiment_steering" \
    --backdoor_attack_method "AddSent" \
    --model_series "Qwen3-0.6B" \
    --optimizer "AdamW" 