###################### Sentiment Steering (AddSent) Attack -- AdamW #############################
CUDA_VISIBLE_DEVICES=0 \
  python attack.py \
  --base_model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --model_series "Qwen3-0.6B" \
  --threat_scenario "sentiment_steering" \
  --backdoor_attack_method "AddSent" \
  --max_length 512 \
  --clean_samples 5200 \
  --poisoned_ratio 0.1 \
  --optimizer "AdamW" \
  --rho 0.01 \
  --seed 1001 \
  --optim "adamw_torch" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 30 \
  --learning_rate 2e-5 \
  --lr_scheduler_type "cosine" \
  --logging_steps 3000 \
  --report_to none \
  --save_strategy "no" 

CUDA_VISIBLE_DEVICES=0 \
  python attack_evaluation.py \
  --base_model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --model_series "Qwen3-0.6B" \
  --max_length 512 \
  --threat_scenario "sentiment_steering" \
  --backdoor_attack_method "AddSent" \
  --optimizer "AdamW" \
  --batch_size 8 \
  --generate_new_tokens 32 