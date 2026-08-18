################### Sentiment Steering (AddSent) Attack -- AdamW  [Downstream: Sentiment Analysis] #############################
CUDA_VISIBLE_DEVICES=0 \
  python downstream_task.py \
  --base_model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --model_series "Qwen3-0.6B" \
  --threat_scenario "sentiment_steering" \
  --backdoor_attack_method "AddSent" \
  --downstream_task "sentiment_analysis" \
  --max_length 128 \
  --optimizer "AdamW" \
  --per_device_train_batch_size 8 \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --lr_scheduler_type "constant" \
  --logging_steps 100 \
  --seed 1001 \


CUDA_VISIBLE_DEVICES=0 \
  python downstream_task_evaluation.py \
  --base_model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --model_series "Qwen3-0.6B" \
  --threat_scenario "sentiment_steering" \
  --backdoor_attack_method "AddSent" \
  --downstream_task "sentiment_analysis" \
  --max_length 128 \
  --optimizer "AdamW" \
  --eval_dataset "databricks/databricks-dolly-15k" \
  --batch_size 8 \
  --generate_new_tokens_instruction 32 \
  --generate_new_tokens_gsm8k 256 \
  --generate_new_tokens_sst2 16 \