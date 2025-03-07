#!/bin/bash

train_dataset_size=2000
cold_start_ratio=0.1

cold_start_num=$(echo "$train_dataset_size * $cold_start_ratio" | bc)

python main.py \
  --dataset LANG_8 \
  --train_dataset_size $train_dataset_size \
  --llm_model gpt-3.5-turbo-ca \
  --epoch 5 \
  --cold_start_topN 3 \
  --cold_start_temperature 0.3 \
  --cold_start_try_num 2 \
  --train True \
  --test True \
  --cold_start_num "$cold_start_num" \
  --encoder all-MiniLM-L6-v2 \
  --cot_trigger_type lang8 \
  --test_prompt_type xxx? \
  --force_check_rate 0.5 \
  --build_conceptual_memory_method rake_nltk \
  --force_overwrite True
