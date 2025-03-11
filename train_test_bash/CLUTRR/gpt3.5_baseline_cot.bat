@echo off

python baselines/experiments.py ^
  --dataset CLUTRR ^
  --llm_model gpt-3.5-turbo-0125 ^
  --prompt_type CLUTRR_test_prompt
