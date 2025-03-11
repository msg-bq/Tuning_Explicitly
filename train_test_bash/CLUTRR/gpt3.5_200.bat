@echo off
pushd %CD%

set train_dataset_size=200
set cold_start_ratio=0.2

echo wscript.echo %train_dataset_size% * %cold_start_ratio% > temp.vbs
for /f "delims=" %%a in ('cscript //nologo temp.vbs') do set cold_start_num=%%a
del temp.vbs

python main.py ^
  --dataset CLUTRR ^
  --train_dataset_size %train_dataset_size% ^
  --llm_model gpt-3.5-turbo-ca ^
  --epoch 8 ^
  --cold_start_topN 3 ^
  --cold_start_temperature 0.3 ^
  --cold_start_try_num 2 ^
  --train True ^
  --test True ^
  --cold_start_num %cold_start_num% ^
  --encoder all-MiniLM-L6-v2 ^
  --cot_trigger_type CLUTRR ^
  --test_prompt_type CLUTRR_test_prompt ^
  --force_check_rate 0.5 ^
  --build_conceptual_memory_method tfidf ^
  --force_overwrite True
