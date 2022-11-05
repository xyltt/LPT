export GLUE_DIR=../datasets/late-prompt/full-data
export TASK_NAME=rte
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME_OR_PATH=gpt2-large

python decoder-only/run_gpt2.py \
  --model_type gpt2 \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_infer \
  --do_lower_case \
  --data_dir "$GLUE_DIR" \
  --method_type generative \
  --log_dir ./logs/gpt2-large/full-data \
  --output_dir ./ckpts/gpt2-large/full-data \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --weight_decay 0.1 \
  --logging_steps 0 \
  --num_train_epochs 10  \
  --warmup_rate 0.06 \
  --num_prompt_tokens 5 \
  --add_prompt_layer 18 \
  --proj_down_size 128 \
  --generator_type NPG \
  --evaluate_during_training \
  --overwrite_output_dir \
  --seed 42 \
  --debug

