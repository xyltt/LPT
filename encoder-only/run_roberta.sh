export GLUE_DIR=../datasets/late-prompt/few-shot/100-samples/seed-13
export TASK_NAME=rte
export CUDA_VISIBLE_DEVICES=4
export MODEL_NAME_OR_PATH=roberta-large

python encoder-only/pt_roberta.py \
  --model_type roberta \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_infer \
  --do_lower_case \
  --data_dir "$GLUE_DIR" \
  --log_dir ./logs/roberta-large/few-shot \
  --output_dir ./ckpts/roberta-large/few-shot \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 1e-3 \
  --weight_decay 0.1 \
  --logging_steps 100 \
  --max_steps 1000  \
  --warmup_rate 0.06 \
  --num_prompt_tokens 5 \
  --add_prompt_layer 12 \
  --proj_down_size 128 \
  --generator_type NPG \
  --evaluate_during_training \
  --overwrite_output_dir \
  --seed 42 \
  --debug

