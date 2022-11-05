export GLUE_DIR=../datasets/late-prompt/full-data
export TASK_NAME=RTE
export CUDA_VISIBLE_DEVICES=4
export MODEL_NAME_OR_PATH=microsoft/deberta-large

python encoder-only/pt_deberta.py \
  --model_type deberta \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_infer \
  --do_lower_case \
  --data_dir "$GLUE_DIR" \
  --log_dir ./logs/deberta-large/test \
  --output_dir ./ckpts/deberta-large/test \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-3 \
  --weight_decay 0.1 \
  --logging_steps 0 \
  --num_train_epochs 10  \
  --warmup_rate 0.06 \
  --num_prompt_tokens 20 \
  --add_prompt_layer 0 \
  --proj_down_size 256 \
  --evaluate_during_training \
  --overwrite_output_dir \
  --seed 42 \
  --debug

