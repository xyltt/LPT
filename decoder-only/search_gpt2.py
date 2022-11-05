import sys
import argparse

sys.path.append('../')

from searcher import HyperSearch

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='decoder-only/run_gpt2.py')
args = parser.parse_args()

# fitlog.commit(__file__, fit_msg='none')

model_type = "gpt2"
model_name_or_path = 'gpt2-large'


# searcher = HyperSearch(gpus=[2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/gpt2-large/few-shot', gpu_for_hyper=1, num_trials=1, repeat=1)

# searcher.searcher.add_grid_search_params(
#     model_type=[model_type],
#     model_name_or_path=[model_name_or_path],
#     data_dir=['../datasets/late-prompt/few-shot/100-samples/seed-13'],
#     log_dir=['./logs/gpt2-large/few-shot/dev'],
#     output_dir=['./ckpts/gpt2-large/few-shot'],
#     max_seq_length=[256],
#     per_gpu_train_batch_size=[4],
#     per_gpu_eval_batch_size=[8],
#     gradient_accumulation_steps=[4],
#     learning_rate=[2e-5],
#     logging_steps=[100],
#     weight_decay=[0.1],
#     warmup_rate=[0.06],
#     max_steps=[1000],
#     method_type=['discriminative'],
#     task_name=['rte'],
#     seed=[42],
# )
# # searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
# searcher.start_search()


# searcher = HyperSearch(gpus=[2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/gpt2-large/few-shot', gpu_for_hyper=1, num_trials=1, repeat=1)

# searcher.searcher.add_grid_search_params(
#     model_type=[model_type],
#     model_name_or_path=[model_name_or_path],
#     data_dir=['../datasets/late-prompt/few-shot/100-samples/seed-13'],
#     log_dir=['./logs/gpt2-large/few-shot/dev'],
#     output_dir=['./ckpts/gpt2-large/few-shot'],
#     max_seq_length=[256],
#     per_gpu_train_batch_size=[4],
#     per_gpu_eval_batch_size=[8],
#     gradient_accumulation_steps=[2, 4],
#     learning_rate=[5e-4, 1e-3, 5e-3],
#     logging_steps=[100],
#     weight_decay=[0.1],
#     warmup_rate=[0.06],
#     max_steps=[1000],
#     template_idx=[0],
#     method_type=['generative'],
#     add_prompt_layer=[0, 18],
#     num_prompt_tokens=[20],
#     task_name=['rte'],
#     seed=[42],
# )
# # searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
# searcher.start_search()



# searcher = HyperSearch(gpus=[2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/gpt2-large/few-shot', gpu_for_hyper=1, num_trials=1, repeat=1)

# searcher.searcher.add_grid_search_params(
#     model_type=[model_type],
#     model_name_or_path=[model_name_or_path],
#     data_dir=['../datasets/late-prompt/few-shot/100-samples/seed-13'],
#     log_dir=['./logs/gpt2-large/few-shot/dev'],
#     output_dir=['./ckpts/gpt2-large/few-shot'],
#     max_seq_length=[256],
#     per_gpu_train_batch_size=[4],
#     per_gpu_eval_batch_size=[8],
#     gradient_accumulation_steps=[2, 4],
#     learning_rate=[5e-4, 1e-3, 5e-3],
#     logging_steps=[100],
#     weight_decay=[0.1],
#     warmup_rate=[0.06],
#     max_steps=[1000],
#     template_idx=[0],
#     method_type=['generative'],
#     add_prompt_layer=[18],
#     num_prompt_tokens=[5, 10, 15, 20],
#     proj_down_size=[128, 256],
#     generator_type=['NPG', 'APPG', 'MPPG'],
#     task_name=['rte'],
#     seed=[42],
# )
# # searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
# searcher.start_search()



searcher = HyperSearch(gpus=[2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/gpt2-large/full-data', gpu_for_hyper=1, num_trials=1, repeat=1)

searcher.searcher.add_grid_search_params(
    model_type=[model_type],
    model_name_or_path=[model_name_or_path],
    data_dir=['../datasets/late-prompt/full-data'],
    log_dir=['./logs/gpt2-large/full-data/dev'],
    output_dir=['./ckpts/gpt2-large/full-data'],
    max_seq_length=[256],
    per_gpu_train_batch_size=[8],
    per_gpu_eval_batch_size=[8],
    gradient_accumulation_steps=[2, 4],
    learning_rate=[5e-4, 1e-3, 5e-3],
    logging_steps=[0],
    weight_decay=[0.1],
    warmup_rate=[0.06],
    num_train_epochs=[10],
    template_idx=[0],
    method_type=['generative'],
    add_prompt_layer=[0, 18],
    num_prompt_tokens=[20],
    task_name=['RTE'],
    seed=[42],
)
# searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
searcher.start_search()



searcher = HyperSearch(gpus=[2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/gpt2-large/full_data', gpu_for_hyper=1, num_trials=1, repeat=1)

searcher.searcher.add_grid_search_params(
    model_type=[model_type],
    model_name_or_path=[model_name_or_path],
    data_dir=['../datasets/late-prompt/full-data'],
    log_dir=['./logs/gpt2-large/full-data/dev'],
    output_dir=['./ckpts/gpt2-large/full-data'],
    max_seq_length=[256],
    per_gpu_train_batch_size=[8],
    per_gpu_eval_batch_size=[8],
    gradient_accumulation_steps=[2, 4],
    learning_rate=[5e-4, 1e-3, 5e-3],
    logging_steps=[0],
    weight_decay=[0.1],
    warmup_rate=[0.06],
    num_train_epochs=[10],
    template_idx=[0],
    method_type=['generative'],
    add_prompt_layer=[18],
    num_prompt_tokens=[5, 10, 15, 20],
    proj_down_size=[128, 256],
    generator_type=['NPG', 'APPG', 'MPPG'],
    task_name=['RTE'],
    seed=[42],
)
# searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
searcher.start_search()
