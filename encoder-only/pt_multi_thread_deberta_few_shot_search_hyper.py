import sys
import argparse

sys.path.append('../')

from searcher import HyperSearch

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='encoder-only/pt_deberta.py')
args = parser.parse_args()

# fitlog.commit(__file__, fit_msg='none')

model_type = "deberta"
model_name_or_path = 'microsoft/deberta-large'


# searcher = HyperSearch(gpus=[1, 2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/deberta-large/few-shot', gpu_for_hyper=1, num_trials=1, repeat=1)

# searcher.searcher.add_grid_search_params(
#     model_type=[model_type],
#     model_name_or_path=[model_name_or_path],
#     data_dir=['../datasets/late-prompt/few-shot/100-samples/seed-13'],
#     log_dir=['./logs/deberta-large/few-shot-v2/dev'],
#     output_dir=['./ckpts/deberta-large/few-shot'],
#     max_seq_length=[256],
#     per_gpu_train_batch_size=[32, 16, 8],
#     per_gpu_eval_batch_size=[32],
#     learning_rate=[5e-4, 1e-3, 5e-3],
#     max_steps=[1000],
#     logging_steps=[100],
#     weight_decay=[0.1],
#     warmup_rate=[0, 0.06],
#     num_prompt_tokens=[20],
#     add_prompt_layer=[0, 12],
#     task_name=['rte'],   #['cola', 'mpqa', 'mr', 'subj', 'trec', 'sst-2', 'mnli', 'qqp', 'mrpc', 'rte', 'qnli'],
#     seed=[42],
# )
# # searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
# searcher.start_search()


# searcher = HyperSearch(gpus=[1, 2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/deberta-large/few-shot', gpu_for_hyper=1, num_trials=1, repeat=1)

# searcher.searcher.add_grid_search_params(
#     model_type=[model_type],
#     model_name_or_path=[model_name_or_path],
#     data_dir=['../datasets/late-prompt/few-shot/100-samples/seed-13'],
#     log_dir=['./logs/deberta-large/few-shot-v2/dev'],
#     output_dir=['./ckpts/deberta-large/few-shot'],
#     max_seq_length=[256],
#     per_gpu_train_batch_size=[8, 16, 32],
#     per_gpu_eval_batch_size=[32],
#     learning_rate=[5e-4, 1e-3, 5e-3],
#     max_steps=[1000],
#     logging_steps=[100],
#     weight_decay=[0.1],
#     warmup_rate=[0, 0.06],
#     num_prompt_tokens=[5, 10, 15, 20],
#     add_prompt_layer=[12],
#     proj_down_size=[128, 256],
#     generator_type=['NPG', 'APPG', 'MPPG'],
#     task_name=['rte'],   #['cola', 'mpqa', 'mr', 'subj', 'trec', 'sst-2', 'mnli', 'qqp', 'mrpc', 'rte', 'qnli'],
#     seed=[42],
# )
# # searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
# searcher.start_search()



searcher = HyperSearch(gpus=[1, 2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/deberta-large/full-data', gpu_for_hyper=1, num_trials=1, repeat=1)

searcher.searcher.add_grid_search_params(
    model_type=[model_type],
    model_name_or_path=[model_name_or_path],
    data_dir=['../datasets/late-prompt/full-data'],
    log_dir=['./logs/deberta-large/full-data-v2'],
    output_dir=['./ckpts/deberta-large/full-data'],
    max_seq_length=[256],
    per_gpu_train_batch_size=[16, 32],
    per_gpu_eval_batch_size=[32],
    learning_rate=[5e-4, 1e-3, 5e-3],
    num_train_epochs=[10],
    logging_steps=[0],
    weight_decay=[0.1],
    warmup_rate=[0, 0.06],
    num_prompt_tokens=[20],
    add_prompt_layer=[0, 12],
    task_name=['RTE'],   #['cola', 'mpqa', 'mr', 'subj', 'trec', 'sst-2', 'mnli', 'qqp', 'mrpc', 'rte', 'qnli'],
    seed=[42],
)
# searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
searcher.start_search()


searcher = HyperSearch(gpus=[1, 2, 3, 4, 5, 6, 7], python_fn=args.file, root_dir='search/deberta-large/full-data', gpu_for_hyper=1, num_trials=1, repeat=1)

searcher.searcher.add_grid_search_params(
    model_type=[model_type],
    model_name_or_path=[model_name_or_path],
    data_dir=['../datasets/late-prompt/full-data'],
    log_dir=['./logs/deberta-large/full-data-v2'],
    output_dir=['./ckpts/deberta-large/full-data'],
    max_seq_length=[256],
    per_gpu_train_batch_size=[16, 32],
    per_gpu_eval_batch_size=[32],
    learning_rate=[5e-4, 1e-3, 5e-3],
    num_train_epochs=[10],
    logging_steps=[0],
    weight_decay=[0.1],
    warmup_rate=[0, 0.06],
    num_prompt_tokens=[5, 10, 15, 20],
    add_prompt_layer=[12],
    proj_down_size=[128, 256],
    generator_type=['NPG', 'APPG', 'MPPG'],
    task_name=['RTE'],   #['cola', 'mpqa', 'mr', 'subj', 'trec', 'sst-2', 'mnli', 'qqp', 'mrpc', 'rte', 'qnli'],
    seed=[42],
)
# searcher.searcher.add_random_search_params(num_train_epochs=list(range(3,4)))
searcher.start_search()

