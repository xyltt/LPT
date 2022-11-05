import csv
import logging
import os
import sys
sys.path.append('../')

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import glue_compute_metrics

from data_loader import PromptDataset
from data.process import processors

logger = logging.getLogger(__name__)


task_mappings = {
    'sst-2': 'sst-2',
    'cola': 'cola',
    'mnli': 'mnli',
    'mnli-mm': 'mnli-mm',
    'qqp': 'qqp',
    'qnli': 'qnli',
    'rte': 'rte',
    'mrpc': 'mrpc',
    'mpqa': 'sst-2',
    'mr': 'sst-2',
    'subj': 'sst-2',
    'trec': 'sst-2',
    'snli': 'qnli',
}


def evaluate(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_dataset = PromptDataset(args, eval_task, tokenizer, data_type='dev')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[-1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[-2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                logits = logits[:, label_ids]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = batch_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = np.append(out_label_ids, batch_labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = glue_compute_metrics(task_mappings[eval_task], preds, out_label_ids)
        results.update(result)
        

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

    return results



def predict(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_list = processor.get_labels()
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_dataset = PromptDataset(args, eval_task, tokenizer, data_type='test')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None

        for batch in tqdm(eval_dataloader, desc="Infering"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[3]
                outputs = model(**inputs)
                logits = outputs[0]
                logits = logits[:, label_ids]

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        output_infer_file = os.path.join(
            eval_output_dir, 
            "{}_{}_{}_{}_{}_{}_{}.tsv".format(
                eval_task, 
                args.generator_type,
                args.add_prompt_layer, 
                args.num_prompt_tokens, 
                args.proj_down_size,
                args.per_gpu_train_batch_size,
                args.learning_rate,
                args.warmup_rate,
            )
        )
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "prediction"])
            for i, pred in enumerate(preds):
                if args.output_mode == "classification":
                    prediction = label_list[pred]
                elif args.output_mode == "regression":
                    prediction = str(pred)
                writer.writerow([i, prediction]) 

