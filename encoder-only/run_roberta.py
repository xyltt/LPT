import json
import logging
import os
import random
import fitlog

import numpy as np
import torch

from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import RobertaConfig, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process

from models.modeling_roberta import RobertaPromptTuning

from arguments import get_args
from data_loader import PromptDataset
from data.process import processors, output_modes
from evaluations import evaluate, predict

logger = logging.getLogger(__name__)

def get_metric_key(task_name):
    if task_name == "cola":
        return "mcc"
    elif task_name == "sst-2":
        return "acc"
    elif task_name == "mrpc":
        return "acc_and_f1"
    elif task_name == "sts-b":
        return "corr"
    elif task_name == "qqp":
        return "acc_and_f1"
    elif task_name == "mnli":
        return "mnli/acc"
    elif task_name == "mnli-mm":
        return "mnli-mm/acc"
    elif task_name == "qnli":
        return "acc"
    elif task_name == "rte":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "hans":
        return "acc"
    elif task_name == "mpqa":
        return "acc"
    elif task_name == "mr":
        return "acc"
    elif task_name == "subj":
        return "acc"
    elif task_name == "trec":
        return "acc"
    elif task_name == "snli":
        return "acc"
    else:
        raise KeyError(task_name)

        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    if args.debug:
        fitlog.debug()
    if args.local_rank in [-1, 0]:
        fitlog.set_log_dir(args.log_dir)
        fitlog.add_hyper(args)

    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir) 

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]  

    
    if args.warmup_steps > 0:
        num_warmup_steps = args.warmup_steps
    else:
        num_warmup_steps = args.warmup_rate * t_total

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level) 

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    best_all_metric = {}
    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )

    set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(args.task_name)
    if args.task_name == 'mnli':
        metric_key = 'avg_acc'


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            inputs["mask_pos"] = batch[-2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()    

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        res_for_display = {}
                        num_metric = 0
                        avg_metric = 0
                        for k, v in results.items():
                            num_metric += 1
                            avg_metric += v
                            res_for_display[k.replace("-", "_")] = v
                        if args.task_name == 'mnli':
                            results[metric_key] = avg_metric / num_metric
                            res_for_display[metric_key] = avg_metric / num_metric
                        fitlog.add_metric({"dev": res_for_display}, step=global_step)
                        if results[metric_key] > best:
                            best = results[metric_key]
                            best_all_metric.update(results)
                            fitlog.add_best_metric({"dev": {metric_key.replace("-", "_"): best}})
                            # save the best model
                            if not args.not_save_model:
                                output_dir = os.path.join(args.output_dir, "best_model")
                                model_to_save = OrderedDict()
                                for n, p in model.named_parameters():
                                    if p.requires_grad:
                                        model_to_save[n] = p
                                model.save_pretrained(output_dir, state_dict=model_to_save)

                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                logger.info("Saving model checkpoint to %s", output_dir)

                                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                # logger.info("Saving optimizer and scheduler states to %s", output_dir)


                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value


                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    fitlog.add_loss(loss_scalar, name="Loss", step=global_step)

                    print(json.dumps({**logs, **{"step": global_step}}))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if (
                args.local_rank == -1 and args.evaluate_during_training and args.logging_steps == 0
        ):
            logs = {}
            results = evaluate(args, model, tokenizer)
            res_for_display = {}
            num_metric = 0
            avg_metric = 0
            for k, v in results.items():
                num_metric += 1
                avg_metric += v
                res_for_display[k.replace("-", "_")] = v
            if args.task_name == 'mnli':
                results[metric_key] = avg_metric / num_metric
                res_for_display[metric_key] = avg_metric / num_metric
            fitlog.add_metric({"dev": res_for_display}, step=global_step)
            if results[metric_key] > best:
                best = results[metric_key]
                best_all_metric.update(results)
                fitlog.add_best_metric({"dev": {metric_key.replace("-", "_"): best}}) 
                # save the best model
                if not args.not_save_model:
                    output_dir = os.path.join(args.output_dir, "best_model")
                    model_to_save = OrderedDict()
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            model_to_save[n] = p
                    model.save_pretrained(output_dir, state_dict=model_to_save)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)

            for key, value in results.items():
                eval_key = "eval_{}".format(key)
                logs[eval_key] = value

            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar

            print(json.dumps({**logs, **{"step": global_step}}))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    logs = {}
    if (
            args.local_rank == -1 and args.evaluate_during_training and
            args.logging_steps > 0 and (global_step-1) % args.logging_steps != 0
    ):
        results = evaluate(args, model, tokenizer)
        res_for_display = {}
        num_metric = 0
        avg_metric = 0
        for k, v in results.items():
            num_metric += 1
            avg_metric += v
            res_for_display[k.replace("-", "_")] = v
        if args.task_name == 'mnli':
            results[metric_key] = avg_metric / num_metric
            res_for_display[metric_key] = avg_metric / num_metric
        fitlog.add_metric({"dev": res_for_display}, step=global_step)
        if results[metric_key] > best:
            best = results[metric_key]
            best_all_metric.update(results)
            fitlog.add_best_metric({"dev": {metric_key.replace("-", "_"): best}}) 
            # save the best model
            if not args.not_save_model:
                output_dir = os.path.join(args.output_dir, "best_model")
                model_to_save = OrderedDict()
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        model_to_save[n] = p
                model.save_pretrained(output_dir, state_dict=model_to_save)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                # logger.info("Saving optimizer and scheduler states to %s", output_dir)

        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value

        learning_rate_scalar = scheduler.get_lr()[0]
        logs["learning_rate"] = learning_rate_scalar

        print(json.dumps({**logs, **{"step": global_step}}))

    if args.local_rank in [-1, 0]:
        fitlog.finish()

    return global_step, tr_loss / global_step, best_all_metric


def main():
    args = get_args()

    # SEARCH
    args.data_dir = os.path.join(args.data_dir, args.task_name)
    args.output_dir = os.path.join(
        args.output_dir, 
        str(args.generator_type),
        str(args.add_prompt_layer),
        str(args.num_prompt_tokens),
        str(args.per_gpu_train_batch_size),
        str(args.learning_rate),
        str(args.warmup_rate),
        str(args.proj_down_size),
        args.task_name
    )


    if not os.path.exists(args.log_dir):
        try:
            os.makedirs(args.log_dir)
        except:
            pass


    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach() 

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = RobertaConfig.from_pretrained(        
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.generator_type = args.generator_type
    config.add_prompt_layer = args.add_prompt_layer
    config.num_prompt_tokens = args.num_prompt_tokens
    config.proj_down_size = args.proj_down_size

    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,        
    )

    model = RobertaPromptTuning.from_pretrained(
        args.model_name_or_path, 
        config=config, 
    )

    for param in model.parameters():
        param.requires_grad = False

    if args.generator_type is not None:
        model.roberta.encoder.add_prompt_generator()
    else:
        model.generate_prompt_embeddings()

    total_model_params = 0
    num_trained_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_trained_params += p.numel()
        else:
            total_model_params += p.numel()

    print("Total Model Parameters: {}, Trainable Parameters: {}".format(total_model_params, num_trained_params))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)  

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = None
    best_all_metric = None
    if args.do_train:
        # train_dataset = load_and_cache_examples_glue(args, args.task_name, tokenizer, data_type='train')
        train_dataset = PromptDataset(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss, best_all_metric = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_infer:

        best_model = os.path.join(args.output_dir, "best_model", "pytorch_model.bin")

        model.load_state_dict(torch.load(best_model), strict=False)
        model.to(args.device) 
        predict(args, model, tokenizer)   
      
    return best_all_metric


if __name__ == "__main__":
    best = main()