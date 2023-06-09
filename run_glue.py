# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = {
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            LongformerConfig           
            #XLMRobertaConfig,
}

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "longformer": (LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer),
    "gpt2": (GPT2Config, GPT2Tokenizer, GPT2LMHeadModel) 
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

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
   
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas= (0.9, 0.98), weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
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
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
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
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            
            preds = outputs[1].detach().cpu().numpy() 
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            num_zer = 0
            
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
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    output_dir = os.path.join(args.output_dir, "checkpoint")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
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
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                for mm in range(len(inputs['input_ids'])):
                    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][mm].detach().cpu().numpy()))
                
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

               
        
        list_pred = list(np.argmax(preds, axis=1))
       
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        print()
        
        test_f1 = f1_score(out_label_ids, preds, average='binary')
        test_acc = accuracy_score(out_label_ids, preds)
        print('Test set: test_f1: {} -- test_acc = {}'.format(test_f1, test_acc))
        test_f1 = f1_score(out_label_ids, preds, average='micro') 
        print('micro f1 = {}'.format(test_f1))
        test_f1 = f1_score(out_label_ids, preds, average='macro')   
        print('macro f1 = {}'.format(test_f1))
        test_f1 = f1_score(out_label_ids, preds, average='binary')   
        print('binary f1 = {}'.format(test_f1))
        test_f1 = f1_score(out_label_ids, preds, average='weighted')
        print('weighted f1 = {}'.format(test_f1))
        test_roc_auc = roc_auc_score(out_label_ids, preds)
        print('auc_score = {}'.format(test_roc_auc))
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def test_text_rel_score(args, model, tokenizer, ifname, agg_method, prefix=""):
    #get the relance score of each text
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    print('data_dir {}'.format(args.data_dir))
    print('ifname is {}'.format(ifname))
    num_sent_text= create_tsvfile(args.data_dir, ifname) #creates a MRPC input file format containing a pair of sentences in each line
    
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = num_sent_text * 2
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running testing {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        ind1 =1
        pred_scores_texts = []
        gt_scores_texts = []
        for batch in tqdm(eval_dataloader, desc="Testing"):
            model.eval()
            min_score_1_2 = 1000
            wmean_score1_2 = 0
            mean_score1_2 = 0
            batch = tuple(t.to(args.device) for t in batch)
            ind1+=1
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                preds = torch.nn.functional.softmax(logits, dim=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                   
            nb_eval_steps += 1
            kk = 0
            for ind_s_i in range(num_sent_text):
                for ind_s_j in range(ind_s_i+1, num_sent_text):
                    score1_2=preds[kk,1]
                    kk+=1
                    wmean_score1_2 += ((1/(ind_s_j-ind_s_i))*score1_2)
                    mean_score1_2 += score1_2
                    if score1_2 < min_score_1_2:
                        min_score_1_2 = score1_2
            

            if agg_method=='min':
                score_text = min_score_1_2
            elif agg_method=='w_mean':
                score_text = wmean_score1_2/kk
            else:
                score_text = mean_score1_2/kk
            if score_text.item()<0.5:
                pred_scores_texts.append(np.int64(0))
            else:
                pred_scores_texts.append(np.int64(1))
            gt_scores_texts.append(out_label_ids[0])
           
           
        
        test_f1 = f1_score(gt_scores_texts, pred_scores_texts, average='binary')
        test_acc = accuracy_score(gt_scores_texts, pred_scores_texts)
        print('Test set: test_f1: {} -- test_acc = {}'.format(test_f1, test_acc))
        results['acc'] = test_acc
        results['f1'] = test_f1

        return results


def predict_sentrel_agg(args, model, tokenizer, ifname, ofname, agg_method, prefix=""):
    #get the relance score of each text and write into file
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    num_sent_text= create_tsvfile(args.data_dir, ifname) #creates a MRPC input file format containing a pair of sentences in each line
    fw = open(os.path.join(args.data_dir,ofname), 'w')
    fr = open(os.path.join(args.data_dir,ifname), 'r')
    texts = fr.readlines()
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = num_sent_text * 2
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running testing {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        ind1 =1
        pred_scores_texts = []
        gt_scores_texts = []
        ind_text = 0
        for batch in tqdm(eval_dataloader, desc="Testing"):
            model.eval()
            min_score_1_2 = 1000
            wmean_score1_2 = 0
            mean_score1_2 = 0
            batch = tuple(t.to(args.device) for t in batch)
            ind1+=1
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                preds = torch.nn.functional.softmax(logits, dim=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            nb_eval_steps += 1
            kk = 0
            for ind_s_i in range(num_sent_text):
                for ind_s_j in range(ind_s_i+1, num_sent_text):
                    score1_2=preds[kk,1]
                    kk+=1
                    wmean_score1_2 += ((1/(ind_s_j-ind_s_i))*score1_2)
                    mean_score1_2 += score1_2
                    if score1_2 < min_score_1_2:
                        min_score_1_2 = score1_2
            if agg_method=='min':
                score_text = min_score_1_2
            elif agg_method=='w_mean':
                score_text = wmean_score1_2/kk
            else:
                score_text = mean_score1_2/kk
            pred_scores_texts.append(score_text)
            fw.write('{}\t{}\n'.format(texts[ind_text].split('\n')[0].strip()[:-1].strip(),score_text))
            ind_text+=1
        fw.close()
        return -1

def predict_sentsensible_agg(args, model, tokenizer, ifname, ofname, agg_method, prefix=""):
    #get the sensibility score of each text and write into file
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    num_sent_text= create_tsvfile_textlevel_sensibility(args.data_dir, ifname) #creates a CoLA input file format containing a pair of consecutive sentences in each line
    fw = open(os.path.join(args.data_dir,ofname), 'w')
    fr = open(os.path.join(args.data_dir,ifname), 'r')
    texts = fr.readlines()
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = num_sent_text-1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running testing {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        ind1 =1
        pred_scores_texts = []
        gt_scores_texts = []
        ind_text = 0
        for batch in tqdm(eval_dataloader, desc="Testing"):
            model.eval()
            min_score_1_2 = 1000
            mean_score1_2 = 0
            max_score_1_2 = 0
            batch = tuple(t.to(args.device) for t in batch)
            ind1+=1
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}

                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2] 
                preds = torch.nn.functional.softmax(logits, dim=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            nb_eval_steps += 1
            kk = 0
            for ind_s_i in range(num_sent_text-1):
                    score1_2=preds[kk,1]
                    print('score is {}'.format(preds[kk,1]))
                    kk+=1
                    print('i={} and j={} score is {}'.format(ind_s_i,ind_s_i+1,score1_2))
                    mean_score1_2 += score1_2
                    if score1_2 < min_score_1_2:
                        min_score_1_2 = score1_2
                    if score1_2 > max_score_1_2:
                        max_score_1_2 = score1_2
            
            if agg_method=='min':
                score_text = min_score_1_2
            elif agg_method=='max':
                score_text = max_score_1_2
            else:
                score_text = mean_score1_2/kk
            pred_scores_texts.append(score_text)
            fw.write('{}\t{}\n'.format(texts[ind_text].split('\n')[0].strip(),score_text))
            ind_text+=1
        fw.close()
        return -1

def predict_textrel(args, model, tokenizer, ifname, ofname, prefix=""):
    #get the relance score of each text and write into file
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    create_tsvfile_textlevel(args.data_dir, ifname) #creates a CoLA input file format containing text in each line
    fw = open(os.path.join(args.data_dir,ofname), 'w')
    fr = open(os.path.join(args.data_dir,ifname), 'r')
    texts = fr.readlines()
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running testing {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        ind1 =1
        pred_scores_texts = []
        gt_scores_texts = []
        ind_text = 0
        for batch in tqdm(eval_dataloader, desc="Testing"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            ind1+=1
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                preds = torch.nn.functional.softmax(logits, dim=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy()                     
            nb_eval_steps += 1
            for ind in range(len(preds)):
                pred_scores_texts.append(preds[ind,1])
                fw.write('{}\t{}\n'.format(texts[ind_text].split('\n')[0].strip().strip(),preds[ind,1]))
                ind_text+=1
        fw.close()
        return -1
 


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
       ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            task=args.task_name.lower()
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    if args.model_type.lower() =='roberta' or args.model_type.lower()=='distilbert' or args.model_type.lower()=='longformer':
        all_token_type_ids=torch.tensor([[0 for i in range(args.max_seq_length)] for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
     
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
   
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def create_tsvfile(data_dir, ifname): 
    fr = open(os.path.join(data_dir,ifname), 'r')
    lines = fr.readlines()
    tsv_file = open(os.path.join(data_dir,'dev.tsv'), 'w')

    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    tsv_writer.writerow(["Quality", "#1 ID", "#2 ID", "#1 String", "2 String"])
    for line in lines:
        parts= line.split('\n')[0].split('</s>')
        num_sent_text = len(parts)-1
        for i in range(num_sent_text):
            for j in range(i+1,num_sent_text):
                tsv_writer.writerow([parts[-1].strip(), '111','222', parts[i].strip(), parts[j].strip()])
    return num_sent_text

def create_tsvfile_textlevel(data_dir, ifname): 
    fr = open(os.path.join(data_dir,ifname), 'r')
    lines = fr.readlines()
    if os.path.exists(os.path.join(data_dir,'dev.tsv')):
        return 
    tsv_file = open(os.path.join(data_dir,'dev.tsv'), 'w')

    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    ind_line=0
    for line in lines:
        line = line.split('\n')[0]
        line = line.split('</s>')
        line = [l.strip() for l in line]
        line = ' '.join(line)
        story = line.replace('  ', ' ')
        tsv_writer.writerow([ind_line, 0, ind_line, story.strip()])
        ind_line +=1

def create_tsvfile_textlevel_sensibility(data_dir, ifname):
    fr = open(os.path.join(data_dir,ifname), 'r')
    lines = fr.readlines()
    tsv_file = open(os.path.join(data_dir,'dev.tsv'), 'w')

    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    ind_line=0
    for line in lines:
        parts= line.split('\n')[0].split('</s>')
        num_sent_text = len(parts)-1
        for i in range(num_sent_text-1):
            st_text=parts[i].strip()+' '+parts[i+1].strip()
            tsv_writer.writerow([ind_line, 0, ind_line, st_text])
            ind_line+=1
    return num_sent_text


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required="The name of the task to train selected in the listc_ " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--test_fname", default=None, type=str,
                        help="The name of the file including one text and its label.")
    parser.add_argument("--do_test_text", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pred_sentscores_agg", action='store_true',
                        help="Whether to predict scores on the dev set and aggregate them to have text level scores.")
    parser.add_argument("--do_pred_textscores", action='store_true',
                        help="Whether to predict scores on the dev set to have text level scores.")
    parser.add_argument("--do_pred_sent_sensible_scores_agg", action='store_true',
                        help="Whether to predict sensibility scores on the dev set to have text level sensiblitiy scores.") 

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

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
    else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
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

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    #label_list= ['0','1','2','3','4']
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels= num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)


    if (args.do_test_text or args.do_pred_sentscores_agg or args.do_pred_textscores or args.do_pred_sent_sensible_scores_agg) and args.local_rank in [-1, 0]:
        args.data_dir= args.data_dir.strip()
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            if args.do_test_text:
                result = test_text_rel_score(args, model, tokenizer, args.test_fname, agg_method='w_mean', prefix=prefix)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)
            if args.do_pred_sentscores_agg:
                ofname = args.test_fname+'_ft_lm_aggsentscores'
                result = predict_sentrel_agg(args, model, tokenizer, args.test_fname, ofname, agg_method='w_mean', prefix=prefix)
                results.update(result)
            if args.do_pred_sent_sensible_scores_agg:
                ofname = args.test_fname+'_ft_lm_aggsent_sensscores'
                result = predict_sentsensible_agg(args, model, tokenizer, args.test_fname, ofname, agg_method='mean', prefix=prefix)
            if args.do_pred_textscores:
                ofname = args.test_fname+'_ft_lm_textscores'
                result = predict_textrel(args, model, tokenizer, args.test_fname, ofname, prefix=prefix)
                results.update(result)

    return results


if __name__ == "__main__":
    main()
