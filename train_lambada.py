import argparse
import os
from collections import defaultdict
from time import perf_counter
import sys
import random

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.optim import RMSprop
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from universal_transformer import datasets, logger, models, tokenizers, vectors

TEMP_WEIGHTS_PATH = "state_dict.pickle"


def run_model_on_dataset(
    model, dataset, config, yield_freq=None, optimizer=None, scheduler=None
):
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=config.batch_size, pin_memory=True
    )

    total_loss = 0
    total_target_loss = 0
    total_examples = 0
    sum_of_words = 0
    preds = []
    #logits = []
    #label_ids = []
    batches_since_yield = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()

    for i, batch in enumerate(dataloader):
        device = torch.device(config.device)
        batch = tuple(t.to(device) for t in batch)
        (input_ids, masks) = batch
        batch_logits = model(
            source_ids=input_ids[:, :-1],
            target_ids=input_ids[:, :-1],
            source_padding_mask=masks[:, :-1],
            target_padding_mask=masks[:, :-1],
        )
        loss = criterion(
            batch_logits.view(-1, batch_logits.size(-1)), input_ids[:, 1:].reshape(-1)
        )

        indices = (~masks[:, 1:]).type(torch.int64).sum(dim=1)
        batch_words = indices.sum()
        sum_of_words += batch_words
        indices = indices.detach().cpu().numpy() - 1
        #print('inidices', indices)
        batch_examples = len(batch[0]) 
        total_examples += batch_examples

        total_loss += loss.item() * batch_words # Convert from mean to sum.

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_logits = batch_logits.detach().cpu().numpy()
        # find positions of last word #1s are the mask, so negate
        #print('~masks')
        #print((~masks).type(torch.int64))

        

        #logits.append(batch_logits)
        target_logits = batch_logits[torch.arange(batch_logits.shape[0]), indices, :]
        preds = np.argmax(target_logits, axis=1)
        target_words = input_ids[:, 1:][torch.arange(input_ids.shape[0]), indices] #[:, indices]
        print(type(preds == target_words))
        print(type(preds))
        print(type(target_words))
        correct += (preds == target_words).sum()


        target_loss = criterion(
            torch.tensor(target_logits).cuda(), torch.tensor(target_words).cuda()
        )
        total_target_loss += target_loss.item() * batch_examples # Convert from mean to sum.

        #print('compare targets')
        #print(target_words)
        #print(input_ids[:, 1:])
        #print('preds')
        #print(preds)
        #preds.extend(np.argmax(batch_logits, axis=1))
        #label_ids.extend(batch[1][-1].detach().cpu().numpy())
        batches_since_yield += 1

        if (
            i == len(dataloader) - 1
            or yield_freq is not None
            and (i + 1) % yield_freq == 0
        ):
            #logits = np.concatenate(logits, axis=0)
            #yield logits, preds, label_ids, total_loss / batches_since_yield
            mean_loss = total_loss / sum_of_words 
            perplexity = np.exp(mean_loss.detach().cpu().numpy())
            accuracy = correct / total_examples
            mean_target_loss = total_target_loss / total_examples
            target_perplexity = np.exp(mean_target_loss)
            yield  mean_loss, perplexity, accuracy, target_perplexity #batches_since_yield
            total_loss = 0
            total_target_loss = 0
            total_examples = 0
            sum_of_words = 0
            preds = []
            logits = []
            label_ids = []
            batches_since_yield = 0
            correct = 0
            #print('batch', i)


def log_step(
    run_type,
    metrics,
    epoch=None,
    **kwargs,
):
    log_dict = {f"{run_type}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    logger.info(log_dict)
    wandb.log(log_dict, **kwargs)

    _step_metrics[run_type].append(metrics)


_step_metrics = defaultdict(lambda: [])


def compute_metrics(
    #logits,
    #preds,
    #label_ids,
    loss,
    perplexity,
    accuracy,
    target_perplexity,
    runtime,
):
    return {
        "loss": loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "target_perplexity": target_perplexity,
        "runtime": runtime
        #"examples_per_second": len(preds) / runtime,
        #"sample_size": len(preds),
    }


def log_summary(run_type):
    metrics_df = pd.DataFrame(_step_metrics[run_type])
    for agg_method in ["min", "max"]:
        for metric, value in metrics_df.agg(agg_method).items():
            wandb.run.summary[f"{run_type}_{metric}_{agg_method}"] = value


def train(config, run):
    # Load stuff based on the config.
    #tokenizer = tokenizers.get_tokenizer(config)

    data = datasets.get_dataset(config, None) #tokenizer)
    config.train_size = len(data.train)
    config.val_size = len(data.val)

    #embedding_matrix = vectors.get_vectors(config, tokenizer)

    model = models.get_model(config, embedding_matrix=None, vocab=data.vocab, embedding_size=256) #embedding_matrix)

    if config.log is not None:
        wandb.watch(model, log=config.log)

    device = torch.device(config.device)
    model.to(device)

    best_performance = None
    step = 0
    for epoch in range(1, config.epochs + 1):
        if config.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == "rmsprop":
            optimizer = RMSprop(model.parameters(), lr=config.lr)
        else:
            raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')

        scheduler = None
        if config.get("learning_rate_decay_schedule", None) is not None:
            if config.learning_rate_decay_schedule == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=len(data.train) * config.epochs,
                )
            else:
                raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')
        model.train()
        mini_batch_start_time = perf_counter()

        #for logits, preds, label_ids, loss in run_model_on_dataset(
        for loss, perplexity, accuracy, target_perplexity in run_model_on_dataset(
            model,
            data.train,
            config,
            yield_freq=config.get("log_freq"),
            optimizer=optimizer,
            scheduler=scheduler,
        ):
            step += 1
            train_metrics = compute_metrics(
                #logits=logits,
                #preds=preds,
                #label_ids=label_ids,
                loss=loss,
                perplexity=perplexity,
                accuracy=accuracy,
                target_perplexity=target_perplexity,
                runtime=perf_counter() - mini_batch_start_time,
            )
            log_step("train", train_metrics, step=step, epoch=epoch)

            # Validate
            model.eval()
            with torch.no_grad():
                start_time = perf_counter()
                #logits, preds, label_ids, loss = iter(
                loss, perplexity, accuracy, target_perplexity = iter(
                    next(run_model_on_dataset(model, data.val, config, yield_freq=None))
                )
                val_metrics = compute_metrics(
                    #logits=logits,
                    #preds=preds,
                    #label_ids=label_ids,
                    loss=loss,
                    perplexity=perplexity,
                    accuracy=accuracy,
                    target_perplexity=target_perplexity,
                    runtime=perf_counter() - start_time,
                )
                log_step("val", val_metrics, step=step, epoch=epoch)
                log_summary("val")

                if config.checkpoint_metric is not None:
                    if (
                        best_performance is None
                        or val_metrics[config.checkpoint_metric] > best_performance
                    ):
                        best_performance = val_metrics[config.checkpoint_metric]
                        torch.save(model.state_dict(), TEMP_WEIGHTS_PATH)

            model.train()  # Need to re-enter training model.

            mini_batch_start_time = perf_counter()

    if config.checkpoint_metric is not None and run.name is not None:
        # Save the best model weights.
        artifact = wandb.Artifact(
            f"{run.name.replace('-', '_')}_best_weights", type="weights"
        )
        artifact.add_file(TEMP_WEIGHTS_PATH)
        run.log_artifact(artifact)


    # now score on test set
    model.load_state_dict(torch.load(TEMP_WEIGHTS_PATH))
    model.eval()
    with torch.no_grad():
        start_time = perf_counter()
        #logits, preds, label_ids, loss = iter(
        loss, perplexity, accuracy, target_perplexity = iter(
            next(run_model_on_dataset(model, data.test, config, yield_freq=None))
        )
        val_metrics = compute_metrics(
            #logits=logits,
            #preds=preds,
            #label_ids=label_ids,
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            target_perplexity=target_perplexity,
            runtime=perf_counter() - start_time,
        )
        log_step("test", val_metrics, step=step, epoch=epoch)
        log_summary("test")



class ConfigWrapper:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getattr__(self, key):
        try:
            return getattr(self.config, key)
        except:
            pass

    def __setattr__(self, key, value):
        if key == "config":
            self.__dict__["config"] = value
        else:
            setattr(self.config, key, value)

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, default=None, metavar="N")
    parser.add_argument("--project", type=str, default="delete_me", metavar="N")

    args, unknown = parser.parse_known_args()

    if args.configs is not None:
        os.environ["WANDB_CONFIG_PATHS"] = args.configs

    run = wandb.init(entity="dfd", project=args.project, name=args.configs[8:])

    config = ConfigWrapper(wandb.config)

    if torch.cuda.is_available():
        config.device = "cuda"
        print('using cuda')
    else:
        config.device = "cpu"
        print('using cpu')

    seed_torch()
    train(config, run)
