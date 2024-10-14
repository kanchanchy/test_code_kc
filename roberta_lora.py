import argparse
from opacus import PrivacyEngine

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
)

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import pdb
from pathlib import Path

torch.manual_seed(1)


# model_name_or_path = "roberta-large"
# task = "qnli"
# device = "cuda:0"
# num_epochs = 10

# peft_config = LoraConfig(
#     task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
# )
# batch_size = 16
# lr = 3e-4
# print(f"lr: {lr}")

# private_training = True
# MAX_GRAD_NORM = 0.1
# DELTA = 1e-6
# EPSILON = 4.0
# model.base_model.model.roberta.encoder.layer[0].attention.self.query.lora_B.default.weight
# model.base_model.model.classifier.modules_to_save.default.dense.weight
# model.base_model.model.classifier.modules_to_save.default.out_proj.weight


def get_tokenizer(args):
    if any(k in args.model_type for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type, padding_side=padding_side
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def prepare_data(args, tokenizer):

    def tokenize_dataset(dataset):
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            if args.task == "qnli":
                outputs = tokenizer(
                    examples["question"],
                    examples["sentence"],
                    truncation=True,
                    max_length=None,
                )
            elif args.task == "sst2":
                outputs = tokenizer(
                    examples["sentence"], truncation=True, max_length=None
                )
            elif args.task == "mnli":
                outputs = tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    max_length=None,
                )
            else:
                raise ValueError(f"Unknown task: {args.task}")
            return outputs

        if args.task == "qnli":
            remove_columns = ["idx", "question", "sentence"]
        elif args.task == "sst2":
            remove_columns = ["idx", "sentence"]
        elif args.task == "mnli":
            remove_columns = ["idx", "premise", "hypothesis"]
        else:
            raise ValueError(f"Unknown task: {args.task}")

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
        )
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        return tokenized_datasets

    eval_split_key = "validation_matched" if args.task == "mnli" else "validation"
    # dataset = load_dataset("glue", args.task)
    ds_train = load_dataset("glue", args.task, split=f"train[:{args.percent}%]")
    ds_valid = load_dataset("glue", args.task, split=eval_split_key)
    ds_train_tokenized = tokenize_dataset(ds_train)
    ds_valid_tokenized = tokenize_dataset(ds_valid)
    num_labels = ds_valid.features["label"].num_classes

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        ds_train_tokenized,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        ds_valid_tokenized,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
    )
    return train_dataloader, eval_dataloader, num_labels


def load_model(args, num_classes, device, from_path=False):
    if from_path:
        lora_str = "_lora" if args.use_lora else ""
        eps_str = f"_eps{args.epsilon}" if args.private_training else ""
        model_name = f"{args.model_type}_{args.task}{lora_str}{eps_str}"
        model = AutoModelForSequenceClassification.from_pretrained(f"models/{model_name}", return_dict=True, num_labels=num_classes)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_type, return_dict=True, num_labels=num_classes)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    #model.to(f"cuda:{args.gpu}")
    model.to(device)
    return model


def run(args, model, train_dataloader, eval_dataloader, num_labels, device):
    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    metric = evaluate.load("glue", args.task)
    #device = torch.device(f"cuda:{args.gpu}")
    criterion = torch.nn.CrossEntropyLoss()

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * args.epochs),
        num_training_steps=(len(train_dataloader) * args.epochs),
    )

    if args.private_training:
        model.train()
        privacy_engine = PrivacyEngine()

        model, optimizer, criterion, train_dataloader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                criterion=criterion,
                target_delta=args.delta,
                target_epsilon=args.epsilon,
                epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
                grad_sample_mode="ghost",
            )
        )

    model.to(device)

    best_valid_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        # for step, batch in enumerate(tqdm(train_dataloader)):
        for step, batch in enumerate(train_dataloader):
            batch.to(device)
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch.labels
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        valid_acc_total = eval_metric["accuracy"]
        if valid_acc_total > best_valid_acc:
            best_valid_acc = valid_acc_total
            # Save model
            if args.private_training:
                # Unwrap opacus.grad_sample.grad_sample_module.GradSampleModule
                model_to_save = model._modules["_module"]
            else:
                model_to_save = model
            lora_str = "_lora" if args.use_lora else ""
            eps_str = f"_eps{args.epsilon}" if args.private_training else ""
            model_name = f"{args.model_type}_{args.task}{lora_str}{eps_str}"
            model_to_save.save_pretrained(f"models/{model_name}")
        print(f"epoch {epoch}:", eval_metric)
        with open("result.txt", 'a') as file:
            file.write(f"epoch {epoch}: " + str(valid_acc_total)+ "\n")

    print(f"Best validation accuracy: {best_valid_acc}")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--private_training", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--percent", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--task", type=str, default="mnli")
    parser.add_argument("--model_type", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    print(args)

    with_lora = "without lora"
    if args.use_lora:
        with_lora = "with lora"

    with open("result.txt", 'a') as file:
        file.write(f"\n\nData Percent: {args.percent}, Learning Rate: {args.lr}, {with_lora}\n....................................\n")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Found device: ", device)

    tokenizer = get_tokenizer(args)
    train_loader, valid_loader, num_labels = prepare_data(args, tokenizer)
    model = load_model(args, num_labels, device, from_path=True)
    run(args, model, train_loader, valid_loader, num_labels, device)
