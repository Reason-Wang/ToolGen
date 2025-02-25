from data.loading import load_datasets
from models.loading import load_model, load_tokenizer
from utils.setting import set_project, set_system, set_args, set_distributed_logging
from dataclasses import field, dataclass
from typing import Optional, Any
import torch
from data.loading import load_datasets
import os
import transformers
from transformers import Trainer
from typing import List
from prompts.templates import null_template
from utils.logging import get_logger
from utils.distributed import get_rank, is_main_process


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: str = field(default="")
    chat: bool = False
    architecture: str = field(default='causal')
    flash_attention: bool = False
    data_path: str = field(default="")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    resume_training: bool = False
    per_device_train_batch_size = 8
    max_length: int = 2048
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    gather_weights: bool = True
    datasets: List[str] = field(default_factory=list)
    dataset_nums: List[int] = field(default_factory=int)
    template: str = field(default="llama-3")
    add_virtual_tokens: bool = False


def train():
    # set_system("src/configs/project_config.json")
    # set_distributed_logging(strict=True)
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_args(args)
    set_project(args)

    # Get rank
    rank = get_rank()
    Logger = get_logger("logs", level="INFO", rank=rank)

    # Load VAgent tokenizer
    tokenizer = load_tokenizer(
        args.model_name_or_path, 
        cache_dir=args.cache_dir,
        virtual_tokens=args.add_virtual_tokens,
    )

    Logger.info("---- Loading Datasets ----")
    dataset, collator = load_datasets(
        chat=args.chat,
        architecture=args.architecture,
        datasets=args.datasets,
        dataset_nums=args.dataset_nums,
        tokenizer=tokenizer,
        max_length=args.max_length,
        template=args.template,
    )
    Logger.info(f"Data length: {len(dataset)}")

    Logger.info("---- Loading Model ----")
    model = load_model(
        args.model_name_or_path,
        architecture=args.architecture,
        tokenizer=tokenizer,
        flash_attention=args.flash_attention,
        cache_dir=args.cache_dir,
        virtual_tokens=args.add_virtual_tokens,
    )

    trainer = Trainer(
        model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=args.resume_training)
    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)

    # Whether to gather weights before saving
    # This is prefered for small models
    if args.gather_weights:
        trainer.save_model(args.output_dir)
    else:
        trainer.deepspeed.save_checkpoint(args.output_dir)


if __name__ == "__main__":
    train()
