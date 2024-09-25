import os
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    LayoutLMv3ForTokenClassification,
    set_seed,
)
from transformers.trainer import Trainer

from helpers import DataCollator, MAX_LEN


@dataclass
class Arguments(TrainingArguments):
    model_dir: str = field(
        default=None,
        metadata={"help": "Path to model, based on `microsoft/layoutlmv3-base`"},
    )
    dataset_dir: str = field(
        default=None,
        metadata={"help": "Path to dataset"},
    )


def load_train_and_dev_dataset(path: str) -> (Dataset, Dataset):
    datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(path, "train.jsonl.gz"),
            "dev": os.path.join(path, "dev.jsonl.gz"),
        },
    )
    return datasets["train"], datasets["dev"]


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    train_dataset, dev_dataset = load_train_and_dev_dataset(args.dataset_dir)
    logger.info(
        "Train dataset size: {}, Dev dataset size: {}".format(
            len(train_dataset), len(dev_dataset)
        )
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_dir, num_labels=MAX_LEN, visual_embed=False
    )
    data_collator = DataCollator()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
