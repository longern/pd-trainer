#!/usr/bin/env python3
import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig

from .trainer import PDTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Base model name or path to finetune.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The file path of the dataset.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Save directory.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=32,
        help="Total number of training epochs to perform. (default: 32)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device accelerator core/CPU for training. (default: 1)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32768,
        help="The maximum sequence length. (default: 32768)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        help="Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`",
    )
    parser.add_argument(
        "--disable_lora",
        action="store_true",
        help="Whether LoRA is disabled.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank. (default: 64)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha. (default: 64)",
    )
    args = parser.parse_args()

    train_dataset = load_dataset("json", data_files=args.dataset)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        save_strategy="no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        num_train_epochs=args.num_train_epochs,
        model_init_kwargs={
            "attn_implementation": args.attn_implementation,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trainer = PDTrainer(
        model=args.model_name_or_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset["train"],
        peft_config=lora_config if not args.disable_lora else None,
    )
    trainer.train()

    if args.save_dir:
        trainer.model.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
