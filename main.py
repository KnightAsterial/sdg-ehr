import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from bitsandbytes import optim
import datasets
import os

OUTPUT_ROOT = os.environ["OUTPUT_ROOT"]
if OUTPUT_ROOT is None:
    print("OUTPUT_ROOT ENVIRONMENT VARIABLE NOT SET")
    exit(1)

DATASET_FILE = os.environ["DATASET_FILE"]
if DATASET_FILE is None:
    print("DATASET_FILE ENVIRONMENT VARIABLE NOT SET")
    exit(1)


def prompt_formatter(example):
  codes = example['codes']

  narrative = ["PATIENT HISTORY:"]
  for idx, visit in enumerate(codes, 1):
      visit_text = (
          f"Visit {idx}:\n"
          f"  Diagnoses: {', '.join(visit) if visit else 'None'}\n"
      )
      narrative.append(visit_text)
  narrative.append("GENERATE A SET OF NURSES NOTES THAT WOULD BE WRITTEN ABOUT THIS PATIENT GIVEN THEIR DIAGNOSES HISTORY.")
  prompt = '\n'.join(narrative)

  example['prompt'] = prompt

  return example

def get_dataset_splits():
    dataset_file = DATASET_FILE
    dataset = datasets.load_dataset('json', data_files=dataset_file)

    processed_dataset = dataset.map(prompt_formatter)
    processed_dataset = processed_dataset.rename_column("notes", "completion")
    processed_dataset = processed_dataset.remove_columns("codes")

    splits = processed_dataset['train'].train_test_split(test_size=0.1)

    return splits



def main():

    output_dir = f"{OUTPUT_ROOT}/finetuned_model"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 300
    logging_steps = 10
    learning_rate = 2.5e-5
    max_grad_norm = 0.3
    max_steps = 3000
    warmup_ratio = 0.03
    # eval_strategy = "steps"
    # eval_steps = 600
    lr_scheduler_type = "constant_with_warmup"
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Load dataset
    splits = get_dataset_splits()


    # Load the 7b mistral model
    model_id = "BioMistral/BioMistral-7B"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set it to a new token to correctly attend to EOS tokens.
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )


    # Add LoRA adapters to the model
    # model = get_peft_model(model, lora_config) # Load and prepare dataset
    model.add_adapter(lora_config)

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        # eval_strategy=eval_strategy,
        # eval_steps=eval_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        weight_decay=0.001,
        report_to="tensorboard"
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=splits['train'],
        eval_dataset=splits['test'],
        packing=False,
        tokenizer=tokenizer,
        max_seq_length=2048
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model(f"{OUTPUT_ROOT}/finetuned_model")

main()
