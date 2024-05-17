import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from bitsandbytes import optim


def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text

def main(args):

    output_dir = f"{YOUR_HF_USERNAME}/llama-7b-qlora-ultrachat"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 1000
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"



    # Load the 7b mistral model
    model_id = "BioMistral/BioMistral-7B"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config) # Load and prepare dataset
    dataset = load_dataset("")  # our dataset path
    train_dataset = dataset["train"]

    train_dataset.set_format("torch") # Maybe??


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
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        push_to_hub=True,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        packing=True,
        dataset_text_field="_______TODO________",
        tokenizer=tokenizer,
        max_seq_length=8192,
        formatting_func=formatting_func,
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model("./finetuned_model")

