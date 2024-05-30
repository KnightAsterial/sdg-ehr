import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, PeftModel
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

MODEL_DIR = os.environ["MODEL_DIR"]
if MODEL_DIR is None:
    print("MODEL_DIR ENVIRONMENT VARIABLE NOT SET")
    exit(1)


def format_single_example(example):

  narrative = ["PATIENT HISTORY:"]
  for idx, visit in enumerate(example, 1):
      visit_text = (
          f"Visit {idx}:\n"
          f"  Diagnoses: {', '.join(visit) if visit else 'None'}\n"
      )
      narrative.append(visit_text)
  narrative.append("GENERATE A SET OF NURSES NOTES THAT WOULD BE WRITTEN ABOUT THIS PATIENT GIVEN THEIR DIAGNOSES HISTORY.")
  prompt = '\n'.join(narrative)

  return prompt



def main():

    # Load the 7b mistral model
    model_id = "BioMistral/BioMistral-7B"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set it to a new token to correctly attend to EOS tokens.
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )


    # Load pretrained model from disk
    # model = get_peft_model(model, lora_config) # Load and prepare dataset
    model = PeftModel.from_pretrained(model, MODEL_DIR)

    test_datum = [["Single liveborn, born in hospital, delivered by cesarean section", "Need for prophylactic vaccination and inoculation against viral hepatitis", "Observation for suspected infectious condition"]]
    prompt = format_single_example(test_datum)

    model_input  = [
                    {"role": "user", "content": prompt}
                    ]   
    
    tokens = tokenizer.apply_chat_template(model_input, return_tensors="pt").to('cuda')
    streamer = TextStreamer(tokenizer)
    output = model.generate(tokens, streamer=streamer, penalty_alpha=0.6, top_k=4, max_new_tokens=2048)
    # output = model.generate(tokens, num_beams=4, max_new_tokens=2048)

    output = output.to('cpu')
    print(output)
    print(tokenizer.decode(token_ids=output, skip_special_tokens=True))


main()
