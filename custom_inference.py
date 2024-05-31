import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, PeftModel
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from bitsandbytes import optim
import datasets
import os
import pickle
import json

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
        diagnosis_history = visit.get('diagnosis history', [])
        visit_text = (
            f"Visit {idx}:\n"
            f"  Diagnoses: {', '.join(diagnosis_history) if diagnosis_history else 'None'}\n"
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load pretrained model from disk
    model = PeftModel.from_pretrained(model, MODEL_DIR)

    # Load the examples from the pickle file
    pickle_file_path = '/home/aakritilakshmanan/halo_text.pkl'
    with open(pickle_file_path, 'rb') as f:
        examples = pickle.load(f)

    # results = []

    output_file_path = os.path.join(OUTPUT_ROOT, 'generated_nurses_notes.json')
    
    for idx, example in enumerate(examples):
        print("Generating item", idx+1)
        prompt = format_single_example(example)
        model_input = [{"role": "user", "content": prompt}]
        tokens = tokenizer.apply_chat_template(model_input, return_tensors="pt").to('cuda')
        # streamer = TextStreamer(tokenizer)

        # Generate the text using the model
        # output = model.generate(tokens, streamer=streamer, penalty_alpha=0.6, top_k=4, repetition_penalty=1.2, temperature=0.7, do_sample=True, max_new_tokens=2048)
        output = model.generate(tokens, penalty_alpha=0.6, top_k=4, repetition_penalty=1.2, temperature=0.7, do_sample=True, max_new_tokens=2048)

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        new_item = {"history": example, "note": output_text}
        # results.append({"history": example, "note": output_text})
        
        # Print each generated note (first 10)
        # print(f"History: {example}\nGenerated Note: {output_text}\n")
        with open(output_file_path, 'a') as f:
            print("Writing item", idx+1)
            json_string = json.dumps(new_item)
            f.write(json_string + "\n")

    # Write the results to a JSON file
    # output_file_path = os.path.join(OUTPUT_ROOT, 'generated_nurses_notes.json')
    # with open(output_file_path, 'w') as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
