# Imports and definitions
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import login
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
from hqq_utils import AutoHQQHFModel, get_size_of_model

def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
    return quant_config

def tokenize_fn(examples):
        return tokenizer(examples["text"])

 def group_texts(examples):
        block_size = 512
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concat["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i + block_size] for i in range(0, total_len, block_size)] for k, t in concat.items()}
        result["labels"] = result["input_ids"].copy()
        return result

def main():
    login(token="hf_YOUR_ACCESS_TOKEN")  # Replace with your token

    torch.manual_seed(0)
    random.seed(0)
    device = 'cuda:0'

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    HQQLinear.set_backend(HQQBackend.PYTORCH)
    print(f"Model size before quantization: {get_size_of_model(model) / (1024**2):.2f} MiB")

    quant_config = get_quant_config_slm(model)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.bfloat16, device=device)
    print(f"Model size after quantization: {get_size_of_model(model) / (1024**2):.2f} MiB")

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        task_type=TaskType.CAUSAL_LM,
        use_dora=True
    )
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, lora_cfg)
    model.to(torch.bfloat16)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])

    lm_dataset = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    training_args = TrainingArguments(
        output_dir="./adapter_ckpt",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("./my_adapter")
    tokenizer.save_pretrained("./my_adapter")

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device, attn_implementation="sdpa").eval()
    lora_model = PeftModel.from_pretrained(base, "./my_adapter").eval()
    merged = lora_model.merge_and_unload().eval()

    hf_repo = "will200112/quantized-llama3-3b"
    merged.push_to_hub(hf_repo, safe_serialization=True)
    tokenizer.push_to_hub(hf_repo)

    print(f"Model and tokenizer uploaded to https://huggingface.co/{hf_repo}")

if __name__ == "__main__":
    main()
