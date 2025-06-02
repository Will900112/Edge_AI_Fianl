import torch, random, os, csv, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from peft import PeftModel
from hqq.utils.patching import prepare_for_inference
from hqq_utils import get_linear_tags_from_model, get_size_of_model, AutoHQQHFModel
import torch.backends.cuda as bk

def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    q4_config = BaseQuantizeConfig(nbits=4, group_size=1024)

    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config

    return quant_config

def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)

    nsamples = test_enc.numel() // model.seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()

def main():
    bk.enable_flash_sdp(False)
    bk.enable_mem_efficient_sdp(True)
    bk.enable_math_sdp(False)
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda:0"

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    adapter_path = "/content"

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa"
    ).eval()

    lora_model = PeftModel.from_pretrained(base, adapter_path).eval()
    merged = lora_model.merge_and_unload()
    merged.eval()

    quant_cfg = get_quant_config_slm(merged)
    AutoHQQHFModel.quantize_model(
        merged,
        quant_config=quant_cfg,
        compute_dtype=torch.float16,
        device=device,
    )

    print(f"Model size after quantization: {get_size_of_model(merged)/(1024**2):.2f} MiB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        merged.config.pad_token_id = tokenizer.eos_token_id

    model = merged
    prepare_for_inference(model.model, backend="gemlite")

    model.forward = torch.compile(
        model.forward,
        mode="max-autotune",
        fullgraph=False,
        dynamic=True,
    )
    model.prefill_forward = model.forward

    cache_device = next(model.parameters()).device
    max_new_tokens = 256
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_new_tokens + 16,
        device=cache_device,
        dtype=torch.float16,
    )

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    for _ in tqdm(range(5), desc="Warm Up..."):
        _ = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tputs = []
    time_record = []

    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = generated[0][input_ids.shape[1]:].shape[0] / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)

    print(f'Prompt: {prompt}\nResponse: {response}\n')
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')
    print(f'Throughput: {org_tput} toks/s')

    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, round(ppl, 2)])
        writer.writerow([1, round(org_tput, 1)])

if __name__ == "__main__":
    main()
