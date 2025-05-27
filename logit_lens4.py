import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# ---------------------- CONFIG -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_name = "meta-llama/Llama-2-7b-chat-hf"
jsonl_path = "300.jsonl"

# ---------------------- LOAD MODEL -------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    cache_dir = "/scratch/rahul.garg/hfCache"
).to("cuda").eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ---------------------- LOAD DATA --------------------
with open(jsonl_path) as f:
    data = [json.loads(l) for l in f]

# ---------------------- HOOKS ------------------------
attn_reps = []
mlp_reps = []
residual_streams = []

def make_hook_post_attn_ln(layer_idx):
    def hook(module, input, output):
        raw_out = output[0] if isinstance(output, tuple) else output
        normed = model.model.layers[layer_idx].post_attention_layernorm(raw_out)
        attn_reps.append((layer_idx, normed.detach()))
    return hook

def make_hook_post_mlp_ln(layer_idx):
    def hook(module, input, output):
        normed = model.model.layers[layer_idx].input_layernorm(output)
        mlp_reps.append((layer_idx, normed.detach()))
    return hook

def make_hook_residual_stream(layer_idx):
    def hook(module, input, output):
        residual_streams.append((layer_idx, output[0].detach()))
    return hook

for idx, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(make_hook_post_attn_ln(idx))
    layer.mlp.register_forward_hook(make_hook_post_mlp_ln(idx))
    layer.register_forward_hook(make_hook_residual_stream(idx))

# ---------------------- UTILS ------------------------
def get_logit_diff(rep, unembedding, token_ids):
    last_rep = rep[:, -1, :]
    logits = last_rep @ unembedding.T
    diff = logits[0, token_ids[0]] - logits[0, token_ids[1]]
    return diff.item()

# ---------------------- MAIN LOOP --------------------
unembedding = model.lm_head.weight
token_a_id = tokenizer.encode("A", add_special_tokens=False)[0]
token_b_id = tokenizer.encode("B", add_special_tokens=False)[0]
answer_token_ids = [token_a_id, token_b_id]

cumulative_attn_diffs = []
cumulative_mlp_diffs = []
final_trunc_diffs = []
residual_diffs_per_layer = []

for item in tqdm(data, desc="Running cumulative logit-lens"):
    prompt = f"""Answer this question based on your knowledge and given information. Just choose only one option strictly and give only the letter as answer.

Question: {item['question']}

Options: 
A: {item['options'][0]}
B: {item['options'][1]}

Information: {item['counter_memory']}

Correct Option:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attn_reps.clear()
    mlp_reps.clear()
    residual_streams.clear()

    with torch.no_grad():
        _ = model(**inputs)

    attn_reps_sorted = [x[1] for x in sorted(attn_reps, key=lambda x: x[0])]
    mlp_reps_sorted = [x[1] for x in sorted(mlp_reps, key=lambda x: x[0])]
    residuals_sorted = [x[1] for x in sorted(residual_streams, key=lambda x: x[0])]

    attn_diffs = [get_logit_diff(rep, unembedding, answer_token_ids) for rep in attn_reps_sorted]
    mlp_diffs = [get_logit_diff(rep, unembedding, answer_token_ids) for rep in mlp_reps_sorted]

    # Cumulative per example
    cumulative_attn_diffs.append(np.cumsum(attn_diffs))
    cumulative_mlp_diffs.append(np.cumsum(mlp_diffs))

    # Final output difference (mean across last residual layer only)
    residual_layer_diffs = [
        get_logit_diff(model.model.norm(res), unembedding, answer_token_ids)
        for res in residuals_sorted
    ]
    residual_diffs_per_layer.append(residual_layer_diffs)

# ---------------------- AGGREGATE --------------------
cumulative_attn_diffs = np.array(cumulative_attn_diffs)
cumulative_mlp_diffs = np.array(cumulative_mlp_diffs)
residual_diffs_per_layer = np.array(residual_diffs_per_layer)

mean_cum_attn = np.mean(cumulative_attn_diffs, axis=0)
std_cum_attn = np.std(cumulative_attn_diffs, axis=0)

mean_cum_mlp = np.mean(cumulative_mlp_diffs, axis=0)
std_cum_mlp = np.std(cumulative_mlp_diffs, axis=0)

mean_resid = np.mean(residual_diffs_per_layer, axis=0)
std_resid = np.std(residual_diffs_per_layer, axis=0)

layers = np.arange(len(mean_cum_attn))

# ---------------------- PLOT -------------------------
plt.figure(figsize=(12, 6))
plt.plot(layers, mean_cum_attn, label="Cumulative Attention Effect", color="#1f77b4", marker='o', linewidth=2)
plt.fill_between(layers, mean_cum_attn - std_cum_attn, mean_cum_attn + std_cum_attn, color="#1f77b4", alpha=0.15)

plt.plot(layers, mean_cum_mlp, label="Cumulative MLP Effect", color="#ff7f0e", marker='s', linewidth=2)
plt.fill_between(layers, mean_cum_mlp - std_cum_mlp, mean_cum_mlp + std_cum_mlp, color="#ff7f0e", alpha=0.15)

plt.plot(layers, mean_resid, label="Residual Stream Effect", color="#2ca02c", marker='^', linewidth=2)
plt.fill_between(layers, mean_resid - std_resid, mean_resid + std_resid, color="#2ca02c", alpha=0.15)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xlabel("Layer", fontsize=14)
plt.ylabel("Logit(A) - Logit(B)", fontsize=14)
# plt.title("Last Token Layerwise Effects on Entity Substituted Parametric Evidence in context", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cum_logit_parametric.pdf", dpi=1200)
plt.show()