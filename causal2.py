import torch 
from transformers import AutoModelForCausalLM , AutoTokenizer
from json_utils import load_json, save_json, load_jsonl, save_jsonl
from tqdm import tqdm
import os
from save_activations import load_model, get_activations
from get_diff_tokens import token_difference, load_tokenizer
from utils import get_target_prob
import sys
import json
import torch.nn.functional as F
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


# a_sents = []
# b_sents = []
# difference = []
# model = load_model("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True , cache_dir = "/scratch/rahul.garg/hfCache").to("cuda")
model.eval()
tokenizer = load_tokenizer("meta-llama/Llama-2-7b-chat-hf")

data = load_jsonl("300.jsonl")


def save_hook(name, store):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        store[name] = out.detach().clone()
    return hook

# --- Probability getter for target token (last token in output) ---
def get_probs_over_tokens(logits, token_a, token_b):
    final_logits = logits[0, -1]  # last token
    selected_logits = final_logits[[token_a, token_b]]
    probs = F.softmax(selected_logits, dim=0)
    return {
        token_a: probs[0].item(),
        token_b: probs[1].item()
    }

# --- Swap hook to replace activations ---
def make_swap_hook(layer_idx, comp_name, positions, activations_a):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        swapped = out.clone()
        for pos in positions:
            swapped[:, pos, :] = activations_a[f"layer_{layer_idx}_{comp_name}"][:, pos, :]
        return (swapped,) + output[1:] if isinstance(output, tuple) else swapped
    return hook

attn_all_effects = []
attn_last_effects = []
mlp_all_effects = []
mlp_last_effects = []

attn_all_effects_de = []
attn_last_effects_de = []
mlp_all_effects_de = []
mlp_last_effects_de = []

num_layers = len(model.model.layers)
final = {}
for i in range(num_layers):
    final[i] = {
        "attn": [] ,
        "attn_last":[],
        "mlp_last": [],
        "mlp": [],
        "attn_de": [] ,
        "mlp_de": [],
        "attn_last_de": [],
        "mlp_last_de": []
    }
    
data = data[:300]

for it , item in tqdm(enumerate(data)):
    text_a = f'''Answer this question based on your knowledge and given information. Just choose only one option strictly and give only the letter as answer.

Question: {item['question']}

Options: 
A: {item['options'][0]}
B: {item['options'][1]}

Information: {item['counter_memory']}

Correct Option:'''

    text_b = f'''Answer this question based on your knowledge and given information. Just choose only one option strictly and give only the letter as answer.

Question: {item['question']}

Options: 
A: {item['options'][0]}
B: {item['options'][1]}

Information: {item['entity_substituted_counter_memory']}

Correct Option:'''
    d, inputs_a, inputs_b = token_difference(tokenizer, text_a, text_b)
    # a_sents.append(inputs_a)
    # b_sents.append(inputs_b)
    # difference.append(d)
    activations_a, activations_b = {}, {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(save_hook(f"layer_{i}_attn", activations_a)))
        hooks.append(layer.mlp.register_forward_hook(save_hook(f"layer_{i}_mlp", activations_a)))
    with torch.no_grad(): 
        model(**inputs_a)
    for h in hooks: 
        h.remove()

    # Save activations for B
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(save_hook(f"layer_{i}_attn", activations_b)))
        hooks.append(layer.mlp.register_forward_hook(save_hook(f"layer_{i}_mlp", activations_b)))
    with torch.no_grad(): 
        model(**inputs_b)
    for h in hooks: 
        h.remove()
        
    with torch.no_grad():
        logits_b = model(**inputs_b).logits
        prob_b = get_probs_over_tokens(logits_b, 319, 350)  #A

        logits_a = model(**inputs_a).logits
        prob_a = get_probs_over_tokens(logits_a, 319, 350)  #B
    
    # print(prob_a)
    
    pr = prob_a[350]
    pr_ = prob_a[319]
    # print(prob_b)
    # print(activations_a)
    # print(activations_b)
    # break
    diff_last = [inputs_b["input_ids"].shape[1] - 1]
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        for component, pos_set, store_ide , store_de in [
            ("attn", d, attn_all_effects, attn_all_effects_de),
            ("attn_last", diff_last, attn_last_effects, attn_last_effects_de),
            ("mlp", d, mlp_all_effects, mlp_all_effects_de),
            ("mlp_last", diff_last, mlp_last_effects, mlp_last_effects_de),
        ]:
            hooks = []
            layer = model.model.layers[i]
            if "attn" in component:
                hooks.append(getattr(layer, "self_attn").register_forward_hook(
                    make_swap_hook(i, "attn", pos_set, activations_b)
                ))
            else:
                hooks.append(getattr(layer, "mlp").register_forward_hook(
                    make_swap_hook(i, "mlp", pos_set, activations_b)
                ))
            with torch.no_grad():
                logits_swap = model(**inputs_a).logits
                prob_swap = get_probs_over_tokens(logits_swap, 319, 350)
                pzr = prob_swap[350]
                pzr_ = prob_swap[319]

            ide = 0.5*((pzr - pr)/pr + (pr_ - pzr_)/pzr_)
            store_ide.append(ide)
            de = pr - pzr
            store_de.append(de)
            final[i][component].append(ide)
            final[i][component + "_de"].append(de)
            
            for h in hooks: 
                h.remove()
                
with open("change_info_pos_ent_sub_to_counter.json", "w") as f:
    json.dump(final, f, indent=2)


    

