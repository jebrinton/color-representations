import os
import torch
import numpy as np
import regex as re

from nnsight import LanguageModel

TRACER_KWARGS = {'scan': False, 'validate': False}


def load_steering_vector(sv_dir: str, color: str, layer_num: int) -> torch.Tensor:
    """
    Loads a steering vector from a specific file path.

    Args:
        sv_dir: The base directory for steering vectors.
        color: The color of the steering vector (e.g., "Red").
        layer_num: The layer number of the steering vector.

    Returns:
        The loaded steering vector as a torch.Tensor.
    """
    # Construct the full file path
    file_path = os.path.join(sv_dir, color, f"{layer_num}.pt")
    
    try:
        steering_vector = torch.load(file_path, map_location=torch.device('cuda'), weights_only=True)
        return steering_vector
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def steer_model_logits(model, prompt, steering_vector, layer_num, alpha, token_positions=[-1]):
    # TODO: change prompt to be a dataloader
    # for layer in model.config.n_layers:
    print(f"Steering vector shape: {steering_vector.shape}")

    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
    with model.trace(input_ids, scan=False, validate=False) as tracer: # TODO: add token_positions
        hidden_layer = model.model.layers[layer_num].output.save()
        hidden_layer += alpha * steering_vector

        logits_intervened = model.lm_head.output[0, -1].save()

    with model.trace(input_ids, scan=False, validate=False) as tracer:
        logits_clean = model.lm_head.output[0, -1].save()

    print(logits_intervened[0])
    print(logits_clean[0])
    # Calculate the difference in logits
    logit_diff = logits_intervened - logits_clean
    
    # Get specific token IDs for colors
    red_token_id = model.tokenizer.encode(" Red", add_special_tokens=False)[0]
    blue_token_id = model.tokenizer.encode(" Blue", add_special_tokens=False)[0]
    green_token_id = model.tokenizer.encode(" Green", add_special_tokens=False)[0]

    print(f"Logit change for ' Red': {logit_diff[red_token_id].item():.4f}")
    print(f"Logit change for ' Blue': {logit_diff[blue_token_id].item():.4f}")
    print(f"Logit change for ' Green': {logit_diff[green_token_id].item():.4f}")

    print(f"Prompt: {prompt}")
    print(f"Alpha: {alpha}")
    
    # Find tokens with biggest changes
    top_k = 10
    top_increases = torch.topk(logit_diff, k=top_k)
    top_decreases = torch.topk(-logit_diff, k=top_k)
    
    print(f"\n=== Top {top_k} tokens with increased logits ===")
    for i, (token_id, diff) in enumerate(zip(top_increases.indices, top_increases.values)):
        token_text = model.tokenizer.decode([token_id.item()])
        print(f"{i+1}. Token ID {token_id.item()}: '{token_text}' | Change: +{diff.item():.4f}")
    
    print(f"\n=== Top {top_k} tokens with decreased logits ===")
    for i, (token_id, diff) in enumerate(zip(top_decreases.indices, top_decreases.values)):
        token_text = model.tokenizer.decode([token_id.item()])
        print(f"{i+1}. Token ID {token_id.item()}: '{token_text}' | Change: -{diff.item():.4f}")
    
    return logits_intervened, logits_clean

def steer_generate(model, prompt, steering_vector, layer_num, alpha, token_positions=[-3, -2, -1]):
    # padding on left in case you batch this in the future (so that -1 is still the last token)
    input_ids = model.tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")["input_ids"] # {input_ids: [batch_size (1), seq_len], attention_mask: [batch_size (1), seq_len]}
    input_len = input_ids.shape[1]

    with model.generate(input_ids, max_new_tokens=17, pad_token_id=model.tokenizer.eos_token_id) as tracer:
        hidden = model.model.layers[layer_num].output.save() # {batch_idx: [seq_len, hidden_size]}
        hidden[-5:, :] += alpha * steering_vector
        generation = model.generator.output.save() # [batch_size, generated_seq_len]

    prompt0_prompt = model.tokenizer.decode(generation[0][:input_len])
    prompt0_generation = model.tokenizer.decode(generation[0][input_len:])
    # print(f"||>> {prompt0_generation}")
    match = re.search(r"#([0-9a-fA-F]{6})", prompt0_generation)
    hex_code = match.group(1) if match else None
    return hex_code


def main():
    sv_dir = "/projectnb/cs599m1/projects/color-representations/outputs/steering_vectors_predict/regular_templates/fundamental_colors"
    
    model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto", dtype=torch.bfloat16)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.generation_config.pad_token_id = model.tokenizer.eos_token_id
    layer_num = 31

    hexes = "#D53A7A, #EFA1BB, #6D5880, #0A47DB, #6CA91D, #4FB1AA, #A226D6, #4AF7E7, #AAADE6, and #5FA716"
    hexes = "#D53A7A and #EFA1BB"

    color = " Orange"
    steering_vector = load_steering_vector(sv_dir, color, layer_num)

    for color in ["_____", "of cherries", "of the sea", "of grass", "of a wall", "of ideas"]:
        prompt = f"""
        Answer in the format #xxxxxx where x is a hex digit. Only output the RGB value, nothing else.
        For example:
        What is the RGB hex code of the color red?
        Answer: #ff0000
        What is the RGB hex code of the color blue?
        Answer: #0000ff
        What is the RGB hex code of the color green?
        Answer: #00ff00
        What is the RGB hex code of the color yellow?
        Answer: #ffff00
        What is the RGB hex code of the color purple?
        Answer: #800080
        What is the RGB hex code of the color orange?
        Answer: #ffa500
        What is the RGB hex code of the color {color}?
        """
        max_alpha = 2.0
        step_size = 0.1

        hex_codes = {}
        for alpha in np.arange(-max_alpha, max_alpha+step_size, step_size):
            hex_codes[alpha] = steer_generate(model, prompt, steering_vector, layer_num, alpha=alpha, token_positions=[-1])
            # decoded = steer_model_logits(model, prompt, steering_vector=steering_vector, layer_num=layer_num, token_positions=[-1], alpha=alpha)

        print(f"Steering with {color} at layer {layer_num}")
        # print(f"Prompt: {prompt}")
        print(f"What is the RGB hex code of the color {color}?")
        for alpha, hex_code in hex_codes.items():
            print(f"Alpha: {alpha:3.2f}, Hex code: #{hex_code}")


if __name__ == "__main__":
    main()

    # steer only at last token position
