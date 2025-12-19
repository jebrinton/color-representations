import os
import torch
import numpy as np
import regex as re
import itertools
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

from nnsight import LanguageModel

TRACER_KWARGS = {'scan': False, 'validate': False}

def save_steering_results(
    results: np.ndarray,
    metadata: Dict,
    output_dir: str,
    filename: str = "steering_results"
) -> Tuple[str, str]:
    """
    Save steering results as NumPy array with JSON metadata.
    
    Args:
        results: Array of shape [n_sv_pairs, n_objects, n_alpha, n_beta] 
                 containing hex codes as strings
        metadata: Dict with keys: sv_pairs, objects, alpha_values, beta_values
        output_dir: Directory to save files
        filename: Base filename (without extension)
    
    Returns:
        Tuple of (array_path, metadata_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    array_path = output_path / f"{filename}.npy"
    metadata_path = output_path / f"{filename}_metadata.json"
    
    np.save(array_path, results)
    
    # Convert numpy types to native Python types for JSON
    json_metadata = {
        "sv_pairs": metadata["sv_pairs"],
        "objects": metadata["objects"],
        "alpha_values": [float(x) for x in metadata["alpha_values"]],
        "beta_values": [float(x) for x in metadata["beta_values"]],
        "shape": list(results.shape),
        "dtype": str(results.dtype)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(json_metadata, f, indent=2)
    
    return str(array_path), str(metadata_path)


def load_steering_results(
    array_path: str,
    metadata_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Load steering results from NumPy array and JSON metadata.
    
    Args:
        array_path: Path to .npy file
        metadata_path: Path to _metadata.json file (auto-detected if None)
    
    Returns:
        Tuple of (results_array, metadata_dict)
    """
    results = np.load(array_path, allow_pickle=True)
    
    if metadata_path is None:
        metadata_path = str(Path(array_path).with_suffix('')) + "_metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return results, metadata


def make_prompts(objects: list[str]) -> list[str]:
    # hexes = "#D53A7A, #EFA1BB, #6D5880, #0A47DB, #6CA91D, #4FB1AA, #A226D6, #4AF7E7, #AAADE6, and #5FA716"
    # hexes = "#D53A7A and #EFA1BB"
    prompts = []
    for object in objects:
        # prompts.append(f"""
        #     Answer in the format #xxxxxx where x is a hex digit. Only output the RGB value, nothing else.
        #     For example:
        #     What is the RGB hex code of the color red?
        #     Answer: #ff0000
        #     What is the RGB hex code of the color blue?
        #     Answer: #0000ff
        #     What is the RGB hex code of the color green?
        #     Answer: #00ff00
        #     What is the RGB hex code of the color yellow?
        #     Answer: #ffff00
        #     What is the RGB hex code of the color purple?
        #     Answer: #800080
        #     What is the RGB hex code of the color orange?
        #     Answer: #ffa500
        #     What is the RGB hex code of {object}?
        #     """)
        prompts.append(f"""
            Answer in the format #xxxxxx where x is a hex digit. Only output the RGB value, nothing else.
            What is the color of {object}?
            Answer: #
        """)
    return prompts

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

def steer_generate(model, prompts, layer_num, steering_vector, alpha, steering_vector_2=None, beta=None):
    # padding on left in case you batch this in the future (so that -1 is still the last token)
    input_ids = model.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")["input_ids"] # {input_ids: [batch_size (1), seq_len], attention_mask: [batch_size (1), seq_len]}
    input_len = input_ids.shape[1]

    with model.generate(input_ids, max_new_tokens=7, pad_token_id=model.tokenizer.eos_token_id) as tracer:
        hidden = model.model.layers[layer_num].output.save() # {batch_idx: [seq_len, hidden_size]}
        hidden[:, :] += alpha * steering_vector
        if steering_vector_2 is not None and beta is not None:
            hidden[-10:-5, :] += beta * steering_vector_2
        generation = model.generator.output.save() # [batch_size, generated_seq_len]

    hex_codes = []
    for i in range(generation.shape[0]):
        gen = model.tokenizer.decode(generation[i][input_len:])
        match = re.search(r"([0-9a-fA-F]{6})", gen)
        hex_code = match.group(1) if match else None
        hex_codes.append(hex_code)
    return hex_codes


def main():
    sv_dir = "/projectnb/cs599m1/projects/color-representations/outputs/steering_vectors_predict/regular_templates/fundamental_colors"
    output_dir = "/projectnb/cs599m1/projects/color-representations/outputs/steering_results"
    
    model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto", dtype=torch.bfloat16)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.generation_config.pad_token_id = model.tokenizer.eos_token_id
    layer_num = 31

    objects = ["fresh milk", "white", "glass", "an apple", "BLORGARs", "an emotion", "a sunset"]
    prompts = make_prompts(objects)

    min_alpha = -0.6
    max_alpha = 2.0
    step_size = 0.2
    my_range = np.arange(min_alpha, max_alpha+step_size, step_size)
    alpha_values = my_range.tolist()
    beta_values = my_range.tolist()

    sv_colors = [" Green", " Orange"]
    sv_pairs = []
    for sv_color_1, sv_color_2 in itertools.combinations(sv_colors, 2):
        sv_pairs.append(f"{sv_color_1}_{sv_color_2}")
    
    # Initialize results array: [n_sv_pairs, n_objects, n_alpha, n_beta]
    n_sv_pairs = len(list(itertools.combinations(sv_colors, 2)))
    n_objects = len(objects)
    n_alpha = len(alpha_values)
    n_beta = len(beta_values)
    
    results = np.empty((n_sv_pairs, n_objects, n_alpha, n_beta), dtype=object)
    
    counter = 0
    sv_pair_idx = 0
    for sv_color_1, sv_color_2 in tqdm(itertools.combinations(sv_colors, 2), desc="SV pairs", position=0, colour="green"):
        sv_1 = load_steering_vector(sv_dir, sv_color_1, layer_num)
        sv_2 = load_steering_vector(sv_dir, sv_color_2, layer_num)
        
        for alpha_idx, alpha in tqdm(enumerate(alpha_values), desc="Alpha values", leave=False, position=1, colour="blue"):
            for beta_idx, beta in tqdm(enumerate(beta_values), desc="Beta values", leave=False, position=2, colour="yellow"):
                counter += 1
                # print(f"{sv_color_1} * {alpha} + {sv_color_2} * {beta}")
                hex_codes = steer_generate(model, prompts, layer_num, sv_1, alpha, sv_2, beta)
                # print(hex_codes)
                
                # Store hex codes for all objects at this (alpha, beta)
                for obj_idx, hex_code in enumerate(hex_codes):
                    results[sv_pair_idx, obj_idx, alpha_idx, beta_idx] = hex_code
        
        sv_pair_idx += 1
    
    print(f"Total number of steers: {counter}")
    
    # Save results
    metadata = {
        "sv_pairs": sv_pairs,
        "objects": objects,
        "alpha_values": alpha_values,
        "beta_values": beta_values
    }
    
    array_path, metadata_path = save_steering_results(results, metadata, output_dir, filename="prefill_dreams31")
    print(f"Saved results to {array_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()