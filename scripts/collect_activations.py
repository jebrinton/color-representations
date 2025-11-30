import os
import sys
import argparse

from colorrep.utils import setup_model, get_device_info

import torch
import json

from tqdm import tqdm

def get_color_token_ids_to_string(colorlist, tokenizer):
    # Create a map from a token_id -> color_string for fast lookups.
    # We check IDs with and without a preceding space, as the model might
    # generate either.
    color_token_ids_to_string = {}
    for color in colorlist:
        # Check without space
        token_ids_no_space = tokenizer(color, add_special_tokens=False).input_ids
        if len(token_ids_no_space) == 1:
            color_token_ids_to_string[token_ids_no_space[0]] = color
        
        # Check with space (more common for generation)
        token_ids_with_space = tokenizer(f" {color}", add_special_tokens=False).input_ids
        if len(token_ids_with_space) == 1:
            color_token_ids_to_string[token_ids_with_space[0]] = color
    return color_token_ids_to_string

def main(model_id, prompts_path, objects_path, colorlist_path, output_dir):
    device, dtype = get_device_info()
    model, tokenizer = setup_model(model_id)

    prompt_templates = json.load(open(prompts_path))
    objects = json.load(open(objects_path))
    colorlist = json.load(open(colorlist_path))

    # create a list of prompts, using the prompt templates and the colors
    prompts = [prompt_template.format(object=object) for prompt_template in prompt_templates for object in objects]

    # Get model configuration
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size 

    sum_acts = {}
    counts = {}

    # initialize dictionaries to computer mean activations
    for color in colorlist:
        sum_acts[color] = torch.zeros(num_layers, hidden_size, device=device, dtype=dtype)
        counts[color] = torch.zeros(num_layers, device=device, dtype=torch.int16)
    
    color_token_ids_to_string = get_color_token_ids_to_string(colorlist, tokenizer)

    print(f"Found {len(color_token_ids_to_string)} unique single-token IDs for {len(colorlist)} colors.")

    batch_size = 16
    layers = model.model.layers
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts"):
        batch_prompts = prompts[i:i + batch_size]
        
        # We want the *activations from the last token of the prompt*
        # and the *first generated token*

        acts = {}
        with model.generate(
            batch_prompts,
            max_new_tokens=1,
        ) as tracer:

            for layer_idx in range(num_layers):
                acts[layer_idx] = layers[layer_idx].output[:, -1, :].save()

            # save model generation
            generation = model.generator.output.save()

        
        # acts is a dictionary of layer_idx -> tensor of shape [batch_size, hidden_size]
        print(len(acts))
        print(acts[0].shape)

        print(type(generation))
        print(generation.shape)

        # Iterate through the batch results
        for j in range(len(batch_prompts)):
            token_id = tracer.output.generation.sequences[j, -1].item()

            # Check if this token ID is one of our target colors
            if token_id in color_token_ids_to_string:
                color = color_token_ids_to_string[token_id]
                
                # This prompt's generation was a valid color.
                # Add the saved prompt activations to the sum.
                for layer_idx in range(num_layers):
                    # .value gives the tensor we saved
                    # [j] indexes into the batch
                    activation_for_this_item = prompt_activations[layer_idx].value[j]
                    sum_acts[color][layer_idx] += activation_for_this_item
                
                # Increment the count for this color (element-wise for the tensor)
                counts[color] += 1

    # --- Calculate means and save ---
    mean_acts = {}
    print("\n--- Summary ---")
    for color in colorlist:
        # Get the count from any layer (they are all the same)
        total_count = counts[color][0].item()
        
        if total_count > 0:
            mean_acts[color] = sum_acts[color] / total_count
            print(f"Color '{color}': Found {total_count} examples.")
        else:
            mean_acts[color] = None # Or keep the zero tensor
            print(f"Color '{color}': Found 0 examples.")

    # Save the dictionary of mean activation tensors
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_activations.pt")
    
    # Filter out colors with no data before saving
    final_mean_acts = {color: tensor for color, tensor in mean_acts.items() if tensor is not None}
    
    torch.save(final_mean_acts, save_path)
    print(f"\nSaved mean activations to {save_path}")

    



if __name__ == "__main__":
    # parse arguments for prompt template file, colors list file, output dir, model id
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.json")
    parser.add_argument("--objects_path", type=str, default="data/objects.json")
    parser.add_argument("--colorlist_path", type=str, default="data/colorlist.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(**args.__dict__)