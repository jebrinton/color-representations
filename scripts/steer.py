import os
import torch

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


def steer_model(model, prompt, steering_vector, layer_num, alpha, token_positions=[-1]):
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



def main():
    sv_dir = "/projectnb/cs599m1/projects/color-representations/outputs/steering_vectors_predict/regular_templates/fundamental_colors"
    color = " Red"
    layer_num = 16

    steering_vector = load_steering_vector(sv_dir, color, layer_num)
    print(steering_vector.shape)

    model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="cuda", dtype=torch.bfloat16)

    # datase)
    for alpha in [-1.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        decoded = steer_model(model, "Answer in one word. What is the color of a blade of grass?", steering_vector=steering_vector, layer_num=layer_num, token_positions=[-1], alpha=alpha)
        print(f"Alpha: {alpha}")



if __name__ == "__main__":
    main()

    # steer only at last token position