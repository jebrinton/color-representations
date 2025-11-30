import os
import torch

from nnsight import LanguageModel


def load_steering_vector(sv_dir: str, color: str, layer_num: int) -> torch.Tensor:
    """
    Loads a steering vector from a specific file path.

    Args:
        sv_dir: The base directory for steering vectors (e.g., "SV_DIR").
        color: The subdirectory for the vector type (e.g., "red").
        module_name: The name of the module, which is also the filename
                     (e.g., "model.model.layers.3.mlp.up_proj.pt").

    Returns:
        The loaded steering vector as a torch.Tensor.
    """
    # Construct the full file path
    file_path = os.path.join(sv_dir, color, f"{layer_num}.pt")

    # Check if the file exists
    if not os.path.exists(file_path):
        return FileNotFoundError(f"Steering vector file not found at: {file_path}")

    print(f"Loading steering vector from: {file_path}")
    
    # Load the tensor from the .pt file
    # We map it to 'cpu' by default; you can change this if needed
    try:
        steering_vector = torch.load(file_path, map_location=torch.device('cuda'))
        return steering_vector
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def steer_model(model, prompt, steering_vector, layer_num, alpha, token_pos=-1):
    # TODO: change prompt to be a dataloader
    # for layer in model.config.n_layers:

    n_new_tokens = 3
    with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
        hidden_layer = model.model.layers[layer_num].output
        hidden_layer += alpha * steering_vector

        out = model.generator.output.save()

    decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
    decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

    print("Prompt: ", decoded_prompt)
    print("Generated Answer: ", decoded_answer)

    return decoded_answer


def main():
    sv_dir = "/projectnb/cs599m1/projects/color-rep/color-representations/outputs_latest/steering_vectors_predict"
    color = "Red"
    layer_num = 16
    steering_vector = load_steering_vector(sv_dir, color, layer_num)

    model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device="auto")
    decoded = steer_model(model, "Answer in one word. The color of #FF0099 is: ", steering_vector=steer_model, layer_num=layer_num, token_pos=-1, alpha=1)
    print(decoded)


if __name__ == "__main__":
    main()

    # steer only at last token position