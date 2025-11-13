import os
import torch


def load_steering_vector(sv_dir: str, color: str, module_name: str) -> torch.Tensor:
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
    file_path = os.path.join(sv_dir, color, module_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: Steering vector file not found at: {file_path}")
        # In a real application, you might raise an exception
        # raise FileNotFoundError(f"Steering vector file not found at: {file_path}")
        return None

    print(f"Loading steering vector from: {file_path}")
    
    # Load the tensor from the .pt file
    # We map it to 'cpu' by default; you can change this if needed
    try:
        steering_vector = torch.load(file_path, map_location=torch.device('cpu'))
        return steering_vector
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def steer_model(model, steering_vector, token_ids):
    """
    Steers the model by adding the steering vector to the last token position.
    """
    prompt = 'The Eiffel Tower is in the city of'
    n_new_tokens = 3
    with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
        out = model.generator.output.save()

    decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
    decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

    print("Prompt: ", decoded_prompt)
    print("Generated Answer: ", decoded_answer)


def main():
    sv_dir = "outputs/templates/fundamental_colors"
    color = "red"
    module_name = "model.model.layers.3.mlp.up_proj.pt"
    steering_vector = load_steering_vector(sv_dir, color, module_name)
    print(steering_vector)

if __name__ == "__main__":
    main()

    # steer only at last token position