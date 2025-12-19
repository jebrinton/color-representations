from nnsight import LanguageModel, CONFIG
from .base_model import BaseModel
import torch

class Llama3_1_8B(BaseModel):
    """Wrapper for Meta-LLaMA 3.1-8B."""

    def __init__(self, api_key: str, device: str = "cuda"):
        CONFIG.set_default_api_key(api_key)
        CONFIG.APP.DEBUG = False
        self.device = device
        self.model = None

    def load(self):
        self.model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map=self.device)
        return self

    def get_hidden_activations(self, text: str):
        activations = {}
        with self.model.trace(text):
            for i, layer in enumerate(self.model.model.layers):
                activations[str(i)] = layer.output.detach().clone().cpu()
            
            out = self.model.lm_head.output.argmax(dim=-1).save()
            # print(text, out)
        # print(activations.keys())

        output_text = self.model.tokenizer.decode(out[0][-1])

        # print(output_text)

        return activations, output_text
    
    # def steer(
    #     self,
    #     text: str,
    #     layer_idx: int,
    #     steering_vector: torch.Tensor,
    #     alpha: float = 1.0,
    # ):
    #     """
    #     Apply a steering vector at a specific layer.
    #     Args:
    #         text: input prompt
    #         layer_idx: transformer layer to steer
    #         steering_vector: [hidden_dim] or [1, 1, hidden_dim]
    #         alpha: steering strength
    #     Returns:
    #         output_text: model output after steering
    #     """
    #     target_device = torch.device(self.device)
    #     steering_vector = steering_vector.to(target_device)
        
    #     # Ensure correct shape [1, 1, hidden_dim]
    #     if steering_vector.dim() == 1:
    #         steering_vector = steering_vector.view(1, 1, -1)
    #     elif steering_vector.dim() == 2:
    #         steering_vector = steering_vector.unsqueeze(0)
        
    #     print(f"Steering vector shape: {steering_vector.shape}")
    #     print(f"Alpha: {alpha}")
    #     # alpha = alpha * 10
        
    #     with self.model.trace(text):
    #         # Get the layer output
    #         layer_output = self.model.model.layers[layer_idx].output
            
    #         # Save BEFORE modification
    #         original = layer_output.save()
            
    #         # Apply steering - create new tensor, don't modify in place
    #         steered_output = layer_output + alpha * steering_vector
            
    #         # Replace the output
    #         self.model.model.layers[layer_idx].output = steered_output
            
    #         # Save AFTER modification
    #         modified = self.model.model.layers[layer_idx].output.save()
            
    #         # Get final output
    #         token_ids = self.model.lm_head.output.argmax(dim=-1).save()
        
    #     # Compute delta outside trace context
    #     delta = modified - original
    #     print(f"Layer output shape: {original.shape}")
    #     print(f"Mean delta: {delta.mean().item():.6f}")
    #     print(f"Max delta: {delta.max().item():.6f}")
    #     print(f"Min delta: {delta.min().item():.6f}")
    #     print(f"Delta std: {delta.std().item():.6f}")
        
    #     return self.model.tokenizer.decode(token_ids[0, -1])

    #     def steer(
    #     self,
    #     text: str,
    #     layer_idx: int,
    #     steering_vector: torch.Tensor,
    #     alpha: float = 1.0,
    #     max_new_tokens: int = 20,
    # ):
    #     """Apply steering and generate text."""
        
    #     # Prepare steering vector
    #     steering_vector = steering_vector.to(self.device).view(1, 1, -1)
        
    #     # Tokenize input
    #     input_ids = self.model.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
    #     # Generate with steering
    #     with self.model.generate(max_new_tokens=max_new_tokens) as generator:
    #         with generator.invoke(input_ids):
    #             h = self.model.model.layers[layer_idx].output
    #             self.model.model.layers[layer_idx].output = h + alpha * steering_vector
        
    #     # Decode output
    #     output = self.model.generator.output
    #     return self.model.tokenizer.decode(output[0], skip_special_tokens=True)
    def steer(
        self,
        text: str,
        layer_idx: int,
        steering_vector,
        alpha: float,
        max_new_tokens: int = 30,
    ):
        with self.model.generate(text, max_new_tokens=max_new_tokens) as tracer:

            # Apply steering at every generation step
            with tracer.all():
                h = self.model.model.layers[layer_idx].output
                self.model.model.layers[layer_idx].output = h + alpha * steering_vector

            # Capture generated tokens
            token_ids = self.model.generator.output.save()

        return self.model.tokenizer.decode(token_ids[0])

    


