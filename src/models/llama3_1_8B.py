from nnsight import LanguageModel, CONFIG
from .base_model import BaseModel

class Llama3_1_8B(BaseModel):
    """Wrapper for Meta-LLaMA 3.1-8B."""

    def __init__(self, api_key: str, device: str = "auto"):
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
    


