from nnsight import LanguageModel, CONFIG
from .base_model import BaseModel

class Gemma2_9B(BaseModel):
    """Wrapper around Gemma 2-9B"""

    def __init__(self, api_key: str, device: str = "auto"):
        CONFIG.set_default_api_key(api_key)
        CONFIG.APP.DEBUG = False
        self.device = device
        self.model = None

    def load(self):
        self.model = LanguageModel("google/gemma-2-9b", device_map=self.device)
        return self

    def get_hidden_activations(self, text: str):
        ignore = ["act_fn", "input_layernorm", "post_attention_layernorm"]
        activations = {}
        with self.model.trace(text):
            for name, module in self.model.model.named_modules():
                if len(list(module.children())) == 0 and not any(term in name for term in ignore):
                    activations[name] = module.output[0]
        return activations
