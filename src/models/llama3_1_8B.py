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
        self.model = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device_map=self.device)
        return self

    def get_hidden_activations(self, text: str):
        # I dont know why this is happening, these layers are not associated with outout so cant get activations. 
        # Need to look more but just adding a bandaid for now. 
        ignore = ["act_fn", "input_layernorm", "post_attention_layernorm", "rotary_emb"] 

        activations = {}
        with self.model.trace(text):
            out = self.model.output.save()
            for name, module in self.model.model.named_modules():
                # print(name)
                if len(list(module.children())) == 0 and not any(term in name for term in ignore):
                    tensor = module.output[0]
                    activations[name] = tensor.detach().clone().cpu()
        
        # decode next-token text output
        logits = out["logits"]                     # [1, seq_len, vocab]
        final_logits = logits[0, -1]               # last step
        token_id = final_logits.argmax().item()
        output_text = self.model.tokenizer.decode(token_id).strip()

        return activations, output_text
    


