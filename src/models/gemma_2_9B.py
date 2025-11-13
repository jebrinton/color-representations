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
            # I dont know why this is happening, these layers are not associated with outout so cant get activations. 
            # Need to look more but just adding a bandaid for now. 
            ignore = ["act_fn", "input_layernorm", "post_attention_layernorm", "rotary_emb", "pre_feedforward_layernorm"] 

            activations = {}
            with self.model.trace(text):
                
                for name, module in self.model.model.named_modules():
                    # print(name)
                    if len(list(module.children())) == 0 and not any(term in name for term in ignore):
                        tensor = module.output[0]
                        activations[name] = tensor.detach().clone().cpu()
                
                out = self.model.lm_head.output.argmax(dim=-1).save()
                print(text, out)


            output_text = self.model.tokenizer.decode(out[0])

            print(output_text)

            return activations, output_text
