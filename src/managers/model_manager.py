from src.models.llama3_1_8B import Llama3_1_8B
from src.models.gemma_2_9B import Gemma2_9B

class ModelManager:

    def __init__(self, config):
        self.config = config
        self.model = None

    def load_model(self):
        name = self.config.model_name.lower()

        # Add more models if we decide to experiment on different families
        if "llama" in name:
            self.model = Llama3_1_8B(self.config.api_key, self.config.device).load()
        elif "gemma" in name:
            self.model = Gemma2_9B(self.config.api_key, self.config.device).load()
        else:
            raise ValueError(f"Unknown model name: {name}")

        return self.model
