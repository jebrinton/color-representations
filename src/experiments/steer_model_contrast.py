import os
import json
import torch

from src.colorep.config import Config
from src.managers.model_manager import ModelManager


class ColorSteeringExp:
    """
    Steers the model using precomputed vectors on a single fixed sentence.
    Sweeps color × layer × alpha and captures outputs.
    """

    def __init__(self, config_path="config/config.yaml"):
        self.config = Config(config_path)

        # Model
        self.model = ModelManager(self.config).load_model()

        # Experiment config
        exp_cfg = self.config.get_experiment()

        self.vector_root = self.config.save_dir
        self.layers = exp_cfg.get("layers", list(range(27, 28)))
        self.alphas = exp_cfg.get("alphas", list(range(-30, 30, 2)))
        self.output_dir = "/projectnb/ivc-ml/divsp/fall2025/CS599/color-representations/outputs"

        os.makedirs(self.output_dir, exist_ok=True)

        # self.device = self.model.model.device
        self.device = "cuda"
        print("Model device: ", self.device)
        self._vector_cache = {}

        # Fixed input sentence
        # self.prompts = ["The color of an apple is",
        #                 "The color of the sea is",
        #                 "The color of ideas is",
        #                 "The color of clear sky is",
        #                 "The color of iphone is",
        #                 "The color of music is",
        #                 "The color of butterfly is",
        #                 "The color of ripe banana is",
        #                 "The color of fresh snow is"]
        self.prompts = [
                        "The color of butterfly is",
                        "The color of ripe banana is",
                        "The color of fresh snow is"]


        # Infer colors from directory structure
        self.colors = sorted([
            d for d in os.listdir(self.vector_root)
            if os.path.isdir(os.path.join(self.vector_root, d))
        ])

        # self.colors = self.colors[:2]

    def _load_vector(self, color: str, layer: int) -> torch.Tensor:
        key = f"{color}:{layer}"

        if key in self._vector_cache:
            return self._vector_cache[key]

        path = os.path.join(self.vector_root, color, f"{layer}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        vec = torch.load(path, map_location="cpu")

        if vec.dim() > 1:
            vec = vec.squeeze()

        vec = vec / vec.norm()
        vec = vec.to(self.device)

        self._vector_cache[key] = vec
        return vec

    def run(self):
        results = {}
        for prompt_ in self.prompts:
            
            for color in self.colors:
                results[color] = {}

                for layer in self.layers:
                    results[color][layer] = {}
                    vec = self._load_vector(color, layer)

                    for alpha in self.alphas:
                        # print(color, layer, alpha)
                        try:
                            output = self.model.steer(
                                text=prompt_,
                                layer_idx=layer,
                                steering_vector=vec,
                                alpha=float(alpha),
                            )
                            print(color, layer, alpha, output)
                            results[color][layer][alpha] = output

                        except Exception as e:
                            print(e)
                            results[color][layer][alpha] = f"[ERROR] {e}"
                
            object_prompted = prompt_.split(" ")[-2]
            self._save(results, object_prompted)
        return results

    def _save(self, results, object_prompted):
        out_path = os.path.join(self.output_dir, f"color_of_{object_prompted}_contrast_sent.json")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] Steering results saved to {out_path}")


if __name__ == "__main__":
    exp = ColorSteeringExp("config/steering_config_contrast.yaml")
    exp.run()
