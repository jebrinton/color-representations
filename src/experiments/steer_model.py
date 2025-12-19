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
        self.layers = exp_cfg.get("layers", list(range(1, 31)))
        self.alphas = exp_cfg.get("alphas", list(range(-100, 101, 10)))
        self.output_dir = "/projectnb/ivc-ml/divsp/fall2025/CS599/color-representations/outputs"

        os.makedirs(self.output_dir, exist_ok=True)

        # self.device = self.model.model.device
        self.device = "cuda"
        print("Model device: ", self.device)
        self._vector_cache = {}

        # Fixed input sentence
        self.prompt = "Answer in one word. What is the color of apple?"


        # Infer colors from directory structure
        self.colors = sorted([
            d for d in os.listdir(self.vector_root)
            if os.path.isdir(os.path.join(self.vector_root, d))
        ])

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
        # allowed_colors = {
        #     " Black", " Blue", " Brown", " Gold", " Green",
        #     " Grey", " Orange", " Pink", " Purple", " Red", " Silver", " White", " Yellow"
        # }

        for color in self.colors:
            # if color not in allowed_colors:
            #     continue
            results[color] = {}

            for layer in self.layers:
                results[color][layer] = {}
                vec = self._load_vector(color, layer)

                for alpha in self.alphas:
                    # print(color, layer, alpha)
                    try:
                        output = self.model.steer(
                            text=self.prompt,
                            layer_idx=layer,
                            steering_vector=vec,
                            alpha=float(alpha),
                        )
                        print(color, layer, alpha, output)
                        results[color][layer][alpha] = output

                    except Exception as e:
                        results[color][layer][alpha] = f"[ERROR] {e}"
                

        self._save(results)
        return results

    def _save(self, results):
        out_path = os.path.join(self.output_dir, "steering_results.json")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] Steering results saved to {out_path}")


if __name__ == "__main__":
    exp = ColorSteeringExp("config/steering_config_contrast.yaml")
    exp.run()
