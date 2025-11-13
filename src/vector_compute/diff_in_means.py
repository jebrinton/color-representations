import torch
from pathlib import Path
import json
from tqdm import tqdm


class DiffInMeansCompute:
  
    def __init__(self, vector_manager, save_dir="outputs/steering_vectors"):
        self.vector_manager = vector_manager
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_and_save(self, model, contrastive_pairs, template_type, color_type, activation_pool="mean"):

        root_dir = self.save_dir / template_type / color_type
        root_dir.mkdir(parents=True, exist_ok=True)

        index = {}

        for color, samples in tqdm(contrastive_pairs.items(), desc=f"Computing {template_type}/{color_type}"):

            pos_sents = samples["positive"]
            neg_sents = samples["negative"]

            
            layer_steering = self._compute_for_color(
                model, pos_sents, neg_sents, activation_pool
            )

            color_dir = root_dir / color
            color_dir.mkdir(parents=True, exist_ok=True)

            index[color] = {}
            for layer, vec in layer_steering.items():
                file_path = color_dir / f"{layer.replace('/', '_')}.pt"
                torch.save(vec, file_path)
                index[color][layer] = str(file_path.resolve())

        index_path = root_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[INFO] Saved all steering vectors to: {root_dir.resolve()}")
        return index


    def _compute_for_color(self, model, pos_sents, neg_sents, activation_pool="mean"):

        pos_acts, neg_acts = {}, {}



        for text in tqdm(pos_sents, desc="Positive", leave=False):
            acts = model.get_hidden_activations(text)
            for name, tensor in acts.items():
                tensor = self._pool_activation(tensor, activation_pool)
                pos_acts.setdefault(name, []).append(tensor)

        
        for text in tqdm(neg_sents, desc="Negative", leave=False):
            acts = model.get_hidden_activations(text)
            for name, tensor in acts.items():
                tensor = self._pool_activation(tensor, activation_pool)
                neg_acts.setdefault(name, []).append(tensor)

        layer_vecs = {}
        for layer_name in pos_acts.keys():
            pos_tensor = torch.stack(pos_acts[layer_name])
            neg_tensor = torch.stack(neg_acts[layer_name])
            layer_vecs[layer_name] = self.vector_manager.diff_in_means(pos_tensor, neg_tensor)

        return layer_vecs


    @staticmethod
    def _pool_activation(tensor: torch.Tensor, mode: str = "mean") -> torch.Tensor:

        if tensor.dim() == 3:  # If its of shape : [batch, seq_len, hidden]
            tensor = tensor.squeeze(0)
        if mode == "mean":
            return tensor.mean(dim=0)
        elif mode == "last":
            return tensor[-1]
        else:
            raise ValueError(f"Invalid activation_pool mode: {mode}")
