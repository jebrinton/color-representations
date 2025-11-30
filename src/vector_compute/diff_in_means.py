import torch
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np

class DiffInMeansCompute:
  
    def __init__(self, vector_manager, save_dir="outputs/steering_vectors"):
        self.vector_manager = vector_manager
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def get_and_save_predictions(self, model, sents, template_type, color_type, activation_pool="mean"):

        # Directory structure: outputs/steering_vectors/<template_type>/<color_type>/
        root_dir = self.save_dir / template_type / color_type
        root_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Predicting colors for {len(sents)} prompts...")

        # Compute the steering vectors from predictions
        steering_vecs = self._compute_for_predictions(model, sents, activation_pool)

        index = {}

        # Save vectors for each predicted color
        for color_name, layer_dict in steering_vecs.items():
            color_dir = root_dir / color_name
            color_dir.mkdir(parents=True, exist_ok=True)

            index[color_name] = {}

            for layer_name, vec in layer_dict.items():
                file_path = color_dir / f"{layer_name.replace('/', '_')}.pt"
                torch.save(vec, file_path)
                index[color_name][layer_name] = str(file_path.resolve())

        # Save index.json
        index_path = root_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[INFO] Saved predicted steering vectors to: {root_dir.resolve()}")
        return index

    def _average_dicts(self, dict_list, activation_pool):
        acc = defaultdict(list)

        # Collect all vectors per key
        for d in dict_list:
            for k, v in d.items():
                mean_v = self._pool_activation(v, activation_pool)
                acc[k].append(mean_v)
                # print(mean_v.shape)

        # Compute averages
        return {k: torch.tensor(np.mean(acc[k], axis=0)) for k in acc}

    def get_and_save(self, model, color_sentences, template_type, color_type, activation_pool="mean"):

        root_dir = self.save_dir / template_type / color_type
        root_dir.mkdir(parents=True, exist_ok=True)

        color_acts = {}
        for color, samples in tqdm(color_sentences.items(), desc=f"Computing {template_type}/{color_type}"):
            sent_acts = []
            for text in tqdm(samples, desc="All prompts", leave=False):
                acts, _ = model.get_hidden_activations(text)
                sent_acts.append(acts)
            
            avg_sent_dict = self._average_dicts(sent_acts, activation_pool)
            
            color_acts[color] = avg_sent_dict

        steering_vecs = {}
        all_colors = list(color_acts.keys())

        # print(all_colors)

        for target_color in all_colors:
            
            layer_vecs = {}
            pos_examples = [color_acts[target_color]]
            neg_examples = []
            for color, exs in color_acts.items():
                if color != target_color:
                    neg_examples.append(color_acts[color])
            
            
            layer_names = pos_examples[0].keys()
            
            
            for layer_name in layer_names:
                # print(neg_examples[0].keys())
                pos_tensor = torch.stack([ex[layer_name] for ex in pos_examples])
                neg_tensor = torch.stack([ex[layer_name] for ex in neg_examples])

                layer_vecs[layer_name] = self.vector_manager.diff_in_means(
                    pos_tensor, neg_tensor
                )

            steering_vecs[target_color] = layer_vecs

        index = {}

        # Save vectors for each predicted color
        for color_name, layer_dict in steering_vecs.items():
            color_dir = root_dir / color_name
            color_dir.mkdir(parents=True, exist_ok=True)

            index[color_name] = {}

            for layer_name, vec in layer_dict.items():
                file_path = color_dir / f"{layer_name.replace('/', '_')}.pt"
                torch.save(vec, file_path)
                index[color_name][layer_name] = str(file_path.resolve())

        # Save index.json
        index_path = root_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"[INFO] Saved predicted steering vectors to: {root_dir.resolve()}")
        return index
                

            

    def _compute_for_predictions(self, model, sents, activation_pool="mean"):
        color_acts = {}
        for text in tqdm(sents, desc="All prompts", leave=False):
            acts, output_text = model.get_hidden_activations(text)
            prompt_acts = {}

            for name, tensor in acts.items():
                tensor = self._pool_activation(tensor, activation_pool)
                prompt_acts.setdefault(name, []).append(tensor)

            color_acts.setdefault(output_text, []).append(prompt_acts)

        steering_vecs = {}
        all_colors = list(color_acts.keys())

        print(all_colors)

        for target_color in all_colors:
            
            layer_vecs = {}
            pos_examples = color_acts[target_color]
            neg_examples = [
                ex for color, exs in color_acts.items()
                if color != target_color
                for ex in exs
            ]
            if len(pos_examples) == 0 or len(neg_examples) == 0:
                continue 

            for layer_name in pos_examples[0].keys():
                pos_tensor = torch.stack([ex[layer_name][0] for ex in pos_examples])
                neg_tensor = torch.stack([ex[layer_name][0] for ex in neg_examples])

                layer_vecs[layer_name] = self.vector_manager.diff_in_means(
                    pos_tensor, neg_tensor
                )

            steering_vecs[target_color] = layer_vecs

        return steering_vecs

    

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


