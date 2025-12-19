import os
import json
import torch
from datetime import datetime, timezone
from pathlib import Path

def save_steering_vector(vector, dataset, model, color, layer, training_samples, base_path):
    vector_dir = Path(base_path) / dataset / model / color
    vector_dir.mkdir(parents=True, exist_ok=True)
    
    vector_path = vector_dir / f"{layer}.pt"
    torch.save(vector, vector_path)
    
    metadata = {
        "vector_id": f"{dataset}_{model}_{color}_{layer}",
        "norm": float(vector.norm()),
        "training_samples": training_samples,
        "dataset": dataset,
        "model": model,
        "color": color,
        "layer": layer,
        "created_utc": datetime.now(timezone.utc)
    }
    
    metadata_path = vector_dir / f"{layer}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_steering_vector_from_dataset(dataset, model, color, layer, base_path):
    vector_path = Path(base_path) / dataset / model / color / f"{layer}.pt"
    metadata_path = Path(base_path) / dataset / model / color / f"{layer}.json"
    
    vector = torch.load(vector_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return vector, metadata


def load_steering_vector(sv_dir: str, color: str, layer_num: int) -> torch.Tensor:
    """
    Loads a steering vector from a specific file path.

    Args:
        sv_dir: The base directory for steering vectors.
        color: The color of the steering vector (e.g., "Red").
        layer_num: The layer number of the steering vector.

    Returns:
        The loaded steering vector as a torch.Tensor.
    """
    # Construct the full file path
    file_path = os.path.join(sv_dir, color, f"{layer_num}.pt")
    
    try:
        steering_vector = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
        return steering_vector
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
