import os
import json
import torch
from datetime import datetime, timezone
from pathlib import Path
from config import STEERING_VECTORS_DIR

def save_steering_vector(vector, dataset, model, color, layer, training_samples, base_path=STEERING_VECTORS_DIR):
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


def load_steering_vector(dataset, model, color, layer, base_path=STEERING_VECTORS_DIR):
    vector_path = Path(base_path) / dataset / model / color / f"{layer}.pt"
    metadata_path = Path(base_path) / dataset / model / color / f"{layer}.json"
    
    vector = torch.load(vector_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return vector, metadata
