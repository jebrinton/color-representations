import os
import json
import logging
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


def get_device_info():
    """
    Determina el mejor dispositivo disponible para PyTorch.
    
    Returns:
        tuple: (device, dtype) - dispositivo y tipo de datos recomendado
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus == 1:
            device = "cuda"
            dtype = torch.bfloat16
        else:
            task_id = int(os.environ.get("SGE_TASK_ID", 1))
            gpu_id = (task_id - 1) % n_gpus
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    return device, dtype


def setup_model(model_id):
    """
    Configura el modelo de lenguaje y tokenizer.

    Args:
        model_id: ID del modelo en HuggingFace

    Returns:
        tuple: (model, submodule, tokenizer)
    """
    device, dtype = get_device_info()

    # Load language model
    model = LanguageModel(model_id, torch_dtype=dtype, device_map=device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def save_json(data, filepath):
    """
    Guarda datos en formato JSON.
    
    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Carga datos desde un archivo JSON.
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        dict: Datos cargados
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(directory):
    """
    Asegura que un directorio existe.
    
    Args:
        directory: Ruta del directorio
    """
    os.makedirs(directory, exist_ok=True)


def setup_logging(log_dir, log_filename="unnamed.log"):
    """
    Configura el sistema de logging.
    
    Args:
        log_dir: Directorio donde guardar el archivo de log
        log_filename: Nombre del archivo de log
    """
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
