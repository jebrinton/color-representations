import yaml
import json
from pathlib import Path

class Config:
    """Config loader supporting YAML or JSON."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        if self.path.suffix in [".yaml", ".yml"]:
            with open(self.path, "r") as f:
                self.cfg = yaml.safe_load(f)
        elif self.path.suffix == ".json":
            with open(self.path, "r") as f:
                self.cfg = json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml or .json")

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    # -------------------------
    # Standard fields
    # -------------------------
    @property
    def model_name(self):
        return self.cfg["model_name"]

    @property
    def api_key(self):
        return self.cfg["api_key"]

    @property
    def device(self):
        return self.cfg.get("device", "auto")

    @property
    def save_dir(self):
        return self.cfg.get("save_dir", "outputs")

    # -------------------------
    # Dataset config block
    # -------------------------
    def get_dataset_config(self):
        """
        Returns dataset configuration block:
            dataset:
                type: synthetic
                path: data/synthetic.json
        """
        return self.cfg.get("dataset", {})

    # -------------------------
    # Experiment parameters
    # -------------------------
    def get_experiment(self):
        return self.cfg.get("experiment", {})
